"""
spec5.instrument.galaxy_photometry — Fiber aperture-loss model for extended
(galaxy) sources.

Unlike stars, BGS galaxies are not point sources: only a fraction of their
total light falls within the spectrograph fiber. This module models that
fraction with a Sersic light profile and the standard (regularised
incomplete-gamma) curve-of-growth formula, optionally convolved with
seeing. Seeing is a circular Gaussian or Moffat PSF.
"""

from functools import lru_cache

import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.stats import ncx2

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Moffat index used by the DESI and Spec-S5 throughput budgets (DESI-347,
# SS5-0022) for atmospheric seeing.
MOFFAT_BETA = 3.5

_PSFS = ("gaussian", "moffat")


def _check_psf(psf):
    if psf not in _PSFS:
        raise ValueError(f"psf must be one of {_PSFS}, got {psf!r}")


def _moffat_alpha(seeing_fwhm, beta):
    """Moffat core radius for a given FWHM."""
    return seeing_fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def _psf_enclosed(radius, seeing_fwhm, psf, beta):
    """Fraction of a centred PSF's flux within ``radius`` (its radial CDF)."""
    if psf == "gaussian":
        sigma = seeing_fwhm * _FWHM_TO_SIGMA
        return 1.0 - np.exp(-0.5 * (radius / sigma) ** 2)
    alpha = _moffat_alpha(seeing_fwhm, beta)
    return 1.0 - (1.0 + (radius / alpha) ** 2) ** (1.0 - beta)


def fiber_capture_fraction(half_light_radius, fiber_diameter, sersic_n=1.0):
    """
    Fraction of a Sersic-profile galaxy's total flux captured within a
    circular fiber aperture.

    Parameters
    ----------
    half_light_radius : float or array-like
        Galaxy half-light (effective) radius, in arcsec. Non-positive or
        non-finite values are treated as a point source (fraction = 1).
    fiber_diameter : float
        Fiber angular diameter, in arcsec.
    sersic_n : float
        Sersic index of the light profile (default: 1.0, an exponential
        disk — typical of star-forming BGS galaxies). 4.0 would model a
        de Vaucouleurs / early-type profile instead.

    Returns
    -------
    float or ndarray
        Fraction of total flux captured by the fiber, in (0, 1].

    Notes
    -----
    For a Sersic profile, the flux enclosed within radius R is given by the
    regularised lower incomplete gamma function:

        f(R) = P(2n, b_n * (R / R_e)^(1/n))

    where ``b_n`` is defined by ``P(2n, b_n) = 0.5`` (i.e. R_e is the
    half-light radius by construction). This is the standard curve-of-growth
    formula for Sersic profiles (e.g. Graham & Driver 2005).
    """
    r_half = np.asarray(half_light_radius, dtype=float)
    point_source = ~np.isfinite(r_half) | (r_half <= 0)

    # Avoid division by zero for point-source entries; result is overwritten below.
    r_half_safe = np.where(point_source, 1.0, r_half)

    fiber_radius = fiber_diameter / 2.0
    bn = gammaincinv(2.0 * sersic_n, 0.5)
    x = bn * (fiber_radius / r_half_safe) ** (1.0 / sersic_n)
    frac = gammainc(2.0 * sersic_n, x)

    frac = np.where(point_source, 1.0, frac)
    return float(frac) if frac.ndim == 0 else frac


def point_source_capture_fraction(fiber_diameter, seeing_fwhm, psf="gaussian",
                                  beta=MOFFAT_BETA):
    """
    Fraction of a point source's flux within a centred circular fiber.

    ``fiber_diameter`` and ``seeing_fwhm`` are in arcsec. ``psf`` is
    ``'gaussian'`` (default) or ``'moffat'`` with index ``beta``.
    """
    _check_psf(psf)
    return _psf_enclosed(0.5 * fiber_diameter, seeing_fwhm, psf, beta)


def _ring_kernel(r, fiber_radius, seeing_fwhm, psf, beta):
    """
    P(light emitted at radius ``r`` from the fiber centre lands in the fiber
    after blurring by the PSF), for any circular PSF.

    Integrates the PSF over shells of radius s around the emission point,
    each weighted by the fraction of its circumference inside the fiber.
    """
    scale = seeing_fwhm if psf == "gaussian" else _moffat_alpha(seeing_fwhm, beta)
    s_edges = np.concatenate([[0.0], np.geomspace(1e-4 * scale, 300.0 * scale, 12000)])
    s_mid = np.sqrt(np.maximum(s_edges[:-1], 1e-6 * scale) * s_edges[1:])
    shell = np.diff(_psf_enclosed(s_edges, seeing_fwhm, psf, beta))

    r = np.asarray(r, dtype=float)
    out = np.empty(r.shape)
    for i in range(0, r.size, 64):   # chunked to bound memory
        rc = r[i:i + 64, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = (rc ** 2 + s_mid ** 2 - fiber_radius ** 2) / (2.0 * rc * s_mid)
        arc = np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi
        out[i:i + 64] = arc @ shell
    # At r = 0 the kernel is the PSF's enclosed fraction, known exactly.
    return np.where(r == 0.0, _psf_enclosed(fiber_radius, seeing_fwhm, psf, beta), out)


@lru_cache(maxsize=64)
def _seeing_capture_table(fiber_diameter, seeing_fwhm, sersic_n, psf="gaussian",
                          beta=MOFFAT_BETA):
    """Capture fraction tabulated against log10(half-light radius)."""
    fiber_radius = 0.5 * fiber_diameter

    if psf == "gaussian":
        sigma = seeing_fwhm * _FWHM_TO_SIGMA
        # Radial shells of the intrinsic profile. Past fiber_radius + 10 sigma
        # a shell contributes nothing, so the profile's outer tail can be
        # dropped.
        edges = np.geomspace(1e-4 * sigma, fiber_radius + 10.0 * sigma, 2001)
        mid = np.sqrt(edges[:-1] * edges[1:])
        # P(light emitted at radius r lands in the fiber after Gaussian
        # blurring): the blurred offset over sigma is Rice distributed, so its
        # square is noncentral chi-squared with 2 degrees of freedom.
        kernel = ncx2.cdf((fiber_radius / sigma) ** 2, 2, (mid / sigma) ** 2)
    else:
        # The Moffat wings fall as r^(-2 beta), so light from much farther out
        # still reaches the fiber. The kernel is smooth in r, so compute it on
        # a coarser grid and interpolate.
        alpha = _moffat_alpha(seeing_fwhm, beta)
        edges = np.geomspace(1e-4 * alpha, fiber_radius + 40.0 * seeing_fwhm, 2001)
        mid = np.sqrt(edges[:-1] * edges[1:])
        r_k = np.concatenate([[0.0], np.geomspace(1e-3 * alpha, edges[-1], 600)])
        kernel = np.interp(mid, r_k, _ring_kernel(r_k, fiber_radius, seeing_fwhm,
                                                  psf, beta))
    kernel_center = point_source_capture_fraction(fiber_diameter, seeing_fwhm, psf, beta)

    log_hlr = np.linspace(-3.0, 3.0, 601)
    bn = gammaincinv(2.0 * sersic_n, 0.5)
    enclosed = gammainc(2.0 * sersic_n,
                        bn * (edges[None, :] / 10.0 ** log_hlr[:, None]) ** (1.0 / sersic_n))
    capture = kernel_center * enclosed[:, 0] + np.sum(kernel * np.diff(enclosed, axis=1), axis=1)
    return log_hlr, capture


def seeing_capture_fraction(half_light_radius, fiber_diameter, seeing_fwhm,
                            sersic_n=1.0, psf="gaussian", beta=MOFFAT_BETA):
    """
    Fraction of a Sersic galaxy's total flux within a centred circular fiber
    after convolution with seeing.

    Parameters
    ----------
    half_light_radius : float or array-like
        Intrinsic half-light radius, arcsec. Non-positive or non-finite
        values are treated as a point source.
    fiber_diameter : float
        Fiber angular diameter, arcsec.
    seeing_fwhm : float
        FWHM of the PSF, arcsec. Must be positive.
    sersic_n : float
        Sersic index of the light profile (default: 1.0).
    psf : {'gaussian', 'moffat'}
        PSF shape (default: 'gaussian').
    beta : float
        Moffat index, used only for ``psf='moffat'`` (default: 3.5).

    Returns
    -------
    float or ndarray
        Fraction of total flux captured by the fiber, in (0, 1).
    """
    if seeing_fwhm <= 0:
        raise ValueError("seeing_fwhm must be positive; use "
                         "fiber_capture_fraction for the no-seeing limit.")
    _check_psf(psf)
    r_half = np.asarray(half_light_radius, dtype=float)
    point_source = ~np.isfinite(r_half) | (r_half <= 0)
    log_hlr, capture = _seeing_capture_table(float(fiber_diameter), float(seeing_fwhm),
                                             float(sersic_n), psf, float(beta))
    log_r = np.log10(np.where(point_source, 1.0, r_half))
    # Radii below the table are indistinguishable from a point source once
    # blurred; radii above it (>1000") are unphysical for a BGS target.
    frac = np.interp(log_r, log_hlr, capture)
    frac = np.where(point_source,
                    point_source_capture_fraction(fiber_diameter, seeing_fwhm, psf, beta),
                    frac)
    return float(frac) if frac.ndim == 0 else frac
