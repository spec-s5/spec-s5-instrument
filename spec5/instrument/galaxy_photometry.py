"""
spec5.instrument.galaxy_photometry — Fiber aperture-loss model for extended
(galaxy) sources.

Unlike stars, BGS galaxies are not point sources: only a fraction of their
total light falls within the spectrograph fiber. This module models that
fraction with a Sersic light profile and the standard (regularised
incomplete-gamma) curve-of-growth formula, optionally convolved with
Gaussian seeing.
"""

from functools import lru_cache

import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.stats import ncx2

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


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


def point_source_capture_fraction(fiber_diameter, seeing_fwhm):
    """
    Fraction of a Gaussian-seeing point source's flux within a centred
    circular fiber. ``fiber_diameter`` and ``seeing_fwhm`` are in arcsec.
    """
    sigma = seeing_fwhm * _FWHM_TO_SIGMA
    return 1.0 - np.exp(-0.5 * (0.5 * fiber_diameter / sigma) ** 2)


@lru_cache(maxsize=64)
def _seeing_capture_table(fiber_diameter, seeing_fwhm, sersic_n):
    """Capture fraction tabulated against log10(half-light radius)."""
    sigma = seeing_fwhm * _FWHM_TO_SIGMA
    fiber_radius = 0.5 * fiber_diameter

    # Radial shells of the intrinsic profile. Past fiber_radius + 10 sigma a
    # shell contributes nothing, so the profile's outer tail can be dropped.
    edges = np.geomspace(1e-4 * sigma, fiber_radius + 10.0 * sigma, 2001)
    mid = np.sqrt(edges[:-1] * edges[1:])
    # P(light emitted at radius r lands in the fiber after Gaussian blurring):
    # the blurred offset over sigma is Rice distributed, so its square is
    # noncentral chi-squared with 2 degrees of freedom.
    kernel = ncx2.cdf((fiber_radius / sigma) ** 2, 2, (mid / sigma) ** 2)
    kernel_center = point_source_capture_fraction(fiber_diameter, seeing_fwhm)

    log_hlr = np.linspace(-3.0, 3.0, 601)
    bn = gammaincinv(2.0 * sersic_n, 0.5)
    enclosed = gammainc(2.0 * sersic_n,
                        bn * (edges[None, :] / 10.0 ** log_hlr[:, None]) ** (1.0 / sersic_n))
    capture = kernel_center * enclosed[:, 0] + np.sum(kernel * np.diff(enclosed, axis=1), axis=1)
    return log_hlr, capture


def seeing_capture_fraction(half_light_radius, fiber_diameter, seeing_fwhm,
                            sersic_n=1.0):
    """
    Fraction of a Sersic galaxy's total flux within a centred circular fiber
    after convolution with Gaussian seeing.

    Parameters
    ----------
    half_light_radius : float or array-like
        Intrinsic half-light radius, arcsec. Non-positive or non-finite
        values are treated as a point source.
    fiber_diameter : float
        Fiber angular diameter, arcsec.
    seeing_fwhm : float
        FWHM of the Gaussian PSF, arcsec. Must be positive.
    sersic_n : float
        Sersic index of the light profile (default: 1.0).

    Returns
    -------
    float or ndarray
        Fraction of total flux captured by the fiber, in (0, 1).
    """
    if seeing_fwhm <= 0:
        raise ValueError("seeing_fwhm must be positive; use "
                         "fiber_capture_fraction for the no-seeing limit.")
    r_half = np.asarray(half_light_radius, dtype=float)
    point_source = ~np.isfinite(r_half) | (r_half <= 0)
    log_hlr, capture = _seeing_capture_table(float(fiber_diameter), float(seeing_fwhm),
                                             float(sersic_n))
    log_r = np.log10(np.where(point_source, 1.0, r_half))
    # Radii below the table are indistinguishable from a point source once
    # blurred; radii above it (>1000") are unphysical for a BGS target.
    frac = np.interp(log_r, log_hlr, capture)
    frac = np.where(point_source,
                    point_source_capture_fraction(fiber_diameter, seeing_fwhm), frac)
    return float(frac) if frac.ndim == 0 else frac
