"""
spec5.instrument.galaxy_photometry — Fiber aperture-loss model for extended
(galaxy) sources.

Unlike stars, BGS galaxies are not point sources: only a fraction of their
total light falls within the spectrograph fiber. This module models that
fraction with a Sersic light profile and the standard (regularised
incomplete-gamma) curve-of-growth formula.
"""

import numpy as np
from scipy.special import gammainc, gammaincinv


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
