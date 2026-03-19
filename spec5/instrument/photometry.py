"""
spec5.instrument.photometry — Photometric colour transformations.

Source
------
RTN-099 (LSST/Rubin technical note): https://rtn-099.lsst.io/
Transformations derived from LSSTComCam ↔ Gaia DR3 (Section 1.3.4).
"""

import warnings

import numpy as np

# Valid BP-RP range from RTN-099
_BPRP_MIN = -0.2
_BPRP_MAX =  3.1


def gaia_g_to_lsst_z(G, bp_rp):
    """
    Convert Gaia G magnitude to LSST ComCam z magnitude.

    Uses the polynomial transformation from RTN-099 (Section 1.3.4):
        z - G = +0.034*(BP-RP)^2 - 0.747*(BP-RP) + 0.416
    RMS residual: 0.014 mag.

    Parameters
    ----------
    G : float or array-like
        Gaia G magnitude(s).
    bp_rp : float or array-like
        Gaia BP-RP colour(s).

    Returns
    -------
    float or ndarray
        LSST z magnitude(s).
    """
    G     = np.asarray(G,     dtype=float)
    bp_rp = np.asarray(bp_rp, dtype=float)

    out_of_range = (bp_rp < _BPRP_MIN) | (bp_rp > _BPRP_MAX)
    if np.any(out_of_range):
        warnings.warn(
            f"{np.sum(out_of_range)} BP-RP value(s) outside the valid range "
            f"[{_BPRP_MIN}, {_BPRP_MAX}]. Transformation may be unreliable."
        )

    return G + 0.034 * bp_rp**2 - 0.747 * bp_rp + 0.416


def lsst_z_to_gaia_g(z, bp_rp):
    """
    Convert LSST ComCam z magnitude to Gaia G magnitude.

    Exact inverse of :func:`gaia_g_to_lsst_z`, using the same polynomial
    from RTN-099 (Section 1.3.4):
        G - z = -0.034*(BP-RP)^2 + 0.747*(BP-RP) - 0.416
    RMS residual: 0.014 mag.

    Parameters
    ----------
    z : float or array-like
        LSST z magnitude(s).
    bp_rp : float or array-like
        Gaia BP-RP colour(s).

    Returns
    -------
    float or ndarray
        Gaia G magnitude(s).
    """
    z     = np.asarray(z,     dtype=float)
    bp_rp = np.asarray(bp_rp, dtype=float)

    out_of_range = (bp_rp < _BPRP_MIN) | (bp_rp > _BPRP_MAX)
    if np.any(out_of_range):
        warnings.warn(
            f"{np.sum(out_of_range)} BP-RP value(s) outside the valid range "
            f"[{_BPRP_MIN}, {_BPRP_MAX}]. Transformation may be unreliable."
        )

    return z - 0.034 * bp_rp**2 + 0.747 * bp_rp - 0.416


def gaia_g_to_lsst_i(G, bp_rp):
    """
    Convert Gaia G magnitude to LSST ComCam i magnitude.

    Uses the polynomial transformation from RTN-099 (Section 1.3.4):
        i - G = +0.099*(BP-RP)^2 - 0.672*(BP-RP) + 0.343
    RMS residual: 0.009 mag.

    Parameters
    ----------
    G : float or array-like
        Gaia G magnitude(s).
    bp_rp : float or array-like
        Gaia BP-RP colour(s).

    Returns
    -------
    float or ndarray
        LSST i magnitude(s).
    """
    G     = np.asarray(G,     dtype=float)
    bp_rp = np.asarray(bp_rp, dtype=float)

    out_of_range = (bp_rp < _BPRP_MIN) | (bp_rp > _BPRP_MAX)
    if np.any(out_of_range):
        warnings.warn(
            f"{np.sum(out_of_range)} BP-RP value(s) outside the valid range "
            f"[{_BPRP_MIN}, {_BPRP_MAX}]. Transformation may be unreliable."
        )

    return G + 0.099 * bp_rp**2 - 0.672 * bp_rp + 0.343


# ---------------------------------------------------------------------------
# Gaia proper-motion error scaling relations
# ---------------------------------------------------------------------------

# From https://www.cosmos.esa.int/web/gaia/science-performance
# σ_parallax [μas] = Tfactor * sqrt(40 + 800*z + 30*z²)
#   z = max(10^(0.4*(13 - 15)), 10^(0.4*(G - 15)))  = max(0.1585, 10^(0.4*(G-15)))
# σ_pm [μas/yr] = pm_scale * σ_parallax  (sky-averaged)
_GAIA_PM_PARAMS = {
    #            Tfactor   pm_scale
    'gaia_dr4': (0.749,    0.54),
    'gaia_dr5': (0.527,    0.27),
}


def gaia_pm_err(G, release='gaia_dr5'):
    """
    Compute Gaia sky-averaged proper motion uncertainty from the ESA
    science-performance scaling relations.

    Source: https://www.cosmos.esa.int/web/gaia/science-performance

    The parallax uncertainty is:
        σ_ϖ [μas] = Tfactor * sqrt(40 + 800*z + 30*z²)
    where z = max(10^(0.4*(13-15)), 10^(0.4*(G-15))),
    and the sky-averaged proper motion uncertainty is:
        σ_μ [μas/yr] = pm_scale * σ_ϖ

    Parameters
    ----------
    G : float or array-like
        Gaia G magnitude(s).
    release : {'gaia_dr4', 'gaia_dr5'}
        Gaia data release (default: 'gaia_dr5').

    Returns
    -------
    float or ndarray
        Proper motion uncertainty in mas/yr.
    """
    if release not in _GAIA_PM_PARAMS:
        raise ValueError(f"release must be one of {list(_GAIA_PM_PARAMS)}")

    G = np.asarray(G, dtype=float)
    tfactor, pm_scale = _GAIA_PM_PARAMS[release]

    z = np.maximum(10 ** (-0.8), 10 ** (0.4 * (G - 15.0)))
    sigma_plx = tfactor * np.sqrt(40.0 + 800.0 * z + 30.0 * z ** 2)   # μas
    sigma_pm  = pm_scale * sigma_plx                                    # μas/yr

    result = sigma_pm * 1e-3   # → mas/yr
    result = np.where(G > 20.7, np.nan, result)
    return result


# Assumed i-z colour for metal-poor Milky Way halo RGB stars ([Fe/H] ~ -2 to -1).
# Both lsst_i_to_lsst_z and lsst_z_to_lsst_i reference this constant so they
# are guaranteed to be exact inverses of each other.
_LSST_IZ_COLOR = 0.2


def lsst_i_to_lsst_z(i):
    """
    Convert LSST i magnitude to LSST z magnitude for metal-poor halo RGB stars.

    Assumes a fixed i-z colour of ``_LSST_IZ_COLOR`` (0.2 mag), appropriate
    for old metal-poor Milky Way halo RGB stars ([Fe/H] ~ -2 to -1).

    Parameters
    ----------
    i : float or array-like
        LSST i magnitude(s).

    Returns
    -------
    float or ndarray
        LSST z magnitude(s).
    """
    return np.asarray(i, dtype=float) - _LSST_IZ_COLOR


def lsst_z_to_lsst_i(z):
    """
    Convert LSST z magnitude to LSST i magnitude for metal-poor halo RGB stars.

    Assumes a fixed i-z colour of ``_LSST_IZ_COLOR`` (0.2 mag), appropriate
    for old metal-poor Milky Way halo RGB stars ([Fe/H] ~ -2 to -1).

    Parameters
    ----------
    z : float or array-like
        LSST z magnitude(s).

    Returns
    -------
    float or ndarray
        LSST i magnitude(s).
    """
    return np.asarray(z, dtype=float) + _LSST_IZ_COLOR
