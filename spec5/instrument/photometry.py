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
