"""
spec5.instrument.mock_observations — Convert simulated stars to mock observations.

Functions
---------
galactocentric_to_observed  : Galactocentric phase-space + luminosity → on-sky observables
observe_with_spec5           : On-sky observables → mock Spec-S5 measurements with noise
"""

import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation

from .stellar_etc import compute_measurement_errors

# ---------------------------------------------------------------------------
# Solar Galactocentric frame parameters (Gravity Collaboration 2019)
# Fixing these explicitly so results are independent of astropy version defaults.
# ---------------------------------------------------------------------------
_GALCEN_DISTANCE = 8.122   # kpc  (Gravity Collaboration 2019)
_GALCEN_V_SUN    = [12.9, 245.6, 7.78]   # km/s  (Reid & Brunthaler 2004 / Drimmel & Poggio 2018)
_Z_SUN           = 0.0208  # kpc  (Bennett & Bovy 2019)

# ---------------------------------------------------------------------------
# Photometry
# ---------------------------------------------------------------------------
_M_BOL_SUN = 4.74   # IAU 2015 nominal solar bolometric absolute magnitude

# Bolometric corrections BC_z = M_bol - M_z for old, metal-poor ([Fe/H] ~ -1.5)
# stellar populations in the LSST z-band.
#
# Derived via BC_z = BC_V + (V−z), using:
#   BC_V from Alonso et al. (1999, A&AS 140, 261) at [Fe/H]=-1.5
#   V−z (AB) from empirical colors of metal-poor stars
#
# Giants: Teff ~ 4500 K, log g ~ 1.5  →  BC_V = -0.60, V−z ~ +1.30  →  BC_z = +0.70
# Dwarfs: Teff ~ 5500 K, log g ~ 4.5  →  BC_V = -0.10, V−z ~ +0.40  →  BC_z = +0.30
#
# Positive BC_z means z is brighter than the bolometric for these cool stars.
# For reference, the Sun (Teff=5778K) has BC_z = +0.23 (Willmer 2018).
_BC_Z = {
    'giant': +0.70,
    'dwarf': +0.30,
}

_VALID_STAR_TYPES = frozenset(_BC_Z)

# ---------------------------------------------------------------------------
# Required input / output column names
# ---------------------------------------------------------------------------
_INPUT_COLS  = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'luminosity')
_OUTPUT_COLS = ('ra', 'dec', 'distance', 'vrad', 'pmra', 'pmdec', 'lsst_z')


def _luminosity_to_lsst_z(luminosity, distance_kpc, star_type):
    """Convert luminosity [L_sun] and heliocentric distance [kpc] to apparent LSST z."""
    bc_z    = _BC_Z[star_type]
    m_bol   = _M_BOL_SUN - 2.5 * np.log10(luminosity)   # absolute bolometric
    m_z_abs = m_bol - bc_z                               # absolute z-band
    mu      = 5.0 * np.log10(distance_kpc * 1e3 / 10.0) # distance modulus
    return m_z_abs + mu


def galactocentric_to_observed(stars, star_type='giant'):
    """
    Convert simulated stars from Galactocentric coordinates to on-sky observables.

    Parameters
    ----------
    stars : numpy structured array or pandas DataFrame
        Must contain columns/fields:
            ``x, y, z``          — Galactocentric Cartesian position [kpc]
            ``vx, vy, vz``       — Galactocentric Cartesian velocity [km/s]
            ``luminosity``       — Stellar luminosity [L_sun] (optional; if
                                   absent, ``lsst_z`` is returned as NaN)
    star_type : {'giant', 'dwarf'}
        Stellar population type, used to select the bolometric correction for
        old, metal-poor ([Fe/H] ~ -1.5) stars:
            giant — Teff ~ 4500 K, log g ~ 1.5, BC_z = -1.70
            dwarf — Teff ~ 5500 K, log g ~ 4.5, BC_z = -0.60

    Returns
    -------
    numpy structured array or pandas DataFrame
        Same type as ``stars``, with columns:
            ``ra``       — ICRS right ascension [deg]
            ``dec``      — ICRS declination [deg]
            ``distance`` — Heliocentric distance [kpc]
            ``vrad``     — Heliocentric line-of-sight velocity [km/s]
            ``pmra``     — Proper motion in RA × cos(dec) [mas/yr]
            ``pmdec``    — Proper motion in Dec [mas/yr]
            ``lsst_z``   — Apparent LSST z-band magnitude

    Notes
    -----
    The Galactocentric frame uses the following Solar parameters:

    - Galactocentric distance: 8.122 kpc (Gravity Collaboration 2019)
    - Solar velocity: (12.9, 245.6, 7.78) km/s (Reid & Brunthaler 2004;
      Drimmel & Poggio 2018)
    - Solar height above midplane: 20.8 pc (Bennett & Bovy 2019)

    ``pmra`` is the standard astrometric convention pm_ra × cos(dec), as
    reported by Gaia and LSST.
    """
    if star_type not in _VALID_STAR_TYPES:
        raise ValueError(
            f"star_type must be one of {sorted(_VALID_STAR_TYPES)}, got {star_type!r}"
        )

    # Detect input type — pandas check is duck-typed to avoid hard dependency
    try:
        import pandas as pd
        is_dataframe = isinstance(stars, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    # Detect whether luminosity is present
    if is_dataframe:
        has_luminosity = 'luminosity' in stars.columns
    else:
        has_luminosity = 'luminosity' in stars.dtype.names

    # Extract columns as plain float arrays
    x  = np.asarray(stars['x'],  dtype=float)
    y  = np.asarray(stars['y'],  dtype=float)
    z  = np.asarray(stars['z'],  dtype=float)
    vx = np.asarray(stars['vx'], dtype=float)
    vy = np.asarray(stars['vy'], dtype=float)
    vz = np.asarray(stars['vz'], dtype=float)

    if has_luminosity:
        lum = np.asarray(stars['luminosity'], dtype=float)
        if np.any(lum <= 0):
            raise ValueError("All luminosity values must be positive")

    # Build Galactocentric SkyCoord with full 6D phase-space
    galcen_frame = coord.Galactocentric(
        galcen_distance=_GALCEN_DISTANCE * u.kpc,
        galcen_v_sun=_GALCEN_V_SUN * u.km / u.s,
        z_sun=_Z_SUN * u.kpc,
    )
    cart = CartesianRepresentation(
        x=x * u.kpc,
        y=y * u.kpc,
        z=z * u.kpc,
        differentials=CartesianDifferential(
            d_x=vx * u.km / u.s,
            d_y=vy * u.km / u.s,
            d_z=vz * u.km / u.s,
        ),
    )
    galcen = coord.SkyCoord(cart, frame=galcen_frame)
    icrs   = galcen.icrs

    distance = icrs.distance.to(u.kpc).value
    lsst_z   = (_luminosity_to_lsst_z(lum, distance, star_type)
                if has_luminosity else np.full(len(distance), np.nan))

    ra    = icrs.ra.deg
    dec   = icrs.dec.deg
    vrad  = icrs.radial_velocity.to(u.km / u.s).value
    pmra  = icrs.pm_ra_cosdec.to(u.mas / u.yr).value
    pmdec = icrs.pm_dec.to(u.mas / u.yr).value

    # Return in the same container type as the input
    if is_dataframe:
        import pandas as pd
        return pd.DataFrame({
            'ra':       ra,
            'dec':      dec,
            'distance': distance,
            'vrad':     vrad,
            'pmra':     pmra,
            'pmdec':    pmdec,
            'lsst_z':   lsst_z,
        })
    else:
        n = len(ra)
        out = np.empty(n, dtype=[
            ('ra',       'f8'),
            ('dec',      'f8'),
            ('distance', 'f8'),
            ('vrad',     'f8'),
            ('pmra',     'f8'),
            ('pmdec',    'f8'),
            ('lsst_z',   'f8'),
        ])
        out['ra']       = ra
        out['dec']      = dec
        out['distance'] = distance
        out['vrad']     = vrad
        out['pmra']     = pmra
        out['pmdec']    = pmdec
        out['lsst_z']   = lsst_z
        return out


def observe_with_spec5(stars, star_type='giant', pm_model='gaia_dr5',
                       texp=3600.0, nexp=3, fiber_diameter=107,
                       vrad_sys=0.6, seed=None):
    """
    Generate mock Spec-S5 observations by computing measurement errors and
    drawing noisy realisations of each observable.

    Parameters
    ----------
    stars : numpy structured array or pandas DataFrame
        Must contain columns/fields:
            ``vrad``    — true heliocentric radial velocity [km/s]
            ``pmra``    — true proper motion in RA × cos(dec) [mas/yr]
            ``pmdec``   — true proper motion in Dec [mas/yr]
            ``distance``— true heliocentric distance [kpc]
            ``lsst_z``  — apparent LSST z magnitude (NaN → all outputs NaN)
    star_type : {'giant', 'dwarf'}
        Stellar population type, passed to ``compute_measurement_errors``.
    pm_model : {'gaia_dr4', 'gaia_dr5', 'lsst1', 'lsst10'}
        Proper-motion reference catalogue (default: 'gaia_dr5').
    texp : float
        Total exposure time in seconds (default: 3600 s).
    nexp : int
        Number of sub-exposures (default: 3).
    fiber_diameter : {107, 120}
        Spec-S5 fiber diameter in microns (default: 107).
    vrad_sys : float
        Systematic radial velocity floor in km/s, added in quadrature
        (default: 0.6 km/s).
    seed : int or None
        Random seed for reproducibility (default: None).

    Returns
    -------
    numpy structured array or pandas DataFrame
        Same type as ``stars``, with columns:

        Observed values (true + Gaussian noise):
            ``vrad_obs``     — observed radial velocity [km/s]
            ``pmra_obs``     — observed proper motion in RA × cos(dec) [mas/yr]
            ``pmdec_obs``    — observed proper motion in Dec [mas/yr]
            ``distance_obs`` — observed heliocentric distance [kpc]

        1-sigma measurement errors:
            ``vrad_err``        — radial velocity error [km/s]
            ``pm_err``          — proper motion error (both components) [mas/yr]
            ``dist_err_frac``   — fractional distance error
            ``snr_median_zarm`` — median SNR per Å in Spec-S5 z-arm

        Stars with NaN ``lsst_z`` have NaN in all output columns.

    Notes
    -----
    Proper motion errors are isotropic: the same ``pm_err`` is used as the
    standard deviation for both ``pmra`` and ``pmdec`` noise draws.

    Distance noise is drawn as a fractional error:
        ``distance_obs = distance * (1 + N(0, dist_err_frac))``.
    """
    # Detect input type
    try:
        import pandas as pd
        is_dataframe = isinstance(stars, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    vrad     = np.asarray(stars['vrad'],     dtype=float)
    pmra     = np.asarray(stars['pmra'],     dtype=float)
    pmdec    = np.asarray(stars['pmdec'],    dtype=float)
    distance = np.asarray(stars['distance'], dtype=float)
    lsst_z   = np.asarray(stars['lsst_z'],  dtype=float)

    n = len(vrad)
    rng = np.random.default_rng(seed)

    # Compute measurement errors for all stars in one vectorised call.
    # Stars with NaN lsst_z are masked out; their outputs remain NaN.
    valid = np.isfinite(lsst_z)

    vrad_err      = np.full(n, np.nan)
    pm_err        = np.full(n, np.nan)
    dist_err_frac = np.full(n, np.nan)
    snr           = np.full(n, np.nan)

    if valid.any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            errs = compute_measurement_errors(
                lsst_z[valid],
                texp=texp, nexp=nexp, fiber_diameter=fiber_diameter,
                pm_model=pm_model, star_type=star_type,
                vrad_sys=vrad_sys,
            )
        vrad_err[valid]      = errs['vrad_err']
        pm_err[valid]        = errs['pm_err']
        dist_err_frac[valid] = errs['dist_err_frac']
        snr[valid]           = errs['snr_median_zarm']

    # Draw noisy observations (NaN errors propagate to NaN observations)
    vrad_obs     = vrad     + rng.normal(0.0, 1.0, n) * vrad_err
    pmra_obs     = pmra     + rng.normal(0.0, 1.0, n) * pm_err
    pmdec_obs    = pmdec    + rng.normal(0.0, 1.0, n) * pm_err
    distance_obs = distance * (1.0 + rng.normal(0.0, 1.0, n) * dist_err_frac)

    if is_dataframe:
        import pandas as pd
        return pd.DataFrame({
            'vrad_obs':        vrad_obs,
            'pmra_obs':        pmra_obs,
            'pmdec_obs':       pmdec_obs,
            'distance_obs':    distance_obs,
            'vrad_err':        vrad_err,
            'pm_err':          pm_err,
            'dist_err_frac':   dist_err_frac,
            'snr_median_zarm': snr,
        })
    else:
        out = np.empty(n, dtype=[
            ('vrad_obs',        'f8'),
            ('pmra_obs',        'f8'),
            ('pmdec_obs',       'f8'),
            ('distance_obs',    'f8'),
            ('vrad_err',        'f8'),
            ('pm_err',          'f8'),
            ('dist_err_frac',   'f8'),
            ('snr_median_zarm', 'f8'),
        ])
        out['vrad_obs']        = vrad_obs
        out['pmra_obs']        = pmra_obs
        out['pmdec_obs']       = pmdec_obs
        out['distance_obs']    = distance_obs
        out['vrad_err']        = vrad_err
        out['pm_err']          = pm_err
        out['dist_err_frac']   = dist_err_frac
        out['snr_median_zarm'] = snr
        return out
