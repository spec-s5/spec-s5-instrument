"""
Spec-S5 Exposure Time Calculator (ETC).

Functions
---------
compute_snr                 : photon counts and SNR for Spec-S5 and DESI
plot_snr_comparison         : side-by-side photon / SNR comparison plot
compute_measurement_errors  : radial-velocity, proper-motion, and distance errors
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from .photometry import (gaia_g_to_lsst_i, gaia_g_to_lsst_z, gaia_pm_err,
                         lsst_z_to_gaia_g, lsst_z_to_lsst_i)

# Physical constants
_HPLANCK = 6.62607015e-27   # erg·s
_CLIGHT  = 2.99792458e10    # cm/s

# Package data directory
_DATA_DIR = Path(__file__).parent / "data"


def compute_snr(m, nexp, texp, fiber_diameter=107):
    """
    Compute photon counts and SNR for DESI and Spec-S5 instruments.

    Parameters
    ----------
    m : float or array-like
        Source AB magnitude(s).
    nexp : int
        Number of exposures.
    texp : float
        Total exposure time in seconds (summed over all sub-exposures).
    fiber_diameter : {107, 120}
        Spec-S5 fiber diameter in microns. 107 µm is the baseline design.

    Returns
    -------
    dict
        Keys: ``lam``, ``lam_t``, ``ph_s5_lam``, ``ph_d_lam``,
        ``snr_s5_lam``, ``snr_d_lam``, ``ph_s5_t``, ``ph_d_t``,
        ``snr_s5_t``, ``snr_d_t``, ``VRAD_ERR``,
        ``fiber_diameter``, ``abmag``, ``exptime``, ``nexp``.
        For scalar ``m`` shapes are ``(L,)`` / ``(fine_L,)``; for array
        ``m`` of length N shapes are ``(N, L)`` / ``(N, fine_L)``.
    """
    scalar_input = np.ndim(m) == 0
    m = np.atleast_1d(np.asarray(m, dtype=float))  # (N,)

    lam_t = np.array([360., 375., 400., 450., 500., 550.,
                      600., 650., 700., 750., 800., 850., 900., 980.]) * 10.0  # Å

    if fiber_diameter == 107:
        thru_s5 = np.array([0.0373, 0.0526, 0.0740, 0.1122, 0.1089, 0.1015,
                             0.1169, 0.1250, 0.1237, 0.1138, 0.1368, 0.1378, 0.1387, 0.0895])
        thru_s5_sky = np.array([0.0959, 0.1343, 0.1871, 0.2813, 0.2716, 0.2529,
                                 0.2920, 0.3133, 0.3109, 0.2865, 0.3444, 0.3471, 0.3497, 0.2261])
        df_s5 = 1.02   # arcsec fiber diameter
    elif fiber_diameter == 120:
        thru_s5 = np.array([0.0422, 0.0596, 0.0837, 0.1269, 0.1231, 0.1147,
                             0.1322, 0.1414, 0.1400, 0.1288, 0.1548, 0.1560, 0.1569, 0.1012])
        thru_s5_sky = np.array([0.0959, 0.1343, 0.1871, 0.2813, 0.2716, 0.2529,
                                 0.2920, 0.3133, 0.3109, 0.2865, 0.3444, 0.3471, 0.3497, 0.2261])
        df_s5 = 1.144  # arcsec fiber diameter
    else:
        raise ValueError("fiber_diameter must be 107 (baseline) or 120 microns.")

    thru_d = np.array([0.0253, 0.0409, 0.0623, 0.0863, 0.0940, 0.0933,
                       0.1117, 0.1216, 0.1231, 0.1155, 0.1433, 0.1456, 0.1440, 0.0845])
    thru_d_sky = np.array([0.0631, 0.1017, 0.1537, 0.2113, 0.2284, 0.2262,
                            0.2716, 0.2976, 0.3033, 0.2859, 0.3558, 0.3619, 0.3586, 0.2107])
    df_d = 1.51   # arcsec

    d_d  = 3.797e2   # cm  (DESI)
    d_s5 = 6.0e2     # cm  (Spec-S5)

    # Sky spectrum (erg/s/cm²/Å/arcsec²) — magnitude-independent, shape (L,)
    flamsky = np.array([1.69e-17, 1.50e-17, 1.22e-17, 1.59e-17, 1.03e-17, 1.02e-17,
                        9.59e-18, 7.89e-18, 8.02e-18, 7.73e-18, 7.76e-18, 7.73e-18,
                        5.86e-18, 8.00e-18])
    phsky = flamsky / (_HPLANCK * _CLIGHT * 1e8 / lam_t)   # photons/s/cm²/Å/arcsec²

    noise_ccd_d  = np.array([3.35] * 6 + [2.72] * 8)
    noise_ccd_s5 = np.array([1.18] * 6 + [2.72] * 8)

    pi = np.pi

    # Source: AB mag → photon flux — broadcast m (N,1) against lam_t (L,) → (N, L)
    fnu  = 10 ** (-0.4 * (m[:, None] + 48.6))              # (N, 1)
    flam = fnu * (_CLIGHT * 1e8) / lam_t**2                 # (N, L)
    phot = flam / (_HPLANCK * _CLIGHT * 1e8 / lam_t)        # (N, L)

    # DESI — noise (L,) broadcasts against phot (N, L)
    ph_d    = phot  * pi * (d_d  / 2) ** 2 * texp * thru_d  # (N, L)
    phsky_d = phsky * pi * (d_d  / 2) ** 2 * texp * pi * (df_d  / 2) ** 2 * thru_d_sky
    noise_d = np.sqrt(phsky_d  + nexp * noise_ccd_d  ** 2)  # (L,)
    snr_d   = ph_d  / noise_d                                # (N, L)

    # Spec-S5
    ph_s5    = phot  * pi * (d_s5 / 2) ** 2 * texp * thru_s5  # (N, L)
    phsky_s5 = phsky * pi * (d_s5 / 2) ** 2 * texp * pi * (df_s5 / 2) ** 2 * thru_s5_sky
    noise_s5 = np.sqrt(phsky_s5 + nexp * noise_ccd_s5 ** 2)   # (L,)
    snr_s5   = ph_s5 / noise_s5                                # (N, L)

    # Fine wavelength grid
    lam = np.linspace(lam_t.min(), lam_t.max(), 1001)

    # Velocity error from Spec-S5 z-arm (7470–9800 Å) — shape (N,)
    i_z      = (lam_t > 7470.) & (lam_t < 9800.)
    VRAD_ERR = 10.0 ** (1.389 - 0.975 * np.log10(np.median(snr_s5[:, i_z], axis=1))
                        - 0.975 * np.log10(0.8))

    result = {
        "lam":            lam,
        "lam_t":          lam_t,
        "ph_s5_lam":      interp1d(lam_t, ph_s5,  kind='cubic', axis=1)(lam),
        "ph_d_lam":       interp1d(lam_t, ph_d,   kind='cubic', axis=1)(lam),
        "snr_s5_lam":     interp1d(lam_t, snr_s5, kind='cubic', axis=1)(lam),
        "snr_d_lam":      interp1d(lam_t, snr_d,  kind='cubic', axis=1)(lam),
        "ph_s5_t":        ph_s5,
        "ph_d_t":         ph_d,
        "snr_s5_t":       snr_s5,
        "snr_d_t":        snr_d,
        "VRAD_ERR":       VRAD_ERR,
        "fiber_diameter": fiber_diameter,
        "abmag":          m,
        "exptime":        texp,
        "nexp":           nexp,
    }

    if scalar_input:
        for key in ("ph_s5_lam", "ph_d_lam", "snr_s5_lam", "snr_d_lam",
                    "ph_s5_t", "ph_d_t", "snr_s5_t", "snr_d_t", "VRAD_ERR", "abmag"):
            result[key] = result[key][0]

    return result


def plot_snr_comparison(result_dict):
    """
    Plot photon rate and SNR comparison for Spec-S5 vs DESI.

    Parameters
    ----------
    result_dict : dict
        Output dictionary from :func:`compute_snr`.
    """
    import matplotlib.pyplot as plt

    lam_t     = result_dict["lam_t"]
    ph_s5_t   = result_dict["ph_s5_t"]
    ph_d_t    = result_dict["ph_d_t"]
    snr_s5_t  = result_dict["snr_s5_t"]
    snr_d_t   = result_dict["snr_d_t"]
    lam       = result_dict["lam"]
    ph_s5_lam  = result_dict["ph_s5_lam"]
    ph_d_lam   = result_dict["ph_d_lam"]
    snr_s5_lam = result_dict["snr_s5_lam"]
    snr_d_lam  = result_dict["snr_d_lam"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Photon rate
    axs[0].plot(lam_t, ph_s5_t,   'o-', label='Spec-S5', color='darkorange')
    axs[0].plot(lam,   ph_s5_lam,  '-',  color='darkorange', alpha=0.7)
    axs[0].plot(lam_t, ph_d_t,    'o-', label='DESI',    color='steelblue')
    axs[0].plot(lam,   ph_d_lam,   '--', color='steelblue')
    axs[0].set(xlabel='Wavelength (Å)', ylabel='Detected Photons / Å',
               ylim=[0, np.max(ph_s5_lam) * 1.1])
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # SNR
    axs[1].plot(lam_t, snr_s5_t,   'o-', label='Spec-S5', color='darkorange')
    axs[1].plot(lam,   snr_s5_lam,  '-',  color='darkorange', alpha=0.7)
    axs[1].plot(lam_t, snr_d_t,    'o-', label='DESI',    color='steelblue')
    axs[1].plot(lam,   snr_d_lam,   '--', color='steelblue')
    axs[1].set(xlabel='Wavelength (Å)', ylabel='SNR / Å',
               ylim=[0, np.max(snr_s5_lam) * 1.1])
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.suptitle(
        f"Spec-S5 vs DESI — SNR Comparison\n"
        f"mag={result_dict['abmag']}, texp={result_dict['exptime']} s, "
        f"fiber={result_dict['fiber_diameter']} µm"
    )
    plt.tight_layout()
    plt.show()


def compute_measurement_errors(magnitude, texp=3600.0, nexp=3,
                                fiber_diameter=107, pm_model='gaia_dr5',
                                star_type='giant', mag_band='lsst_z',
                                bp_rp=None, vrad_sys=0.6):
    """
    Compute measurement errors for stars observed with Spec-S5.

    Parameters
    ----------
    magnitude : float or array-like
        Magnitude(s) of the star(s) in the band specified by ``mag_band``.
    texp : float
        Total exposure time in seconds (default: 3600 s).
    nexp : int
        Number of sub-exposures (default: 3).
    fiber_diameter : {107, 120}
        Spec-S5 fiber diameter in microns (default: 107).
    pm_model : {'gaia_dr5', 'lsst1', 'lsst10'}
        Proper-motion reference catalog (default: 'gaia_dr5').
    star_type : {'giant', 'dwarf'}
        Stellar type for distance error (giant: log g < 3.8; default: 'giant').
    mag_band : {'lsst_z', 'gaia_g'}
        Photometric band of ``magnitude`` (default: 'lsst_z').
    bp_rp : float or array-like, optional
        Gaia BP-RP colour(s), used when ``mag_band='gaia_g'``.
        Defaults to 1.2 for giants and 0.8 for dwarfs.
    vrad_sys : float
        Systematic radial velocity error floor in km/s, added in quadrature
        with the statistical error (default: 0.6 km/s).

    Returns
    -------
    dict
        ``vrad_err``        : radial velocity error [km/s]
        ``pm_err``          : proper motion error [mas/yr]
        ``dist_err_frac``   : fractional distance error (0–1)
        ``snr_median_zarm`` : median SNR in Spec-S5 z-arm (7470–9800 Å)
        plus echoed inputs: ``pm_model``, ``star_type``, ``magnitude``,
        ``texp``, ``nexp``.
        Scalar input returns scalar values; array input returns arrays.
    """
    scalar_input = np.ndim(magnitude) == 0
    input_mags = np.atleast_1d(np.asarray(magnitude, dtype=float))  # (N,)

    # Convert input to all required bands explicitly
    if mag_band == 'gaia_g':
        if bp_rp is None:
            bp_rp = 1.2 if star_type == 'giant' else 0.8
        gaia_g_mags = input_mags
        lsst_z_mags = gaia_g_to_lsst_z(input_mags, bp_rp)
        lsst_i_mags = gaia_g_to_lsst_i(input_mags, bp_rp)
    elif mag_band == 'lsst_z':
        if bp_rp is None:
            bp_rp = 1.2 if star_type == 'giant' else 0.8
        gaia_g_mags = lsst_z_to_gaia_g(input_mags, bp_rp)
        lsst_z_mags = input_mags
        lsst_i_mags = lsst_z_to_lsst_i(input_mags)
    else:
        raise ValueError("mag_band must be 'lsst_z' or 'gaia_g'")

    # 1. SNR — single vectorised call over all magnitudes
    snr_result      = compute_snr(m=lsst_z_mags, nexp=nexp, texp=texp, fiber_diameter=fiber_diameter)
    lam_t           = snr_result['lam_t']
    i_z             = (lam_t > 7470.) & (lam_t < 9800.)
    snr_median_zarm = np.median(snr_result['snr_s5_t'][:, i_z], axis=1)  # (N,)
    vrad_err        = np.sqrt(snr_result['VRAD_ERR'] ** 2 + vrad_sys ** 2)  # (N,)

    # 2. Proper-motion error
    _GAIA_ANALYTICAL = ('gaia_dr4', 'gaia_dr5')
    _LSST_PM_FILES   = {'lsst1': 'LSST1_DECAM.csv', 'lsst10': 'LSST10_DECAM.csv'}

    if pm_model in _GAIA_ANALYTICAL:
        # Analytical scaling relations from ESA science-performance page
        pm_err = gaia_pm_err(gaia_g_mags, release=pm_model)          # (N,)
    elif pm_model in _LSST_PM_FILES:
        pm_data = np.loadtxt(_DATA_DIR / _LSST_PM_FILES[pm_model], delimiter=',', skiprows=1)
        pm_mags_data, pm_errs_data = pm_data[:, 0], pm_data[:, 1]
        out_of_range = (lsst_i_mags < pm_mags_data.min()) | (lsst_i_mags > pm_mags_data.max())
        if out_of_range.any():
            warnings.warn(
                f"{out_of_range.sum()} magnitude(s) outside the {pm_model} data range "
                f"[{pm_mags_data.min():.2f}, {pm_mags_data.max():.2f}]. Extrapolating."
            )
        pm_err = interp1d(pm_mags_data, pm_errs_data, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')(lsst_i_mags)  # (N,)
    else:
        raise ValueError(f"pm_model must be one of {list(_GAIA_ANALYTICAL) + list(_LSST_PM_FILES)}")

    # 3. Fractional distance error — interpolator built once, evaluated on all SNRs
    dist_file_map = {
        'giant': 'dist_err_vs_snr_giant.csv',
        'dwarf': 'dist_err_vs_snr_dwarf.csv',
    }
    if star_type not in dist_file_map:
        raise ValueError("star_type must be 'giant' or 'dwarf'")

    dist_data     = np.loadtxt(_DATA_DIR / dist_file_map[star_type], delimiter=',', comments='#')
    dist_snr, dist_fracs = dist_data[:, 0], dist_data[:, 1]
    dist_err_frac = interp1d(dist_snr, dist_fracs, kind='cubic',
                             bounds_error=False,
                             fill_value=(dist_fracs[0], dist_fracs[-1]))(snr_median_zarm)  # (N,)

    result = {
        'vrad_err':        vrad_err,
        'pm_err':          pm_err,
        'dist_err_frac':   dist_err_frac,
        'snr_median_zarm': snr_median_zarm,
        'pm_model':        pm_model,
        'star_type':       star_type,
        'magnitude':       input_mags,
        'texp':            texp,
        'nexp':            nexp,
    }

    if scalar_input:
        for key in ('vrad_err', 'pm_err', 'dist_err_frac', 'snr_median_zarm', 'magnitude'):
            result[key] = result[key][0]

    return result
