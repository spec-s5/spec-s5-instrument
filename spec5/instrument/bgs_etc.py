"""
Spec-S5 Exposure Time Calculator for DESI Bright Galaxy Survey (BGS)
targets.

Functions
---------
compute_bgs_snr            : r-arm continuum SNR for Spec-S5 and DESI,
                              accounting for fiber aperture losses on
                              extended (galaxy) sources
compute_redshift_success   : binary redshift-success probability from a
                              DESI DR1-calibrated empirical relation

See docs/bgs_plan.md and docs/bgs_path_b_results.md for the research behind
these models.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from .galaxy_photometry import fiber_capture_fraction
from .stellar_etc import _photon_counting_core

_DATA_DIR = Path(__file__).parent / "data"

# DESI r-arm wavelength range (Guy et al. 2023; data.desi.lbl.gov/doc/glossary) —
# the continuum band relevant to BGS redshift features (4000 Å break,
# Ca H&K, Hα/[OIII] at BGS's z<~0.4), unlike the z-arm used for stellar RV.
_R_ARM_MIN, _R_ARM_MAX = 5760.0, 7620.0

# Reference aperture for FIBERFLUX_R-convention magnitudes (Legacy Survey /
# DESI targeting photometry), matching the aperture used to calibrate
# compute_redshift_success's data files. See docs/bgs_path_b_results.md.
_DESI_REFERENCE_FIBER_DIAMETER = 1.51  # arcsec

_BGS_SUCCESS_FILES = {
    "bright": "bgs_redshift_success_bright.csv",
}


def compute_bgs_snr(r_fiber_mag, nexp, texp, fiber_diameter=107,
                     half_light_radius=1.5, sersic_n=1.0):
    """
    Compute Spec-S5 and DESI r-arm continuum SNR for a BGS galaxy.

    Parameters
    ----------
    r_fiber_mag : float or array-like
        Galaxy r-band AB magnitude as captured within DESI's reference
        1.51" fiber aperture — i.e. the same convention as the Legacy
        Survey/DESI targeting ``FIBERFLUX_R`` quantity (not total/model
        magnitude).
    nexp : int
        Number of exposures.
    texp : float
        Total exposure time in seconds (summed over all sub-exposures).
    fiber_diameter : {107, 120}
        Spec-S5 fiber diameter in microns.
    half_light_radius : float or array-like
        Galaxy half-light radius, arcsec (default: 1.5", matching DESI's
        TSNR2_BGS template convention; Guy et al. 2023).
    sersic_n : float
        Sersic index of the assumed light profile (default: 1.0, exponential
        disk).

    Returns
    -------
    dict
        Keys: ``lam_t``, ``snr_s5_t``, ``snr_d_t`` (per-Å SNR at the 14
        throughput nodes), ``snr_r_s5``, ``snr_r_d`` (median continuum SNR
        in the r-arm range), ``capture_frac_s5``, ``capture_frac_d``
        (fiber flux-capture fractions), plus echoed inputs. Scalar input
        returns scalar outputs for the per-target keys.
    """
    scalar_input = np.ndim(r_fiber_mag) == 0
    m = np.atleast_1d(np.asarray(r_fiber_mag, dtype=float))  # (N,)

    # df_s5 / df_d are fixed by fiber_diameter; a point-source probe call
    # is the simplest way to read them back without duplicating constants.
    probe = _photon_counting_core(m, nexp, texp, fiber_diameter=fiber_diameter)
    df_s5, df_d = probe["df_s5"], probe["df_d"]

    hlr = np.broadcast_to(
        np.atleast_1d(np.asarray(half_light_radius, dtype=float)), m.shape
    )
    capture_s5 = fiber_capture_fraction(hlr, df_s5, sersic_n=sersic_n)
    capture_d  = fiber_capture_fraction(hlr, df_d,  sersic_n=sersic_n)

    # r_fiber_mag already represents the flux captured within DESI's
    # reference 1.51" aperture, so the DESI channel needs no further
    # correction. The Spec-S5 channel is scaled by the *relative* capture
    # fraction between the two fiber sizes for a source of this half-light
    # radius (see galaxy_photometry.fiber_capture_fraction).
    aperture_loss_s5 = capture_s5 / capture_d
    aperture_loss_d  = np.ones_like(m)

    core = _photon_counting_core(m, nexp, texp, fiber_diameter=fiber_diameter,
                                  aperture_loss_s5=aperture_loss_s5,
                                  aperture_loss_d=aperture_loss_d)
    lam_t = core["lam_t"]
    i_r = (lam_t > _R_ARM_MIN) & (lam_t < _R_ARM_MAX)

    snr_r_s5 = np.median(core["snr_s5"][:, i_r], axis=1)
    snr_r_d  = np.median(core["snr_d"][:, i_r], axis=1)

    result = {
        "lam_t":              lam_t,
        "snr_s5_t":           core["snr_s5"],
        "snr_d_t":            core["snr_d"],
        "snr_r_s5":           snr_r_s5,
        "snr_r_d":            snr_r_d,
        "capture_frac_s5":    capture_s5,
        "capture_frac_d":     capture_d,
        "fiber_diameter":     fiber_diameter,
        "half_light_radius":  hlr,
        "sersic_n":           sersic_n,
        "r_fiber_mag":        m,
        "exptime":            texp,
        "nexp":               nexp,
    }

    if scalar_input:
        for key in ("snr_r_s5", "snr_r_d", "capture_frac_s5", "capture_frac_d",
                    "half_light_radius", "r_fiber_mag"):
            result[key] = result[key][0]

    return result


def _load_success_curve(bgs_type):
    if bgs_type not in _BGS_SUCCESS_FILES:
        raise ValueError(
            f"bgs_type must be one of {list(_BGS_SUCCESS_FILES)}; a 'faint' "
            "calibration has not yet been derived (see docs/bgs_plan.md §2.4)."
        )
    data = np.loadtxt(_DATA_DIR / _BGS_SUCCESS_FILES[bgs_type], delimiter=",", comments="#")
    mags, rates = data[:, 0], data[:, 1]
    return interp1d(mags, rates, kind="cubic", bounds_error=False,
                     fill_value=(rates[0], rates[-1]))


def compute_redshift_success(r_fiber_mag, instrument="spec5", bgs_type="bright",
                              nexp=1, texp=180.0, fiber_diameter=107,
                              half_light_radius=1.5, sersic_n=1.0):
    """
    Predict the probability of a successful DESI/Spec-S5 redshift
    measurement for a BGS galaxy, following DESI's own binary
    redshift-recovery definition (ZWARN==0 & DELTACHI2>40 &
    SPECTYPE=='GALAXY'; Hahn et al. 2023).

    The underlying relation is purely empirical, calibrated against the
    public DESI DR1 BGS_BRIGHT LSS catalog (see
    docs/bgs_path_b_results.md) — there is no continuous redshift-error
    model here, matching DESI's own Δχ²-based framework.

    Parameters
    ----------
    r_fiber_mag : float or array-like
        Galaxy r-band AB magnitude in DESI's reference 1.51" fiber-flux
        convention (see ``compute_bgs_snr``).
    instrument : {'spec5', 'desi'}
        Which instrument to predict for. ``'desi'`` evaluates the
        DR1-calibrated curve directly. ``'spec5'`` additionally translates
        the input magnitude by a mirror-area + fiber-aperture-loss
        equivalent offset, derived from the actual SNR ratio between the
        two instruments (the BGS analogue of
        ``stellar_etc._DESI_MIRROR_MAG_OFFSET`` — computed per-case here
        rather than as a fixed constant, since the aperture-loss term
        depends on ``fiber_diameter``/``half_light_radius`` and does not
        cancel the way it does for point-source stars).
    bgs_type : {'bright'}
        BGS target class. Only 'bright' is currently calibrated.
    nexp, texp : int, float
        Number of exposures / total exposure time in seconds, used only
        when ``instrument='spec5'`` to compute its SNR (default: 1 x 180s,
        the Hahn et al. 2023 BGS Bright nominal single-visit exposure).
    fiber_diameter : {107, 120}
        Spec-S5 fiber diameter in microns, used only for ``instrument='spec5'``.
    half_light_radius, sersic_n : float or array-like
        Passed through to ``compute_bgs_snr``.

    Returns
    -------
    float or ndarray
        Redshift-success probability, in [0, 1].
    """
    scalar_input = np.ndim(r_fiber_mag) == 0
    m = np.atleast_1d(np.asarray(r_fiber_mag, dtype=float))
    curve = _load_success_curve(bgs_type)

    if instrument == "desi":
        p = curve(m)
    elif instrument == "spec5":
        snr = compute_bgs_snr(m, nexp=nexp, texp=texp, fiber_diameter=fiber_diameter,
                               half_light_radius=half_light_radius, sersic_n=sersic_n)
        ratio = snr["snr_r_s5"] / snr["snr_r_d"]
        # Effective DESI-equivalent magnitude giving the same r-arm SNR that
        # Spec-S5 actually achieves (Poisson-dominated SNR ~ 10^(-0.2 m)).
        m_desi_equiv = m - 5.0 * np.log10(ratio)
        p = curve(m_desi_equiv)
    else:
        raise ValueError("instrument must be 'spec5' or 'desi'")

    p = np.clip(p, 0.0, 1.0)
    return float(p[0]) if scalar_input else p
