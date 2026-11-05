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

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

from .galaxy_photometry import point_source_capture_fraction, seeing_capture_fraction
from .stellar_etc import (_DESI_FIBER_ANGULAR_DIAMETER,
                          _SPEC5_FIBER_ANGULAR_DIAMETER, _photon_counting_core)

_DATA_DIR = Path(__file__).parent / "data"

# DESI r-arm wavelength range (Guy et al. 2023; data.desi.lbl.gov/doc/glossary) —
# the continuum band relevant to BGS redshift features (4000 Å break,
# Ca H&K, Hα/[OIII] at BGS's z<~0.4), unlike the z-arm used for stellar RV.
_R_ARM_MIN, _R_ARM_MAX = 5760.0, 7620.0

# FIBERFLUX_R is the predicted flux in a 1.5" fiber under 1" FWHM Gaussian
# seeing (Dey et al. 2019; Hahn et al. 2023 §2.1).
_FIBERFLUX_DIAMETER = 1.5      # arcsec
_FIBERFLUX_SEEING_FWHM = 1.0   # arcsec

# DESI's exposure-time calculator scales exposures to an effective time at
# nominal dark conditions, seeing FWHM 1.1" (Hahn et al. 2023 §4.1), so the
# DR1 calibration curve corresponds to DESI under these conditions at the BGS
# Bright nominal 1 x 180 s, whatever the realized exposure lengths were.
_DESI_REF_NEXP = 1
_DESI_REF_TEXP = 180.0         # s
_NOMINAL_SEEING_FWHM = 1.1     # arcsec

# Seeing PSF. The throughput budgets (DESI-347, SS5-0022) model seeing as a
# Moffat profile with beta = 3.5. Seeing FWHMs in this module are the FWHM of
# that Moffat at the focal plane, before the losses each throughput table
# already applies as fixed factors (instrument blur, fiber lateral offsets).
_PSF = "moffat"

# Moffat FWHM behind the point-source fiber loss built into each throughput
# table (see stellar_etc and docs/bgs_path_b_results.md, "Seeing"). Both
# budgets use the same convention: seeing before the instrument's own optics,
# whose blur they charge separately. The two values differ in which
# conditions they assume, not in what they include:
# - Spec-S5, 0.96": SS5-0022 v1.1, the g-band median of the seeing DESI
#   measured at the Mayall with its guide cameras, after removing DESI's
#   optical aberrations and the camera pixelization ("DESI Seeing" tab).
# - DESI, 1.1": DESI-347-v17's nominal design value, which DESI's simulator
#   specsim calls atmospheric seeing (FWHM at 6355 A, scaling as
#   lambda^-0.2), with DESI's blur and offsets added separately.
SPEC5_TABLE_SEEING_FWHM = 0.96   # arcsec
DESI_TABLE_SEEING_FWHM = 1.1     # arcsec

# Calibration validity limits: the r_fiber < 22.9 survey cut (Hahn et al. 2023
# Eq. 2) and the SHAPE_R > 1" cut applied to the calibration sample.
_R_FIBER_SAMPLE_LIMIT = 22.9
_MIN_CALIBRATED_HALF_LIGHT_RADIUS = 1.0  # arcsec

_BGS_SUCCESS_FILES = {
    "bright": "bgs_redshift_success_bright.csv",
}


def _fiber_source_scale(half_light_radius, fiber_angular_diameter, sersic_n, seeing_fwhm,
                        table_seeing_fwhm, psf=_PSF):
    """
    Photon-count multiplier for a galaxy of FIBERFLUX-convention magnitude,
    relative to a point source of that AB magnitude in the throughput tables.

    The tables already include a fixed point-source fiber loss, at the seeing
    they assume, and FIBERFLUX already excludes light outside its reference
    fiber. Undoing both, and applying the galaxy's capture in the fiber
    actually used under the observing seeing, gives

        C(gal*PSF_obs, D) / C(PSF_table, D) / C(gal*PSF_LS, 1.5")

    This can exceed 1. A point source gets
    C(PSF_obs, D) / C(PSF_table, D) / C(PSF_LS, 1.5"): worse seeing loses light
    from the fiber, and its FIBERFLUX mag is fainter than its total mag.

    PSF_obs and PSF_table have shape ``psf`` (Moffat by default, as in the
    throughput budgets). PSF_LS is the Gaussian of the FIBERFLUX definition.
    """
    capture = seeing_capture_fraction(half_light_radius, fiber_angular_diameter,
                                      seeing_fwhm, sersic_n=sersic_n, psf=psf)
    capture_psf = point_source_capture_fraction(fiber_angular_diameter, table_seeing_fwhm,
                                                psf=psf)
    capture_fiberflux = seeing_capture_fraction(half_light_radius, _FIBERFLUX_DIAMETER,
                                                _FIBERFLUX_SEEING_FWHM, sersic_n=sersic_n,
                                                psf="gaussian")
    return capture / capture_psf / capture_fiberflux


def compute_bgs_snr(r_fiber_mag, nexp, texp, fiber_diameter=107,
                     half_light_radius=1.5, sersic_n=1.0,
                     seeing_fwhm=SPEC5_TABLE_SEEING_FWHM, sky_scale=1.0,
                     desi_seeing_fwhm=DESI_TABLE_SEEING_FWHM):
    """
    Compute Spec-S5 and DESI r-arm continuum SNR for a BGS galaxy.

    Parameters
    ----------
    r_fiber_mag : float or array-like
        Galaxy r-band AB fiber magnitude in the Legacy Surveys
        ``FIBERFLUX_R`` convention: predicted flux in a 1.5" fiber under 1"
        FWHM Gaussian seeing (not total/model magnitude).
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
    seeing_fwhm : float
        Spec-S5 seeing FWHM during the observation, arcsec: a Moffat PSF with
        beta = 3.5 at the focal plane, before Spec-S5's own blur and fiber
        offset losses, which its throughput table already includes (default:
        0.96", the value assumed by that table, SS5-0022: the median seeing
        DESI measured at the Mayall, without DESI's optics). It has the same
        meaning as ``desi_seeing_fwhm``, so the same night gives both the same
        value. Sets the fiber capture relative to the table's own PSF, so worse
        seeing lowers source photons for galaxies and point sources alike.
    sky_scale : float
        Multiplier on the tabulated sky (default: 1.0). The table is a
        moonless-continuum sky model (Hanuschik 2003, via SS5-0022 v1), about
        r = 21.3 AB mag/arcsec². With the default, ``texp`` is therefore an
        effective exposure under that sky, the same
        sense as DESI's EFFTIME. Set above 1 to treat ``texp`` as a real
        exposure under brighter sky, e.g. ``10**(0.4 * (21.3 - mu_r))`` for
        a sky of ``mu_r`` mag/arcsec² (this scales the spectrum grey).
    desi_seeing_fwhm : float
        DESI seeing FWHM, arcsec, defined as for ``seeing_fwhm`` (default:
        1.1", DESI's nominal design value and the one assumed by the DESI
        throughput table, DESI-347-v17).

    Returns
    -------
    dict
        Keys: ``lam_t``, ``snr_s5_t``, ``snr_d_t`` (per-Å SNR at the 14
        throughput nodes), ``snr_r_s5``, ``snr_r_d`` (median continuum SNR
        in the r-arm range), ``capture_frac_s5``, ``capture_frac_d``
        (fraction of total galaxy flux in the fiber, after seeing),
        ``source_scale_s5``, ``source_scale_d`` (photon-count multipliers
        relative to the point-source tables), plus echoed inputs. Scalar input
        returns scalar outputs for the per-target keys.
    """
    scalar_input = np.ndim(r_fiber_mag) == 0
    m = np.atleast_1d(np.asarray(r_fiber_mag, dtype=float))  # (N,)

    if fiber_diameter not in _SPEC5_FIBER_ANGULAR_DIAMETER:
        raise ValueError("fiber_diameter must be 107 (baseline) or 120 microns.")
    df_s5 = _SPEC5_FIBER_ANGULAR_DIAMETER[fiber_diameter]
    df_d = _DESI_FIBER_ANGULAR_DIAMETER

    hlr = np.broadcast_to(
        np.atleast_1d(np.asarray(half_light_radius, dtype=float)), m.shape
    )
    capture_s5 = seeing_capture_fraction(hlr, df_s5, seeing_fwhm, sersic_n=sersic_n, psf=_PSF)
    capture_d  = seeing_capture_fraction(hlr, df_d,  desi_seeing_fwhm, sersic_n=sersic_n,
                                         psf=_PSF)
    source_scale_s5 = _fiber_source_scale(hlr, df_s5, sersic_n, seeing_fwhm,
                                          SPEC5_TABLE_SEEING_FWHM)
    source_scale_d  = _fiber_source_scale(hlr, df_d,  sersic_n, desi_seeing_fwhm,
                                          DESI_TABLE_SEEING_FWHM)

    core = _photon_counting_core(m, nexp, texp, fiber_diameter=fiber_diameter,
                                  source_scale_s5=source_scale_s5,
                                  source_scale_d=source_scale_d,
                                  sky_scale=sky_scale)
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
        "source_scale_s5":    source_scale_s5,
        "source_scale_d":     source_scale_d,
        "fiber_diameter":     fiber_diameter,
        "half_light_radius":  hlr,
        "sersic_n":           sersic_n,
        "seeing_fwhm":        seeing_fwhm,
        "desi_seeing_fwhm":   desi_seeing_fwhm,
        "sky_scale":          sky_scale,
        "r_fiber_mag":        m,
        "exptime":            texp,
        "nexp":               nexp,
    }

    if scalar_input:
        for key in ("snr_r_s5", "snr_r_d", "capture_frac_s5", "capture_frac_d",
                    "source_scale_s5", "source_scale_d", "half_light_radius", "r_fiber_mag", "snr_s5_t", "snr_d_t"):
            result[key] = result[key][0]

    return result


@lru_cache(maxsize=None)
def _load_success_curve(bgs_type):
    if bgs_type not in _BGS_SUCCESS_FILES:
        raise ValueError(
            f"bgs_type must be one of {list(_BGS_SUCCESS_FILES)}; a 'faint' "
            "calibration has not yet been derived (see docs/bgs_plan.md §2.4)."
        )
    data = np.loadtxt(_DATA_DIR / _BGS_SUCCESS_FILES[bgs_type], delimiter=",", comments="#")
    mags, rates, counts = data[:, 0], data[:, 1], data[:, 2]
    # The binned rates are noisy on the bright plateau, and PCHIP only
    # preserves monotonicity the data already has. Isotonic regression makes
    # success non-increasing with magnitude, so that added depth can never
    # lower the predicted success.
    monotone = isotonic_regression(rates, weights=counts, increasing=False).x
    pchip = PchipInterpolator(mags, monotone)

    def curve(m):
        # Bright end: flat, so clamp. Faint end: hold the last bin to the
        # sample limit, then 0, since no targets fainter than that exist to
        # calibrate against.
        p = pchip(np.clip(m, mags[0], mags[-1]))
        return np.where(m > _R_FIBER_SAMPLE_LIMIT, 0.0, p)

    return curve


@lru_cache(maxsize=None)
def _desi_reference_snr_table():
    """(log10 r-arm SNR, magnitude) for DESI's reference exposure, SNR ascending."""
    mags = np.arange(45.0, 5.0, -0.005)
    core = _photon_counting_core(mags, _DESI_REF_NEXP, _DESI_REF_TEXP)
    i_r = (core["lam_t"] > _R_ARM_MIN) & (core["lam_t"] < _R_ARM_MAX)
    return np.log10(np.median(core["snr_d"][:, i_r], axis=1)), mags


def _desi_equivalent_mag(snr_r, half_light_radius, sersic_n):
    """
    Fiber magnitude at which DESI reaches r-arm SNR ``snr_r`` for this galaxy,
    at its reference exposure and nominal conditions.

    Solves SNR_DESI(m') = snr_r exactly, including sky and read noise. A
    monotonic table is inverted rather than calling a root-finder per target
    so that large mock catalogs stay fast; the table spacing gives errors
    well below 1e-4 mag.
    """
    log_snr, mags = _desi_reference_snr_table()
    with np.errstate(divide="ignore"):
        point_source_mag = np.interp(np.log10(snr_r), log_snr, mags)
    scale = _fiber_source_scale(half_light_radius, _DESI_FIBER_ANGULAR_DIAMETER,
                                sersic_n, _NOMINAL_SEEING_FWHM, DESI_TABLE_SEEING_FWHM)
    return point_source_mag + 2.5 * np.log10(scale)


def compute_redshift_success(r_fiber_mag, instrument="spec5", bgs_type="bright",
                              nexp=1, texp=180.0, fiber_diameter=107,
                              half_light_radius=1.5, sersic_n=1.0,
                              seeing_fwhm=SPEC5_TABLE_SEEING_FWHM, sky_scale=1.0):
    """
    Predict the probability of a successful DESI/Spec-S5 redshift
    measurement for a BGS galaxy, using the Hahn et al. 2023 criteria that
    don't need a deep comparison spectrum: ZWARN==0 & DELTACHI2>40 &
    SPECTYPE=='GALAXY'.

    The underlying relation is purely empirical, calibrated against the
    public DESI DR1 BGS_BRIGHT LSS catalog (see
    docs/bgs_path_b_results.md) — there is no continuous redshift-error
    model here, matching DESI's own Δχ²-based framework.

    Parameters
    ----------
    r_fiber_mag : float or array-like
        Galaxy r-band AB magnitude in the ``FIBERFLUX_R`` convention (see
        ``compute_bgs_snr``).
    instrument : {'spec5', 'desi'}
        Which instrument to predict for. ``'desi'`` evaluates the
        DR1-calibrated curve directly. ``'spec5'`` first finds the magnitude
        at which DESI, at its reference 1 x 180 s effective exposure in
        nominal conditions, reaches the r-arm SNR that Spec-S5 achieves for
        this galaxy and exposure, then evaluates the curve there.
    bgs_type : {'bright'}
        BGS target class. Only 'bright' is currently calibrated.
    nexp, texp : int, float
        Number of exposures / total exposure time in seconds, used only
        when ``instrument='spec5'`` to compute its SNR (default: 1 x 180s,
        the Hahn et al. 2023 BGS Bright nominal single-visit exposure).
    fiber_diameter : {107, 120}
        Spec-S5 fiber diameter in microns, used only for ``instrument='spec5'``.
    half_light_radius, sersic_n : float or array-like
        Galaxy shape, passed through to ``compute_bgs_snr``. Radii below 1"
        raise a warning: the calibration sample has SHAPE_R > 1".
    seeing_fwhm, sky_scale : float
        Spec-S5 observing conditions, used only for ``instrument='spec5'``;
        see ``compute_bgs_snr`` for what ``seeing_fwhm`` includes (default:
        0.96", as assumed by the Spec-S5 throughput table). The DESI reference
        stays at its nominal 1.1", because the calibration curve was measured
        there.

    Returns
    -------
    float or ndarray
        Redshift-success probability, in [0, 1], non-increasing with
        magnitude. Magnitudes brighter than the calibration table clamp to
        the brightest value. Past the r_fiber < 22.9 sample limit, where the
        calibrated rate is still ~0.39, it returns 0: this is a step, chosen
        to be conservative where no data exist.
    """
    scalar_input = np.ndim(r_fiber_mag) == 0
    m = np.atleast_1d(np.asarray(r_fiber_mag, dtype=float))
    curve = _load_success_curve(bgs_type)

    hlr = np.broadcast_to(
        np.atleast_1d(np.asarray(half_light_radius, dtype=float)), m.shape
    )
    if np.any(hlr < _MIN_CALIBRATED_HALF_LIGHT_RADIUS):
        warnings.warn(
            f"half_light_radius below {_MIN_CALIBRATED_HALF_LIGHT_RADIUS}\" is outside "
            "the BGS calibration sample (SHAPE_R > 1\"); see docs/bgs_path_b_results.md.",
            stacklevel=2,
        )

    if instrument == "desi":
        p = curve(m)
    elif instrument == "spec5":
        snr = compute_bgs_snr(m, nexp=nexp, texp=texp, fiber_diameter=fiber_diameter,
                               half_light_radius=hlr, sersic_n=sersic_n,
                               seeing_fwhm=seeing_fwhm, sky_scale=sky_scale)
        p = curve(_desi_equivalent_mag(snr["snr_r_s5"], hlr, sersic_n))
    else:
        raise ValueError("instrument must be 'spec5' or 'desi'")

    p = np.clip(p, 0.0, 1.0)
    return float(p[0]) if scalar_input else p
