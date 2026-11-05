"""
Unit tests for spec5.instrument.bgs_etc and spec5.instrument.galaxy_photometry.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from spec5.instrument.bgs_etc import (
    _R_FIBER_SAMPLE_LIMIT,
    _TABLE_SEEING_FWHM,
    _desi_equivalent_mag,
    _fiber_source_scale,
    _load_success_curve,
    compute_bgs_snr,
    compute_redshift_success,
)
from spec5.instrument.galaxy_photometry import (
    fiber_capture_fraction,
    point_source_capture_fraction,
    seeing_capture_fraction,
)

MAGS = [17.5, 19.5, 20.5, 21.5]

# Expected values evaluated at nexp=1, texp=180.0, half_light_radius=1.5,
# fiber_diameter=107 (baseline Spec-S5 design).
EXPECTED_SNR_R_S5 = {17.5: 22.480359, 19.5: 7.607643, 20.5: 3.922270, 21.5: 1.825424}
EXPECTED_SNR_R_D  = {17.5: 17.383986, 19.5: 5.592132, 20.5: 2.769602, 21.5: 1.247007}

EXPECTED_P_DESI  = {17.5: 0.989070, 19.5: 0.986045, 20.5: 0.986045, 21.5: 0.950944}
EXPECTED_P_SPEC5 = {17.5: 0.989070, 19.5: 0.986045, 20.5: 0.986045, 21.5: 0.984514}


@pytest.mark.parametrize("m", MAGS)
def test_compute_bgs_snr(m):
    r = compute_bgs_snr(m, nexp=1, texp=180.0, half_light_radius=1.5)
    assert r["snr_r_s5"] == pytest.approx(EXPECTED_SNR_R_S5[m], rel=1e-5)
    assert r["snr_r_d"] == pytest.approx(EXPECTED_SNR_R_D[m], rel=1e-5)


def test_compute_bgs_snr_scalar_input_returns_1d_wavelength_arrays():
    r = compute_bgs_snr(19.5, nexp=1, texp=180.0, half_light_radius=1.5)
    assert r["snr_s5_t"].shape == r["lam_t"].shape
    assert r["snr_d_t"].shape == r["lam_t"].shape
    assert np.ndim(r["snr_r_s5"]) == 0


def test_compute_bgs_snr_vectorised_matches_scalar():
    scalar_results = [compute_bgs_snr(m, nexp=1, texp=180.0, half_light_radius=1.5)
                       for m in MAGS]
    vec = compute_bgs_snr(np.array(MAGS), nexp=1, texp=180.0, half_light_radius=1.5)
    for i, m in enumerate(MAGS):
        assert vec["snr_r_s5"][i] == pytest.approx(scalar_results[i]["snr_r_s5"], rel=1e-10)
        assert vec["snr_r_d"][i] == pytest.approx(scalar_results[i]["snr_r_d"], rel=1e-10)


def test_point_source_capture_is_the_psf_capture():
    r = compute_bgs_snr(19.5, nexp=1, texp=180.0, half_light_radius=0.0, seeing_fwhm=1.1)
    assert r["capture_frac_s5"] == pytest.approx(point_source_capture_fraction(1.02, 1.1))
    assert r["capture_frac_d"] == pytest.approx(point_source_capture_fraction(1.51, 1.1))


def test_smaller_fiber_captures_less_extended_flux():
    # Spec-S5's baseline 107 um fiber (1.02") is smaller than DESI's (1.51"),
    # so for the same extended source it should capture a smaller fraction.
    r = compute_bgs_snr(19.5, nexp=1, texp=180.0, half_light_radius=1.5)
    assert 0.0 < r["capture_frac_s5"] < r["capture_frac_d"] < 1.0


@pytest.mark.parametrize("m", MAGS)
def test_compute_redshift_success_desi(m):
    p = compute_redshift_success(m, instrument="desi")
    assert p == pytest.approx(EXPECTED_P_DESI[m], rel=1e-4)


@pytest.mark.parametrize("m", MAGS)
def test_compute_redshift_success_spec5(m):
    p = compute_redshift_success(m, instrument="spec5")
    assert p == pytest.approx(EXPECTED_P_SPEC5[m], rel=1e-4)


def test_redshift_success_is_probability():
    p = compute_redshift_success(np.linspace(16.0, 23.0, 20), instrument="spec5")
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_redshift_success_past_sample_limit_is_zero():
    limit = _R_FIBER_SAMPLE_LIMIT
    assert compute_redshift_success(limit, instrument="desi") > 0.3
    assert compute_redshift_success(limit + 0.01, instrument="desi") == 0.0
    assert compute_redshift_success(30.0, instrument="desi") == 0.0


def test_calibration_curve_is_monotone():
    p = _load_success_curve("bright")(np.linspace(16.0, _R_FIBER_SAMPLE_LIMIT, 3000))
    assert np.all(np.diff(p) <= 1e-12)


def test_calibration_has_no_step_at_old_binning_edge():
    # The table used to stop at 22, with a step from 0.80 to 0 right after it.
    p = _load_success_curve("bright")(np.array([21.95, 22.05]))
    assert p[0] - p[1] < 0.1 and p[1] > 0.7


def test_redshift_success_bright_extrapolation_clamps():
    # Brighter than the table's bright edge, clamp to the brightest
    # tabulated rate rather than extrapolating toward zero.
    bright_edge = compute_redshift_success(17.05, instrument="desi")
    assert compute_redshift_success(10.0, instrument="desi") == pytest.approx(bright_edge)


def test_redshift_success_bad_instrument_raises():
    with pytest.raises(ValueError):
        compute_redshift_success(19.5, instrument="not-a-real-instrument")


def test_redshift_success_bad_bgs_type_raises():
    with pytest.raises(ValueError):
        compute_redshift_success(19.5, bgs_type="faint")


def test_fiber_capture_fraction_point_source():
    assert fiber_capture_fraction(0.0, 1.51) == pytest.approx(1.0)
    assert fiber_capture_fraction(np.nan, 1.51) == pytest.approx(1.0)


def test_fiber_capture_fraction_decreases_with_size():
    fracs = [fiber_capture_fraction(hlr, 1.51) for hlr in [0.5, 1.5, 3.0, 8.0]]
    assert all(f1 > f2 for f1, f2 in zip(fracs, fracs[1:]))


def test_fiber_capture_fraction_vectorised():
    hlr = np.array([0.0, 0.5, 1.5, 5.0])
    out = fiber_capture_fraction(hlr, 1.51)
    assert out.shape == hlr.shape
    for i, h in enumerate(hlr):
        assert out[i] == pytest.approx(fiber_capture_fraction(float(h), 1.51), rel=1e-10)


TEXP_GRID = [30.0, 60.0, 180.0, 600.0, 1800.0, 7200.0]
FAINT_MAGS = np.linspace(19.0, 25.0, 61)


def test_success_non_decreasing_in_texp():
    p = np.array([compute_redshift_success(FAINT_MAGS, texp=t) for t in TEXP_GRID])
    assert np.all(np.diff(p, axis=0) >= -1e-12)
    # and exposure time has a real effect, not just a non-negative one
    assert compute_redshift_success(22.5, texp=1800.0) > compute_redshift_success(22.5, texp=60.0) + 0.3


def test_success_non_decreasing_in_total_exposure():
    p = np.array([compute_redshift_success(FAINT_MAGS, nexp=n, texp=n * 180.0)
                  for n in [1, 2, 4, 8, 16]])
    assert np.all(np.diff(p, axis=0) >= -1e-12)


@pytest.mark.parametrize("m", [19.0, 20.5, 21.5, 22.0, 22.5])
@pytest.mark.parametrize("hlr", [1.0, 1.5, 4.0])
def test_desi_equivalent_mag_is_exact_inversion(m, hlr):
    target = compute_bgs_snr(m, 1, 180.0, half_light_radius=hlr)["snr_r_s5"]
    exact = brentq(
        lambda x: compute_bgs_snr(x, 1, 180.0, half_light_radius=hlr)["snr_r_d"] - target,
        10.0, 30.0, xtol=1e-8)
    got = _desi_equivalent_mag(np.atleast_1d(target), np.atleast_1d(hlr), 1.0)[0]
    assert got == pytest.approx(exact, abs=1e-4)


def test_spec5_gain_is_not_the_source_poisson_shift():
    # In the sky-limited r-arm the true equivalent-magnitude gain is smaller
    # than the m - 5 log10(SNR ratio) shortcut implies.
    m = 22.0
    r = compute_bgs_snr(m, 1, 180.0)
    shortcut = 5.0 * np.log10(r["snr_r_s5"] / r["snr_r_d"])
    exact = m - _desi_equivalent_mag(np.atleast_1d(r["snr_r_s5"]), np.array([1.5]), 1.0)[0]
    assert 0.0 < exact < shortcut - 0.3


@pytest.mark.parametrize("fwhm, expected", [(0.8, 0.690), (1.1, 0.807), (1.4, 0.878)])
def test_aperture_penalty_is_relative_to_point_source(fwhm, expected):
    # hlr=1.5", n=1: PSF-aware penalty is well above the intrinsic ratio of
    # 0.54, because the throughput tables already hold point-source losses.
    # Here the tables and the observation share the same seeing.
    scale_s5 = _fiber_source_scale(1.5, 1.02, 1.0, fwhm, table_seeing_fwhm=fwhm)
    scale_d = _fiber_source_scale(1.5, 1.51, 1.0, fwhm, table_seeing_fwhm=fwhm)
    assert scale_s5 / scale_d == pytest.approx(expected, abs=2e-3)


SEEING_GRID = [0.6, 0.8, 1.0, 1.1, 1.3, 1.5, 1.8]


@pytest.mark.parametrize("hlr", [0.0, 0.05, 0.5, 1.0, 1.5, 3.0, 8.0])
def test_success_non_increasing_in_seeing(hlr):
    mags = np.linspace(20.0, 25.0, 51)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.array([compute_redshift_success(mags, half_light_radius=hlr, seeing_fwhm=s)
                      for s in SEEING_GRID])
    assert np.all(np.diff(p, axis=0) <= 1e-12)
    # and seeing has a real effect somewhere in this range
    assert np.max(p[0] - p[-1]) > 0.01


@pytest.mark.parametrize("fiber", [1.02, 1.144, 1.51])
@pytest.mark.parametrize("seeing", [0.8, 1.5])
def test_point_source_scales_with_capture_at_observing_seeing(fiber, seeing):
    ratio = _fiber_source_scale(0.0, fiber, 1.0, seeing) / \
        _fiber_source_scale(0.0, fiber, 1.0, _TABLE_SEEING_FWHM)
    expected = point_source_capture_fraction(fiber, seeing) / \
        point_source_capture_fraction(fiber, _TABLE_SEEING_FWHM)
    assert ratio == pytest.approx(expected)


def test_compact_source_has_no_relative_aperture_penalty():
    r = compute_bgs_snr(19.5, 1, 180.0, half_light_radius=0.0)
    assert r["source_scale_s5"] == pytest.approx(r["source_scale_d"])
    # FIBERFLUX of a point source is fainter than its total mag by C(PSF_LS, 1.5")
    assert r["source_scale_d"] == pytest.approx(1.0 / point_source_capture_fraction(1.5, 1.0))


def test_desi_fiberflux_not_charged_a_second_fiber_loss():
    # At the FIBERFLUX reference seeing and (nearly) its fiber, the galaxy's
    # shape cancels: the fiber magnitude already encodes its aperture loss.
    scales = [_fiber_source_scale(h, 1.51, 1.0, 1.0, table_seeing_fwhm=1.0)
              for h in [1.0, 1.5, 4.0]]
    assert scales == pytest.approx([1.0 / point_source_capture_fraction(1.51, 1.0)] * 3, rel=0.015)


def test_seeing_capture_limits():
    # Compact galaxy -> PSF; much better seeing than the galaxy -> intrinsic profile
    assert seeing_capture_fraction(1e-3, 1.5, 1.0) == pytest.approx(
        point_source_capture_fraction(1.5, 1.0), abs=1e-4)
    assert seeing_capture_fraction(1.5, 1.51, 0.01) == pytest.approx(
        fiber_capture_fraction(1.5, 1.51), abs=1e-3)
    with pytest.raises(ValueError):
        seeing_capture_fraction(1.5, 1.51, 0.0)


def test_seeing_capture_vectorised_with_point_sources():
    hlr = np.array([0.0, np.nan, 0.5, 1.5, 5.0])
    out = seeing_capture_fraction(hlr, 1.51, 1.1)
    for i, h in enumerate(hlr):
        assert out[i] == pytest.approx(seeing_capture_fraction(float(h), 1.51, 1.1))


def test_brighter_sky_lowers_spec5_success():
    p = [compute_redshift_success(22.0, sky_scale=s) for s in [1.0, 2.0, 5.0]]
    assert p[0] > p[1] > p[2]


def test_sky_scale_leaves_desi_calibration_alone():
    assert compute_redshift_success(22.0, instrument="desi", sky_scale=5.0) == \
        compute_redshift_success(22.0, instrument="desi")


def test_warns_outside_calibrated_size_range():
    with pytest.warns(UserWarning, match="calibration sample"):
        compute_redshift_success(np.array([19.5, 20.0]), half_light_radius=np.array([0.5, 1.5]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_redshift_success(19.5, half_light_radius=1.0)


def test_nan_magnitude_propagates():
    p = compute_redshift_success(np.array([np.nan, 20.0]), instrument="spec5")
    assert np.isnan(p[0]) and np.isfinite(p[1])


def test_bad_fiber_diameter_raises_value_error():
    with pytest.raises(ValueError):
        compute_bgs_snr(19.5, 1, 180.0, fiber_diameter=100)
