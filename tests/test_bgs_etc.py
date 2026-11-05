"""
Unit tests for spec5.instrument.bgs_etc and spec5.instrument.galaxy_photometry.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from spec5.instrument.bgs_etc import (
    DESI_TABLE_SEEING_FWHM,
    _R_FIBER_SAMPLE_LIMIT,
    SPEC5_TABLE_SEEING_FWHM,
    _desi_equivalent_mag,
    _fiber_source_scale,
    _load_success_curve,
    compute_bgs_snr,
    compute_redshift_success,
)
from spec5.instrument.galaxy_photometry import (
    _FWHM_TO_SIGMA,
    _moffat_alpha,
    _ring_kernel,
    fiber_capture_fraction,
    point_source_capture_fraction,
    seeing_capture_fraction,
)

MAGS = [17.5, 19.5, 20.5, 21.5]

# Expected values evaluated at nexp=1, texp=180.0, half_light_radius=1.5,
# fiber_diameter=107 (baseline Spec-S5 design).
EXPECTED_SNR_R_S5 = {17.5: 22.469989, 19.5: 7.603293, 20.5: 3.919652, 21.5: 1.824044}
EXPECTED_SNR_R_D  = {17.5: 18.494695, 19.5: 6.052377, 20.5: 3.036061, 21.5: 1.381136}

EXPECTED_P_DESI  = {17.5: 0.989070, 19.5: 0.986045, 20.5: 0.986045, 21.5: 0.950944}
EXPECTED_P_SPEC5 = {17.5: 0.989070, 19.5: 0.986045, 20.5: 0.986045, 21.5: 0.980413}


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
    assert r["capture_frac_s5"] == pytest.approx(
        point_source_capture_fraction(1.02, 1.1, psf="moffat"))
    assert r["capture_frac_d"] == pytest.approx(
        point_source_capture_fraction(1.51, 1.1, psf="moffat"))


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
    # In the sky-limited r-arm SNR scales with flux, so the true
    # equivalent-magnitude gain is about half the m - 5 log10(SNR ratio)
    # shortcut, which assumes source-Poisson noise.
    m = 22.0
    r = compute_bgs_snr(m, 1, 180.0)
    shortcut = 5.0 * np.log10(r["snr_r_s5"] / r["snr_r_d"])
    exact = m - _desi_equivalent_mag(np.atleast_1d(r["snr_r_s5"]), np.array([1.5]), 1.0)[0]
    assert 0.0 < exact < 0.7 * shortcut


@pytest.mark.parametrize("fwhm, expected", [(0.8, 0.720), (1.1, 0.812), (1.4, 0.873)])
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


@pytest.mark.parametrize("fiber, table", [(1.02, SPEC5_TABLE_SEEING_FWHM),
                                          (1.144, SPEC5_TABLE_SEEING_FWHM),
                                          (1.51, DESI_TABLE_SEEING_FWHM)])
@pytest.mark.parametrize("seeing", [0.8, 1.5])
def test_point_source_scales_with_capture_at_observing_seeing(fiber, table, seeing):
    ratio = _fiber_source_scale(0.0, fiber, 1.0, seeing, table) / \
        _fiber_source_scale(0.0, fiber, 1.0, table, table)
    expected = point_source_capture_fraction(fiber, seeing, psf="moffat") / \
        point_source_capture_fraction(fiber, table, psf="moffat")
    assert ratio == pytest.approx(expected)


def test_compact_source_defaults_to_each_table_psf():
    # By default each instrument is observed at its own table's seeing (0.96"
    # for Spec-S5, 1.1" for DESI), so a point source is unscaled apart from
    # the FIBERFLUX convention.
    r = compute_bgs_snr(19.5, 1, 180.0, half_light_radius=0.0)
    fiberflux = point_source_capture_fraction(1.5, 1.0, psf="gaussian")
    assert r["seeing_fwhm"] == SPEC5_TABLE_SEEING_FWHM
    assert r["desi_seeing_fwhm"] == DESI_TABLE_SEEING_FWHM
    assert r["source_scale_s5"] == pytest.approx(1.0 / fiberflux)
    assert r["source_scale_d"] == pytest.approx(1.0 / fiberflux)


def test_compact_source_in_desi_nominal_seeing():
    # In 1.1" seeing Spec-S5 loses the extra light pushed out of its 1.02"
    # fiber relative to its 0.96" table.
    r = compute_bgs_snr(19.5, 1, 180.0, half_light_radius=0.0, seeing_fwhm=1.1)
    fiberflux = point_source_capture_fraction(1.5, 1.0, psf="gaussian")
    s5_seeing_loss = (point_source_capture_fraction(1.02, 1.1, psf="moffat")
                      / point_source_capture_fraction(1.02, 0.96, psf="moffat"))
    assert r["source_scale_s5"] == pytest.approx(s5_seeing_loss / fiberflux)
    assert 0.8 < s5_seeing_loss < 0.85


def test_desi_seeing_option_leaves_spec5_alone():
    a = compute_bgs_snr(20.5, 1, 180.0)
    b = compute_bgs_snr(20.5, 1, 180.0, desi_seeing_fwhm=1.4)
    assert b["snr_r_s5"] == pytest.approx(a["snr_r_s5"])
    assert b["snr_r_d"] < a["snr_r_d"]


def test_desi_fiberflux_not_charged_a_second_fiber_loss():
    # At the FIBERFLUX reference seeing and (nearly) its fiber, the galaxy's
    # shape cancels: the fiber magnitude already encodes its aperture loss.
    # Uses the FIBERFLUX convention's Gaussian PSF so the two captures match.
    scales = [_fiber_source_scale(h, 1.51, 1.0, 1.0, 1.0, psf="gaussian")
              for h in [1.0, 1.5, 4.0]]
    assert scales == pytest.approx(
        [1.0 / point_source_capture_fraction(1.51, 1.0, psf="gaussian")] * 3, rel=0.015)


def test_seeing_capture_limits():
    # Compact galaxy -> PSF; much better seeing than the galaxy -> intrinsic profile
    assert seeing_capture_fraction(1e-3, 1.5, 1.0, psf="gaussian") == pytest.approx(
        point_source_capture_fraction(1.5, 1.0, psf="gaussian"), abs=1e-4)
    assert seeing_capture_fraction(1.5, 1.51, 0.01, psf="gaussian") == pytest.approx(
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


# Moffat seeing model, checked against the throughput budget and Monte Carlo.

def test_moffat_point_capture_matches_budget():
    # SS5-0022 "Lateral offset throughput est": zero-offset star signal 0.4256
    # for beta = 3.5, 0.96" FWHM, 107 µm = 1.0218" fiber.
    assert point_source_capture_fraction(1.0218, 0.96, psf="moffat") == \
        pytest.approx(0.4256, rel=1e-3)


def test_moffat_galaxy_capture_matches_budget():
    # Same tab: 0.35" Sersic galaxy, 0.86" FWHM, zero offset gives 0.3428.
    # Reproduced for an exponential profile (n = 1).
    assert seeing_capture_fraction(0.35, 1.0218, 0.86, sersic_n=1.0, psf="moffat") == \
        pytest.approx(0.3428, rel=2e-3)


def test_ring_kernel_reproduces_gaussian_rice_kernel():
    from scipy.stats import ncx2
    fiber_radius, fwhm = 0.51, 1.1
    sigma = fwhm * _FWHM_TO_SIGMA
    r = np.array([0.05, 0.3, 0.51, 0.8, 1.5, 3.0])
    exact = ncx2.cdf((fiber_radius / sigma) ** 2, 2, (r / sigma) ** 2)
    np.testing.assert_allclose(_ring_kernel(r, fiber_radius, fwhm, "gaussian", 3.5),
                               exact, atol=5e-4)


@pytest.mark.parametrize("hlr, fiber, fwhm", [(0.5, 1.02, 0.96), (1.5, 1.02, 1.1),
                                              (1.5, 1.51, 1.1), (3.0, 1.51, 0.8)])
def test_moffat_galaxy_capture_matches_monte_carlo(hlr, fiber, fwhm):
    rng = np.random.default_rng(12345)
    n_samp, beta = 1_000_000, 3.5
    # Exponential disk: enclosed light follows Gamma(2), with b_1 = 1.678.
    from scipy.special import gammaincinv
    r_gal = hlr * rng.gamma(2.0, 1.0, n_samp) / gammaincinv(2.0, 0.5)
    # Moffat radial offsets by inverting its enclosed-light fraction.
    alpha = _moffat_alpha(fwhm, beta)
    r_psf = alpha * np.sqrt(rng.uniform(size=n_samp) ** (1.0 / (1.0 - beta)) - 1.0)
    th1, th2 = rng.uniform(0.0, 2.0 * np.pi, (2, n_samp))
    x = r_gal * np.cos(th1) + r_psf * np.cos(th2)
    y = r_gal * np.sin(th1) + r_psf * np.sin(th2)
    mc = np.mean(x ** 2 + y ** 2 < (fiber / 2.0) ** 2)
    got = seeing_capture_fraction(hlr, fiber, fwhm, psf="moffat")
    assert got == pytest.approx(mc, abs=1.5e-3)


def test_moffat_capture_limits():
    assert seeing_capture_fraction(1e-3, 1.02, 1.1, psf="moffat") == pytest.approx(
        point_source_capture_fraction(1.02, 1.1, psf="moffat"), abs=5e-4)
    assert seeing_capture_fraction(1.5, 1.51, 0.01, psf="moffat") == pytest.approx(
        fiber_capture_fraction(1.5, 1.51), abs=1e-3)
    # Moffat wings put less light in a fiber than a Gaussian of the same FWHM.
    assert point_source_capture_fraction(1.51, 1.1, psf="moffat") < \
        point_source_capture_fraction(1.51, 1.1, psf="gaussian")


def test_unknown_psf_raises():
    with pytest.raises(ValueError):
        seeing_capture_fraction(1.5, 1.51, 1.1, psf="airy")


def test_public_capture_functions_default_to_moffat():
    # The package's PSF model is Moffat; direct callers get it by default.
    assert point_source_capture_fraction(1.02, 0.96) == \
        point_source_capture_fraction(1.02, 0.96, psf="moffat")
    assert seeing_capture_fraction(1.5, 1.02, 0.96) == \
        seeing_capture_fraction(1.5, 1.02, 0.96, psf="moffat")
