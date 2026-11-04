"""
Unit tests for spec5.instrument.bgs_etc and spec5.instrument.galaxy_photometry.
"""

import numpy as np
import pytest

from spec5.instrument.bgs_etc import compute_bgs_snr, compute_redshift_success
from spec5.instrument.galaxy_photometry import fiber_capture_fraction

MAGS = [17.5, 19.5, 20.5, 21.5]

# Expected values evaluated at nexp=1, texp=180.0, half_light_radius=1.5,
# fiber_diameter=107 (baseline Spec-S5 design).
EXPECTED_SNR_R_S5 = {17.5: 15.546860, 19.5: 4.744678, 20.5: 2.267571, 21.5: 0.994018}
EXPECTED_SNR_R_D  = {17.5: 13.059599, 19.5: 3.845298, 20.5: 1.801807, 21.5: 0.779652}

EXPECTED_P_DESI  = {17.5: 0.990307, 19.5: 0.984078, 20.5: 0.986692, 21.5: 0.951211}
EXPECTED_P_SPEC5 = {17.5: 0.984300, 19.5: 0.984276, 20.5: 0.985452, 21.5: 0.985343}


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


def test_point_source_has_full_capture_fraction():
    r = compute_bgs_snr(19.5, nexp=1, texp=180.0, half_light_radius=0.0)
    assert r["capture_frac_s5"] == pytest.approx(1.0)
    assert r["capture_frac_d"] == pytest.approx(1.0)


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
