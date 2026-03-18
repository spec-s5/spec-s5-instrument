"""
Unit tests for spec_s5_instrument.etc.compute_snr.
"""

import numpy as np
import pytest

from spec_s5_instrument.etc import compute_snr

MAGS = [19, 20, 21, 22, 23]

# Expected values evaluated at nexp=3, texp=3600, fiber_diameter=107
EXPECTED_VRAD_ERR = {
    19: 0.436025,
    20: 1.070315,
    21: 2.627312,
    22: 6.449286,
    23: 15.831121,
}

EXPECTED_SNR_S5_T = {
    19: [92.0885, 109.7718, 131.8408, 120.0333, 125.6408, 105.7087,
         101.3273, 102.2016, 90.2221, 79.3526, 79.0961, 72.6606, 76.6033, 46.278],
    20: [36.6611, 43.701, 52.4868, 47.7861, 50.0185, 42.0834,
         40.3391, 40.6872, 35.9181, 31.5908, 31.4887, 28.9267, 30.4963, 18.4236],
    21: [14.595, 17.3977, 20.8954, 19.024, 19.9127, 16.7537,
         16.0593, 16.1979, 14.2992, 12.5765, 12.5359, 11.5159, 12.1408, 7.3346],
    22: [5.8104, 6.9261, 8.3186, 7.5736, 7.9274, 6.6698,
         6.3933, 6.4485, 5.6926, 5.0068, 4.9906, 4.5846, 4.8333, 2.9199],
    23: [2.3132, 2.7573, 3.3117, 3.0151, 3.156, 2.6553,
         2.5452, 2.5672, 2.2663, 1.9932, 1.9868, 1.8252, 1.9242, 1.1625],
}

EXPECTED_SNR_MEDIAN_ZARM = {
    19: 77.849716,
    20: 30.992530,
    21: 12.338348,
    22:  4.911985,
    23:  1.955496,
}


@pytest.mark.parametrize("m", MAGS)
def test_vrad_err(m):
    r = compute_snr(m=m, nexp=3, texp=3600.0)
    assert r["VRAD_ERR"] == pytest.approx(EXPECTED_VRAD_ERR[m], rel=1e-4)


@pytest.mark.parametrize("m", MAGS)
def test_snr_s5_t(m):
    r = compute_snr(m=m, nexp=3, texp=3600.0)
    np.testing.assert_allclose(r["snr_s5_t"], EXPECTED_SNR_S5_T[m], rtol=1e-4)


@pytest.mark.parametrize("m", MAGS)
def test_snr_median_zarm(m):
    r = compute_snr(m=m, nexp=3, texp=3600.0)
    i_z = (r["lam_t"] > 7470.) & (r["lam_t"] < 9800.)
    snr_median = np.median(r["snr_s5_t"][i_z])
    assert snr_median == pytest.approx(EXPECTED_SNR_MEDIAN_ZARM[m], rel=1e-4)
