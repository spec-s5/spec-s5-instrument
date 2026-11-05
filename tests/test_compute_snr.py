"""
Unit tests for spec5.instrument.stellar_etc.compute_snr.
"""

import numpy as np
import pytest

from spec5.instrument.stellar_etc import compute_snr

MAGS = [19, 20, 21, 22, 23]

# Expected values evaluated at nexp=3, texp=3600, fiber_diameter=107
EXPECTED_VRAD_ERR = {
    19: 0.848254,
    20: 1.568332,
    21: 3.198593,
    22: 7.109870,
    23: 16.669293,
}

EXPECTED_SNR_S5_T = {
    19: [34.7704, 40.6569, 46.9456, 52.3707, 49.9709, 45.2559,
         45.8564, 45.6961, 43.0245, 39.3583, 41.0746, 39.3213, 39.1986, 27.8865],
    20: [19.8210, 23.2677, 27.0431, 28.9815, 28.2107, 25.1512,
         25.1570, 25.1408, 23.2939, 21.0828, 21.7108, 20.5236, 20.8067, 14.0288],
    21: [10.3448, 12.2114, 14.3284, 14.5494, 14.5287, 12.6931,
         12.4984, 12.5327, 11.3993, 10.1981, 10.3579, 9.6709, 9.9692, 6.3963],
    22: [4.8743, 5.7848, 6.8524, 6.6222, 6.7610, 5.8027,
         5.6407, 5.6715, 5.0847, 4.5100, 4.5361, 4.1998, 4.3789, 2.7208],
    23: [2.1167, 2.5212, 3.0060, 2.8140, 2.9123, 2.4722,
         2.3851, 2.4019, 2.1357, 1.8854, 1.8865, 1.7390, 1.8240, 1.1151],
}

EXPECTED_SNR_MEDIAN_ZARM = {
    19: 39.339789,
    20: 20.944793,
    21: 10.083673,
    22:  4.444482,
    23:  1.854714,
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


# Physics checks, independent of the pinned values above.

@pytest.mark.parametrize("snr_key, ph_key", [("snr_s5_t", "ph_s5_t"), ("snr_d_t", "ph_d_t")])
def test_bright_source_is_source_poisson_limited(snr_key, ph_key):
    # With source photons in the variance, a very bright star has SNR = sqrt(N).
    r = compute_snr(m=5.0, nexp=3, texp=3600.0)
    np.testing.assert_allclose(r[snr_key], np.sqrt(r[ph_key]), rtol=1e-3)


@pytest.mark.parametrize("snr_key", ["snr_s5_t", "snr_d_t"])
@pytest.mark.parametrize("bright_mag, per_mag", [(5.0, 10 ** 0.2), (32.0, 10 ** 0.4)])
def test_snr_slope_limits(snr_key, bright_mag, per_mag):
    # SNR scales as sqrt(flux) when source-limited, and as flux when
    # background-limited.
    r1 = compute_snr(m=bright_mag, nexp=3, texp=3600.0)
    r2 = compute_snr(m=bright_mag + 1.0, nexp=3, texp=3600.0)
    np.testing.assert_allclose(r1[snr_key] / r2[snr_key], per_mag, rtol=1e-3)
