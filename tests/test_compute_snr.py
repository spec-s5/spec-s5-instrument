"""
Unit tests for spec5.instrument.stellar_etc.compute_snr.
"""

import numpy as np
import pytest

from spec5.instrument.stellar_etc import compute_snr

MAGS = [19, 20, 21, 22, 23]

# Expected values evaluated at nexp=3, texp=3600, fiber_diameter=107
EXPECTED_VRAD_ERR = {
    19: 0.837497,
    20: 1.545124,
    21: 3.144261,
    22: 6.977976,
    23: 16.346059,
}

EXPECTED_SNR_S5_T = {
    19: [35.2056, 41.1047, 47.4924, 53.0238, 50.5677, 45.8222,
         46.4364, 46.2794, 43.5704, 39.8655, 41.6214, 39.8507, 39.7286, 28.3014],
    20: [20.1029, 23.5582, 27.3984, 29.3990, 28.5959, 25.5140,
         25.5258, 25.5123, 23.6378, 21.3997, 22.0486, 20.8468, 21.1353, 14.2722],
    21: [10.5169, 12.3896, 14.5483, 14.7935, 14.7607, 12.9066,
         12.7113, 12.7481, 11.5940, 10.3748, 10.5426, 9.8444, 10.1500, 6.5205],
    22: [4.9667, 5.8813, 6.9731, 6.7464, 6.8832, 5.9122,
         5.7476, 5.7802, 5.1806, 4.5957, 4.6242, 4.2813, 4.4656, 2.7770],
    23: [2.1601, 2.5667, 3.0637, 2.8701, 2.9689, 2.5219,
         2.4329, 2.4506, 2.1781, 1.9230, 1.9247, 1.7741, 1.8617, 1.1388],
}

EXPECTED_SNR_MEDIAN_ZARM = {
    19: 39.858114,
    20: 21.267523,
    21: 10.262424,
    22:  4.530664,
    23:  1.892340,
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
