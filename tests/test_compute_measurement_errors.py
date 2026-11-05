"""
Unit tests for spec5.instrument.stellar_etc.compute_measurement_errors.
"""

import math
import warnings

import pytest

from spec5.instrument.stellar_etc import compute_measurement_errors

MAGS = [19, 20, 21, 22, 23]

# Expected values evaluated at texp=3600, nexp=3, fiber_diameter=107,
# pm_model='gaia_dr5', star_type='giant'.
# pm_err is NaN for z >= 21 because the default bp_rp=1.2 converts those to G > 20.7.
EXPECTED = {
    19: dict(vrad_err=1.039007, pm_err=0.055601, dist_err_frac=0.345694, snr_median_zarm=39.339789),
    20: dict(vrad_err=1.679186, pm_err=0.125927, dist_err_frac=0.480979, snr_median_zarm=20.944793),
    21: dict(vrad_err=3.254381, pm_err=None,      dist_err_frac=0.575208, snr_median_zarm=10.083673),
    22: dict(vrad_err=7.135142, pm_err=None,      dist_err_frac=0.628257, snr_median_zarm=4.444482),
    23: dict(vrad_err=16.680087, pm_err=None,      dist_err_frac=0.631276, snr_median_zarm=1.854714),
}


@pytest.mark.parametrize("m", MAGS)
def test_vrad_err(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_measurement_errors(m)
    assert r["vrad_err"] == pytest.approx(EXPECTED[m]["vrad_err"], rel=1e-4)


@pytest.mark.parametrize("m", MAGS)
def test_pm_err(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_measurement_errors(m)
    expected = EXPECTED[m]["pm_err"]
    if expected is None:
        assert math.isnan(r["pm_err"])
    else:
        assert r["pm_err"] == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("m", MAGS)
def test_dist_err_frac(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_measurement_errors(m)
    assert r["dist_err_frac"] == pytest.approx(EXPECTED[m]["dist_err_frac"], rel=1e-4)


@pytest.mark.parametrize("m", MAGS)
def test_snr_median_zarm(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_measurement_errors(m)
    assert r["snr_median_zarm"] == pytest.approx(EXPECTED[m]["snr_median_zarm"], rel=1e-4)


# When the input is Gaia G, the Gaia DR5 PM lookup uses G directly.
# G > 20.7 returns NaN.
EXPECTED_PM_ERR_GAIA_G = {
    19: 0.040103,
    20: 0.087718,
    21: None,      # G=21 > 20.7
    22: None,      # G=22 > 20.7
    23: None,      # G=23 > 20.7
}

@pytest.mark.parametrize("m", MAGS)
def test_pm_err_gaia_g_input(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_measurement_errors(m, mag_band='gaia_g')
    expected = EXPECTED_PM_ERR_GAIA_G[m]
    if expected is None:
        assert math.isnan(r["pm_err"])
    else:
        assert r["pm_err"] == pytest.approx(expected, rel=1e-4)
