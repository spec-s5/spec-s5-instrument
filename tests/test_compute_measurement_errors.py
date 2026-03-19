"""
Unit tests for spec5.instrument.stellar_etc.compute_measurement_errors.
"""

import warnings

import pytest

from spec5.instrument.stellar_etc import compute_measurement_errors

MAGS = [19, 20, 21, 22, 23]

# Expected values evaluated at texp=3600, nexp=3, fiber_diameter=107,
# pm_model='gaia_dr5', star_type='giant'
EXPECTED = {
    19: dict(vrad_err=0.741699, pm_err=0.055601, dist_err_frac=0.164930, snr_median_zarm=77.849716),
    20: dict(vrad_err=1.227019, pm_err=0.125927, dist_err_frac=0.403254, snr_median_zarm=30.992530),
    21: dict(vrad_err=2.694952, pm_err=0.301494, dist_err_frac=0.554783, snr_median_zarm=12.338348),
    22: dict(vrad_err=6.477136, pm_err=0.741982, dist_err_frac=0.623753, snr_median_zarm=4.911985),
    23: dict(vrad_err=15.842486, pm_err=1.848216, dist_err_frac=0.631276, snr_median_zarm=1.955496),
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
    assert r["pm_err"] == pytest.approx(EXPECTED[m]["pm_err"], rel=1e-4)


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


# When the input is Gaia G, the Gaia DR5 PM lookup uses G directly, so pm_err
# matches the original values (before the z→G conversion was introduced).
EXPECTED_PM_ERR_GAIA_G = {
    19: 0.040103,
    20: 0.087718,
    21: 0.205896,
    22: 0.502024,
    23: 1.245542,
}

@pytest.mark.parametrize("m", MAGS)
def test_pm_err_gaia_g_input(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_measurement_errors(m, mag_band='gaia_g')
    assert r["pm_err"] == pytest.approx(EXPECTED_PM_ERR_GAIA_G[m], rel=1e-4)
