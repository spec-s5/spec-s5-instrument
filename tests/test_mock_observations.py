"""
Unit tests for spec5.instrument.mock_observations.
"""

import numpy as np
import pytest

from spec5.instrument.mock_observations import (
    _BC_Z,
    _GALCEN_DISTANCE,
    _M_BOL_SUN,
    _luminosity_to_lsst_z,
    galactocentric_to_observed,
    observe_with_spec5,
)

# ---------------------------------------------------------------------------
# Shared test fixture — 5 stars as a numpy structured array
# ---------------------------------------------------------------------------

_DTYPE = [('x','f8'), ('y','f8'), ('z','f8'),
          ('vx','f8'), ('vy','f8'), ('vz','f8'),
          ('luminosity','f8')]

STARS = np.array([
    (-8.122,  0.0,  0.0,    0.0,  220.0,   0.0,  10.0),
    ( 0.0,    8.0,  0.5,  -50.0,  -50.0,  20.0,  50.0),
    (-5.0,   -5.0, -1.0,  100.0,  100.0,  10.0, 100.0),
    (-15.0,   3.0,  2.0,  -80.0,   30.0,  -5.0, 200.0),
    ( 2.0,    2.0,  8.0,   50.0,  -80.0,  30.0,   1.0),
], dtype=_DTYPE)

# Regression values computed from a reference run
EXPECTED_GIANT = {
    'ra':       [12.790393, 285.954447, 189.793714,  87.889697, 240.042832],
    'dec':      [-27.168146,  11.499491, -72.596561,  56.424404,   1.127702],
    'distance': [0.020800,   11.410350,   5.982370,   7.760443,  13.043082],
    'vrad':     [7.796512, -251.509930, 166.766300,  -4.268506,  -7.542463],
    'pmra':     [250.787111, -1.785476,  -0.063288,   2.628330,  -3.426883],
    'pmdec':    [-146.980832, -2.541667,   1.101886,  -5.824304,  -4.057311],
    'lsst_z':   [3.130318,   15.079070,  12.924366,  12.736858,  19.616901],
}
EXPECTED_DWARF = {
    'lsst_z':   [3.530318,   15.479070,  13.324366,  13.136858,  20.016901],
}

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_invalid_star_type_raises():
    with pytest.raises(ValueError, match="star_type"):
        galactocentric_to_observed(STARS, star_type='subdwarf')


def test_no_luminosity_returns_nan_lsst_z():
    dtype_nolum = [('x','f8'), ('y','f8'), ('z','f8'), ('vx','f8'), ('vy','f8'), ('vz','f8')]
    stars_nolum = np.empty(len(STARS), dtype=dtype_nolum)
    for col in ('x', 'y', 'z', 'vx', 'vy', 'vz'):
        stars_nolum[col] = STARS[col]
    out = galactocentric_to_observed(stars_nolum)
    assert np.all(np.isnan(out['lsst_z']))
    # All other columns should still be finite
    for col in ('ra', 'dec', 'distance', 'vrad', 'pmra', 'pmdec'):
        assert np.all(np.isfinite(out[col]))


def test_negative_luminosity_raises():
    bad = STARS.copy()
    bad['luminosity'][0] = -1.0
    with pytest.raises(ValueError, match="luminosity"):
        galactocentric_to_observed(bad)


def test_zero_luminosity_raises():
    bad = STARS.copy()
    bad['luminosity'][0] = 0.0
    with pytest.raises(ValueError, match="luminosity"):
        galactocentric_to_observed(bad)


# ---------------------------------------------------------------------------
# Output type mirrors input type
# ---------------------------------------------------------------------------

def test_output_is_structured_array_for_structured_input():
    out = galactocentric_to_observed(STARS)
    assert isinstance(out, np.ndarray)
    assert out.dtype.names is not None   # structured array has named fields


def test_output_is_dataframe_for_dataframe_input():
    pd = pytest.importorskip("pandas")
    df_in = pd.DataFrame({col: STARS[col] for col in STARS.dtype.names})
    df_out = galactocentric_to_observed(df_in)
    assert isinstance(df_out, pd.DataFrame)


def test_dataframe_and_structured_array_agree():
    pd = pytest.importorskip("pandas")
    df_in  = pd.DataFrame({col: STARS[col] for col in STARS.dtype.names})
    out_sa = galactocentric_to_observed(STARS)
    out_df = galactocentric_to_observed(df_in)
    for col in ('ra', 'dec', 'distance', 'vrad', 'pmra', 'pmdec', 'lsst_z'):
        np.testing.assert_allclose(out_sa[col], out_df[col].values, rtol=1e-10)


# ---------------------------------------------------------------------------
# Output shape and column names
# ---------------------------------------------------------------------------

_OUTPUT_COLS = ('ra', 'dec', 'distance', 'vrad', 'pmra', 'pmdec', 'lsst_z')

def test_output_columns():
    out = galactocentric_to_observed(STARS)
    assert set(out.dtype.names) == set(_OUTPUT_COLS)


def test_output_length():
    out = galactocentric_to_observed(STARS)
    assert len(out) == len(STARS)


def test_single_star_returns_length_one_array():
    out = galactocentric_to_observed(STARS[:1])
    assert len(out) == 1
    assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------

def test_ra_in_range():
    out = galactocentric_to_observed(STARS)
    assert np.all((out['ra'] >= 0) & (out['ra'] < 360))


def test_dec_in_range():
    out = galactocentric_to_observed(STARS)
    assert np.all((out['dec'] >= -90) & (out['dec'] <= 90))


def test_distance_positive():
    out = galactocentric_to_observed(STARS)
    assert np.all(out['distance'] > 0)


def test_distance_modulus_scaling():
    """Two identical stars at 10× different distances differ by exactly 5 mag."""
    near = np.array([(-8.122, 1.0, 0.0,  0.0, 220.0, 0.0, 10.0)], dtype=_DTYPE)
    far  = np.array([(-8.122, 10.0, 0.0, 0.0, 220.0, 0.0, 10.0)], dtype=_DTYPE)
    out_near = galactocentric_to_observed(near)
    out_far  = galactocentric_to_observed(far)
    delta_mag = out_far['lsst_z'][0] - out_near['lsst_z'][0]
    expected  = 5.0 * np.log10(out_far['distance'][0] / out_near['distance'][0])
    assert delta_mag == pytest.approx(expected, rel=1e-6)


def test_giant_brighter_in_z_than_dwarf():
    """Giants (Teff~4500K) have larger BC_z than dwarfs (Teff~5500K), so they are
    brighter in z for the same bolometric luminosity: M_z = M_bol - BC_z is smaller."""
    out_g = galactocentric_to_observed(STARS, star_type='giant')
    out_d = galactocentric_to_observed(STARS, star_type='dwarf')
    assert np.all(out_g['lsst_z'] < out_d['lsst_z'])


def test_giant_dwarf_z_difference():
    """The z-band difference equals the difference in bolometric corrections:
    m_z(dwarf) - m_z(giant) = BC_z(giant) - BC_z(dwarf) = +0.70 - 0.30 = +0.40."""
    out_g = galactocentric_to_observed(STARS, star_type='giant')
    out_d = galactocentric_to_observed(STARS, star_type='dwarf')
    expected_diff = _BC_Z['giant'] - _BC_Z['dwarf']   # = +0.70 - 0.30 = +0.40
    np.testing.assert_allclose(
        out_d['lsst_z'] - out_g['lsst_z'],
        np.full(len(STARS), expected_diff),
        rtol=1e-10,
    )


def test_solar_luminosity_at_10pc():
    """L=1 L_sun at exactly 10 pc heliocentric should give m_z = M_z (distance modulus = 0)."""
    # Place star 10 pc (0.010 kpc) along +Z from the Sun.
    # Sun is at (x=-galcen_distance, y=0, z=z_sun) in Galactocentric.
    from spec5.instrument.mock_observations import _Z_SUN
    star = np.array([
        (-_GALCEN_DISTANCE, 0.0, _Z_SUN + 0.010, 0.0, 0.0, 0.0, 1.0)
    ], dtype=_DTYPE)
    out = galactocentric_to_observed(star, star_type='giant')
    assert out['distance'][0] == pytest.approx(0.010, rel=1e-4)
    expected_mz = _M_BOL_SUN - _BC_Z['giant']   # M_z = M_bol - BC_z = 4.74 - 0.70 = 4.04
    assert out['lsst_z'][0] == pytest.approx(expected_mz, abs=0.01)


# ---------------------------------------------------------------------------
# Luminosity helper
# ---------------------------------------------------------------------------

def test_luminosity_helper_distance_scaling():
    """Apparent magnitude increases by 5 for ×10 distance."""
    m1 = _luminosity_to_lsst_z(10.0, 1.0,  'giant')
    m2 = _luminosity_to_lsst_z(10.0, 10.0, 'giant')
    assert m2 - m1 == pytest.approx(5.0, rel=1e-10)


def test_luminosity_helper_luminosity_scaling():
    """Doubling luminosity brightens by 2.5*log10(2) ≈ 0.753 mag."""
    m1 = _luminosity_to_lsst_z(1.0, 5.0, 'dwarf')
    m2 = _luminosity_to_lsst_z(2.0, 5.0, 'dwarf')
    assert m1 - m2 == pytest.approx(2.5 * np.log10(2), rel=1e-10)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", _OUTPUT_COLS)
def test_regression_giant(col):
    out = galactocentric_to_observed(STARS, star_type='giant')
    np.testing.assert_allclose(out[col], EXPECTED_GIANT[col], rtol=1e-4)


@pytest.mark.parametrize("col", ['lsst_z'])
def test_regression_dwarf(col):
    out = galactocentric_to_observed(STARS, star_type='dwarf')
    np.testing.assert_allclose(out[col], EXPECTED_DWARF[col], rtol=1e-4)


# ===========================================================================
# observe_with_spec5
# ===========================================================================

# Shared fixture: on-sky observables for the 5 test stars (giant)
@pytest.fixture(scope='module')
def observed_stars():
    return galactocentric_to_observed(STARS, star_type='giant')


# Expected error values (deterministic — no RNG involved)
EXPECTED_ERRORS = {
    'vrad_err':        [0.600000, 0.600139, 0.600003, 0.600002, 0.967323],
    'pm_err':          [0.00184179, 0.00531844, 0.00209805, 0.00195657, 0.09129012],
    'dist_err_frac':   [0.056332, 0.056332, 0.056332, 0.056332, 0.315718],
    'snr_median_zarm': [173432166.787589, 2881.569605, 20965.754589,
                        24918.043927, 44.105892],
}

# Expected noisy observations for seed=42
EXPECTED_OBS = {
    'vrad_obs':     [7.979342,   -252.134065, 167.216573,  -3.704166,  -9.429744],
    'pmra_obs':     [250.784712,   -1.784796,  -0.063952,   2.628297,  -3.504758],
    'pmdec_obs':    [-146.979213,  -2.537530,   1.102024,  -5.822098,  -4.014632],
    'distance_obs': [0.019793,    11.647370,   5.659229,   8.144466,  12.837490],
}


# ---------------------------------------------------------------------------
# Output type mirrors input type
# ---------------------------------------------------------------------------

def test_observe_output_is_structured_array(observed_stars):
    out = observe_with_spec5(observed_stars, seed=0)
    assert isinstance(out, np.ndarray)
    assert out.dtype.names is not None


def test_observe_output_is_dataframe_for_dataframe_input(observed_stars):
    pd = pytest.importorskip('pandas')
    df_in = pd.DataFrame({col: observed_stars[col] for col in observed_stars.dtype.names})
    out = observe_with_spec5(df_in, seed=0)
    assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Output shape and columns
# ---------------------------------------------------------------------------

_OBS_COLS = ('vrad_obs', 'pmra_obs', 'pmdec_obs', 'distance_obs',
             'vrad_err', 'pm_err', 'dist_err_frac', 'snr_median_zarm')

def test_observe_output_columns(observed_stars):
    out = observe_with_spec5(observed_stars, seed=0)
    assert set(out.dtype.names) == set(_OBS_COLS)


def test_observe_output_length(observed_stars):
    out = observe_with_spec5(observed_stars, seed=0)
    assert len(out) == len(observed_stars)


# ---------------------------------------------------------------------------
# Reproducibility and seeding
# ---------------------------------------------------------------------------

def test_observe_seed_reproducibility(observed_stars):
    out1 = observe_with_spec5(observed_stars, seed=42)
    out2 = observe_with_spec5(observed_stars, seed=42)
    np.testing.assert_array_equal(out1['vrad_obs'], out2['vrad_obs'])


def test_observe_different_seeds_differ(observed_stars):
    out1 = observe_with_spec5(observed_stars, seed=1)
    out2 = observe_with_spec5(observed_stars, seed=2)
    assert not np.allclose(out1['vrad_obs'], out2['vrad_obs'])


# ---------------------------------------------------------------------------
# NaN propagation
# ---------------------------------------------------------------------------

def test_observe_nan_lsst_z_gives_nan_outputs(observed_stars):
    stars_nan = observed_stars.copy()
    stars_nan['lsst_z'][2] = np.nan
    out = observe_with_spec5(stars_nan, seed=0)
    for col in _OBS_COLS:
        assert np.isnan(out[col][2]), f'{col}[2] should be NaN'
    # Other stars should be finite
    for col in _OBS_COLS:
        assert np.isfinite(out[col][0]), f'{col}[0] should be finite'


def test_observe_all_nan_lsst_z(observed_stars):
    stars_nan = observed_stars.copy()
    stars_nan['lsst_z'][:] = np.nan
    out = observe_with_spec5(stars_nan, seed=0)
    for col in _OBS_COLS:
        assert np.all(np.isnan(out[col]))


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------

def test_observe_errors_positive(observed_stars):
    out = observe_with_spec5(observed_stars, seed=0)
    assert np.all(out['vrad_err'][np.isfinite(out['vrad_err'])] > 0)
    assert np.all(out['dist_err_frac'][np.isfinite(out['dist_err_frac'])] > 0)


def test_observe_vrad_sys_floor(observed_stars):
    """vrad_err should be at least vrad_sys (default 0.6 km/s) for all stars."""
    out = observe_with_spec5(observed_stars, seed=0)
    finite = np.isfinite(out['vrad_err'])
    assert np.all(out['vrad_err'][finite] >= 0.6)


def test_observe_vrad_sys_zero_gives_smaller_errors(observed_stars):
    out_sys  = observe_with_spec5(observed_stars, vrad_sys=0.6, seed=0)
    out_stat = observe_with_spec5(observed_stars, vrad_sys=0.0, seed=0)
    finite = np.isfinite(out_sys['vrad_err']) & np.isfinite(out_stat['vrad_err'])
    assert np.all(out_sys['vrad_err'][finite] >= out_stat['vrad_err'][finite])


def test_observe_distance_obs_positive(observed_stars):
    """Observed distance should be positive for all stars."""
    out = observe_with_spec5(observed_stars, seed=0)
    finite = np.isfinite(out['distance_obs'])
    assert np.all(out['distance_obs'][finite] > 0)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", EXPECTED_ERRORS)
def test_observe_regression_errors(observed_stars, col):
    out = observe_with_spec5(observed_stars, seed=42)
    np.testing.assert_allclose(out[col], EXPECTED_ERRORS[col], rtol=2e-4)


@pytest.mark.parametrize("col", EXPECTED_OBS)
def test_observe_regression_noisy(observed_stars, col):
    out = observe_with_spec5(observed_stars, seed=42)
    np.testing.assert_allclose(out[col], EXPECTED_OBS[col], rtol=1e-4)
