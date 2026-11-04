"""
Unit tests for spec5.instrument.mock_observations.observe_bgs_with_spec5
and observe_bgs_with_desi.
"""

import numpy as np
import pytest

from spec5.instrument.mock_observations import (
    observe_bgs_with_desi,
    observe_bgs_with_spec5,
)

_DTYPE = [('r_fiber_mag', 'f8'), ('half_light_radius', 'f8'), ('redshift', 'f8')]

GALAXIES = np.array([
    (18.0, 2.0, 0.10),
    (20.5, 1.5, 0.15),
    (21.8, 3.0, 0.20),
    (np.nan, 1.5, 0.10),  # NaN magnitude -> all outputs NaN/False
], dtype=_DTYPE)


@pytest.mark.parametrize("observe_fn", [observe_bgs_with_spec5, observe_bgs_with_desi])
def test_nan_magnitude_propagates(observe_fn):
    out = observe_fn(GALAXIES, seed=0)
    assert np.isnan(out['redshift_obs'][-1])
    assert not out['success'][-1]
    assert np.isnan(out['success_prob'][-1])


@pytest.mark.parametrize("observe_fn", [observe_bgs_with_spec5, observe_bgs_with_desi])
def test_success_prob_in_unit_interval(observe_fn):
    out = observe_fn(GALAXIES, seed=0)
    valid = np.isfinite(out['success_prob'])
    assert valid.sum() == len(GALAXIES) - 1
    assert np.all(out['success_prob'][valid] >= 0.0)
    assert np.all(out['success_prob'][valid] <= 1.0)


@pytest.mark.parametrize("observe_fn", [observe_bgs_with_spec5, observe_bgs_with_desi])
def test_redshift_obs_matches_truth_on_success(observe_fn):
    out = observe_fn(GALAXIES, seed=0)
    valid = out['success']
    np.testing.assert_allclose(out['redshift_obs'][valid], GALAXIES['redshift'][valid])


@pytest.mark.parametrize("observe_fn", [observe_bgs_with_spec5, observe_bgs_with_desi])
def test_failure_gives_nan_redshift(observe_fn):
    out = observe_fn(GALAXIES, seed=0)
    failed = ~out['success'] & np.isfinite(out['success_prob'])
    assert np.all(np.isnan(out['redshift_obs'][failed]))


def test_seed_reproducibility():
    out1 = observe_bgs_with_spec5(GALAXIES, seed=123)
    out2 = observe_bgs_with_spec5(GALAXIES, seed=123)
    np.testing.assert_array_equal(out1['success'], out2['success'])


def test_dataframe_roundtrip():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        'r_fiber_mag': [18.0, 20.5],
        'half_light_radius': [2.0, 1.5],
        'redshift': [0.1, 0.15],
    })
    out = observe_bgs_with_spec5(df, seed=0)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ['redshift_obs', 'success', 'success_prob']


def test_very_faint_mostly_fails_very_bright_mostly_succeeds():
    dtype = [('r_fiber_mag', 'f8'), ('half_light_radius', 'f8'), ('redshift', 'f8')]
    n = 500
    bright = np.array([(18.0, 2.0, 0.1)] * n, dtype=dtype)
    faint  = np.array([(22.5, 2.0, 0.1)] * n, dtype=dtype)

    out_bright = observe_bgs_with_spec5(bright, seed=1)
    out_faint  = observe_bgs_with_spec5(faint, seed=1)

    assert out_bright['success'].mean() > 0.9
    assert out_faint['success'].mean() < out_bright['success'].mean()
