"""
Unit tests for spec5.instrument.fiber.
"""

import numpy as np
import pytest

from spec5.instrument.fiber import (
    _allocate_targets,
    _cell_area_in_fov,
    mock_fiber_assign,
)

# ---------------------------------------------------------------------------
# Shared fixture — 200k sources uniform in a 60' radius FoV
# ---------------------------------------------------------------------------

_FOV_RADIUS = 60.0  # arcmin
_DTYPE = [('x', 'f8'), ('y', 'f8'), ('zmag', 'f8')]


def _uniform_catalog(n, radius, seed=0):
    rng = np.random.default_rng(seed)
    r     = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    cat = np.empty(n, dtype=_DTYPE)
    cat['x']    = r * np.cos(theta)
    cat['y']    = r * np.sin(theta)
    cat['zmag'] = rng.uniform(15.0, 22.0, n)
    return cat


CATALOG = _uniform_catalog(200_000, _FOV_RADIUS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_cell_area_sums_to_fov_area():
    for n_grid_cells in (1, 10, 65, 100):
        area = _cell_area_in_fov(n_grid_cells, _FOV_RADIUS)
        assert area.sum() == pytest.approx(np.pi * _FOV_RADIUS**2, rel=2e-3)
        assert area.max() <= (2.0 * _FOV_RADIUS / n_grid_cells)**2 + 1e-9


@pytest.mark.parametrize('n_targets', [1, 5000, 7999, 13000, 31000])
def test_allocate_targets_exact_total(n_targets):
    area = _cell_area_in_fov(100, _FOV_RADIUS)
    alloc = _allocate_targets(n_targets, area, np.random.default_rng(1))
    assert alloc.sum() == n_targets
    quota = n_targets * area / area.sum()
    assert np.all(alloc >= np.floor(quota))
    assert np.all(alloc <= np.ceil(quota))

# ---------------------------------------------------------------------------
# Target counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('priority_column', [None, 'zmag'])
@pytest.mark.parametrize('n_fibers', [5000, 8000, 13000, 20000, 31000])
def test_dense_field_returns_requested_count(n_fibers, priority_column):
    out = mock_fiber_assign(CATALOG, n_fibers, _FOV_RADIUS,
                            priority_column=priority_column, seed=2)
    # Exact up to cells that run out of objects (Poisson fluctuations)
    assert len(out) == pytest.approx(n_fibers, rel=0.01)
    assert len(out) <= n_fibers


def test_n_passes_multiplies_fibers():
    out = mock_fiber_assign(CATALOG, 6500, _FOV_RADIUS, n_passes=2, seed=3)
    assert len(out) == pytest.approx(13000, rel=0.01)


def test_sparse_field_returns_everything_in_fov():
    """Every cell has spare fibers (~130 fibers vs ~8 objects per cell)."""
    cat = _uniform_catalog(1000, 1.2 * _FOV_RADIUS, seed=4)
    in_fov = np.hypot(cat['x'], cat['y']) <= _FOV_RADIUS
    out = mock_fiber_assign(cat, 13000, _FOV_RADIUS, n_grid_cells=10, seed=5)
    assert len(out) == in_fov.sum()
    np.testing.assert_array_equal(out, cat[in_fov])


def test_objects_outside_fov_never_selected():
    cat = _uniform_catalog(50_000, 2.0 * _FOV_RADIUS, seed=6)
    out = mock_fiber_assign(cat, 5000, _FOV_RADIUS, seed=7)
    assert np.all(np.hypot(out['x'], out['y']) <= _FOV_RADIUS)


def test_clustered_catalog_is_density_capped():
    """A dense clump keeps only its share of fibers; unused fibers are not moved."""
    rng = np.random.default_rng(8)
    clump = np.empty(100_000, dtype=_DTYPE)
    clump['x'] = rng.normal(0.0, 1.0, len(clump))
    clump['y'] = rng.normal(0.0, 1.0, len(clump))
    clump['zmag'] = 20.0
    out = mock_fiber_assign(clump, 13000, _FOV_RADIUS, seed=9)
    # The clump covers roughly a 3' radius, i.e. (3/60)^2 of the FoV area
    assert len(out) < 0.01 * 13000

# ---------------------------------------------------------------------------
# Spatial uniformity
# ---------------------------------------------------------------------------

def test_edge_density_matches_interior():
    out = mock_fiber_assign(CATALOG, 100_000, _FOV_RADIUS, seed=10)
    r = np.hypot(out['x'], out['y'])
    edge = 0.9 * _FOV_RADIUS
    dens_inner = np.sum(r < edge) / (np.pi * edge**2)
    dens_outer = np.sum(r >= edge) / (np.pi * (_FOV_RADIUS**2 - edge**2))
    assert dens_outer / dens_inner == pytest.approx(1.0, abs=0.02)

# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

def _cell_index(cat, n_grid_cells=100):
    size = 2.0 * _FOV_RADIUS / n_grid_cells
    ix = ((cat['x'] + _FOV_RADIUS) // size).astype(int)
    iy = ((cat['y'] + _FOV_RADIUS) // size).astype(int)
    return ix * n_grid_cells + iy


@pytest.mark.parametrize('ascending', [True, False])
def test_priority_within_cell(ascending):
    out = mock_fiber_assign(CATALOG, 20000, _FOV_RADIUS, priority_column='zmag',
                            priority_order_ascending=ascending, seed=11)
    selected = np.isin(CATALOG['zmag'], out['zmag'])
    cell = _cell_index(CATALOG)
    sign = 1.0 if ascending else -1.0
    for c in np.unique(cell[selected])[:200]:
        in_cell = cell == c
        kept    = sign * CATALOG['zmag'][in_cell & selected]
        dropped = sign * CATALOG['zmag'][in_cell & ~selected]
        if len(dropped):
            assert kept.max() <= dropped.min()


def test_nan_priority_chosen_last():
    """A NaN-priority object is only kept once its cell has no finite ones left."""
    cat = CATALOG.copy()
    cat['zmag'][::2] = np.nan
    out = mock_fiber_assign(cat, 20000, _FOV_RADIUS, priority_column='zmag', seed=12)
    kept = np.isin(cat[['x', 'y']], out[['x', 'y']])
    cell = _cell_index(cat)
    finite = np.isfinite(cat['zmag'])
    cells_with_dropped_finite = np.unique(cell[finite & ~kept])
    assert not np.any(np.isin(cell[~finite & kept], cells_with_dropped_finite))
    assert np.isfinite(out['zmag']).mean() > 0.99

# ---------------------------------------------------------------------------
# Reproducibility and container types
# ---------------------------------------------------------------------------

def test_seed_reproducible():
    a = mock_fiber_assign(CATALOG, 13000, _FOV_RADIUS, seed=13)
    b = mock_fiber_assign(CATALOG, 13000, _FOV_RADIUS, seed=13)
    c = mock_fiber_assign(CATALOG, 13000, _FOV_RADIUS, seed=14)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_output_in_input_order():
    out = mock_fiber_assign(CATALOG, 13000, _FOV_RADIUS, priority_column='zmag', seed=15)
    idx = np.flatnonzero(np.isin(CATALOG['zmag'], out['zmag']))
    np.testing.assert_array_equal(out, CATALOG[idx])


def test_dataframe_input_returns_dataframe():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(CATALOG)
    out_df = mock_fiber_assign(df, 13000, _FOV_RADIUS, priority_column='zmag', seed=16)
    out_np = mock_fiber_assign(CATALOG, 13000, _FOV_RADIUS, priority_column='zmag', seed=16)
    assert isinstance(out_df, pd.DataFrame)
    assert list(out_df.columns) == ['x', 'y', 'zmag']
    np.testing.assert_array_equal(out_df['zmag'].to_numpy(), out_np['zmag'])

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_priority_column_raises():
    with pytest.raises(ValueError, match="not found"):
        mock_fiber_assign(CATALOG, 1000, _FOV_RADIUS, priority_column='rmag')


@pytest.mark.parametrize('kwargs', [
    dict(n_fibers=0),
    dict(n_passes=0),
    dict(fov_radius_arcmin=0.0),
    dict(n_grid_cells=0),
])
def test_invalid_arguments_raise(kwargs):
    args = dict(n_fibers=1000, fov_radius_arcmin=_FOV_RADIUS) | kwargs
    with pytest.raises(ValueError):
        mock_fiber_assign(CATALOG, **args)
