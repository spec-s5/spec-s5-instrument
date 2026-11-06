"""
spec5.instrument.fiber — Simple mock fiber assignment.

Functions
---------
mock_fiber_assign : Spatially uniform, density-limited subsample of a catalog
                    within a circular field of view
"""

import numpy as np

# Sub-samples per cell side (minimum) and per FoV diameter (minimum) used to
# estimate the area of each grid cell that lies inside the circular FoV.
_AREA_SUBSAMPLES_PER_CELL = 8
_AREA_SUBSAMPLES_PER_FOV  = 1024


def _cell_area_in_fov(n_grid_cells, fov_radius):
    """
    Area of each grid cell inside a circle of radius ``fov_radius``.

    The circle is centred on an ``n_grid_cells × n_grid_cells`` grid spanning
    ``[-fov_radius, fov_radius]`` on each axis. Areas are estimated from a
    regular sub-grid of points, in the same units as ``fov_radius`` squared.
    Returns a flat array indexed by ``ix * n_grid_cells + iy``.
    """
    k = max(_AREA_SUBSAMPLES_PER_CELL,
            int(np.ceil(_AREA_SUBSAMPLES_PER_FOV / n_grid_cells)))
    n_sub    = n_grid_cells * k
    sub_size = 2.0 * fov_radius / n_sub
    centers  = -fov_radius + (np.arange(n_sub) + 0.5) * sub_size
    inside   = np.hypot(centers[:, None], centers[None, :]) <= fov_radius
    counts   = inside.reshape(n_grid_cells, k, n_grid_cells, k).sum(axis=(1, 3))
    return counts.ravel() * sub_size**2


def _allocate_targets(n_targets, cell_area, rng):
    """
    Split ``n_targets`` fibers across grid cells in proportion to cell area.

    Each cell gets ``floor(q)`` fibers, where ``q`` is its fractional quota.
    The remaining fibers go to distinct cells, drawn with probability
    proportional to ``q - floor(q)``. The total is exactly ``n_targets``, and
    the expected allocation of every cell is ``q``.
    """
    quota      = n_targets * cell_area / cell_area.sum()
    allocation = np.floor(quota).astype(int)
    remainder  = quota - allocation
    n_extra    = int(round(n_targets - allocation.sum()))
    if n_extra > 0:
        extra = rng.choice(len(quota), size=n_extra, replace=False,
                           p=remainder / remainder.sum())
        allocation[extra] += 1
    return allocation


def mock_fiber_assign(catalog, n_fibers, fov_radius_arcmin, n_passes=1,
                      n_grid_cells=100, priority_column=None,
                      priority_order_ascending=True, seed=None):
    """
    Select a spatially uniform, density-limited subsample of a catalog.

    The circular field of view (FoV) is divided into an
    ``n_grid_cells × n_grid_cells`` Cartesian grid. The
    ``n_fibers × n_passes`` fibers are spread over the grid in proportion to
    each cell's area inside the FoV, which sets a maximum target surface
    density. Each cell then keeps up to its allocation of objects, chosen by
    priority (or at random). Cells with fewer objects than their allocation
    keep all of them, and their unused fibers are not moved elsewhere, so
    clustered catalogs return fewer targets than ``n_fibers × n_passes``. The
    cap applies even when the catalog has fewer objects than fibers.

    Parameters
    ----------
    catalog : numpy structured array or pandas DataFrame
        Must contain columns/fields:
            ``x, y`` — Tangent-plane position relative to the FoV centre
                       [arcmin]
        and ``priority_column``, if one is given.
    n_fibers : int
        Number of fibers per pass.
    fov_radius_arcmin : float
        Radius of the circular FoV [arcmin].
    n_passes : int
        Number of passes over the field (default: 1). The total number of
        fibers is ``n_fibers × n_passes``.
    n_grid_cells : int
        Number of grid cells per axis (default: 100). Smaller cells enforce
        uniformity on finer scales.
    priority_column : str or None
        Column used to rank objects within each grid cell, e.g. ``'zmag'``
        (default: None). If None, objects in a cell are chosen at random.
    priority_order_ascending : bool
        If True (default), smaller values of ``priority_column`` are chosen
        first (e.g. brighter magnitudes). If False, larger values are chosen
        first. NaN values are always chosen last.
    seed : int or None
        Random seed for reproducibility (default: None). It sets how leftover
        fibers are spread across cells, how ties in ``priority_column`` are
        broken, and which objects are picked when no priority is given.

    Returns
    -------
    numpy structured array or pandas DataFrame
        Same type as ``catalog``: the selected rows, in their input order.

    Notes
    -----
    - Priority is applied within each grid cell, not across the field. A faint
      object in a sparse cell can be selected while a brighter one in a crowded
      cell is not. This is not a global magnitude cut.
    - For a catalog that is dense everywhere in the FoV, exactly
      ``min(n_fibers × n_passes, N_in_fov)`` objects are returned.
    - Fiber collisions, patrol radii and minimum separations are not modelled.
    """
    if n_fibers < 1 or n_passes < 1:
        raise ValueError(
            f"n_fibers and n_passes must be at least 1, got {n_fibers} and {n_passes}"
        )
    if fov_radius_arcmin <= 0:
        raise ValueError(f"fov_radius_arcmin must be positive, got {fov_radius_arcmin}")
    if n_grid_cells < 1:
        raise ValueError(f"n_grid_cells must be at least 1, got {n_grid_cells}")

    # Detect input type — pandas check is duck-typed to avoid hard dependency
    try:
        import pandas as pd
        is_dataframe = isinstance(catalog, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    columns = catalog.columns if is_dataframe else catalog.dtype.names
    if priority_column is not None and priority_column not in columns:
        raise ValueError(f"Column {priority_column!r} not found in catalog")

    rng = np.random.default_rng(seed)

    x = np.asarray(catalog['x'], dtype=float)
    y = np.asarray(catalog['y'], dtype=float)

    # Indices of objects inside the FoV, in random order so that ties (or the
    # absence of a priority) are broken at random
    in_fov = np.flatnonzero(np.hypot(x, y) <= fov_radius_arcmin)
    in_fov = rng.permutation(in_fov)

    if priority_column is not None:
        priority = np.asarray(catalog[priority_column], dtype=float)[in_fov]
        if not priority_order_ascending:
            priority = -priority
        in_fov = in_fov[np.argsort(priority, kind='stable')]

    # Grid cell of each object
    cell_size = 2.0 * fov_radius_arcmin / n_grid_cells
    ix = np.clip(((x[in_fov] + fov_radius_arcmin) // cell_size).astype(int),
                 0, n_grid_cells - 1)
    iy = np.clip(((y[in_fov] + fov_radius_arcmin) // cell_size).astype(int),
                 0, n_grid_cells - 1)
    cell = ix * n_grid_cells + iy

    # Group by cell (stable, so priority order is kept within each cell) and
    # rank objects within their cell
    order  = np.argsort(cell, kind='stable')
    in_fov = in_fov[order]
    cell   = cell[order]
    first  = np.searchsorted(cell, cell, side='left')
    rank   = np.arange(len(cell)) - first

    cell_area  = _cell_area_in_fov(n_grid_cells, fov_radius_arcmin)
    allocation = _allocate_targets(n_fibers * n_passes, cell_area, rng)

    selected = np.zeros(len(x), dtype=bool)
    selected[in_fov[rank < allocation[cell]]] = True

    if is_dataframe:
        return catalog.loc[selected].copy()
    return catalog[selected]
