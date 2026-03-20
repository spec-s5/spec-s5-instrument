import numpy as np


def mock_fiber_assign(
    catalog,
    n_fibers,
    fov_radius_arcmin,
    n_passes=1,
    n_grid_cells=100,
    priority_column=None,
    priority_order_ascending=True,
    random_seed=None,
):    
    """
    Perform spatially uniform subsampling of a catalog within a circular
    field of view (FoV), optionally prioritizing sources by a given column.

    The FoV is restricted to a circle and partitioned into a 2D Cartesian grid.
    A per-cell target count is derived from the desired global surface density
    implied by `n_fibers × n_passes`. Subsampling proceeds in two stages:

    1. First pass:
        Each grid cell contributes up to a fixed number of targets
        (`max_targets_per_cell`). If a cell contains fewer objects than this
        cap, all objects are retained.

    2. Optional rescaling pass:
        If the total number of selected targets exceeds the global target
        (`n_fibers × n_passes`), counts in each cell are uniformly rescaled
        (via deterministic truncation) to enforce the global constraint while
        preserving spatial uniformity.

    Within each grid cell:
        - If `priority_column` is provided: objects are sorted globally by this
          column (ascending), and higher-priority objects are preferentially
          retained within each cell.
        - If `priority_column` is None: selection preserves the original row
          order (no prioritization or randomization).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Input catalog containing at minimum:
        - 'x' : float
        - 'y' : float
          Coordinates in arcminutes, centered on the FoV.

        If `priority_column` is provided, that column must also exist.

    n_fibers : int
        Number of fibers per pass.

    fov_radius_arcmin : float
        Radius of the circular FoV in arcminutes.

    n_passes : int, optional
        Number of passes over the field. The effective target count is
        `n_fibers × n_passes`.

    n_grid_cells : int, optional
        Number of grid cells per axis used to approximate spatial uniformity.
        Larger values provide finer spatial control but increase discretization
        noise. Default is 100.

    priority_column : str or None, optional
        Column used to prioritize selection (ascending order).
        Example: 'DECam_z' (smaller = brighter).
        If None, no prioritization is applied.
        
    priority_order_ascending : bool, optional
        If True, prioritize objects in ascending order of the priority column.
        If False, prioritize objects in descending order of the priority column.

    random_seed : int or None, optional
        RNG seed (currently unused; reserved for future stochastic variants).

    Returns
    -------
    subsampled_catalog : pandas.DataFrame
        Spatially uniform subset with size ≤ n_fibers × n_passes.

    Notes
    -----
    - The method enforces a *maximum surface density*, not an exact target count.
      The final number of objects may be less than the requested total due to:
        * finite sampling within cells,
        * integer truncation during per-cell allocation,
        * spatial inhomogeneity in the input catalog.

    - Each populated grid cell contributes at least one object if
      `max_targets_per_cell ≥ 1`, which is enforced in this implementation.

    - The algorithm is deterministic given fixed input ordering and
      `priority_column`. No stochastic subsampling is performed.

    - Spatial uniformity is approximate and depends on the grid resolution.
      Edge cells intersecting the circular FoV are treated as full cells,
      which can introduce small (~percent-level) density biases near the boundary.

    - The approach approximates a uniform fiber density but does not model
      fiber collision constraints or minimum separation effects.
    """

    rng = np.random.default_rng(random_seed)

    # --- 1. Restrict to circular FoV ---
    radial_distance = np.hypot(catalog["x"], catalog["y"])
    catalog = catalog.loc[radial_distance <= fov_radius_arcmin].copy()

    max_targets = n_fibers * n_passes
    if len(catalog) <= max_targets:
        return catalog

    # --- 2. Compute density and per-cell quota ---
    fov_area = np.pi * fov_radius_arcmin**2
    target_surface_density = max_targets / fov_area

    cell_size = 2 * fov_radius_arcmin / n_grid_cells
    cell_area = cell_size**2

    max_targets_per_cell = int(np.floor(target_surface_density * cell_area))
    if max_targets_per_cell < 1:
        max_targets_per_cell = 1

    # --- 3. Assign grid cell indices ---
    grid_edges = np.linspace(-fov_radius_arcmin, fov_radius_arcmin, n_grid_cells + 1)

    x_cell_index = np.clip(
        np.digitize(catalog["x"], grid_edges) - 1, 0, n_grid_cells - 1
    )
    y_cell_index = np.clip(
        np.digitize(catalog["y"], grid_edges) - 1, 0, n_grid_cells - 1
    )

    catalog = catalog.assign(grid_cell=list(zip(x_cell_index, y_cell_index)))

    # --- 4. Optional prioritization ---
    if priority_column is not None:
        if priority_column not in catalog.columns:
            raise ValueError(f"Column '{priority_column}' not found in catalog.")
        catalog = catalog.sort_values(priority_column, ascending=priority_order_ascending)

    # --- 5. First-pass selection ---
    initial_selection = (
        catalog.groupby("grid_cell", group_keys=False)
        .head(max_targets_per_cell)
    )

    if len(initial_selection) <= max_targets:
        return initial_selection.drop(columns="grid_cell").copy()

    # --- 6. Downscale to match global target ---
    scaling_factor = max_targets / len(initial_selection)

    final_selection = (
        initial_selection.groupby("grid_cell", group_keys=False)
        .apply(lambda df: df.head(int(np.floor(len(df) * scaling_factor))))
    )

    return final_selection.drop(columns="grid_cell").copy()
