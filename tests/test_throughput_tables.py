"""
Provenance tests for the ETC throughput and sky tables in stellar_etc.

The reference values are exported directly from the Spec-S5 Systems
Engineering Throughput Budget, SS5-0022 v1.1, into tests/data/ (see the
header of each CSV for the source cells). The code stores the tables rounded
to 4 decimals (3 significant figures for the sky), so these tests check both
that the tables were copied correctly and that they aren't edited by accident.
"""

from pathlib import Path

import numpy as np
import pytest

from spec5.instrument.stellar_etc import (
    _DESI_THRU,
    _DESI_THRU_SKY,
    _SKY_FLAM,
    _SPEC5_THRU,
    _SPEC5_THRU_SKY,
    _THRU_WAVELENGTH,
)

_DATA = Path(__file__).parent / "data"


def _read_csv(name):
    """Read a CSV with a '#' comment header into a structured array."""
    lines = [ln for ln in (_DATA / name).read_text().splitlines() if not ln.startswith("#")]
    return np.genfromtxt(lines, delimiter=",", names=True)


BUDGET = _read_csv("ss5-0022_v1.1_throughput.csv")
LATERAL = _read_csv("ss5-0022_v1.1_lateral_offset_capture.csv")

# Rounding to 4 decimals costs at most 0.17% at the smallest value (0.0313).
RTOL = 2e-3


def test_wavelength_nodes():
    np.testing.assert_allclose(_THRU_WAVELENGTH, BUDGET["wavelength_nm"] * 10.0)


@pytest.mark.parametrize("table, column", [
    (_DESI_THRU, "desi_star"),
    (_DESI_THRU_SKY, "desi_sky"),
    (_SPEC5_THRU[107], "spec5_star"),
    (_SPEC5_THRU_SKY, "spec5_sky"),
])
def test_tables_match_budget(table, column):
    np.testing.assert_allclose(table, BUDGET[column], rtol=RTOL)
    # and are the budget values rounded to 4 decimals
    np.testing.assert_allclose(table, np.round(BUDGET[column], 4), atol=1e-12)


def test_sky_matches_budget():
    # Stored to 3 significant figures, which costs up to 0.5%.
    np.testing.assert_allclose(_SKY_FLAM, BUDGET["sky_flam"], rtol=5e-3)


def test_desi_source_curve_is_the_star_curve():
    # The budget defines DESI star = galaxy x 1.24; the code must use the star
    # curve, since the BGS model treats both source tables as point sources.
    np.testing.assert_allclose(BUDGET["desi_star"], 1.24 * BUDGET["desi_galaxy"], rtol=1e-12)
    np.testing.assert_allclose(_DESI_THRU / BUDGET["desi_galaxy"], 1.24, rtol=RTOL)


def test_120um_curve_follows_budget_fiber_capture_ratio():
    # The budget has no 120 µm star curve. It is the 107 µm star curve times
    # the budget's 120/107 point-source capture ratio at 0.96", evaluated at
    # each node's budgeted lateral error.
    lateral = BUDGET["lateral_error_um"]
    ratio = (np.interp(lateral, LATERAL["lateral_um"], LATERAL["star_120um"])
             / np.interp(lateral, LATERAL["lateral_um"], LATERAL["star_107um"]))
    assert np.all((ratio > 1.15) & (ratio < 1.16))
    np.testing.assert_allclose(_SPEC5_THRU[120], np.round(BUDGET["spec5_star"] * ratio, 4),
                               atol=1e-12)
