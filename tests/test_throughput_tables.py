"""
Provenance tests for the ETC throughput and sky tables in stellar_etc.

Reference values are copied at full precision from the Spec-S5 Systems
Engineering Throughput Budget, SS5-0022 v1.1, "Comparison to DESI" tab
(throughputs) and "Noise" tab row 5 (sky). The code stores them rounded to
4 decimals (3 significant figures for the sky), so these tests catch any
change to the tables that is not a deliberate update of the source.
"""

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

SS5_0022_V1_1 = {
    "wavelength_nm": [360, 375, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 980],
    "desi_star": [0.0313319, 0.0507348, 0.0772698, 0.106978, 0.116554, 0.115718, 0.138483,
                  0.150784, 0.152706, 0.143204, 0.177713, 0.180486, 0.178606, 0.104772],
    "desi_galaxy": [0.0252676, 0.0409152, 0.0623143, 0.0862723, 0.0939951, 0.093321, 0.11168,
                    0.1216, 0.12315, 0.115487, 0.143317, 0.145554, 0.144037, 0.0844932],
    "desi_sky": [0.0631172, 0.101732, 0.153695, 0.211302, 0.228432, 0.226192, 0.271592,
                 0.297554, 0.3033, 0.285886, 0.355832, 0.361885, 0.358608, 0.210657],
    "spec5_star": [0.0365411, 0.0515628, 0.0724791, 0.109929, 0.106691, 0.0993712, 0.114473,
                   0.122426, 0.121162, 0.111476, 0.133965, 0.135022, 0.135848, 0.0876275],
    "spec5_sky": [0.0959219, 0.134309, 0.187148, 0.28134, 0.271649, 0.252873, 0.292008,
                  0.313267, 0.310862, 0.286488, 0.344442, 0.347149, 0.349746, 0.226071],
    "sky_flam": [1.68787e-17, 1.49917e-17, 1.21914e-17, 1.58641e-17, 1.03416e-17,
                 1.02069e-17, 9.58694e-18, 7.89322e-18, 8.0232e-18, 7.72816e-18,
                 7.76317e-18, 7.7298e-18, 5.85829e-18, 8.00025e-18],
}

# Rounding to 4 decimals costs at most 0.17% at the smallest value (0.0313).
RTOL = 2e-3


def test_wavelength_nodes():
    np.testing.assert_allclose(_THRU_WAVELENGTH, np.array(SS5_0022_V1_1["wavelength_nm"]) * 10.0)


@pytest.mark.parametrize("table, key", [
    (_DESI_THRU, "desi_star"),
    (_DESI_THRU_SKY, "desi_sky"),
    (_SPEC5_THRU[107], "spec5_star"),
    (_SPEC5_THRU_SKY, "spec5_sky"),
])
def test_tables_match_budget(table, key):
    np.testing.assert_allclose(table, SS5_0022_V1_1[key], rtol=RTOL)


def test_sky_matches_budget():
    # Stored to 3 significant figures, which costs up to 0.5%.
    np.testing.assert_allclose(_SKY_FLAM, SS5_0022_V1_1["sky_flam"], rtol=5e-3)


def test_desi_source_curve_is_the_star_curve():
    # The budget defines DESI star = galaxy x 1.24; the code must use the star
    # curve, since the BGS model treats both source tables as point sources.
    np.testing.assert_allclose(_DESI_THRU / np.array(SS5_0022_V1_1["desi_galaxy"]),
                               1.24, rtol=RTOL)


def test_120um_curve_follows_budget_fiber_capture_ratio():
    # The budget has no 120 µm star curve; it is the 107 µm curve times the
    # budget's 120/107 point-source capture ratio at 0.96" (1.154-1.156 at the
    # budgeted lateral errors). 4-decimal rounding adds up to ~0.3% scatter.
    ratio = _SPEC5_THRU[120] / _SPEC5_THRU[107]
    assert np.all((ratio > 1.150) & (ratio < 1.160))
