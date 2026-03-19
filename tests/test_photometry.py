"""
Unit tests for spec5.instrument.photometry.
"""

import warnings

import numpy as np
import pytest

from spec5.instrument.photometry import (
    gaia_g_to_lsst_i,
    gaia_g_to_lsst_z,
    lsst_i_to_lsst_z,
    lsst_z_to_gaia_g,
)

# ---------------------------------------------------------------------------
# gaia_g_to_lsst_z
# ---------------------------------------------------------------------------

# Expected: z = G + 0.034*bp_rp**2 - 0.747*bp_rp + 0.416
@pytest.mark.parametrize("G, bp_rp, expected", [
    (18.0, 0.0,  18.0 + 0.034*0.0**2 - 0.747*0.0  + 0.416),
    (18.0, 1.5,  18.0 + 0.034*1.5**2 - 0.747*1.5  + 0.416),
    (20.0, 2.0,  20.0 + 0.034*2.0**2 - 0.747*2.0  + 0.416),
    (15.0, 3.0,  15.0 + 0.034*3.0**2 - 0.747*3.0  + 0.416),
])
def test_gaia_g_to_lsst_z_values(G, bp_rp, expected):
    assert gaia_g_to_lsst_z(G, bp_rp) == pytest.approx(expected, rel=1e-6)


def test_gaia_g_to_lsst_z_array():
    G     = np.array([18.0, 19.0, 20.0])
    bp_rp = np.array([1.0,  1.5,  2.0])
    result = gaia_g_to_lsst_z(G, bp_rp)
    expected = G + 0.034 * bp_rp**2 - 0.747 * bp_rp + 0.416
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_gaia_g_to_lsst_z_out_of_range_warns():
    with pytest.warns(UserWarning, match="BP-RP"):
        gaia_g_to_lsst_z(18.0, bp_rp=4.0)


def test_gaia_g_to_lsst_z_in_range_no_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaia_g_to_lsst_z(18.0, bp_rp=1.5)


# ---------------------------------------------------------------------------
# gaia_g_to_lsst_i
# ---------------------------------------------------------------------------

# Expected: i = G + 0.099*bp_rp**2 - 0.672*bp_rp + 0.343
@pytest.mark.parametrize("G, bp_rp, expected", [
    (18.0, 0.0,  18.0 + 0.099*0.0**2 - 0.672*0.0  + 0.343),
    (18.0, 1.5,  18.0 + 0.099*1.5**2 - 0.672*1.5  + 0.343),
    (20.0, 2.0,  20.0 + 0.099*2.0**2 - 0.672*2.0  + 0.343),
    (15.0, 3.0,  15.0 + 0.099*3.0**2 - 0.672*3.0  + 0.343),
])
def test_gaia_g_to_lsst_i_values(G, bp_rp, expected):
    assert gaia_g_to_lsst_i(G, bp_rp) == pytest.approx(expected, rel=1e-6)


def test_gaia_g_to_lsst_i_array():
    G     = np.array([18.0, 19.0, 20.0])
    bp_rp = np.array([1.0,  1.5,  2.0])
    result = gaia_g_to_lsst_i(G, bp_rp)
    expected = G + 0.099 * bp_rp**2 - 0.672 * bp_rp + 0.343
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_gaia_g_to_lsst_i_out_of_range_warns():
    with pytest.warns(UserWarning, match="BP-RP"):
        gaia_g_to_lsst_i(18.0, bp_rp=4.0)


def test_gaia_g_to_lsst_i_in_range_no_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaia_g_to_lsst_i(18.0, bp_rp=1.5)


# i band is brighter than z for red stars (positive i-z colour)
def test_gaia_g_to_lsst_i_brighter_than_z():
    bp_rp = np.array([1.0, 1.5, 2.0, 2.5])
    G = 18.0
    assert np.all(gaia_g_to_lsst_i(G, bp_rp) > gaia_g_to_lsst_z(G, bp_rp))


# ---------------------------------------------------------------------------
# lsst_i_to_lsst_z
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("i", [15.0, 18.0, 20.0, 23.0])
def test_lsst_i_to_lsst_z_offset(i):
    # i-z = 0.2 for metal-poor halo RGB, so z = i - 0.2
    assert lsst_i_to_lsst_z(i) == pytest.approx(i - 0.2, rel=1e-6)


def test_lsst_i_to_lsst_z_array():
    i = np.array([18.0, 19.0, 20.0])
    np.testing.assert_allclose(lsst_i_to_lsst_z(i), i - 0.2, rtol=1e-6)


# Consistency: i->z should agree with Gaia->z minus Gaia->i at typical RGB colours
def test_lsst_i_to_lsst_z_consistent_with_gaia_transforms():
    G, bp_rp = 18.0, 1.5   # typical metal-poor halo RGB
    i = gaia_g_to_lsst_i(G, bp_rp)
    z_from_i  = lsst_i_to_lsst_z(i)
    z_from_g  = gaia_g_to_lsst_z(G, bp_rp)
    # fixed offset (0.2) should be close to the Gaia-derived i-z at BP-RP~1.5
    assert z_from_i == pytest.approx(z_from_g, abs=0.1)


# ---------------------------------------------------------------------------
# lsst_z_to_gaia_g
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G, bp_rp", [
    (18.0, 0.0),
    (18.0, 1.5),
    (20.0, 2.0),
    (15.0, 3.0),
])
def test_lsst_z_to_gaia_g_roundtrip(G, bp_rp):
    z = gaia_g_to_lsst_z(G, bp_rp)
    assert lsst_z_to_gaia_g(z, bp_rp) == pytest.approx(G, rel=1e-6)


def test_lsst_z_to_gaia_g_array():
    G     = np.array([18.0, 19.0, 20.0])
    bp_rp = np.array([1.0,  1.5,  2.0])
    z = gaia_g_to_lsst_z(G, bp_rp)
    np.testing.assert_allclose(lsst_z_to_gaia_g(z, bp_rp), G, rtol=1e-6)


def test_lsst_z_to_gaia_g_out_of_range_warns():
    with pytest.warns(UserWarning, match="BP-RP"):
        lsst_z_to_gaia_g(18.0, bp_rp=4.0)
