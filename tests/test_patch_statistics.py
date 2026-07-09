"""Tests for the area-weighted per-region aggregator (pandas only)."""
import numpy as np
import pandas as pd

from surface_morphometrics import patch_statistics as ps


def _df():
    return pd.DataFrame({
        "patch_number": [1, 1, 2, 2, 0],
        "curvedness_VV": [0.1, 0.3, 0.2, 0.0, 0.9],
        "thickness": [3.0, 4.0, 3.5, 3.5, 9.0],
        "area": [1.0, 1.0, 2.0, 2.0, 1.0],
    })


def test_aggregate_regions_area_weighted():
    df = pd.DataFrame({"component_number": [1, 1], "thickness": [2.0, 4.0], "area": [3.0, 1.0]})
    r = ps.aggregate_regions(df, "component_number", ["thickness"])
    # weighted mean = (2*3 + 4*1) / 4 = 2.5
    assert abs(r.iloc[0]["thickness_mean"] - 2.5) < 1e-9
    assert r.iloc[0]["total_area"] == 4.0


def test_aggregate_regions_drops_zero_and_nan():
    df = pd.DataFrame({
        "patch_number": [1, 1, 1],
        "curvedness_VV": [0.1, 0.3, 0.0],   # the 0 is dropped
        "thickness": [3.0, 4.0, np.nan],    # the NaN is dropped
        "area": [1.0, 1.0, 1.0],
    })
    r = ps.aggregate_regions(df, "patch_number", ["curvedness_VV", "thickness"]).iloc[0]
    assert abs(r["curvedness_VV_mean"] - 0.2) < 1e-9
    assert abs(r["thickness_mean"] - 3.5) < 1e-9
    assert r["n_triangles"] == 3            # n_triangles counts all region rows


def test_aggregate_regions_excludes_region_zero():
    r = ps.aggregate_regions(_df(), "patch_number", ["thickness"])
    assert set(r["region_id"]) == {1, 2}    # region 0 excluded


def test_effective_thickness_fallback():
    from surface_morphometrics.generate_patches import effective_thickness
    assert effective_thickness(4.0, [4, 4], [0, 2]) == 4.0          # valid center used
    got = effective_thickness(np.nan, [np.nan, 6, 8], [0, 1, 3])    # center NaN -> patch mean
    assert abs(got - (6 * 0.5 + 8 * 0.25) / 0.75) < 1e-9
    assert np.isnan(effective_thickness(np.nan, [np.nan, np.nan], [0, 2]))  # no thickness -> NaN
    assert np.isnan(effective_thickness(0.0, [-1.0], [0]))          # non-positive -> NaN


def _hg_df():
    # Two patches; patch 2's center thickness is NaN (exercises the fallback).
    return pd.DataFrame({
        "area": 1.0,
        "ribo_patch_number":         [1, 1, 1,  2, 2, 2],
        "ribo_patch_center":         [1, 0, 0,  2, 0, 0],
        "ribo_patch_center_distance":[0, 2, 4,  0, 1, 3],
        "ribo_protein_distance":     [10, 12, 14, 20, 22, 24],
        "thickness":                 [4, 4, 4,  np.nan, 6, 8],
    })


def test_headgroup_at_center_and_invariant():
    df = _hg_df()
    hg = ps.compute_headgroup_distances(df, "ribo_patch_number", "ribo_protein_distance")
    assert abs(hg[1] - 8.0) < 1e-9                 # 10 - 4/2, at the center triangle
    assert abs(hg[2] - (20 - (5 / 0.75) / 2)) < 1e-9   # center NaN -> weighted patch mean
    # The whole point of the fix: headgroup is never larger than min_protein_distance.
    agg = ps.aggregate_regions(df, "ribo_patch_number", ["ribo_protein_distance"],
                               min_cols={"min_protein_distance": "ribo_protein_distance"})
    for rid in (1, 2):
        mp = agg.set_index("region_id").loc[rid, "min_protein_distance"]
        assert hg[rid] <= mp + 1e-9
