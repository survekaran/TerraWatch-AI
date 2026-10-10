"""Unit tests for false alarm area calculation (Task 3a).

Verifies the precise geometric definition:
Total polygon area (m2) not overlapping the reference mask.
Specifically proves:
- Polygons that merely touch the reference mask (boundary intersection only)
  contribute 100% of their area to false alarm area.
- Disjoint polygons contribute 100% of their area.
- Partially overlapping polygons contribute only their non-overlapping area.
- Completely contained polygons contribute 0.0 false alarm area.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Polygon

from geoai04.metrics import compute_false_alarm_area_m2


def test_false_alarm_area_definitions():
    # Affine transform: 1 pixel = 1 meter
    transform = Affine.translation(0, 0)
    crs = "EPSG:32633"

    # Reference mask: 10x10 square from x=[0, 10], y=[0, 10]
    # In raster coordinates: rows 0..10, cols 0..10
    ref_mask = np.zeros((30, 30), dtype=np.uint8)
    ref_mask[0:10, 0:10] = 1

    # Case 1: Polygon completely inside reference (x: 2..8, y: 2..8), area = 36 m2
    poly_inside = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])

    # Case 2: Polygon completely disjoint (x: 20..25, y: 20..25), area = 25 m2
    poly_disjoint = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])

    # Case 3: Polygon that MERELY TOUCHES the reference boundary (x: 10..15, y: 0..10)
    # Shares the boundary edge x=10 with ref_mask, but 0 area inside ref_mask. Area = 50 m2
    poly_touching = Polygon([(10, 0), (15, 0), (15, 10), (10, 10)])

    # Case 4: Polygon with partial 50% overlap (x: 5..15, y: 0..10), area = 100 m2
    # Overlaps ref_mask on x: 5..10 (50 m2), outside on x: 10..15 (50 m2)
    poly_partial = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])

    # Test 1: Inside polygon has 0 false alarm area
    gdf_inside = gpd.GeoDataFrame([{"geometry": poly_inside, "area_m2": 36.0}], crs=crs)
    fa_inside = compute_false_alarm_area_m2(gdf_inside, ref_mask, transform, crs)
    assert fa_inside == 0.0, f"Expected 0.0 false alarm for fully contained polygon, got {fa_inside}"

    # Test 2: Disjoint polygon has 100% false alarm area
    gdf_disjoint = gpd.GeoDataFrame([{"geometry": poly_disjoint, "area_m2": 25.0}], crs=crs)
    fa_disjoint = compute_false_alarm_area_m2(gdf_disjoint, ref_mask, transform, crs)
    assert fa_disjoint == 25.0, f"Expected 25.0 false alarm for disjoint polygon, got {fa_disjoint}"

    # Test 3: Touching polygon has 100% false alarm area (merely touching must NOT be treated as overlap!)
    gdf_touching = gpd.GeoDataFrame([{"geometry": poly_touching, "area_m2": 50.0}], crs=crs)
    fa_touching = compute_false_alarm_area_m2(gdf_touching, ref_mask, transform, crs)
    assert fa_touching == 50.0, f"Expected 50.0 false alarm for touching polygon, got {fa_touching}"

    # Test 4: Partially overlapping polygon has exactly non-overlapping area (50.0 m2)
    gdf_partial = gpd.GeoDataFrame([{"geometry": poly_partial, "area_m2": 100.0}], crs=crs)
    fa_partial = compute_false_alarm_area_m2(gdf_partial, ref_mask, transform, crs)
    assert fa_partial == 50.0, f"Expected 50.0 false alarm for partial overlap, got {fa_partial}"

    # Test 5: All combined
    gdf_all = gpd.GeoDataFrame([
        {"geometry": poly_inside, "area_m2": 36.0},
        {"geometry": poly_disjoint, "area_m2": 25.0},
        {"geometry": poly_touching, "area_m2": 50.0},
        {"geometry": poly_partial, "area_m2": 100.0},
    ], crs=crs)
    fa_all = compute_false_alarm_area_m2(gdf_all, ref_mask, transform, crs)
    # Expected total false alarm area = 0.0 + 25.0 + 50.0 + 50.0 = 125.0 m2
    assert fa_all == 125.0, f"Expected 125.0 false alarm for combined polygons, got {fa_all}"
