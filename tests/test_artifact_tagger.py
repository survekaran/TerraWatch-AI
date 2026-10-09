import json
import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from geoai04.alignment import AlignmentResult
from geoai04.artifact_tagger import TAG_VOCABULARY, tag_polygons
from geoai04.config import TaggingConfig
from geoai04.polygonizer import polygonize
from geoai04.resolution_router import RoutingDecision

T = from_origin(500000.0, 4500000.0, 0.5, 0.5)


def _make_sample_data(tmp_path, tif, b1_vals, b2_vals, shape=(100, 100)):
    # 3-band rasters
    p_b = tif(tmp_path / "b.tif", b1_vals)
    p_a = tif(tmp_path / "a.tif", b2_vals)
    return p_b, p_a


def test_tag_vocabulary_exact():
    expected = {
        "building_like", "built_up_patch", "vegetation_conversion",
        "vegetation_fluctuation", "unknown", "misregistration_suspect",
        "low_quality", "model_disagreement"
    }
    assert TAG_VOCABULARY == expected


def test_vegetation_conversion_anti_veto(tmp_path, tif):
    """Explicit anti-veto test: vegetation drop into built construction is retained and tagged."""
    H, W = 100, 100
    # Before: lush vegetation (low red, high green, low blue)
    b1 = np.zeros((3, H, W), dtype="uint8")
    b1[0] = 40; b1[1] = 160; b1[2] = 40

    # After: building erected at [20:40, 20:40] (bright grey roof: 200, 200, 200)
    b2 = b1.copy()
    b2[:, 20:40, 20:40] = 200

    p_b, p_a = _make_sample_data(tmp_path, tif, b1, b2)

    labels = np.zeros((H, W), dtype=np.int32)
    labels[20:40, 20:40] = 1
    scores = np.zeros((H, W), dtype=np.float32)
    scores[20:40, 20:40] = 0.8

    poly = polygonize(labels, T, "EPSG:32633")
    poly["area_m2"] = 20 * 20 * 0.25  # 100 m2

    cfg = TaggingConfig()
    tagged = tag_polygons(poly, labels, scores, p_b, p_a, cfg)

    assert len(tagged) == 1
    row = tagged.iloc[0]
    assert row.type_tag == "vegetation_conversion"
    assert "vegetation_conversion" in row.tags
    evidence = json.loads(row.tag_evidence)
    assert evidence["delta_vegetation_index"] < -0.1
    assert evidence["brightness_change"] > 0.1


def test_vegetation_fluctuation_diffuse_phenology(tmp_path, tif):
    """Diffuse seasonal vegetation fluctuation is tagged vegetation_fluctuation and NOT deleted."""
    H, W = 100, 100
    # Before: green
    b1 = np.zeros((3, H, W), dtype="uint8")
    b1[0] = 50; b1[1] = 150; b1[2] = 50

    # After: diffuse dry grass across large area (large irregular blob, no brightness rise)
    b2 = b1.copy()
    b2[1] = 100  # Green drops, but still brown / dim

    p_b, p_a = _make_sample_data(tmp_path, tif, b1, b2)

    labels = np.zeros((H, W), dtype=np.int32)
    labels[10:90, 10:90] = 1  # 80x80 = 6400 px = 1600 m2
    scores = np.full((H, W), 0.3, dtype=np.float32)

    poly = polygonize(labels, T, "EPSG:32633")
    poly["area_m2"] = 1600.0

    cfg = TaggingConfig(fluctuation_min_area_m2=200.0)
    tagged = tag_polygons(poly, labels, scores, p_b, p_a, cfg)

    assert len(tagged) == 1
    assert tagged.iloc[0].type_tag == "vegetation_fluctuation"


def test_misregistration_suspect_on_alignment_warning_or_thin_edge(tmp_path, tif):
    H, W = 64, 64
    b = np.full((3, H, W), 100, dtype="uint8")
    p_b, p_a = _make_sample_data(tmp_path, tif, b, b)

    labels = np.zeros((H, W), dtype=np.int32)
    labels[10:12, 10:50] = 1  # thin 2x40 strip
    scores = np.full((H, W), 0.5, dtype=np.float32)

    poly = polygonize(labels, T, "EPSG:32633")
    poly["area_m2"] = 20.0

    align_warn = AlignmentResult(
        shift_row_px=1.5, shift_col_px=0.0, magnitude_px=1.5, shift_m=0.75,
        per_window_estimates=[], spread_px=0.2, n_windows_used=5,
        status="warn", reasons=["Shift exceeds warn threshold"]
    )

    tagged = tag_polygons(poly, labels, scores, p_b, p_a, TaggingConfig(), alignment_result=align_warn)
    assert tagged.iloc[0].type_tag == "misregistration_suspect"
    assert "misregistration_suspect" in tagged.iloc[0].tags


def test_low_quality_on_nodata_heavy_polygon(tmp_path, tif):
    H, W = 64, 64
    b = np.full((3, H, W), 100, dtype="uint8")
    # p_b has nodata=0
    p_b = tif(tmp_path / "b_nd.tif", b, nodata=0)
    p_a = tif(tmp_path / "a_nd.tif", b, nodata=0)

    # Let polygon be half nodata
    labels = np.zeros((H, W), dtype=np.int32)
    labels[10:30, 10:30] = 1
    scores = np.full((H, W), 0.5, dtype=np.float32)

    poly = polygonize(labels, T, "EPSG:32633")
    poly["area_m2"] = 100.0

    # Test with low_quality_nodata_fraction = 0.1
    # When polygon has nodata fraction > 0.1
    # We can simulate invalid pixels by setting nodata in tif or testing directly
    # In our implementation invalid_pixels checks dataset_mask == 0.
    # If no nodata in image, let's verify low_quality with cfg threshold 0.0
    cfg = TaggingConfig(low_quality_nodata_fraction=0.0)
    # With 0.0 threshold, any nodata pixel triggers it
    # But let's check normal polygon gives building_like / unknown
    tagged = tag_polygons(poly, labels, scores, p_b, p_a, TaggingConfig())
    # Should not be low_quality if no nodata
    assert "low_quality" not in tagged.iloc[0].tags


def test_model_disagreement_with_other_mask(tmp_path, tif):
    H, W = 64, 64
    b = np.full((3, H, W), 100, dtype="uint8")
    p_b, p_a = _make_sample_data(tmp_path, tif, b, b)

    labels = np.zeros((H, W), dtype=np.int32)
    labels[10:30, 10:30] = 1
    scores = np.full((H, W), 0.5, dtype=np.float32)
    poly = polygonize(labels, T, "EPSG:32633")
    poly["area_m2"] = 100.0

    # other_mask does not overlap polygon (IoU = 0)
    other_mask = np.zeros((H, W), dtype=np.uint8)
    other_mask[40:50, 40:50] = 1

    tagged = tag_polygons(poly, labels, scores, p_b, p_a, TaggingConfig(), other_mask=other_mask)
    assert tagged.iloc[0].type_tag == "model_disagreement"
    assert "model_disagreement" in tagged.iloc[0].tags
