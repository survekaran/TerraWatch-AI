import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from geoai04.alignment import AlignmentResult
from geoai04.config import SeverityConfig
from geoai04.severity import evaluate_severity

TIER_ORDER = {"Low": 0, "Medium": 1, "Critical": 2, "Review": 3}


def _make_polygons(records):
    # records: list of dict(area_m2, confidence, tags)
    rows = []
    for i, r in enumerate(records):
        geom = box(i * 10, 0, i * 10 + 5, 5)
        rows.append({
            "change_id": f"CHG-{i+1:04d}",
            "geometry": geom,
            "area_m2": r["area_m2"],
            "confidence": r["confidence"],
            "tags": r.get("tags", "building_like"),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:32633")


def test_mode_thresholds_and_monotonicity():
    # Test polygons spanning low, medium, critical ranges
    polys = _make_polygons([
        {"area_m2": 50.0, "confidence": 0.4},    # Low in both
        {"area_m2": 250.0, "confidence": 0.55},  # Medium in emergency, Low in enforcement
        {"area_m2": 600.0, "confidence": 0.65},  # Critical in emergency, Low/Med in enforcement
        {"area_m2": 1500.0, "confidence": 0.90}, # Critical in both
    ])

    cfg_em = SeverityConfig(mode="emergency")
    cfg_enf = SeverityConfig(mode="enforcement")

    res_em = evaluate_severity(polys, cfg_em)
    res_enf = evaluate_severity(polys, cfg_enf)

    # Monotonicity check: emergency mode should never rank a polygon lower than enforcement mode
    for t_em, t_enf in zip(res_em["severity_tier"], res_enf["severity_tier"]):
        assert TIER_ORDER[t_em] >= TIER_ORDER[t_enf]

    # Specific assertions
    assert res_em.iloc[1]["severity_tier"] == "Medium"
    assert res_enf.iloc[1]["severity_tier"] == "Low"
    assert res_em.iloc[2]["severity_tier"] == "Critical"


def test_review_override_on_tags_and_alignment():
    polys = _make_polygons([
        {"area_m2": 2000.0, "confidence": 0.95, "tags": "building_like"},
        {"area_m2": 2000.0, "confidence": 0.95, "tags": "misregistration_suspect;building_like"},
        {"area_m2": 2000.0, "confidence": 0.95, "tags": "low_quality"},
        {"area_m2": 2000.0, "confidence": 0.95, "tags": "model_disagreement"},
    ])

    cfg = SeverityConfig(mode="enforcement")
    res = evaluate_severity(polys, cfg)

    # Clean polygon
    assert res.iloc[0]["severity_tier"] == "Critical"
    assert res.iloc[0]["base_severity"] == "Critical"
    assert res.iloc[0]["review_reasons"] == ""

    # Overridden polygons
    for i in (1, 2, 3):
        row = res.iloc[i]
        assert row["severity_tier"] == "Review"
        assert row["base_severity"] == "Critical"
        assert row["review_reasons"] != ""
        assert row["quality_flag"] != "nominal"


def test_run_level_alignment_warning_overrides_all_to_review():
    polys = _make_polygons([
        {"area_m2": 1000.0, "confidence": 0.9, "tags": "building_like"},
        {"area_m2": 50.0, "confidence": 0.3, "tags": "building_like"},
    ])
    align_fail = AlignmentResult(
        shift_row_px=3.0, shift_col_px=0.0, magnitude_px=3.0, shift_m=1.5,
        per_window_estimates=[], spread_px=0.1, n_windows_used=5,
        status="fail", reasons=["Exceeds fail threshold"]
    )

    res = evaluate_severity(polys, SeverityConfig(), alignment_result=align_fail, alignment_override=True)
    assert (res["severity_tier"] == "Review").all()
    assert (res["quality_flag"] == "alignment_override").all()


def test_percentile_fallback_when_few_polygons():
    polys = _make_polygons([
        {"area_m2": 200.0, "confidence": 0.7},
        {"area_m2": 800.0, "confidence": 0.7},
    ])
    cfg = SeverityConfig(area_rule="percentile", min_polygons_for_percentile=10)
    # Only 2 polygons, should fall back cleanly without error
    res = evaluate_severity(polys, cfg)
    assert len(res) == 2
