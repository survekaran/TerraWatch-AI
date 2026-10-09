import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from geoai04.config import load_config
from geoai04.pipeline import run_pipeline
from geoai04.validator import InputValidationError

EXPECTED_FILES = ["change_mask.tif", "change_scores.tif", "change_polygons.gpkg", "change_polygons.geojson",
                  "change_summary.csv", "provenance.json", "run.log", "validation_report.json"]


@pytest.fixture(scope="module")
def run(synthetic_pair, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return run_pipeline(synthetic_pair["before"], synthetic_pair["after"], load_config(), out_root=out, run_name="r1")


def test_all_outputs_written(run):
    for f in EXPECTED_FILES:
        assert (run.run_dir / f).is_file(), f


def test_polygons_match_known_areas(run, synthetic_pair):
    known = sorted(c["area_m2"] for c in synthetic_pair["meta"]["changes"])
    got = sorted(run.polygons.area_m2)
    assert len(got) == 3
    for k, g in zip(known, got):
        assert g == pytest.approx(k, rel=0.03)


def test_geopackage_reload_preserves_crs_geometry_and_attributes(run):
    gdf = gpd.read_file(run.run_dir / "change_polygons.gpkg", layer="change_polygons")
    assert gdf.crs.to_epsg() == 32633
    assert len(gdf) == 3 and gdf.geometry.is_valid.all() and not gdf.geometry.is_empty.any()
    for c in ["change_id", "area_m2", "centroid_lon", "centroid_lat", "change_magnitude", "confidence", "source_pair_id", "detector_used"]:
        assert c in gdf.columns
    assert np.allclose(sorted(gdf.area_m2), sorted(run.polygons.area_m2))


def test_geojson_reload_is_wgs84(run):
    gdf = gpd.read_file(run.run_dir / "change_polygons.geojson")
    assert gdf.crs.to_epsg() == 4326 and len(gdf) == 3 and gdf.geometry.is_valid.all()
    assert gdf.geometry.representative_point().x.between(14.9, 15.1).all()
    json.loads((run.run_dir / "change_polygons.geojson").read_text())


def test_rasters_keep_crs_and_transform(run, synthetic_pair):
    with rasterio.open(synthetic_pair["before"]) as src:
        crs, tr, shape = src.crs, src.transform, src.shape
    for f in ("change_mask.tif", "change_scores.tif"):
        with rasterio.open(run.run_dir / f) as r:
            assert r.crs == crs and r.transform == tr and r.shape == shape
    with rasterio.open(run.run_dir / "change_mask.tif") as r:
        assert set(np.unique(r.read(1))) <= {0, 1, 255}


def test_provenance_content(run):
    p = json.loads((run.run_dir / "provenance.json").read_text())
    assert p["status"] == "completed" and p["phase"] == 1
    assert p["detector"]["learned_model_used"] is False and p["detector"]["name"] == "baseline_difference"
    assert p["alignment"]["status"] == "not_run"
    assert len(p["inputs"]["before"]["sha256"]) == 64
    assert p["counts"]["polygons_after_area_filter"] == 3
    assert p["counts"]["components_raw"] >= p["counts"]["polygons_before_area_filter"]
    assert any("not implemented" in s.lower() or "not available" in s.lower() for s in p["not_implemented"])
    assert "rasterio" in p["software"] and p["parameters"]["detection"]["threshold_method"] == "otsu"


def test_reproducible_for_same_inputs(synthetic_pair, tmp_path):
    a = run_pipeline(synthetic_pair["before"], synthetic_pair["after"], load_config(), out_root=tmp_path, run_name="a")
    b = run_pipeline(synthetic_pair["before"], synthetic_pair["after"], load_config(), out_root=tmp_path, run_name="b")
    assert a.polygons.drop(columns="geometry").equals(b.polygons.drop(columns="geometry"))
    assert all(x.equals(y) for x, y in zip(a.polygons.geometry, b.polygons.geometry))


def test_inputs_are_unchanged(synthetic_pair, tmp_path):
    import hashlib
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = (h(synthetic_pair["before"]), h(synthetic_pair["after"]))
    run_pipeline(synthetic_pair["before"], synthetic_pair["after"], load_config(), out_root=tmp_path, run_name="x")
    assert before == (h(synthetic_pair["before"]), h(synthetic_pair["after"]))


def test_empty_result_exports_cleanly(tmp_path, tif):
    rng = np.random.default_rng(5)
    d = rng.integers(0, 255, (3, 64, 64)).astype("uint8")
    a, b = tif(tmp_path / "a.tif", d), tif(tmp_path / "b.tif", d)
    cfg = load_config(overrides={"detection": {"threshold_method": "fixed", "fixed_threshold": 0.5}})
    res = run_pipeline(a, b, cfg, out_root=tmp_path / "o", run_name="empty")
    assert res.status == "completed" and res.polygons.empty
    assert json.loads((res.run_dir / "change_polygons.geojson").read_text())["features"] == []
    assert len(gpd.read_file(res.run_dir / "change_polygons.gpkg")) == 0
    assert any("No change polygons" in w for w in res.warnings)


def test_invalid_input_fails_loudly_and_writes_failed_provenance(tmp_path, tif, synthetic_pair):
    nocrs = tif(tmp_path / "nocrs.tif", np.random.default_rng(0).integers(0, 255, (3, 64, 64)).astype("uint8"), crs=None)
    with pytest.raises(InputValidationError) as exc:
        run_pipeline(nocrs, synthetic_pair["after"], load_config(), out_root=tmp_path / "o", run_name="bad")
    p = json.loads((exc.value.run_dir / "provenance.json").read_text())
    assert p["status"] == "failed_validation" and p["errors"]
