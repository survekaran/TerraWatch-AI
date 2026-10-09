import hashlib
import json
import pytest

from geoai04.config import load_config
from geoai04.pipeline import run_pipeline


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_reproducibility_identical_outputs_and_manifest(synthetic_pair, tmp_path):
    """Re-running with same inputs and config must yield identical raster bytes and polygon attributes."""
    cfg = load_config()

    res1 = run_pipeline(
        synthetic_pair["before"], synthetic_pair["after"], cfg=cfg,
        out_root=tmp_path, run_name="run_rep1"
    )
    res2 = run_pipeline(
        synthetic_pair["before"], synthetic_pair["after"], cfg=cfg,
        out_root=tmp_path, run_name="run_rep2"
    )

    # Compare raster files (bit-for-bit identical)
    for raster_file in ["change_scores.tif", "change_mask.tif"]:
        sha1 = _sha256(res1.run_dir / raster_file)
        sha2 = _sha256(res2.run_dir / raster_file)
        assert sha1 == sha2, f"Raster {raster_file} differed across identical runs!"

    # Compare polygon attributes and geometry
    df1 = res1.polygons.drop(columns=["geometry"])
    df2 = res2.polygons.drop(columns=["geometry"])
    assert df1.equals(df2), "Polygon attributes differed across runs!"

    for g1, g2 in zip(res1.polygons.geometry, res2.polygons.geometry):
        assert g1.equals(g2), "Polygon geometry differed across runs!"

    # Check provenance manifest and snapshot
    assert (res1.run_dir / "config_snapshot.yaml").is_file()
    prov1 = json.loads((res1.run_dir / "provenance.json").read_text())
    assert "manifest_sha256" in prov1
    assert "config_hash_sha256" in prov1
    assert "change_polygons.gpkg" in prov1["manifest_sha256"]
    assert prov1["cloud_shadow_screening"] == "not_performed"
