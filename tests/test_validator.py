import numpy as np

from geoai04.validator import validate_pair


def rgb(v=100, size=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (3, size, size)).astype("uint8") if v is None else np.full((3, size, size), v, "uint8")


def test_valid_synthetic_pair_passes(synthetic_pair):
    rep = validate_pair(synthetic_pair["before"], synthetic_pair["after"])
    assert rep.ok, rep.errors
    assert abs(rep.before.gsd_x_m - 0.5) < 1e-6
    assert rep.before.valid_fraction == 1.0


def test_missing_file_fails(tmp_path, synthetic_pair):
    rep = validate_pair(tmp_path / "nope.tif", synthetic_pair["after"])
    assert not rep.ok and any("not found" in e for e in rep.errors)


def test_not_a_raster_fails(tmp_path, synthetic_pair):
    bad = tmp_path / "bad.tif"
    bad.write_text("this is not a tiff")
    rep = validate_pair(bad, synthetic_pair["after"])
    assert not rep.ok


def test_missing_crs_fails_and_is_never_assigned(tmp_path, tif, synthetic_pair):
    nocrs = tif(tmp_path / "nocrs.tif", rgb(None), crs=None)
    rep = validate_pair(nocrs, synthetic_pair["after"])
    assert not rep.ok and any("No CRS" in e for e in rep.errors)


def test_crs_mismatch_fails(tmp_path, tif):
    a = tif(tmp_path / "a.tif", rgb(None), crs="EPSG:32633")
    b = tif(tmp_path / "b.tif", rgb(None), crs="EPSG:32634")
    rep = validate_pair(a, b)
    assert not rep.ok and any("CRS mismatch" in e for e in rep.errors)


def test_dimension_mismatch_fails_without_resampling(tmp_path, tif):
    a = tif(tmp_path / "a.tif", rgb(None, 64))
    b = tif(tmp_path / "b.tif", rgb(None, 48))
    rep = validate_pair(a, b)
    assert not rep.ok and any("same pixel grid" in e for e in rep.errors)


def test_band_count_mismatch_fails(tmp_path, tif):
    a = tif(tmp_path / "a.tif", rgb(None))
    b = tif(tmp_path / "b.tif", rgb(None)[:1])
    rep = validate_pair(a, b)
    assert not rep.ok and any("Band count" in e for e in rep.errors)


def test_no_overlap_fails(tmp_path, tif):
    a = tif(tmp_path / "a.tif", rgb(None), origin=(500000.0, 4500000.0))
    b = tif(tmp_path / "b.tif", rgb(None), origin=(600000.0, 4600000.0))
    rep = validate_pair(a, b)
    assert not rep.ok and any("overlap" in e for e in rep.errors)


def test_constant_image_fails(tmp_path, tif):
    a = tif(tmp_path / "a.tif", rgb(0))
    b = tif(tmp_path / "b.tif", rgb(None))
    rep = validate_pair(a, b)
    assert not rep.ok and any("constant" in e for e in rep.errors)


def test_all_nodata_fails(tmp_path, tif):
    a = tif(tmp_path / "a.tif", np.zeros((3, 32, 32), "uint8"), nodata=0)
    b = tif(tmp_path / "b.tif", rgb(None, 32))
    rep = validate_pair(a, b)
    assert not rep.ok and any("valid" in e.lower() for e in rep.errors)


def test_partial_nodata_warns_and_is_measured(tmp_path, tif):
    d = rgb(None, 64) + 1
    d[:, :16, :] = 0
    a = tif(tmp_path / "a.tif", d, nodata=0)
    b = tif(tmp_path / "b.tif", rgb(None, 64))
    rep = validate_pair(a, b)
    assert rep.ok
    assert abs(rep.before.valid_fraction - 0.75) < 0.01
    assert any("NoData" in w for w in rep.warnings)


def test_dtype_difference_warns(tmp_path, tif):
    a = tif(tmp_path / "a.tif", rgb(None), dtype="uint8")
    b = tif(tmp_path / "b.tif", rgb(None).astype("uint16") * 30, dtype="uint16")
    rep = validate_pair(a, b)
    assert rep.ok and any("Data types differ" in w for w in rep.warnings)


def test_report_is_serialisable(synthetic_pair):
    import json
    json.dumps(validate_pair(synthetic_pair["before"], synthetic_pair["after"]).to_dict())
