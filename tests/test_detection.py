import numpy as np

from geoai04.baseline_detector import apply_threshold, normalise, run_baseline, smooth_scores
from geoai04.config import DetectionConfig, PostprocessingConfig
from geoai04.metrics import pixel_metrics
from geoai04.postprocessing import clean_mask


def test_normalise_by_dtype():
    cfg = DetectionConfig()
    assert normalise(np.array([255], "uint8"), "uint8", cfg)[0] == 1.0
    assert abs(normalise(np.array([5000], "uint16"), "uint16", cfg)[0] - 0.5) < 1e-6
    assert normalise(np.array([2.0], "float32"), "float32", cfg)[0] == 1.0  # clipped


def test_baseline_finds_known_change(synthetic_pair):
    import rasterio
    res = run_baseline(synthetic_pair["before"], synthetic_pair["after"], DetectionConfig())
    with rasterio.open(synthetic_pair["reference"]) as s:
        ref = s.read(1) > 0
    m = pixel_metrics(res.mask, ref, res.valid)
    assert m["iou"] > 0.9 and m["recall"] > 0.95
    assert res.scores.min() >= 0 and res.scores.max() <= 1


def test_identical_images_produce_no_change(tmp_path, tif):
    rng = np.random.default_rng(1)
    d = rng.integers(0, 255, (3, 64, 64)).astype("uint8")
    a, b = tif(tmp_path / "a.tif", d), tif(tmp_path / "b.tif", d)
    cfg = DetectionConfig(threshold_method="fixed", fixed_threshold=0.15)
    res = run_baseline(a, b, cfg)
    assert res.mask.sum() == 0 and res.scores.max() == 0


def test_nodata_pixels_are_excluded(tmp_path, tif):
    rng = np.random.default_rng(2)
    base = rng.integers(10, 200, (3, 64, 64)).astype("uint8")
    changed = base.copy()
    changed[:, :, :] = 255 - base  # everything differs strongly
    base[:, :20, :] = 0   # nodata block in 'before'
    a = tif(tmp_path / "a.tif", base, nodata=0)
    b = tif(tmp_path / "b.tif", changed)
    res = run_baseline(a, b, DetectionConfig(threshold_method="fixed", fixed_threshold=0.3, smoothing_sigma_px=0))
    assert not res.valid[:20].any()
    assert not res.mask[:20].any()
    assert res.scores[:20].max() == 0


def test_threshold_methods():
    rng = np.random.default_rng(3)
    s = rng.random((50, 50)).astype("float32") * 0.1
    s[:10, :10] = 0.8
    v = np.ones_like(s, bool)
    for method in ("otsu", "fixed", "percentile"):
        mask, thr = apply_threshold(s, v, DetectionConfig(threshold_method=method))
        assert mask[:10, :10].all() and thr > 0


def test_smoothing_does_not_bleed_nodata():
    s = np.ones((20, 20), "float32")
    v = np.ones((20, 20), bool)
    v[:, 10:] = False
    s[~v] = 0
    out = smooth_scores(s, v, 2.0)
    assert np.allclose(out[v], 1.0, atol=1e-4) and (out[~v] == 0).all()


def test_morphology_and_mmu_remove_noise():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True        # 900 px real region
    mask[70, 70] = True              # isolated noise pixel
    mask[60:63, 60:63] = True        # 9 px blob, below MMU
    cfg = PostprocessingConfig(opening_radius_px=1, closing_radius_px=1, min_mapping_unit_m2=25.0)
    res = clean_mask(mask, np.ones_like(mask), cfg, pixel_area_m2=0.25)  # MMU = 100 px
    assert res.components_after_mmu == 1 and res.mmu_pixels == 100
    assert res.mask[20, 20] and not res.mask[70, 70] and not res.mask[61, 61]
