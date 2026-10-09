import numpy as np
import pytest

from geoai04.metrics import pixel_metrics


def test_perfect_prediction():
    m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
    r = pixel_metrics(m, m)
    assert r["iou"] == 1.0 and r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0


def test_known_confusion():
    ref = np.zeros((10, 10), bool); ref[:5] = True            # 50 positives
    pred = np.zeros((10, 10), bool); pred[:3] = True; pred[8:9] = True   # 30 TP, 10 FP
    r = pixel_metrics(pred, ref)
    assert (r["tp"], r["fp"], r["fn"]) == (30, 10, 20)
    assert r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(0.6)
    assert r["iou"] == pytest.approx(30 / 60) and r["f1"] == pytest.approx(60 / 90)


def test_undefined_ratios_are_none_and_valid_mask_applies():
    z = np.zeros((4, 4), bool)
    r = pixel_metrics(z, z)
    assert r["iou"] is None and r["precision"] is None
    ref = np.ones((4, 4), bool); valid = np.zeros((4, 4), bool); valid[0] = True
    assert pixel_metrics(z, ref, valid)["fn"] == 4


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pixel_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
