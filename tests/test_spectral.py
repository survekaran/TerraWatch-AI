import numpy as np
import pytest

from geoai04.config import AppConfig
from geoai04.spectral import (
    SPECTRAL_DETECTOR_NAME,
    compute_spectral_index,
    normalized_difference,
    run_spectral_detector,
)


def test_normalized_difference_known_values():
    b1 = np.array([0.8, 0.2, 0.5, 0.0])
    b2 = np.array([0.2, 0.8, 0.5, 0.0])

    nd = normalized_difference(b1, b2)
    # (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    assert nd[0] == pytest.approx(0.6)
    # (0.2 - 0.8) / (0.2 + 0.8) = -0.6
    assert nd[1] == pytest.approx(-0.6)
    # Equal values -> 0.0
    assert nd[2] == pytest.approx(0.0)
    # Zero denominator safe -> 0.0
    assert nd[3] == pytest.approx(0.0)


def test_ndvi_ndbi_mndwi_computation():
    bands = {
        "nir": np.array([0.7, 0.1]),
        "red": np.array([0.1, 0.7]),
        "swir1": np.array([0.8, 0.2]),
        "green": np.array([0.4, 0.3]),
    }

    # NDVI = (nir - red) / (nir + red)
    ndvi, err = compute_spectral_index(bands, "ndvi")
    assert err is None
    assert ndvi[0] == pytest.approx((0.7 - 0.1) / (0.7 + 0.1))
    assert ndvi[1] == pytest.approx((0.1 - 0.7) / (0.1 + 0.7))

    # NDBI = (swir1 - nir) / (swir1 + nir)
    ndbi, err = compute_spectral_index(bands, "ndbi")
    assert err is None
    assert ndbi[0] == pytest.approx((0.8 - 0.7) / (0.8 + 0.7))

    # MNDWI = (green - swir1) / (green + swir1)
    mndwi, err = compute_spectral_index(bands, "mndwi")
    assert err is None
    assert mndwi[0] == pytest.approx((0.4 - 0.8) / (0.4 + 0.8))


def test_missing_band_returns_unavailable_with_reason():
    incomplete = {"red": np.array([0.5])}
    ndvi, err = compute_spectral_index(incomplete, "ndvi")
    assert ndvi is None
    assert "nir" in err.lower()


def test_band_order_not_assumed():
    # Bands can be in any key order or naming case
    bands = {"NIR": np.array([0.6]), "RED": np.array([0.2])}
    ndvi, err = compute_spectral_index(bands, "NDVI")
    assert err is None
    assert ndvi[0] == pytest.approx(0.5)


def test_vegetation_proxy_exg_not_called_ndvi():
    rgb = {
        "red": np.array([0.2, 0.5]),
        "green": np.array([0.8, 0.1]),
        "blue": np.array([0.2, 0.1]),
    }
    exg, err = compute_spectral_index(rgb, "vegetation_proxy_exg")
    assert err is None
    # 2*0.8 - 0.2 - 0.2 = 1.2, scaled / 2.0 = 0.6
    assert exg[0] == pytest.approx(0.6)
    # 2*0.1 - 0.5 - 0.1 = -0.4, scaled / 2.0 = -0.2
    assert exg[1] == pytest.approx(-0.2)


def test_spectral_detector_end_to_end(tmp_path, tif):
    # Create 4-band test rasters (B, G, R, NIR)
    # Date 1: High NIR (vegetation)
    b1 = np.zeros((4, 32, 32), dtype="uint8")
    b1[0] = 50   # blue
    b1[1] = 100  # green
    b1[2] = 40   # red
    b1[3] = 200  # nir

    # Date 2: Low NIR, high red in a patch (vegetation cleared)
    b2 = b1.copy()
    b2[2, 10:20, 10:20] = 180  # red up
    b2[3, 10:20, 10:20] = 50   # nir down

    p_before = tif(tmp_path / "ms_b.tif", b1)
    p_after = tif(tmp_path / "ms_a.tif", b2)

    cfg = AppConfig()
    cfg.bands.band_map = {"blue": 1, "green": 2, "red": 3, "nir": 4}
    cfg.detection.threshold_method = "fixed"
    cfg.detection.fixed_threshold = 0.1

    res, deltas = run_spectral_detector(p_before, p_after, cfg)

    assert res.detector == SPECTRAL_DETECTOR_NAME
    assert "ndvi" in deltas
    assert res.mask[15, 15]  # Changed patch detected
    assert not res.mask[2, 2]  # Unchanged background
    assert deltas["ndvi"][15, 15] < -0.3  # NDVI dropped significantly
