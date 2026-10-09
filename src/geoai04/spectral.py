"""Spectral indices and index-mode change detector.

Computes physical spectral indices (NDVI, NDBI, MNDWI, NDWI) ONLY when required bands
are explicitly mapped in config bands.band_map. Band order is never assumed.

Provides a clearly labelled RGB proxy: vegetation_proxy_exg (Excess Green: 2g - r - b),
which is NEVER called NDVI.

Computes change scores as root-mean-square difference of available indices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from .baseline_detector import DetectionResult, apply_threshold, normalise, smooth_scores
from .config import AppConfig, DetectionConfig

log = logging.getLogger(__name__)

SPECTRAL_DETECTOR_NAME = "spectral_index_difference"


def normalized_difference(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Safe normalized difference (b1 - b2) / (b1 + b2), bounded in [-1.0, 1.0]."""
    den = b1 + b2
    num = b1 - b2
    valid = np.abs(den) > 1e-6
    out = np.zeros_like(b1, dtype=np.float32)
    np.divide(num, den, out=out, where=valid)
    return np.clip(out, -1.0, 1.0)


def excess_green_proxy(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vegetation proxy for RGB imagery: ExG = 2*G - R - B on [0, 1] normalized bands.

    Named vegetation_proxy_exg. This is a heuristic color proxy and is NEVER called NDVI.
    Normalized by 2.0 to span [-1.0, 1.0].
    """
    exg = 2.0 * g - r - b
    return np.clip(exg / 2.0, -1.0, 1.0).astype(np.float32)


def compute_spectral_index(
    bands_dict: dict[str, np.ndarray],
    index_name: str,
) -> tuple[np.ndarray | None, str | None]:
    """Compute a single spectral index from named normalized band arrays.

    Returns (index_array, None) if successful, or (None, reason) if required bands are missing.
    """
    b = {k.lower(): v for k, v in bands_dict.items()}
    idx = index_name.lower()

    if idx == "ndvi":
        if "nir" not in b or "red" not in b:
            return None, "NDVI requires 'nir' and 'red' bands."
        return normalized_difference(b["nir"], b["red"]), None

    if idx == "ndbi":
        if "swir1" not in b or "nir" not in b:
            return None, "NDBI requires 'swir1' and 'nir' bands."
        return normalized_difference(b["swir1"], b["nir"]), None

    if idx == "mndwi":
        if "green" not in b or "swir1" not in b:
            return None, "MNDWI requires 'green' and 'swir1' bands."
        return normalized_difference(b["green"], b["swir1"]), None

    if idx == "ndwi":
        if "green" not in b or "nir" not in b:
            return None, "NDWI requires 'green' and 'nir' bands."
        return normalized_difference(b["green"], b["nir"]), None

    if idx == "vegetation_proxy_exg":
        if "red" not in b or "green" not in b or "blue" not in b:
            return None, "vegetation_proxy_exg requires 'red', 'green', and 'blue' bands."
        return excess_green_proxy(b["red"], b["green"], b["blue"]), None

    return None, f"Unknown spectral index: {index_name}"


def _windows(width: int, height: int, tile: int):
    for r in range(0, height, tile):
        for c in range(0, width, tile):
            yield Window(c, r, min(tile, width - c), min(tile, height - r))


def run_spectral_detector(
    before_path: str | Path,
    after_path: str | Path,
    cfg: AppConfig,
) -> tuple[DetectionResult, dict[str, np.ndarray]]:
    """Run index-mode change detector on grid-identical rasters.

    Change score = root-mean-square of normalized absolute changes in available indices.
    Returns (DetectionResult, delta_rasters_dict).
    """
    band_map = {k.lower(): v for k, v in cfg.bands.band_map.items()}
    det_cfg = cfg.detection

    with rasterio.open(before_path) as sb, rasterio.open(after_path) as sa:
        width, height = sb.width, sb.height
        needed_bands = sorted(set(band_map.values()))

        # Determine which indices can be computed
        indices_to_compute = []
        if "nir" in band_map and "red" in band_map:
            indices_to_compute.append("ndvi")
        if "swir1" in band_map and "nir" in band_map:
            indices_to_compute.append("ndbi")
        if "green" in band_map and "swir1" in band_map:
            indices_to_compute.append("mndwi")
        if cfg.bands.vegetation_proxy_exg and all(k in band_map for k in ("red", "green", "blue")):
            indices_to_compute.append("vegetation_proxy_exg")

        if not indices_to_compute:
            raise ValueError(
                f"No spectral indices could be computed with configured band_map: {band_map}. "
                "Band order is never assumed. Provide required bands or use baseline detector."
            )

        scores = np.zeros((height, width), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        delta_rasters = {idx: np.zeros((height, width), dtype=np.float32) for idx in indices_to_compute}

        for w in _windows(width, height, det_cfg.tile_size):
            r0, c0 = int(w.row_off), int(w.col_off)
            sl = (slice(r0, r0 + int(w.height)), slice(c0, c0 + int(w.width)))

            mb = sb.dataset_mask(window=w) > 0
            ma = sa.dataset_mask(window=w) > 0
            ok = mb & ma

            bands_b = {}
            bands_a = {}
            for name, band_idx in band_map.items():
                raw_b = sb.read(band_idx, window=w)
                raw_a = sa.read(band_idx, window=w)
                if raw_b.dtype.kind == "f":
                    ok &= np.isfinite(raw_b)
                if raw_a.dtype.kind == "f":
                    ok &= np.isfinite(raw_a)
                bands_b[name] = normalise(raw_b, sb.dtypes[band_idx - 1], det_cfg)
                bands_a[name] = normalise(raw_a, sa.dtypes[band_idx - 1], det_cfg)

            sq_diffs = []
            for idx_name in indices_to_compute:
                arr_b, _ = compute_spectral_index(bands_b, idx_name)
                arr_a, _ = compute_spectral_index(bands_a, idx_name)
                # Signed delta (after - before)
                delta = arr_a - arr_b
                delta_rasters[idx_name][sl] = np.where(ok, delta, 0.0)
                # Normalized absolute change (indices span [-1, 1], max span 2.0)
                norm_change = np.abs(delta) / 2.0
                sq_diffs.append(norm_change ** 2)

            # RMS change score
            rms = np.sqrt(np.mean(sq_diffs, axis=0)).astype(np.float32)
            scores[sl] = np.where(ok, rms, 0.0)
            valid[sl] = ok

    scores = smooth_scores(scores, valid, det_cfg.smoothing_sigma_px)
    mask, thr = apply_threshold(scores, valid, det_cfg)

    log.info(
        "Spectral detector: indices=%s threshold=%.4f changed=%.3f%% of valid pixels",
        indices_to_compute, thr, 100.0 * mask.sum() / max(1, valid.sum())
    )

    det_result = DetectionResult(
        scores=scores,
        valid=valid,
        mask=mask,
        threshold=thr,
        threshold_method=det_cfg.threshold_method,
        band_indices=needed_bands,
        detector=SPECTRAL_DETECTOR_NAME,
    )
    return det_result, delta_rasters
