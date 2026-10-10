"""Module D: transparent baseline change detector (no machine learning).

Score = root-mean-square difference between the normalised pixel values of the two
dates over the first N bands, in the range 0..1. It is *not* AI inference; it is a
thresholded image difference and is labelled that way in every output.

Band identity is not assumed. Spectral indices (NDVI, NDBI, ...) require a known band
map and are added in Phase 2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from skimage.filters import threshold_otsu

from .config import DetectionConfig

log = logging.getLogger(__name__)
DETECTOR_NAME = "baseline_difference"


@dataclass
class DetectionResult:
    scores: np.ndarray        # float32 (H, W), 0..1, zero where invalid
    valid: np.ndarray         # bool (H, W)
    mask: np.ndarray          # bool (H, W) raw thresholded mask
    threshold: float
    threshold_method: str
    band_indices: list[int]
    detector: str = DETECTOR_NAME


def normalise(arr: np.ndarray, dtype: str | np.dtype, cfg: DetectionConfig) -> np.ndarray:
    """Scale raw pixel values to roughly 0..1 using the data type (documented in config)."""
    dt = np.dtype(dtype)
    if dt == np.uint8:
        scale = cfg.uint8_scale
    elif dt.kind in "ui":
        scale = cfg.uint16_scale if dt.itemsize == 2 else float(np.iinfo(dt).max)
    else:
        scale = 1.0
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) / np.float32(scale)
    return np.clip(out, 0.0, 1.0)


def _windows(width: int, height: int, tile: int):
    for r in range(0, height, tile):
        for c in range(0, width, tile):
            yield Window(c, r, min(tile, width - c), min(tile, height - r))


def compute_global_mean_std(path: str | Path, bands: list[int], max_tile: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean and standard deviation over valid finite pixels."""
    with rasterio.open(path) as src:
        sums = np.zeros(len(bands), dtype=np.float64)
        sq_sums = np.zeros(len(bands), dtype=np.float64)
        counts = np.zeros(len(bands), dtype=np.int64)

        for w in _windows(src.width, src.height, max_tile):
            arr = src.read(bands, window=w)
            mask = src.dataset_mask(window=w) > 0
            if arr.dtype.kind == "f":
                mask &= np.isfinite(arr).all(axis=0)
            if not mask.any():
                continue
            for i in range(len(bands)):
                vals = arr[i][mask].astype(np.float64)
                sums[i] += vals.sum()
                sq_sums[i] += (vals ** 2).sum()
                counts[i] += vals.size

        means, stds = [], []
        for i in range(len(bands)):
            n = max(1, counts[i])
            m = float(sums[i] / n)
            v = max(0.0, float(sq_sums[i] / n - m ** 2))
            means.append(m)
            stds.append(float(np.sqrt(v)))

        return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def compute_robust_median_mad(
    before_path: str | Path,
    after_path: str | Path,
    bands: list[int],
    cfg: DetectionConfig,
    max_iters: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-band median and MAD on pixels outside initial change mask, iterated at most twice."""
    with rasterio.open(before_path) as sb, rasterio.open(after_path) as sa:
        all_b: list[list[np.ndarray]] = [[] for _ in bands]
        all_a: list[list[np.ndarray]] = [[] for _ in bands]

        for w in _windows(sb.width, sb.height, cfg.tile_size):
            rb, ra = sb.read(bands, window=w), sa.read(bands, window=w)
            ok = (sb.dataset_mask(window=w) > 0) & (sa.dataset_mask(window=w) > 0)
            if rb.dtype.kind == "f":
                ok &= np.isfinite(rb).all(axis=0)
            if ra.dtype.kind == "f":
                ok &= np.isfinite(ra).all(axis=0)
            if not ok.any():
                continue
            for bi in range(len(bands)):
                all_b[bi].append(rb[bi][ok].astype(np.float32))
                all_a[bi].append(ra[bi][ok].astype(np.float32))

        b_vals = [np.concatenate(all_b[bi]) if all_b[bi] else np.array([0.0], dtype=np.float32) for bi in range(len(bands))]
        a_vals = [np.concatenate(all_a[bi]) if all_a[bi] else np.array([0.0], dtype=np.float32) for bi in range(len(bands))]

        n_px = len(b_vals[0])
        unchanged_mask = np.ones(n_px, dtype=bool)

        med_b = np.zeros(len(bands), dtype=np.float32)
        mad_b = np.zeros(len(bands), dtype=np.float32)
        med_a = np.zeros(len(bands), dtype=np.float32)
        mad_a = np.zeros(len(bands), dtype=np.float32)

        for it in range(max_iters):
            if not unchanged_mask.any():
                unchanged_mask = np.ones(n_px, dtype=bool)

            for bi in range(len(bands)):
                vb = b_vals[bi][unchanged_mask]
                va = a_vals[bi][unchanged_mask]
                mb = float(np.median(vb))
                ma = float(np.median(va))
                med_b[bi] = mb
                med_a[bi] = ma
                mad_b[bi] = max(float(np.median(np.abs(vb - mb)) * 1.4826), 1e-4)
                mad_a[bi] = max(float(np.median(np.abs(va - ma)) * 1.4826), 1e-4)

            if it < max_iters - 1:
                diffs_sq = np.zeros(n_px, dtype=np.float32)
                for bi in range(len(bands)):
                    matched_a = (a_vals[bi] - med_a[bi]) * (mad_b[bi] / mad_a[bi]) + med_b[bi]
                    scale = cfg.uint8_scale if sa.dtypes[0] == "uint8" else 1.0
                    d = (matched_a - b_vals[bi]) / scale
                    diffs_sq += d ** 2
                diffs = np.sqrt(diffs_sq / len(bands))
                pct_thresh = float(np.percentile(diffs, 80.0))
                unchanged_mask = diffs <= pct_thresh

        return med_b, mad_b, med_a, mad_a


def compute_scores(before_path: str | Path, after_path: str | Path, cfg: DetectionConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Windowed difference score and validity mask for two grid-identical rasters."""
    with rasterio.open(before_path) as sb, rasterio.open(after_path) as sa:
        if (sb.width, sb.height) != (sa.width, sa.height) or sb.count != sa.count:
            raise ValueError("Rasters must share size and band count (run validation first).")
        n = min(sb.count, cfg.max_bands)
        idx = list(range(1, n + 1))
        scores = np.zeros((sb.height, sb.width), dtype=np.float32)
        valid = np.zeros((sb.height, sb.width), dtype=bool)

        method = getattr(cfg, "radiometric_norm_method", "mean_std")
        apply_rad_norm = getattr(cfg, "radiometric_normalization", False) and method != "off"

        mean_b, std_b, mean_a, std_a = None, None, None, None
        med_b, mad_b, med_a, mad_a = None, None, None, None

        if apply_rad_norm:
            if method == "robust":
                med_b, mad_b, med_a, mad_a = compute_robust_median_mad(before_path, after_path, idx, cfg, max_iters=2)
                log.info("Robust radiometric normalization enabled: matching median/MAD (med_b=%s, mad_b=%s)", med_b, mad_b)
            else:
                mean_b, std_b = compute_global_mean_std(before_path, idx, cfg.tile_size)
                mean_a, std_a = compute_global_mean_std(after_path, idx, cfg.tile_size)
                log.info("Radiometric normalization enabled: matching 'after' bands to 'before' (mean_b=%s, std_b=%s)", mean_b, std_b)

        for w in _windows(sb.width, sb.height, cfg.tile_size):
            rb, ra = sb.read(idx, window=w), sa.read(idx, window=w)
            ok = (sb.dataset_mask(window=w) > 0) & (sa.dataset_mask(window=w) > 0)
            if rb.dtype.kind == "f":
                ok &= np.isfinite(rb).all(axis=0)
            if ra.dtype.kind == "f":
                ok &= np.isfinite(ra).all(axis=0)

            # Apply radiometric matching if enabled
            if apply_rad_norm:
                ra_matched = np.zeros_like(ra, dtype=np.float32)
                if method == "robust" and med_b is not None:
                    for bi in range(n):
                        ra_matched[bi] = (ra[bi].astype(np.float32) - med_a[bi]) * (mad_b[bi] / mad_a[bi]) + med_b[bi]
                elif mean_b is not None and std_b is not None:
                    for bi in range(n):
                        sa_std = max(float(std_a[bi]), 1e-4)
                        ra_matched[bi] = (ra[bi].astype(np.float32) - mean_a[bi]) * (std_b[bi] / sa_std) + mean_b[bi]
                ra_proc = ra_matched
            else:
                ra_proc = ra

            d = normalise(ra_proc, sa.dtypes[0], cfg) - normalise(rb, sb.dtypes[0], cfg)
            s = np.sqrt((d ** 2).mean(axis=0)).astype(np.float32)
            r0, c0 = int(w.row_off), int(w.col_off)
            sl = (slice(r0, r0 + int(w.height)), slice(c0, c0 + int(w.width)))
            scores[sl] = np.where(ok, s, 0.0)
            valid[sl] = ok
    return scores, valid, idx


def smooth_scores(scores: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """NoData-aware Gaussian smoothing (normalised convolution)."""
    if sigma <= 0:
        return scores
    v = valid.astype(np.float32)
    num = ndimage.gaussian_filter(scores * v, sigma)
    den = ndimage.gaussian_filter(v, sigma)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    out[~valid] = 0.0
    return out


def apply_threshold(scores: np.ndarray, valid: np.ndarray, cfg: DetectionConfig) -> tuple[np.ndarray, float]:
    """Threshold the score raster. Returns (boolean mask, threshold used)."""
    vals = scores[valid]
    if vals.size == 0:
        raise ValueError("No valid pixels to threshold.")
    method = cfg.threshold_method
    if method == "fixed":
        thr = cfg.fixed_threshold
    elif method == "percentile":
        thr = float(np.percentile(vals, cfg.percentile))
    elif method == "otsu":
        thr = float(threshold_otsu(vals)) if float(vals.max()) > float(vals.min()) else cfg.otsu_floor
        thr = max(thr, cfg.otsu_floor)
    else:
        raise ValueError(f"Unknown threshold method {method!r}")
    mask = (scores >= thr) & valid
    return mask, float(thr)


def run_baseline(before_path: str | Path, after_path: str | Path, cfg: DetectionConfig) -> DetectionResult:
    """Run the full baseline detector: score, smooth, threshold."""
    scores, valid, idx = compute_scores(before_path, after_path, cfg)
    scores = smooth_scores(scores, valid, cfg.smoothing_sigma_px)
    mask, thr = apply_threshold(scores, valid, cfg)
    log.info("Baseline detector: bands=%s method=%s threshold=%.4f changed=%.3f%% of valid pixels",
             idx, cfg.threshold_method, thr, 100.0 * mask.sum() / max(1, valid.sum()))
    return DetectionResult(scores=scores, valid=valid, mask=mask, threshold=thr, threshold_method=cfg.threshold_method, band_indices=idx)
