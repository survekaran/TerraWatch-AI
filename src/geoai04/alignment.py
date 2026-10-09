"""Alignment gate and sub-pixel registration assessment for bi-temporal raster pairs.

SIGN CONVENTION:
The estimated shifts (shift_row_px, shift_col_px) represent the translation vector
that must be applied to the AFTER image to register it with the BEFORE image:
    registered_after(r, c) = after(r - shift_row_px, c - shift_col_px)
Equivalently, if features in the AFTER image appear displaced by (+dr, +dc) pixels
relative to the BEFORE image, the correction vector is (-dr, -dc).
scipy.ndimage.shift(after, (shift_row_px, shift_col_px)) brings AFTER into alignment
with BEFORE.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation

from .config import AlignmentConfig

log = logging.getLogger(__name__)

INTERPOLATION_ORDERS = {
    "nearest": 0,
    "linear": 1,
    "spline": 3,
}


@dataclass
class AlignmentResult:
    shift_row_px: float | None
    shift_col_px: float | None
    magnitude_px: float | None
    shift_m: float | None
    per_window_estimates: list[dict[str, Any]]
    spread_px: float | None
    n_windows_used: int
    status: str  # "pass" | "warn" | "fail" | "unknown"
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlignmentError(Exception):
    """Raised when alignment gate fails and allow_misaligned is False."""

    def __init__(self, result: AlignmentResult, run_dir: Path | None = None):
        self.result = result
        self.run_dir = run_dir
        super().__init__(
            f"Alignment gate failed (status={result.status}, shift={result.magnitude_px:.2f}px): "
            + "; ".join(result.reasons)
        )


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert multiband (C, H, W) or singleband raster to normalized float32 grayscale."""
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.shape[0] >= 3:
        # Standard luminance weights for first 3 bands (RGB)
        gray = (
            arr[0].astype(np.float32) * 0.2989
            + arr[1].astype(np.float32) * 0.5870
            + arr[2].astype(np.float32) * 0.1140
        )
    else:
        gray = arr.mean(axis=0).astype(np.float32)
    min_v, max_v = gray.min(), gray.max()
    if max_v > min_v:
        gray = (gray - min_v) / (max_v - min_v)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    return gray


def _sample_windows(width: int, height: int, win_size: int, n_target: int) -> list[Window]:
    """Generate a distributed grid of candidate windows across the raster."""
    win_w = min(win_size, width)
    win_h = min(win_size, height)
    if win_w >= width and win_h >= height:
        return [Window(0, 0, width, height)]

    grid_dim = int(math.ceil(math.sqrt(n_target)))
    row_starts = np.linspace(0, max(0, height - win_h), grid_dim, dtype=int)
    col_starts = np.linspace(0, max(0, width - win_w), grid_dim, dtype=int)

    windows = []
    for r in row_starts:
        for c in col_starts:
            windows.append(Window(int(c), int(r), win_w, win_h))
            if len(windows) >= n_target:
                return windows
    return windows


def estimate_alignment(
    before_path: str | Path,
    after_path: str | Path,
    cfg: AlignmentConfig | None = None,
    gsd_x_m: float | None = None,
    gsd_y_m: float | None = None,
) -> AlignmentResult:
    """Estimate sub-pixel translation between BEFORE and AFTER rasters using phase cross correlation.

    Memory-bounded: samples candidate windows using windowed reads and computes median shift.
    Sign convention: returns the vector (shift_row_px, shift_col_px) to apply to AFTER to align with BEFORE.
    """
    cfg = cfg or AlignmentConfig()
    before_path, after_path = Path(before_path), Path(after_path)

    with rasterio.open(before_path) as sb, rasterio.open(after_path) as sa:
        if (sb.width, sb.height) != (sa.width, sa.height):
            return AlignmentResult(
                shift_row_px=None,
                shift_col_px=None,
                magnitude_px=None,
                shift_m=None,
                per_window_estimates=[],
                spread_px=None,
                n_windows_used=0,
                status="unknown",
                reasons=["Raster dimensions differ; cannot measure alignment."],
            )

        width, height = sb.width, sb.height
        candidate_windows = _sample_windows(width, height, cfg.window_size_px, cfg.n_windows)
        usable_shifts: list[tuple[float, float]] = []
        window_estimates: list[dict[str, Any]] = []

        bands_to_read = list(range(1, min(sb.count, sa.count, 3) + 1))

        for w in candidate_windows:
            mb = sb.dataset_mask(window=w) > 0
            ma = sa.dataset_mask(window=w) > 0
            valid_mask = mb & ma
            valid_frac = float(valid_mask.mean())

            if valid_frac < 0.85:
                continue

            rb = sb.read(bands_to_read, window=w)
            ra = sa.read(bands_to_read, window=w)

            gb = _to_gray(rb)
            ga = _to_gray(ra)

            # Mask invalid areas before Sobel
            gb[~valid_mask] = 0.0
            ga[~valid_mask] = 0.0

            grad_b = sobel(gb)
            grad_a = sobel(ga)

            tex_b = float(grad_b[valid_mask].std()) if valid_mask.any() else 0.0
            tex_a = float(grad_a[valid_mask].std()) if valid_mask.any() else 0.0
            min_tex = min(tex_b, tex_a)

            if min_tex < cfg.texture_threshold:
                continue

            try:
                shift, error, _ = phase_cross_correlation(
                    grad_b, grad_a, upsample_factor=cfg.upsample_factor
                )
                sr, sc = float(shift[0]), float(shift[1])
                usable_shifts.append((sr, sc))
                window_estimates.append({
                    "col_off": int(w.col_off),
                    "row_off": int(w.row_off),
                    "width": int(w.width),
                    "height": int(w.height),
                    "shift_row_px": sr,
                    "shift_col_px": sc,
                    "texture": min_tex,
                    "error": float(error),
                })
            except Exception as e:
                log.debug("Phase correlation error on window %s: %s", w, e)
                continue

    n_used = len(usable_shifts)
    if n_used < cfg.min_windows:
        reasons = [
            f"Too few textured, valid windows found ({n_used} usable < {cfg.min_windows} required)."
        ]
        log.warning("Alignment gate unknown: %s", reasons[0])
        return AlignmentResult(
            shift_row_px=None,
            shift_col_px=None,
            magnitude_px=None,
            shift_m=None,
            per_window_estimates=window_estimates,
            spread_px=None,
            n_windows_used=n_used,
            status="unknown",
            reasons=reasons,
        )

    row_shifts = [s[0] for s in usable_shifts]
    col_shifts = [s[1] for s in usable_shifts]

    med_row = float(np.median(row_shifts))
    med_col = float(np.median(col_shifts))
    magnitude = float(math.hypot(med_row, med_col))

    # Spread across windows (standard deviation of distance from median)
    dists = [math.hypot(r - med_row, c - med_col) for r, c in usable_shifts]
    spread = float(np.std(dists)) if len(dists) > 1 else 0.0

    shift_m = None
    if gsd_x_m is not None and gsd_y_m is not None:
        shift_m = float(math.hypot(med_col * gsd_x_m, med_row * gsd_y_m))

    reasons = []
    if spread > cfg.spread_warn_px:
        reasons.append(
            f"High spread across windows ({spread:.2f} px > {cfg.spread_warn_px:.2f} px): "
            "genuine change or parallax may be contaminating the estimate."
        )

    if magnitude >= cfg.fail_px:
        status = "fail"
        reasons.append(
            f"Estimated misregistration of {magnitude:.2f} px exceeds failure threshold ({cfg.fail_px:.2f} px)."
        )
    elif magnitude >= cfg.warn_px:
        status = "warn"
        reasons.append(
            f"Estimated misregistration of {magnitude:.2f} px exceeds warning threshold ({cfg.warn_px:.2f} px)."
        )
    elif spread > cfg.spread_warn_px:
        status = "warn"
    else:
        status = "pass"
        reasons.append(
            f"Alignment within acceptable tolerance ({magnitude:.2f} px < {cfg.warn_px:.2f} px)."
        )

    log.info(
        "Alignment estimate: status=%s, shift=(%.2f, %.2f) px, magnitude=%.2f px (%.2f m), "
        "spread=%.2f px over %d windows",
        status, med_row, med_col, magnitude, shift_m or 0.0, spread, n_used
    )

    return AlignmentResult(
        shift_row_px=med_row,
        shift_col_px=med_col,
        magnitude_px=magnitude,
        shift_m=shift_m,
        per_window_estimates=window_estimates,
        spread_px=spread,
        n_windows_used=n_used,
        status=status,
        reasons=reasons,
    )


def apply_alignment_correction(
    after_path: str | Path,
    out_path: str | Path,
    shift_row_px: float,
    shift_col_px: float,
    interpolation: str = "linear",
) -> Path:
    """Apply estimated sub-pixel translation to the AFTER raster and write the corrected raster.

    The original input file is never modified.
    Pixels shifted in from outside the raster domain are masked as NoData.
    """
    after_path, out_path = Path(after_path), Path(out_path)
    order = INTERPOLATION_ORDERS.get(interpolation, 1)

    with rasterio.open(after_path) as src:
        prof = src.profile.copy()
        count = src.count
        height, width = src.height, src.width
        nodata = src.nodata

        # Determine nodata value if none specified
        if nodata is None:
            if np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
                nodata = 0
            else:
                nodata = -9999.0
            prof["nodata"] = nodata

        prof["compress"] = "deflate"

        # Boundary tracking: 1 inside, 0 outside
        boundary = np.ones((height, width), dtype=np.float32)
        shifted_boundary = (
            ndimage.shift(boundary, (shift_row_px, shift_col_px), order=0, cval=0.0) > 0.5
        )

        orig_mask = src.dataset_mask() > 0
        shifted_mask = (
            ndimage.shift(orig_mask.astype(np.float32), (shift_row_px, shift_col_px), order=0, cval=0.0) > 0.5
        )
        final_valid = shifted_boundary & shifted_mask

        data = src.read()
        shifted_bands = np.zeros_like(data)

        for b in range(count):
            band = data[b].astype(np.float32)
            cval = float(nodata)
            shifted = ndimage.shift(band, (shift_row_px, shift_col_px), order=order, cval=cval)
            shifted[~final_valid] = cval
            if np.issubdtype(np.dtype(src.dtypes[b]), np.integer):
                shifted = np.clip(np.round(shifted), 0, np.iinfo(src.dtypes[b]).max)
            shifted_bands[b] = shifted.astype(src.dtypes[b])

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(shifted_bands)
        dst.write_mask((final_valid.astype(np.uint8) * 255))

    log.info(
        "Wrote alignment-corrected raster to %s (applied shift: (%.2f, %.2f) px, order=%d)",
        out_path, shift_row_px, shift_col_px, order
    )
    return out_path
