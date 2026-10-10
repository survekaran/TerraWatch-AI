"""Evaluation metrics against a reference mask (valid pixels only and geometric false alarm area)."""
from __future__ import annotations

from typing import Any
import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely.ops import unary_union


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def pixel_metrics(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Confusion counts plus precision, recall, F1 and IoU; undefined ratios are None."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs reference {ref.shape}")
    if valid is not None:
        pred, ref = pred[valid], ref[valid]
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum()); tn = int((~pred & ~ref).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn), "iou": _ratio(tp, tp + fp + fn),
    }


def compute_false_alarm_area_m2(
    gdf: gpd.GeoDataFrame,
    ref_mask: np.ndarray,
    transform: Any,
    crs: Any = None,
) -> float:
    """Compute total polygon area (m2) not overlapping the reference mask.

    Boundary intersections (merely touching) do not count as overlap and
    contribute 100% of their area to false alarm.
    """
    if len(gdf) == 0 or not np.any(ref_mask > 0):
        return float(gdf["area_m2"].sum()) if len(gdf) and "area_m2" in gdf else 0.0

    ref_shapes = [
        shape(geom)
        for geom, val in rasterio.features.shapes(ref_mask.astype(np.uint8), transform=transform)
        if val > 0
    ]
    if not ref_shapes:
        return float(gdf["area_m2"].sum()) if "area_m2" in gdf else 0.0

    ref_union = unary_union(ref_shapes)
    total_fa_area = 0.0
    for geom in gdf.geometry:
        diff = geom.difference(ref_union)
        total_fa_area += diff.area

    return float(round(total_fa_area, 4))
