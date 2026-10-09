"""Pixel-level evaluation metrics against a reference mask (valid pixels only)."""
from __future__ import annotations

import numpy as np


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
