"""Synthetic, georeferenced test pair with a known change and reference mask.

THIS IS SYNTHETIC TEST DATA. It is not satellite imagery and must never be presented as
such. Its coordinates are arbitrary. It exists so the pipeline can be tested with exact
ground truth (known areas) without downloading anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

# (name, row0, row1, col0, col1, kind) in pixel coordinates, end-exclusive
CHANGES = [
    ("new_building_A", 100, 160, 120, 200, "new"),
    ("new_building_B", 220, 260, 330, 370, "new"),
    ("demolished_building", 330, 390, 150, 210, "demolished"),
]
EXISTING_BUILDING = ("existing_building", 60, 100, 380, 440)  # unchanged, same in both dates


def _terrain(rng: np.random.Generator, size: int) -> np.ndarray:
    low = ndimage.gaussian_filter(rng.normal(size=(size, size)), 18)
    low = (low - low.min()) / (low.max() - low.min())
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = 70 + 40 * low
    img[..., 1] = 110 + 50 * low
    img[..., 2] = 60 + 25 * low
    img[size // 2 - 3: size // 2 + 3, :, :] = (130, 128, 122)  # road
    return img


def generate_synthetic_pair(out_dir: str | Path, size: int = 512, gsd: float = 0.5, seed: int = 42,
                            epsg: int = 32633, origin: tuple[float, float] = (500000.0, 4500000.0)) -> dict:
    """Write before.tif, after.tif, reference_mask.tif and synthetic_pair_meta.json; return the metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    before = _terrain(rng, size)
    roof, bare = np.array([205, 200, 190], np.float32), np.array([150, 125, 95], np.float32)

    name, r0, r1, c0, c1 = EXISTING_BUILDING
    before[r0:r1, c0:c1] = roof
    after = before.copy()
    ref = np.zeros((size, size), dtype=np.uint8)
    for _, r0, r1, c0, c1, kind in CHANGES:
        if kind == "new":
            after[r0:r1, c0:c1] = roof
        else:
            before[r0:r1, c0:c1] = roof
            after[r0:r1, c0:c1] = bare
        ref[r0:r1, c0:c1] = 1
    # nuisance differences that are NOT change: slight brightness gain and independent sensor noise
    after = after * 1.03 + 2.0
    before = before + rng.normal(0, 2.0, before.shape)
    after = after + rng.normal(0, 2.0, after.shape)
    before, after = (np.clip(a, 0, 255).astype(np.uint8).transpose(2, 0, 1) for a in (before, after))

    transform = from_origin(origin[0], origin[1], gsd, gsd)
    prof = dict(driver="GTiff", height=size, width=size, crs=f"EPSG:{epsg}", transform=transform, compress="deflate")
    with rasterio.open(out / "before.tif", "w", count=3, dtype="uint8", **prof) as d:
        d.write(before)
    with rasterio.open(out / "after.tif", "w", count=3, dtype="uint8", **prof) as d:
        d.write(after)
    with rasterio.open(out / "reference_mask.tif", "w", count=1, dtype="uint8", **prof) as d:
        d.write(ref, 1)
    meta = {
        "synthetic": True,
        "warning": "SYNTHETIC test data with arbitrary coordinates. Not satellite imagery.",
        "epsg": epsg, "gsd_m": gsd, "size_px": size, "seed": seed, "origin_xy": list(origin),
        "changes": [{"name": n, "rows": [r0, r1], "cols": [c0, c1], "kind": k,
                     "area_m2": (r1 - r0) * (c1 - c0) * gsd * gsd} for n, r0, r1, c0, c1, k in CHANGES],
        "total_changed_area_m2": float(ref.sum()) * gsd * gsd,
    }
    (out / "synthetic_pair_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
