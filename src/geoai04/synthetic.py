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


def generate_synthetic_pair(
    out_dir: str | Path,
    size: int = 512,
    gsd: float = 0.5,
    seed: int = 42,
    epsg: int = 32633,
    origin: tuple[float, float] = (500000.0, 4500000.0),
    shift_px: tuple[float, float] | None = None,
    add_phenology: bool = False,
    add_nodata_block: bool = False,
    add_vegetation_conversion: bool = False,
) -> dict:
    """Write before.tif, after.tif, reference_mask.tif and synthetic_pair_meta.json; return the metadata.

    Extended with optional Phase 2 test modes:
      shift_px: translation (row, col) applied to AFTER; reference mask stays in original frame.
      add_phenology: diffuse green-to-brown region that is NOT built change.
      add_nodata_block: zero/NoData block in upper corner.
      add_vegetation_conversion: building erected on deep green terrain.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    before = _terrain(rng, size)
    roof, bare = np.array([205, 200, 190], np.float32), np.array([150, 125, 95], np.float32)

    name, r0, r1, c0, c1 = EXISTING_BUILDING
    before[r0:r1, c0:c1] = roof
    after = before.copy()
    ref = np.zeros((size, size), dtype=np.uint8)

    changes_list = list(CHANGES)
    if add_vegetation_conversion:
        # Building erected on green pasture
        vr0, vr1, vc0, vc1 = 410, 460, 60, 110
        changes_list.append(("veg_conversion_building", vr0, vr1, vc0, vc1, "new"))
        # Force green pasture in BEFORE
        before[vr0 - 10: vr1 + 10, vc0 - 10: vc1 + 10] = np.array([30, 170, 30], np.float32)
        after[vr0 - 10: vr1 + 10, vc0 - 10: vc1 + 10] = np.array([30, 170, 30], np.float32)

    for _, r0, r1, c0, c1, kind in changes_list:
        if kind == "new":
            after[r0:r1, c0:c1] = roof
        else:
            before[r0:r1, c0:c1] = roof
            after[r0:r1, c0:c1] = bare
        ref[r0:r1, c0:c1] = 1

    if add_phenology:
        # Diffuse green-to-brown seasonal change (NOT a built change, ref remains 0)
        pr0, pr1, pc0, pc1 = 300, 420, 240, 380
        after[pr0:pr1, pc0:pc1, 1] = np.clip(after[pr0:pr1, pc0:pc1, 1] * 0.45, 0, 255)
        after[pr0:pr1, pc0:pc1, 0] = np.clip(after[pr0:pr1, pc0:pc1, 0] * 1.35 + 20, 0, 255)

    # Nuisance differences that are NOT change: slight brightness gain and sensor noise
    after = after * 1.03 + 2.0
    before = before + rng.normal(0, 2.0, before.shape)
    after = after + rng.normal(0, 2.0, after.shape)

    if shift_px is not None:
        sr, sc = shift_px
        for b in range(3):
            after[..., b] = ndimage.shift(after[..., b], (sr, sc), order=1, mode="nearest")

    before, after = (np.clip(a, 0, 255).astype(np.uint8).transpose(2, 0, 1) for a in (before, after))

    nodata_val = None
    if add_nodata_block:
        before[:, :32, :32] = 0
        after[:, :32, :32] = 0
        nodata_val = 0

    transform = from_origin(origin[0], origin[1], gsd, gsd)
    prof = dict(
        driver="GTiff",
        height=size,
        width=size,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="deflate",
        nodata=nodata_val,
    )
    with rasterio.open(out / "before.tif", "w", count=3, dtype="uint8", **prof) as d:
        d.write(before)
    with rasterio.open(out / "after.tif", "w", count=3, dtype="uint8", **prof) as d:
        d.write(after)
    with rasterio.open(out / "reference_mask.tif", "w", count=1, dtype="uint8", **prof) as d:
        d.write(ref, 1)

    meta = {
        "synthetic": True,
        "warning": "SYNTHETIC test data with arbitrary coordinates. Not satellite imagery.",
        "epsg": epsg,
        "gsd_m": gsd,
        "size_px": size,
        "seed": seed,
        "origin_xy": list(origin),
        "shift_px": shift_px,
        "add_phenology": add_phenology,
        "add_nodata_block": add_nodata_block,
        "add_vegetation_conversion": add_vegetation_conversion,
        "changes": [
            {
                "name": n,
                "rows": [r0, r1],
                "cols": [c0, c1],
                "kind": k,
                "area_m2": (r1 - r0) * (c1 - c0) * gsd * gsd,
            }
            for n, r0, r1, c0, c1, k in changes_list
        ],
        "total_changed_area_m2": float(ref.sum()) * gsd * gsd,
    }
    (out / "synthetic_pair_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
