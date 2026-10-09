"""Module G (part 1): morphological filtering, connected components, minimum mapping unit."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import disk

from .config import PostprocessingConfig

log = logging.getLogger(__name__)
_STRUCT = {4: ndimage.generate_binary_structure(2, 1), 8: ndimage.generate_binary_structure(2, 2)}


@dataclass
class CleanResult:
    mask: np.ndarray      # bool, cleaned
    labels: np.ndarray    # int32, 0 = background
    components_raw: int
    components_after_morphology: int
    components_after_mmu: int
    mmu_pixels: int


def clean_mask(mask: np.ndarray, valid: np.ndarray, cfg: PostprocessingConfig, pixel_area_m2: float, connectivity: int = 8) -> CleanResult:
    """Opening, closing, connected components, then removal of regions below the MMU (m2)."""
    struct = _STRUCT[connectivity]
    raw_labels, n_raw = ndimage.label(mask, structure=struct)
    m = mask.copy()
    if cfg.opening_radius_px > 0:
        m = ndimage.binary_opening(m, structure=disk(cfg.opening_radius_px))
    if cfg.closing_radius_px > 0:
        m = ndimage.binary_closing(m, structure=disk(cfg.closing_radius_px))
    m &= valid
    labels, n_morph = ndimage.label(m, structure=struct)
    mmu_pixels = max(1, int(math.ceil(cfg.min_mapping_unit_m2 / pixel_area_m2))) if cfg.min_mapping_unit_m2 > 0 else 1
    if n_morph:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= mmu_pixels
        keep[0] = False
        m = keep[labels]
        labels, n_final = ndimage.label(m, structure=struct)
    else:
        n_final = 0
    log.info("Postprocessing: components raw=%d, after morphology=%d, after MMU (%d px)=%d", n_raw, n_morph, mmu_pixels, n_final)
    return CleanResult(mask=m, labels=labels.astype(np.int32), components_raw=int(n_raw),
                       components_after_morphology=int(n_morph), components_after_mmu=int(n_final), mmu_pixels=mmu_pixels)
