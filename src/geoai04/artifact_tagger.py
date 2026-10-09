"""Artifact tagger and polygon evidence extractor.

Evaluates explainable per-polygon geometric, spectral, and contextual evidence,
recording metrics in a tag_evidence JSON string and assigning:
  - type_tag: single primary tag
  - tags: semicolon-separated list of all tags that apply

TAG VOCABULARY (EXACTLY THESE):
  building_like, built_up_patch, vegetation_conversion, vegetation_fluctuation,
  unknown, misregistration_suspect, low_quality, model_disagreement

TAG PRECEDENCE ORDER (for primary type_tag):
  1. low_quality: NoData fraction inside polygon exceeds threshold or invalid inputs.
  2. misregistration_suspect: thin, elongated shape near strong edges and/or alignment status is warn/fail.
  3. model_disagreement: evaluated ONLY when a secondary detector mask (other_mask) is provided; IoU < threshold.
  4. vegetation_conversion: vegetation evidence dropped AND (built-up rose OR brightness rose OR compact shape).
  5. vegetation_fluctuation: diffuse vegetation drop/rise with no built-up or brightness rise.
  6. building_like: compact shape within building area range (high-res routes only).
  7. built_up_patch: coarse multispectral routes with built-up / brightness increase.
  8. unknown: insufficient evidence to confidently assign a specific tag.

CRITICAL POLICY:
  - Tags NEVER delete or discard polygons. There is no NDVI veto.
  - Cloud and shadow detection is NOT claimed. Provenance explicitly records
    cloud_shadow_screening = 'not_performed'.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from scipy import ndimage
from skimage.filters import sobel

from .alignment import AlignmentResult
from .baseline_detector import normalise
from .config import TaggingConfig
from .resolution_router import RoutingDecision
from .spectral import compute_spectral_index

log = logging.getLogger(__name__)

TAG_VOCABULARY = {
    "building_like",
    "built_up_patch",
    "vegetation_conversion",
    "vegetation_fluctuation",
    "unknown",
    "misregistration_suspect",
    "low_quality",
    "model_disagreement",
}


def _geom_metrics(geom) -> dict[str, float]:
    """Calculate compactness, MRR width/length, extent, and elongation in CRS units."""
    area = float(geom.area)
    perimeter = float(geom.length)
    compactness = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
    compactness = float(np.clip(compactness, 0.0, 1.0))

    mrr = geom.minimum_rotated_rectangle
    mrr_area = float(mrr.area)
    extent = area / mrr_area if mrr_area > 0 else 0.0
    extent = float(np.clip(extent, 0.0, 1.0))

    # Calculate width and length of MRR
    coords = list(mrr.exterior.coords)
    if len(coords) >= 4:
        d1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        d2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        width = min(d1, d2)
        length = max(d1, d2)
    else:
        width = math.sqrt(area)
        length = width

    elongation = length / max(width, 1e-3)
    return {
        "compactness": round(compactness, 4),
        "extent": round(extent, 4),
        "min_rot_rect_width_m": round(width, 2),
        "min_rot_rect_length_m": round(length, 2),
        "elongation": round(elongation, 2),
    }


def tag_polygons(
    polygons: gpd.GeoDataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    before_path: str | Path,
    after_path: str | Path,
    cfg: TaggingConfig,
    alignment_result: AlignmentResult | None = None,
    routing_decision: RoutingDecision | None = None,
    delta_rasters: dict[str, np.ndarray] | None = None,
    other_mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Compute per-polygon explainable evidence, assign primary type_tag and all tags."""
    if polygons.empty:
        out = polygons.copy()
        out["type_tag"] = []
        out["tags"] = []
        out["tag_evidence"] = []
        return out

    out = polygons.copy()
    label_ids = out["label_id"].to_numpy()

    # Load raster bands for evidence extraction
    with rasterio.open(before_path) as sb, rasterio.open(after_path) as sa:
        bands_count = min(sb.count, sa.count)
        rb = sb.read(list(range(1, min(3, bands_count) + 1)))
        ra = sa.read(list(range(1, min(3, bands_count) + 1)))
        mask_b = sb.dataset_mask() == 0
        mask_a = sa.dataset_mask() == 0

    invalid_pixels = mask_b | mask_a

    # Luminance and chromatic change
    if rb.shape[0] >= 3:
        lum_b = (rb[0] * 0.2989 + rb[1] * 0.5870 + rb[2] * 0.1140) / 255.0
        lum_a = (ra[0] * 0.2989 + ra[1] * 0.5870 + ra[2] * 0.1140) / 255.0
        diff_rgb = np.sqrt(np.mean(((ra.astype(float) - rb.astype(float)) / 255.0) ** 2, axis=0))
    else:
        lum_b = rb.mean(axis=0) / 255.0
        lum_a = ra.mean(axis=0) / 255.0
        diff_rgb = np.abs(lum_a - lum_b)

    delta_lum = lum_a - lum_b
    edge_map = sobel(lum_b) > cfg.misregistration_edge_threshold

    # Region-aggregated evidence
    mean_scores = ndimage.mean(scores, labels, label_ids)
    mean_delta_lum = ndimage.mean(delta_lum, labels, label_ids)
    mean_chromatic = ndimage.mean(diff_rgb, labels, label_ids)
    nodata_fractions = ndimage.mean(invalid_pixels.astype(np.float32), labels, label_ids)
    edge_fractions = ndimage.mean(edge_map.astype(np.float32), labels, label_ids)

    # Vegetation and built-up index deltas
    delta_veg_values = None
    veg_index_name = "unavailable"
    if delta_rasters and "ndvi" in delta_rasters:
        delta_veg_values = ndimage.mean(delta_rasters["ndvi"], labels, label_ids)
        veg_index_name = "ndvi"
    elif delta_rasters and "vegetation_proxy_exg" in delta_rasters:
        delta_veg_values = ndimage.mean(delta_rasters["vegetation_proxy_exg"], labels, label_ids)
        veg_index_name = "vegetation_proxy_exg"
    elif rb.shape[0] >= 3:
        # Fallback to computing ExG on the fly
        exg_b = (2.0 * rb[1] - rb[0] - rb[2]) / 510.0
        exg_a = (2.0 * ra[1] - ra[0] - ra[2]) / 510.0
        delta_exg = exg_a - exg_b
        delta_veg_values = ndimage.mean(delta_exg, labels, label_ids)
        veg_index_name = "vegetation_proxy_exg"

    delta_built_values = None
    if delta_rasters and "ndbi" in delta_rasters:
        delta_built_values = ndimage.mean(delta_rasters["ndbi"], labels, label_ids)

    policy = routing_decision.policy if routing_decision else "high_res_rgb"
    align_status = alignment_result.status if alignment_result else "pass"
    align_shift = alignment_result.magnitude_px if alignment_result else 0.0

    type_tags = []
    all_tags_list = []
    evidences = []

    for i, row in out.iterrows():
        geom = row.geometry
        area_m2 = float(row.get("area_m2", geom.area))
        g_metrics = _geom_metrics(geom)

        score_val = float(mean_scores[i])
        lum_val = float(mean_delta_lum[i])
        chroma_val = float(mean_chromatic[i])
        nodata_frac = float(nodata_fractions[i])
        edge_frac = float(edge_fractions[i])
        delta_veg = float(delta_veg_values[i]) if delta_veg_values is not None else None
        delta_built = float(delta_built_values[i]) if delta_built_values is not None else None

        # Model disagreement IoU if other_mask is supplied
        iou_val = None
        if other_mask is not None:
            region_mask = labels == row["label_id"]
            intersection = np.logical_and(region_mask, other_mask > 0).sum()
            union = np.logical_or(region_mask, other_mask > 0).sum()
            iou_val = float(intersection / union) if union > 0 else 0.0

        evidence = {
            **g_metrics,
            "mean_change_score": round(score_val, 4),
            "brightness_change": round(lum_val, 4),
            "chromatic_change": round(chroma_val, 4),
            "vegetation_index": veg_index_name,
            "delta_vegetation_index": round(delta_veg, 4) if delta_veg is not None else "unavailable",
            "delta_built_up_index": round(delta_built, 4) if delta_built is not None else "unavailable",
            "nodata_fraction": round(nodata_frac, 4),
            "adjacent_edge_fraction": round(edge_frac, 4),
            "other_mask_iou": round(iou_val, 4) if iou_val is not None else "not_evaluated",
        }
        evidences.append(json.dumps(evidence))

        tags = set()

        # Check conditions for each tag
        # 1. low_quality
        if nodata_frac >= cfg.low_quality_nodata_fraction:
            tags.add("low_quality")

        # 2. misregistration_suspect
        width_m = g_metrics["min_rot_rect_width_m"]
        elong = g_metrics["elongation"]
        is_thin_elongated_edge = (
            elong >= cfg.misregistration_min_elongation
            and edge_frac >= cfg.misregistration_edge_threshold
            and width_m <= (align_shift + 1.0) * cfg.misregistration_width_factor_px
        )
        if align_status in {"warn", "fail"} or is_thin_elongated_edge:
            tags.add("misregistration_suspect")

        # 3. model_disagreement
        if iou_val is not None and iou_val < cfg.disagreement_iou_threshold:
            tags.add("model_disagreement")

        veg_dropped = delta_veg is not None and delta_veg < -cfg.veg_drop_threshold
        built_rose = delta_built is not None and delta_built > cfg.built_rise_threshold
        bright_rose = lum_val > cfg.brightness_rise_threshold
        compact_shape = g_metrics["compactness"] >= cfg.min_compactness_building

        # 4. vegetation_fluctuation (diffuse natural variation: large area / low compactness & no built/brightness rise)
        veg_changed = delta_veg is not None and abs(delta_veg) >= cfg.veg_drop_threshold
        diffuse_shape = (
            g_metrics["compactness"] < cfg.max_compactness_diffuse
            or area_m2 >= cfg.fluctuation_min_area_m2
        )
        is_fluctuation = veg_changed and diffuse_shape and not (built_rose or bright_rose)
        if is_fluctuation:
            tags.add("vegetation_fluctuation")

        # 5. vegetation_conversion (real encroachment: vegetation decreased AND (built/brightness increased or compact non-diffuse building))
        if veg_dropped and (built_rose or bright_rose or (compact_shape and not is_fluctuation)):
            tags.add("vegetation_conversion")

        # 6. building_like (high-res routes)
        if (
            policy == "high_res_rgb"
            and compact_shape
            and cfg.min_building_area_m2 <= area_m2 <= cfg.max_building_area_m2
        ):
            tags.add("building_like")

        # 7. built_up_patch (coarse routes or non-building scale built change)
        if (
            policy == "multispectral_coarse"
            and (built_rose or bright_rose)
        ):
            tags.add("built_up_patch")

        # If no specific tags triggered, assign unknown
        if not tags:
            tags.add("unknown")

        # Determine single primary type_tag by strict precedence order:
        # Edge artifacts get misregistration_suspect, while real conversions retain semantic tag
        if "low_quality" in tags and nodata_frac >= cfg.low_quality_nodata_fraction:
            primary = "low_quality"
        elif is_thin_elongated_edge:
            primary = "misregistration_suspect"
        elif "model_disagreement" in tags:
            primary = "model_disagreement"
        elif "vegetation_conversion" in tags:
            primary = "vegetation_conversion"
        elif "vegetation_fluctuation" in tags:
            primary = "vegetation_fluctuation"
        elif "building_like" in tags:
            primary = "building_like"
        elif "built_up_patch" in tags:
            primary = "built_up_patch"
        elif "misregistration_suspect" in tags:
            primary = "misregistration_suspect"
        else:
            primary = "unknown"

        type_tags.append(primary)
        all_tags_list.append(";".join(sorted(tags)))

    out["type_tag"] = type_tags
    out["tags"] = all_tags_list
    out["tag_evidence"] = evidences
    return out
