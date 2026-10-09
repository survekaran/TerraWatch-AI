"""Module H: physical and detector attributes for each change polygon.

Area, projected centroid and lon/lat centroid are computed on the *unsimplified* geometry
in a metric CRS, then geometries stay in the scene CRS (reprojected only at export).

Three concepts are kept separate and none is a validated physical truth:
  * area_m2          physical size of the mapped region
  * change_magnitude mean change score inside the region (0..1)
  * confidence       mean margin-based detector support over region pixels:
                     mean(clip((score - threshold) / (1 - threshold), 0, 1)).
                     This is an UNCALIBRATED detector-support measure, NOT a probability.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .geometry_utils import as_pyproj, select_metric_crs

log = logging.getLogger(__name__)

ATTRIBUTE_COLUMNS = [
    "change_id", "area_m2", "centroid_lon", "centroid_lat", "centroid_x", "centroid_y",
    "projected_crs", "change_magnitude", "confidence", "pixel_count", "source_pair_id", "detector_used",
]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_id(before_path: str | Path, after_path: str | Path) -> str:
    """Short deterministic ID derived from the two input files' contents."""
    return hashlib.sha256((file_sha256(before_path) + file_sha256(after_path)).encode()).hexdigest()[:12]


def build_attributes(raw: gpd.GeoDataFrame, labels: np.ndarray, scores: np.ndarray, threshold: float,
                     transform, width: int, height: int, source_pair_id: str, detector_used: str) -> gpd.GeoDataFrame:
    """Attach area, centroids, magnitude, confidence and provenance fields to polygons."""
    if raw.empty:
        return gpd.GeoDataFrame(columns=ATTRIBUTE_COLUMNS + ["geometry"], geometry="geometry", crs=raw.crs)
    crs = as_pyproj(raw.crs)
    metric = select_metric_crs(crs, transform, width, height)
    metric_geoms = raw.to_crs(metric).geometry
    cent_m = metric_geoms.centroid
    lonlat = Transformer.from_crs(metric, 4326, always_xy=True).transform(cent_m.x.values, cent_m.y.values)

    ids = raw["label_id"].to_numpy()
    magnitude = ndimage.mean(scores, labels, ids)

    # Informative margin-based support: clip((score - threshold) / (1 - threshold), 0, 1)
    denom = max(1.0 - threshold, 1e-6)
    margin = np.clip((scores - threshold) / denom, 0.0, 1.0)
    support = ndimage.mean(margin, labels, ids)
    pixels = ndimage.sum(np.ones_like(scores), labels, ids)

    out = raw.copy()
    out["area_m2"] = np.round(metric_geoms.area.values, 3)
    out["centroid_x"] = np.round(cent_m.x.values, 3)
    out["centroid_y"] = np.round(cent_m.y.values, 3)
    out["centroid_lon"] = np.round(np.asarray(lonlat[0]), 7)
    out["centroid_lat"] = np.round(np.asarray(lonlat[1]), 7)
    out["projected_crs"] = metric.to_string()
    out["change_magnitude"] = np.round(np.asarray(magnitude, dtype=float), 4)
    out["confidence"] = np.round(np.asarray(support, dtype=float), 4)
    out["pixel_count"] = np.asarray(pixels, dtype=int)
    out["source_pair_id"] = source_pair_id
    out["detector_used"] = detector_used
    out = out.sort_values(["area_m2", "label_id"], ascending=[False, True]).reset_index(drop=True)
    out["change_id"] = [f"CHG-{i + 1:04d}" for i in range(len(out))]
    return out[ATTRIBUTE_COLUMNS + ["label_id", "geometry"]]


def filter_by_area(gdf: gpd.GeoDataFrame, min_area_m2: float) -> gpd.GeoDataFrame:
    """Drop polygons smaller than min_area_m2 (square metres, measured in the metric CRS)."""
    if gdf.empty or min_area_m2 <= 0:
        return gdf
    kept = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
    log.info("Area filter (>= %.1f m2): %d -> %d polygons", min_area_m2, len(gdf), len(kept))
    return kept
