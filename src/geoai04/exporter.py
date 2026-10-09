"""Module I: raster and vector exports.

GeoPackage keeps the scene CRS. GeoJSON is written in EPSG:4326 (the RFC 7946 convention).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

log = logging.getLogger(__name__)
NODATA_SCORE = -9999.0
NODATA_MASK = 255


def write_rasters(run_dir: Path, scores: np.ndarray, mask: np.ndarray, valid: np.ndarray, crs, transform) -> dict[str, str]:
    """Write change_scores.tif (float32) and change_mask.tif (uint8: 1 changed, 0 none, 255 nodata)."""
    h, w = scores.shape
    base = dict(driver="GTiff", height=h, width=w, count=1, crs=crs, transform=transform, compress="deflate")
    s = np.where(valid, scores, NODATA_SCORE).astype(np.float32)
    m = np.where(valid, mask.astype(np.uint8), NODATA_MASK).astype(np.uint8)
    with rasterio.open(run_dir / "change_scores.tif", "w", dtype="float32", nodata=NODATA_SCORE, **base) as dst:
        dst.write(s, 1)
    with rasterio.open(run_dir / "change_mask.tif", "w", dtype="uint8", nodata=NODATA_MASK, **base) as dst:
        dst.write(m, 1)
    return {"change_scores": "change_scores.tif", "change_mask": "change_mask.tif"}


def write_vectors(run_dir: Path, gdf: gpd.GeoDataFrame) -> dict[str, str]:
    """Write change_polygons.gpkg (scene CRS), change_polygons.geojson (EPSG:4326) and change_summary.csv."""
    gpkg, gj, csv = run_dir / "change_polygons.gpkg", run_dir / "change_polygons.geojson", run_dir / "change_summary.csv"
    gdf = gdf.drop(columns=["label_id"], errors="ignore")
    gdf.to_file(gpkg, layer="change_polygons", driver="GPKG")
    if gdf.empty:
        gj.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    else:
        gdf.to_crs(4326).to_file(gj, driver="GeoJSON")
    pd_df = gdf.drop(columns=["geometry"])
    pd_df.to_csv(csv, index=False)
    log.info("Wrote %d polygon(s) to GeoPackage, GeoJSON and CSV", len(gdf))
    return {"change_polygons_gpkg": gpkg.name, "change_polygons_geojson": gj.name, "change_summary_csv": csv.name}
