"""Module G (part 2): raster regions to repaired, then conservatively simplified, polygons."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

log = logging.getLogger(__name__)


def _polygonal(geom):
    """Keep only polygonal parts of a (possibly repaired) geometry."""
    if geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
    return unary_union(parts) if parts else None


def polygonize(labels: np.ndarray, transform, crs, connectivity: int = 8) -> gpd.GeoDataFrame:
    """One repaired polygon (with holes) per labelled region, in the raster CRS.

    Uses rasterio.features.shapes, so the affine transform and CRS are preserved exactly.
    """
    recs: dict[int, list] = {}
    for geom, value in features.shapes(labels.astype(np.int32), mask=labels > 0, transform=transform, connectivity=connectivity):
        recs.setdefault(int(value), []).append(shape(geom))
    rows = []
    for label_id in sorted(recs):
        g = unary_union(recs[label_id])
        if not g.is_valid:
            g = make_valid(g)
        g = _polygonal(g)
        if g is not None:
            rows.append({"label_id": label_id, "geometry": g})
    gdf = gpd.GeoDataFrame(rows, columns=["label_id", "geometry"], geometry="geometry", crs=crs)
    log.info("Polygonized %d region(s)", len(gdf))
    return gdf


def simplify(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """Conservative topology-preserving simplification (tolerance in CRS units)."""
    if tolerance <= 0 or gdf.empty:
        return gdf
    out = gdf.copy()
    simp = out.geometry.simplify(tolerance, preserve_topology=True)
    out["geometry"] = [g if g.is_valid else (_polygonal(make_valid(g)) or o) for g, o in zip(simp, out.geometry)]
    return out
