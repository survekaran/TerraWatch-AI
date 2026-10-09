"""CRS, ground-sample-distance and pixel-area helpers.

All physical measurements (area, GSD) are made in a metric CRS. If the scene CRS is a
projected CRS in metres (and not Web Mercator, which distorts area) it is used as-is;
otherwise a WGS84 UTM zone chosen from the scene centre is used. The choice is returned
so it can be recorded in the provenance file.
"""
from __future__ import annotations

import math

from pyproj import CRS, Transformer
from rasterio.transform import array_bounds

_DISTORTING_EPSG = {3857, 900913, 102100, 102113}


def as_pyproj(crs) -> CRS:
    """Convert a rasterio/pyproj/str CRS into a pyproj CRS."""
    if isinstance(crs, CRS):
        return crs
    return CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)


def linear_unit_factor(crs) -> float:
    """Metres per CRS linear unit for a projected CRS."""
    crs = as_pyproj(crs)
    return float(crs.axis_info[0].unit_conversion_factor) if crs.axis_info else 1.0


def centre_lonlat(crs, transform, width: int, height: int) -> tuple[float, float]:
    crs = as_pyproj(crs)
    x, y = transform @ (width / 2.0, height / 2.0)
    if crs.is_geographic:
        return float(x), float(y)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    return float(lon), float(lat)


def select_metric_crs(crs, transform, width: int, height: int) -> CRS:
    """Pick the CRS used for area, length and projected-centroid calculations."""
    crs = as_pyproj(crs)
    if crs.is_projected and abs(linear_unit_factor(crs) - 1.0) < 1e-9:
        epsg = crs.to_epsg()
        if epsg not in _DISTORTING_EPSG:
            return crs
    lon, lat = centre_lonlat(crs, transform, width, height)
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def _shoelace(pts: list[tuple[float, float]]) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _to_metric(points, crs: CRS, metric: CRS):
    if metric == crs:
        return [(float(x), float(y)) for x, y in points]
    tr = Transformer.from_crs(crs, metric, always_xy=True)
    return [tr.transform(x, y) for x, y in points]


def pixel_area_m2(crs, transform, width: int, height: int) -> float:
    """Area in square metres of the pixel at the scene centre."""
    crs = as_pyproj(crs)
    metric = select_metric_crs(crs, transform, width, height)
    c, r = width // 2, height // 2
    corners = [transform @ (c, r), transform @ (c + 1, r), transform @ (c + 1, r + 1), transform @ (c, r + 1)]
    return _shoelace(_to_metric(corners, crs, metric))


def estimate_gsd_m(crs, transform, width: int, height: int) -> tuple[float, float, str]:
    """Ground sample distance (x, y) in metres at the scene centre, plus how it was derived."""
    crs = as_pyproj(crs)
    metric = select_metric_crs(crs, transform, width, height)
    c, r = width // 2, height // 2
    p0, px, py = transform @ (c, r), transform @ (c + 1, r), transform @ (c, r + 1)
    m0, mx, my = _to_metric([p0, px, py], crs, metric)
    gx = math.hypot(mx[0] - m0[0], mx[1] - m0[1])
    gy = math.hypot(my[0] - m0[0], my[1] - m0[1])
    if metric == crs:
        method = f"affine transform in projected metric CRS ({metric.to_string()})"
    else:
        method = f"affine transform converted at scene centre via {metric.to_string()} (estimate)"
    return gx, gy, method


def scene_bounds(transform, width: int, height: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) for a raster grid."""
    return array_bounds(height, width, transform)
