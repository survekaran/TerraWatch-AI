"""Module J (helper): fully offline before/after viewer.

Folium/Leaflet pages load Leaflet from a CDN, which breaks the local-only requirement,
so the viewer is a single self-contained HTML document: PNG data URIs, an SVG polygon
overlay and a comparison slider. No external resources are referenced.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from shapely.geometry import MultiPolygon, Polygon

TIER_COLOR = "#ef4444"


def _read_rgb(path: str | Path, max_dim: int) -> tuple[np.ndarray, int, int, tuple[int, int]]:
    with rasterio.open(path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        ow, oh = max(1, int(round(src.width * scale))), max(1, int(round(src.height * scale)))
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        data = src.read(bands, out_shape=(3, oh, ow), resampling=Resampling.bilinear).astype(np.float32)
        mask = src.read_masks(1, out_shape=(oh, ow), resampling=Resampling.nearest) > 0
        data[:, ~mask] = np.nan
        return data, src.width, src.height, (ow, oh)


def render_pair_previews(before: str | Path, after: str | Path, max_dim: int = 1400) -> tuple[bytes, bytes, tuple[int, int], tuple[int, int]]:
    """PNG bytes for both dates using one shared contrast stretch. Returns (before, after, preview_size, full_size)."""
    b, w, h, size = _read_rgb(before, max_dim)
    a, _, _, _ = _read_rgb(after, max_dim)
    both = np.concatenate([b.reshape(3, -1), a.reshape(3, -1)], axis=1)
    finite = both[:, np.isfinite(both).all(axis=0)]
    lo, hi = (np.percentile(finite, 2), np.percentile(finite, 98)) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0

    def to_png(x: np.ndarray) -> bytes:
        y = np.nan_to_num((x - lo) / (hi - lo), nan=0.0)
        arr = (np.clip(y, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    return to_png(b), to_png(a), size, (w, h)


def _poly_path(poly: Polygon, inv, sx: float, sy: float) -> str:
    def ring(coords):
        pts = [inv @ (x, y) for x, y in coords]
        return "M" + "L".join(f"{c * sx:.1f},{r * sy:.1f}" for c, r in pts) + "Z"
    return " ".join([ring(poly.exterior.coords)] + [ring(i.coords) for i in poly.interiors])


def build_split_view_html(before_png: bytes, after_png: bytes, polygons: gpd.GeoDataFrame, transform,
                          preview_size: tuple[int, int], full_size: tuple[int, int], height_px: int = 620) -> str:
    """Self-contained before/after viewer with a slider, polygon overlay and click-to-inspect."""
    ow, oh = preview_size
    sx, sy = ow / full_size[0], oh / full_size[1]
    inv = ~transform
    paths = []
    for _, row in polygons.iterrows():
        geoms = row.geometry.geoms if isinstance(row.geometry, MultiPolygon) else [row.geometry]
        d = " ".join(_poly_path(g, inv, sx, sy) for g in geoms)
        info = {k: row[k] for k in ("change_id", "area_m2", "change_magnitude", "confidence", "centroid_lon", "centroid_lat") if k in row}
        paths.append((d, info))
    b64 = lambda b: base64.b64encode(b).decode()
    svg = "".join(
        f'<path class="poly" data-info="{html.escape(json.dumps(info, default=float))}" d="{d}" fill-rule="evenodd"/>'
        for d, info in paths)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font:13px system-ui,sans-serif;background:#0b1220;color:#e6edf7}}
#bar{{display:flex;gap:14px;align-items:center;padding:6px 10px}}
#wrap{{position:relative;width:100%;max-height:{height_px}px;aspect-ratio:{ow}/{oh};margin:0 auto;overflow:hidden;background:#000}}
#wrap img,#wrap svg{{position:absolute;inset:0;width:100%;height:100%}}
#clip{{position:absolute;inset:0;overflow:hidden}}
#div{{position:absolute;top:0;bottom:0;width:2px;background:#fff;pointer-events:none;box-shadow:0 0 6px #000}}
.poly{{fill:{TIER_COLOR};fill-opacity:.28;stroke:{TIER_COLOR};stroke-width:2;cursor:pointer;vector-effect:non-scaling-stroke}}
.poly:hover,.poly.sel{{fill-opacity:.55;stroke:#fff}}
.tag{{position:absolute;top:8px;padding:2px 9px;border-radius:99px;background:rgba(8,14,26,.8);font-weight:700;z-index:3}}
#info{{padding:6px 10px;min-height:20px;color:#9fb3cf}}
input[type=range]{{flex:1}}
</style></head><body>
<div id="bar"><b>BEFORE</b><input id="s" type="range" min="0" max="100" value="50"><b>AFTER</b>
<label><input id="t" type="checkbox" checked> change polygons</label></div>
<div id="wrap">
<img src="data:image/png;base64,{b64(after_png)}" alt="after">
<div id="clip"><img src="data:image/png;base64,{b64(before_png)}" alt="before"></div>
<svg id="ov" viewBox="0 0 {ow} {oh}" preserveAspectRatio="none">{svg}</svg>
<div id="div"></div><span class="tag" style="left:8px">BEFORE</span><span class="tag" style="right:8px">AFTER</span></div>
<div id="info">Click a polygon to inspect it. Drag the slider to compare dates.</div>
<script>
const s=document.getElementById('s'),c=document.getElementById('clip'),d=document.getElementById('div');
function u(){{c.style.clipPath='inset(0 '+(100-s.value)+'% 0 0)';d.style.left=s.value+'%'}}s.oninput=u;u();
document.getElementById('t').onchange=e=>document.getElementById('ov').style.display=e.target.checked?'block':'none';
document.querySelectorAll('.poly').forEach(p=>p.onclick=()=>{{document.querySelectorAll('.poly').forEach(x=>x.classList.remove('sel'));p.classList.add('sel');
const i=JSON.parse(p.dataset.info);document.getElementById('info').textContent=Object.entries(i).map(([k,v])=>k+': '+v).join('  |  ')}});
</script></body></html>"""
