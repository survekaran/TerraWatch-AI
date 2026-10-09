"""Module J: fully offline, interactive before/after change viewer.

Single self-contained HTML document with:
- Zero external requests (no CDNs, no external tiles, no fonts, no links)
- Pan and zoom (vanilla JS mouse drag & wheel)
- Before / after comparison slider
- Severity-tier colored polygon overlays with distinct Review styling
- Interactive toggles for severity tier, type tag, and polygons
- Optional change-score heatmap overlay (color-mapped PNG data URI)
- Rich polygon inspection popup showing attributes, tags, and tag_evidence
- Offline review decisions (accept/reject/flag) with JSON Blob download
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from shapely.geometry import MultiPolygon, Polygon

TIER_COLORS = {
    "Critical": "#ef4444",  # Red
    "Medium": "#f59e0b",    # Amber
    "Low": "#10b981",       # Emerald
    "Review": "#a855f7",    # Purple / Magenta
}


def _read_rgb(path: str | Path, max_dim: int) -> tuple[np.ndarray, int, int, tuple[int, int]]:
    with rasterio.open(path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        ow, oh = max(1, int(round(src.width * scale))), max(1, int(round(src.height * scale)))
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        data = src.read(bands, out_shape=(3, oh, ow), resampling=Resampling.bilinear).astype(np.float32)
        mask = src.read_masks(1, out_shape=(oh, ow), resampling=Resampling.nearest) > 0
        data[:, ~mask] = np.nan
        return data, src.width, src.height, (ow, oh)


def render_pair_previews(
    before: str | Path, after: str | Path, max_dim: int = 1400
) -> tuple[bytes, bytes, tuple[int, int], tuple[int, int]]:
    """PNG bytes for both dates using one shared contrast stretch."""
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


def render_heatmap_preview(
    scores_path: str | Path,
    preview_size: tuple[int, int],
) -> bytes:
    """Render a color-mapped transparent PNG data URI of change scores."""
    ow, oh = preview_size
    with rasterio.open(scores_path) as src:
        scores = src.read(1, out_shape=(oh, ow), resampling=Resampling.bilinear)
        mask = src.dataset_mask(out_shape=(oh, ow), resampling=Resampling.nearest) > 0

    valid = mask & (scores >= 0)
    # Color mapping: transparent blue to yellow to red
    rgba = np.zeros((oh, ow, 4), dtype=np.uint8)
    s = np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)

    # Red: starts ramping above 0.3
    rgba[..., 0] = (np.clip((s - 0.2) / 0.8, 0, 1) * 255).astype(np.uint8)
    # Green: peaks in mid scores
    rgba[..., 1] = (np.clip(1.0 - np.abs(s - 0.5) * 2.0, 0, 1) * 220).astype(np.uint8)
    # Blue: higher at lower scores
    rgba[..., 2] = (np.clip((0.6 - s) / 0.6, 0, 1) * 255).astype(np.uint8)
    # Alpha: transparent for very low scores (<0.1) and opaque for high scores
    rgba[..., 3] = np.where(valid & (s > 0.1), (np.clip(s, 0.2, 0.8) * 255).astype(np.uint8), 0)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _poly_path(poly: Polygon, inv, sx: float, sy: float) -> str:
    def ring(coords):
        pts = [inv @ (x, y) for x, y in coords]
        return "M" + "L".join(f"{c * sx:.1f},{r * sy:.1f}" for c, r in pts) + "Z"

    return " ".join([ring(poly.exterior.coords)] + [ring(i.coords) for i in poly.interiors])


def build_split_view_html(
    before_png: bytes,
    after_png: bytes,
    polygons: gpd.GeoDataFrame,
    transform,
    preview_size: tuple[int, int],
    full_size: tuple[int, int],
    height_px: int = 680,
    heatmap_png: bytes | None = None,
) -> str:
    """Build a fully self-contained offline HTML viewer."""
    ow, oh = preview_size
    sx, sy = ow / full_size[0], oh / full_size[1]
    inv = ~transform

    paths = []
    tiers_present = set()
    types_present = set()

    for _, row in polygons.iterrows():
        geoms = row.geometry.geoms if isinstance(row.geometry, MultiPolygon) else [row.geometry]
        d = " ".join(_poly_path(g, inv, sx, sy) for g in geoms)

        tier = str(row.get("severity_tier", "Low"))
        type_tag = str(row.get("type_tag", "unknown"))
        tiers_present.add(tier)
        types_present.add(type_tag)

        # Collect attributes for inspection
        info = {}
        for col in [
            "change_id", "severity_tier", "base_severity", "type_tag", "tags",
            "quality_flag", "review_reasons", "confidence", "change_magnitude",
            "area_m2", "centroid_lon", "centroid_lat", "tag_evidence",
        ]:
            if col in row and row[col] is not None:
                val = row[col]
                info[col] = val

        color = TIER_COLORS.get(tier, "#ef4444")
        is_review = tier == "Review"
        dash_attr = ' stroke-dasharray="4 2"' if is_review else ""

        paths.append((d, info, tier, type_tag, color, dash_attr))

    b64 = lambda b: base64.b64encode(b).decode()

    svg_elements = []
    for d, info, tier, type_tag, color, dash_attr in paths:
        raw_json = json.dumps(info, default=str)
        escaped_info = html.escape(raw_json)
        svg_elements.append(
            f'<path class="poly" data-tier="{tier}" data-type="{type_tag}" '
            f'data-info="{escaped_info}" d="{d}" fill="{color}" stroke="{color}"{dash_attr} fill-rule="evenodd"/>'
        )
    svg = "".join(svg_elements)

    heatmap_layer = ""
    heatmap_toggle = ""
    if heatmap_png:
        heatmap_layer = f'<img id="hm" src="data:image/png;base64,{b64(heatmap_png)}" alt="heatmap" style="display:none;opacity:0.65">'
        heatmap_toggle = '<label><input id="hm_t" type="checkbox"> Heatmap</label>'

    tier_checkboxes = "".join(
        f'<label style="color:{TIER_COLORS.get(t, "#fff")}"><input type="checkbox" class="tier-filter" value="{t}" checked> {t}</label> '
        for t in sorted(TIER_COLORS.keys()) if t in tiers_present or len(tiers_present) == 0
    )

    type_checkboxes = "".join(
        f'<label><input type="checkbox" class="type-filter" value="{t}" checked> {t}</label> '
        for t in sorted(types_present)
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TerraWatch Offline Viewer</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 12px/1.4 system-ui, -apple-system, sans-serif; background: #0a0f1d; color: #e2e8f0; }}
#toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 8px 12px; background: #11192e; border-bottom: 1px solid #1e293b; }}
#controls {{ display: flex; gap: 8px; align-items: center; }}
#viewport-container {{ position: relative; width: 100%; height: {height_px}px; overflow: hidden; background: #000; cursor: grab; }}
#viewport-container.panning {{ cursor: grabbing; }}
#viewport {{ position: absolute; transform-origin: 0 0; width: {ow}px; height: {oh}px; }}
#viewport img, #viewport svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
#viewport svg {{ pointer-events: auto; }}
#clip {{ position: absolute; inset: 0; overflow: hidden; pointer-events: none; }}
#div {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; pointer-events: none; box-shadow: 0 0 8px rgba(0,0,0,0.8); z-index: 10; }}
.poly {{ fill-opacity: 0.28; stroke-width: 2; cursor: pointer; vector-effect: non-scaling-stroke; transition: fill-opacity 0.15s, stroke 0.15s; }}
.poly:hover, .poly.sel {{ fill-opacity: 0.65; stroke: #ffffff !important; stroke-width: 3 !important; }}
.poly[data-tier="Review"] {{ stroke-width: 2.5; stroke-dasharray: 4 2; }}
.badge {{ position: absolute; top: 8px; padding: 3px 8px; border-radius: 4px; background: rgba(15,23,42,0.85); font-weight: 600; font-size: 11px; z-index: 20; border: 1px solid #334155; }}
#panel {{ display: grid; grid-template-columns: 1fr 320px; gap: 10px; padding: 10px; background: #0f172a; border-top: 1px solid #1e293b; }}
#info-box {{ min-height: 80px; padding: 8px; background: #1e293b; border-radius: 6px; font-family: monospace; font-size: 11px; max-height: 140px; overflow-y: auto; white-space: pre-wrap; }}
#review-actions {{ display: flex; flex-direction: column; gap: 6px; padding: 8px; background: #1e293b; border-radius: 6px; }}
.btn {{ padding: 4px 8px; background: #2563eb; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; }}
.btn:hover {{ background: #1d4ed8; }}
.btn-red {{ background: #dc2626; }} .btn-red:hover {{ background: #b91c1c; }}
.btn-green {{ background: #16a34a; }} .btn-green:hover {{ background: #15803d; }}
.btn-purple {{ background: #9333ea; }} .btn-purple:hover {{ background: #7e22ce; }}
input[type=range] {{ flex: 1; min-width: 80px; }}
label {{ cursor: pointer; font-size: 11px; margin-right: 4px; }}
.legend-box {{ display: flex; gap: 12px; align-items: center; background: #1e293b; padding: 4px 8px; border-radius: 4px; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.dot {{ width: 10px; height: 10px; border-radius: 2px; }}
</style>
</head>
<body>

<div id="toolbar">
  <div id="controls">
    <b>BEFORE</b>
    <input id="slider" type="range" min="0" max="100" value="50">
    <b>AFTER</b>
  </div>
  <div class="legend-box">
    <label><input id="poly_t" type="checkbox" checked> Polygons</label>
    {heatmap_toggle}
    <button id="reset_view" class="btn" style="background:#475569">Reset View</button>
  </div>
  <div class="legend-box">
    <span style="color:#94a3b8">Tiers:</span>
    {tier_checkboxes}
  </div>
  <div class="legend-box">
    <span style="color:#94a3b8">Types:</span>
    {type_checkboxes}
  </div>
</div>

<div id="viewport-container">
  <div id="viewport">
    <img src="data:image/png;base64,{b64(after_png)}" alt="after">
    <div id="clip">
      <img src="data:image/png;base64,{b64(before_png)}" alt="before">
    </div>
    {heatmap_layer}
    <div id="div"></div>
    <svg id="ov" viewBox="0 0 {ow} {oh}" preserveAspectRatio="none">{svg}</svg>
  </div>
  <span class="badge" style="left:8px">BEFORE (T1)</span>
  <span class="badge" style="right:8px">AFTER (T2)</span>
</div>

<div id="panel">
  <div>
    <b style="color:#94a3b8">INSPECTION & EVIDENCE</b>
    <div id="info-box">Click any change polygon on the map to inspect its evidence and attributes.</div>
  </div>
  <div id="review-actions">
    <b style="color:#94a3b8">HUMAN-IN-THE-LOOP REVIEW</b>
    <div style="display:flex;gap:4px">
      <button id="btn_accept" class="btn btn-green" disabled>Accept</button>
      <button id="btn_reject" class="btn btn-red" disabled>Reject</button>
      <button id="btn_flag" class="btn btn-purple" disabled>Flag Review</button>
    </div>
    <input id="review_note" type="text" placeholder="Optional review note / reason..." style="background:#0f172a;color:#fff;border:1px solid #334155;padding:4px;border-radius:4px;font-size:11px" disabled>
    <button id="btn_export_reviews" class="btn" style="background:#0284c7">Download review_decisions.json</button>
    <span id="review_status" style="font-size:10px;color:#94a3b8">0 decisions recorded</span>
  </div>
</div>

<script>
// --- State ---
let selectedId = null;
let selectedPolyData = null;
const reviewDecisions = [];

// --- Slider ---
const slider = document.getElementById('slider');
const clip = document.getElementById('clip');
const divider = document.getElementById('div');
function updateSlider() {{
  const val = slider.value;
  clip.style.clipPath = 'inset(0 ' + (100 - val) + '% 0 0)';
  divider.style.left = (val * {ow} / 100) + 'px';
}}
slider.oninput = updateSlider;
updateSlider();

// --- Pan & Zoom ---
const container = document.getElementById('viewport-container');
const viewport = document.getElementById('viewport');
let scale = 1.0, panX = 0, panY = 0;
let isPanning = false, startX = 0, startY = 0;

function updateTransform() {{
  viewport.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
}}

// Center initially
panX = (container.clientWidth - {ow}) / 2;
panY = (container.clientHeight - {oh}) / 2;
updateTransform();

container.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const rect = container.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const delta = e.deltaY < 0 ? 1.15 : 0.85;
  const newScale = Math.min(Math.max(0.4, scale * delta), 15.0);

  // Zoom towards cursor
  panX = mouseX - (mouseX - panX) * (newScale / scale);
  panY = mouseY - (mouseY - panY) * (newScale / scale);
  scale = newScale;
  updateTransform();
}}, {{ passive: false }});

container.addEventListener('mousedown', (e) => {{
  if (e.target.tagName.toLowerCase() === 'path') return;
  isPanning = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
  container.classList.add('panning');
}});

window.addEventListener('mousemove', (e) => {{
  if (!isPanning) return;
  panX = e.clientX - startX;
  panY = e.clientY - startY;
  updateTransform();
}});

window.addEventListener('mouseup', () => {{
  isPanning = false;
  container.classList.remove('panning');
}});

document.getElementById('reset_view').onclick = () => {{
  scale = 1.0;
  panX = (container.clientWidth - {ow}) / 2;
  panY = (container.clientHeight - {oh}) / 2;
  updateTransform();
}};

// --- Layer Visibility & Filters ---
const polyToggle = document.getElementById('poly_t');
polyToggle.onchange = (e) => {{
  document.getElementById('ov').style.display = e.target.checked ? 'block' : 'none';
}};

const hmToggle = document.getElementById('hm_t');
if (hmToggle) {{
  hmToggle.onchange = (e) => {{
    const hm = document.getElementById('hm');
    if (hm) hm.style.display = e.target.checked ? 'block' : 'none';
  }};
}}

function applyFilters() {{
  const activeTiers = new Set(Array.from(document.querySelectorAll('.tier-filter:checked')).map(el => el.value));
  const activeTypes = new Set(Array.from(document.querySelectorAll('.type-filter:checked')).map(el => el.value));
  document.querySelectorAll('.poly').forEach(p => {{
    const t = p.dataset.tier;
    const ty = p.dataset.type;
    const show = activeTiers.has(t) && activeTypes.has(ty);
    p.style.display = show ? 'block' : 'none';
  }});
}}
document.querySelectorAll('.tier-filter, .type-filter').forEach(cb => cb.onchange = applyFilters);

// --- Inspection ---
const infoBox = document.getElementById('info-box');
const btnAccept = document.getElementById('btn_accept');
const btnReject = document.getElementById('btn_reject');
const btnFlag = document.getElementById('btn_flag');
const noteInput = document.getElementById('review_note');

document.querySelectorAll('.poly').forEach(p => {{
  p.onclick = (e) => {{
    e.stopPropagation();
    document.querySelectorAll('.poly').forEach(x => x.classList.remove('sel'));
    p.classList.add('sel');
    const info = JSON.parse(p.dataset.info);
    selectedId = info.change_id;
    selectedPolyData = info;

    let txt = `ID: ${{info.change_id}}  |  Tier: ${{info.severity_tier}} (Base: ${{info.base_severity || info.severity_tier}})\n`;
    txt += `Tag: ${{info.type_tag}}  |  All Tags: ${{info.tags || 'none'}}\n`;
    txt += `Area: ${{info.area_m2}} m²  |  Confidence: ${{info.confidence}}  |  Magnitude: ${{info.change_magnitude}}\n`;
    if (info.quality_flag && info.quality_flag !== 'nominal') txt += `Quality Flag: ${{info.quality_flag}}\n`;
    if (info.review_reasons) txt += `Review Triggers: ${{info.review_reasons}}\n`;
    if (info.tag_evidence) {{
      try {{
        const ev = typeof info.tag_evidence === 'string' ? JSON.parse(info.tag_evidence) : info.tag_evidence;
        txt += `\n--- Evidence Metrics ---\n` + JSON.stringify(ev, null, 2);
      }} catch (err) {{
        txt += `\nEvidence: ${{info.tag_evidence}}`;
      }}
    }}
    infoBox.textContent = txt;

    btnAccept.disabled = false;
    btnReject.disabled = false;
    btnFlag.disabled = false;
    noteInput.disabled = false;
  }};
}});

// --- Review Decision Recording ---
function recordDecision(action) {{
  if (!selectedId) return;
  const note = noteInput.value.trim();
  const decision = {{
    change_id: selectedId,
    decision: action,
    note: note,
    timestamp: new Date().toISOString(),
    attributes: selectedPolyData
  }};
  // Replace if exists, else append
  const idx = reviewDecisions.findIndex(d => d.change_id === selectedId);
  if (idx >= 0) reviewDecisions[idx] = decision;
  else reviewDecisions.push(decision);

  document.getElementById('review_status').textContent = `${{reviewDecisions.length}} decision(s) recorded`;
  noteInput.value = '';
}}

btnAccept.onclick = () => recordDecision('accepted');
btnReject.onclick = () => recordDecision('rejected');
btnFlag.onclick = () => recordDecision('flagged_review');

// --- Export Review Decisions (Offline Blob Download) ---
document.getElementById('btn_export_reviews').onclick = () => {{
  const blob = new Blob([JSON.stringify(reviewDecisions, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'review_decisions.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}};
</script>
</body>
</html>
"""
