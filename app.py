"""Streamlit interface for the GEOAI 04 Phase 1 prototype. Run: streamlit run app.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from geoai04.config import load_config  # noqa: E402
from geoai04.pipeline import PHASE1_NOT_IMPLEMENTED, run_pipeline  # noqa: E402
from geoai04.synthetic import generate_synthetic_pair  # noqa: E402
from geoai04.validator import InputValidationError, validate_pair  # noqa: E402
from geoai04.viewer import build_split_view_html, render_pair_previews  # noqa: E402

st.set_page_config(page_title="GEOAI 04 Change Detection", layout="wide")
st.title("GEOAI 04: bi-temporal change detection (Phase 1)")
st.caption("Local prototype. Baseline image-difference detector only; no AI model is used in this phase.")

defaults = load_config()
with st.sidebar:
    st.header("Detection")
    method = st.selectbox("Threshold method", ["otsu", "fixed", "percentile"], index=["otsu", "fixed", "percentile"].index(defaults.detection.threshold_method))
    fixed = st.slider("Fixed threshold", 0.01, 0.9, float(defaults.detection.fixed_threshold), 0.01, disabled=method != "fixed")
    pct = st.slider("Percentile", 50.0, 99.9, float(defaults.detection.percentile), 0.1, disabled=method != "percentile")
    floor = st.slider("Otsu floor", 0.0, 0.5, float(defaults.detection.otsu_floor), 0.01, disabled=method != "otsu")
    sigma = st.slider("Smoothing sigma (px)", 0.0, 5.0, float(defaults.detection.smoothing_sigma_px), 0.5)
    st.header("Filtering")
    opening = st.number_input("Opening radius (px)", 0, 10, int(defaults.postprocessing.opening_radius_px))
    closing = st.number_input("Closing radius (px)", 0, 20, int(defaults.postprocessing.closing_radius_px))
    mmu = st.number_input("Minimum mapping unit (m²)", 0.0, 1e6, float(defaults.postprocessing.min_mapping_unit_m2), 5.0)
    simp = st.number_input("Simplify tolerance (px)", 0.0, 5.0, float(defaults.polygons.simplify_tolerance_px), 0.1)
    st.header("Status of modules")
    for line in PHASE1_NOT_IMPLEMENTED:
        st.caption("• " + line)

cfg = load_config(overrides={
    "detection": {"threshold_method": method, "fixed_threshold": fixed, "percentile": pct, "otsu_floor": floor, "smoothing_sigma_px": sigma},
    "postprocessing": {"opening_radius_px": int(opening), "closing_radius_px": int(closing), "min_mapping_unit_m2": float(mmu)},
    "polygons": {"simplify_tolerance_px": float(simp)},
})

st.subheader("1. Inputs")
workdir = Path(st.session_state.setdefault("workdir", tempfile.mkdtemp(prefix="geoai04_")))
c1, c2 = st.columns(2)
fb = c1.file_uploader("BEFORE GeoTIFF (T1)", type=["tif", "tiff"], key="fb")
fa = c2.file_uploader("AFTER GeoTIFF (T2)", type=["tif", "tiff"], key="fa")
with st.expander("Or use local files / the synthetic test pair"):
    pb = st.text_input("Path to BEFORE GeoTIFF")
    pa = st.text_input("Path to AFTER GeoTIFF")
    if st.button("Generate SYNTHETIC test pair"):
        generate_synthetic_pair(ROOT / "data" / "sample")
        st.session_state["paths"] = (str(ROOT / "data/sample/before.tif"), str(ROOT / "data/sample/after.tif"), True)
    st.caption("The synthetic pair is generated test data with arbitrary coordinates. It is not satellite imagery.")


def _save(up, name):
    p = workdir / name
    p.write_bytes(up.getbuffer())
    return str(p)


before = after = None
synthetic = False
if fb and fa:
    before, after = _save(fb, "before_" + fb.name), _save(fa, "after_" + fa.name)
elif pb and pa:
    before, after = pb, pa
elif "paths" in st.session_state:
    before, after, synthetic = st.session_state["paths"]

if before and after:
    report = validate_pair(before, after, cfg.validation)
    st.subheader("2. Validation")
    (st.success if report.ok else st.error)("Inputs are valid." if report.ok else "Inputs rejected: " + " | ".join(report.errors))
    for w in report.warnings:
        st.warning(w)
    st.dataframe(pd.DataFrame([{"check": c.name, "status": c.status, "message": c.message} for c in report.checks]), hide_index=True)
    st.info("Alignment gate: not implemented in Phase 1 (arrives in Phase 2). Only grid-level compatibility is checked.")
    if st.button("Run detection", type="primary", disabled=not report.ok):
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                st.write("Detecting changes (baseline difference), filtering, polygonizing, exporting...")
                res = run_pipeline(before, after, cfg, out_root=ROOT / "outputs",
                                   input_notes="SYNTHETIC test pair" if synthetic else None)
                status.update(label=f"Done in {res.elapsed_s:.1f}s", state="complete")
            st.session_state["result"] = (res, before, after)
        except InputValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # surfaced to the user, full trace is in run.log
            st.error(f"Processing failed: {exc}")

if "result" in st.session_state:
    res, before, after = st.session_state["result"]
    st.subheader("3. Results")
    k = res.counts
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Change polygons", k["polygons_after_area_filter"], help=f"{k['polygons_before_area_filter']} before the area filter")
    m2.metric("Changed area (m²)", f"{k['total_changed_area_m2']:,.0f}")
    m3.metric("Threshold used", f"{k['threshold']:.3f}", help=k["threshold_method"])
    m4.metric("Run time", f"{res.elapsed_s:.1f}s")
    st.caption("Detector used: baseline_difference (thresholded image difference). Learned model: not used.")
    for w in res.warnings:
        st.warning(w)
    b_png, a_png, psize, fsize = render_pair_previews(before, after)
    with rasterio.open(before) as src:
        tr = src.transform
    components.html(build_split_view_html(b_png, a_png, res.polygons, tr, psize, fsize), height=720, scrolling=True)

    df = res.polygons.drop(columns=["geometry", "label_id"], errors="ignore")
    st.markdown("**Change polygons**")
    if df.empty:
        st.info("No polygons to show. Try a lower threshold or a smaller minimum mapping unit.")
    else:
        amin, amax = float(df.area_m2.min()), float(df.area_m2.max())
        f1, f2 = st.columns(2)
        area_rng = f1.slider("Area (m²)", amin, max(amax, amin + 1), (amin, max(amax, amin + 1)))
        conf_min = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        view = df[(df.area_m2 >= area_rng[0]) & (df.area_m2 <= area_rng[1]) & (df.confidence >= conf_min)]
        st.dataframe(view, hide_index=True)
    d = res.run_dir
    st.markdown("**Downloads**")
    cols = st.columns(4)
    for col, (label, fn, mime) in zip(cols, [("GeoPackage", "change_polygons.gpkg", "application/geopackage+sqlite3"),
                                             ("GeoJSON", "change_polygons.geojson", "application/geo+json"),
                                             ("CSV summary", "change_summary.csv", "text/csv"),
                                             ("Provenance", "provenance.json", "application/json")]):
        col.download_button(label, (d / fn).read_bytes(), file_name=fn, mime=mime)
    st.caption(f"All outputs were written to: {d}")
