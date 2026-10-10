"""Streamlit interface for GEOAI 04 Change Detection. Run: streamlit run app.py"""
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

from geoai04.alignment import AlignmentError, estimate_alignment  # noqa: E402
from geoai04.config import load_config  # noqa: E402
from geoai04.pipeline import PHASE2_NOT_IMPLEMENTED, run_pipeline  # noqa: E402
from geoai04.resolution_router import route_inputs  # noqa: E402
from geoai04.synthetic import generate_synthetic_pair  # noqa: E402
from geoai04.validator import InputValidationError, validate_pair  # noqa: E402
from geoai04.viewer import build_split_view_html, render_heatmap_preview, render_pair_previews  # noqa: E402

st.set_page_config(page_title="GEOAI 04 Change Intelligence", layout="wide")
st.title("GEOAI 04: bi-temporal change detection (Phase 2)")
st.caption("Local geospatial change intelligence. Alignment gate, resolution routing, artifact tagging, and severity engine.")

defaults = load_config()

with st.sidebar:
    st.header("Operational Mode")
    mode = st.selectbox(
        "Severity Mode",
        ["emergency", "enforcement"],
        index=0 if defaults.severity.mode == "emergency" else 1,
        help="Emergency: recall-weighted (lower thresholds). Enforcement: precision-weighted (higher thresholds).",
    )

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

    st.header("Alignment Options")
    auto_correct = st.checkbox(
        "Auto-correct sub-pixel shift",
        value=bool(defaults.alignment.auto_correct),
        help="Shift AFTER raster to align with BEFORE raster and write corrected GeoTIFF into run folder.",
    )

    st.header("Multispectral Band Map")
    with st.expander("Configure band map (optional)"):
        st.caption("Default unset: band order is NEVER assumed. Specify 1-based band indices if known.")
        b_blue = st.number_input("Blue band index (1-based)", 0, 32, 0)
        b_green = st.number_input("Green band index (1-based)", 0, 32, 0)
        b_red = st.number_input("Red band index (1-based)", 0, 32, 0)
        b_nir = st.number_input("NIR band index (1-based)", 0, 32, 0)
        b_swir1 = st.number_input("SWIR1 band index (1-based)", 0, 32, 0)
    st.header("Status of modules")
    for line in PHASE2_NOT_IMPLEMENTED:
        st.caption("• " + line)

configured_band_map = {}
if b_blue > 0: configured_band_map["blue"] = int(b_blue)
if b_green > 0: configured_band_map["green"] = int(b_green)
if b_red > 0: configured_band_map["red"] = int(b_red)
if b_nir > 0: configured_band_map["nir"] = int(b_nir)
if b_swir1 > 0: configured_band_map["swir1"] = int(b_swir1)

cfg = load_config(overrides={
    "detection": {"threshold_method": method, "fixed_threshold": fixed, "percentile": pct, "otsu_floor": floor, "smoothing_sigma_px": sigma},
    "postprocessing": {"opening_radius_px": int(opening), "closing_radius_px": int(closing), "min_mapping_unit_m2": float(mmu)},
    "polygons": {"simplify_tolerance_px": float(simp)},
    "alignment": {"auto_correct": auto_correct},
    "severity": {"mode": mode},
    "bands": {"band_map": configured_band_map},
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
    st.subheader("2. Validation & Assessment Gate")
    (st.success if report.ok else st.error)("Inputs are valid." if report.ok else "Inputs rejected: " + " | ".join(report.errors))
    for w in report.warnings:
        st.warning(w)
    st.dataframe(pd.DataFrame([{"check": c.name, "status": c.status, "message": c.message} for c in report.checks]), hide_index=True)

    allow_misaligned = False
    if report.ok:
        gsd_x = report.before.gsd_x_m if report.before else None
        gsd_y = report.before.gsd_y_m if report.before else None

        # Alignment Gate preview
        align_res = estimate_alignment(before, after, cfg.alignment, gsd_x_m=gsd_x, gsd_y_m=gsd_y)
        st.markdown("#### Alignment Assessment Gate")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Status", align_res.status.upper())
        ac2.metric("Shift (pixels)", f"{align_res.magnitude_px:.2f} px" if align_res.magnitude_px is not None else "Unknown")
        ac3.metric("Shift (metres)", f"{align_res.shift_m:.2f} m" if align_res.shift_m is not None else "Unknown")
        ac4.metric("Window Spread", f"{align_res.spread_px:.2f} px" if align_res.spread_px is not None else "Unknown")

        st.caption(
            "Sign convention: estimated shift is the translation vector to apply to AFTER to register with BEFORE. "
            f"Evaluated across {align_res.n_windows_used} textured windows."
        )

        for r in align_res.reasons:
            if align_res.status == "fail":
                st.error(r)
            elif align_res.status == "warn":
                st.warning(r)
            else:
                st.info(r)

        if align_res.status == "fail":
            allow_misaligned = st.checkbox("Run anyway: results will be unreliable", value=False)

        # Resolution Router preview
        routing_dec = route_inputs(report.before, report.after, cfg)
        st.markdown("#### Resolution Router Decision")
        rc1, rc2 = st.columns(2)
        rc1.write(f"**Policy:** `{routing_dec.policy}`")
        rc1.write(f"**Detector Selected:** `{routing_dec.detector}`")
        rc2.write(f"**Output Granularity:** {routing_dec.output_granularity}")
        for r in routing_dec.reasons:
            st.caption(f"• {r}")
        for w in routing_dec.warnings:
            st.warning(w)

        can_run = report.ok and (align_res.status != "fail" or allow_misaligned)

        if st.button("Run detection", type="primary", disabled=not can_run):
            try:
                with st.status("Running pipeline...", expanded=True) as status:
                    st.write(f"Running detection ({routing_dec.detector}), tagging, severity ({mode})...")
                    res = run_pipeline(
                        before,
                        after,
                        cfg,
                        out_root=ROOT / "outputs",
                        mode=mode,
                        allow_misaligned=allow_misaligned,
                        auto_correct=auto_correct,
                        input_notes="SYNTHETIC test pair" if synthetic else None,
                    )
                    status.update(label=f"Done in {res.elapsed_s:.1f}s", state="complete")
                st.session_state["result"] = (res, before, after)
            except InputValidationError as exc:
                st.error(str(exc))
            except AlignmentError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

if "result" in st.session_state:
    res, before, after = st.session_state["result"]
    st.subheader("3. Results & Intelligence")
    k = res.counts
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Change Polygons", k["polygons_after_area_filter"], help=f"{k['polygons_before_area_filter']} before area filter")
    m2.metric("Changed Area (m²)", f"{k['total_changed_area_m2']:,.0f}")
    m3.metric("Threshold Used", f"{k['threshold']:.3f}", help=k["threshold_method"])
    m4.metric("Severity Mode", res.routing.policy if res.routing else mode)
    m5.metric("Runtime", f"{res.elapsed_s:.1f}s")

    if any("Implausible change fraction" in str(w) for w in res.warnings):
        warn_text = next(str(w) for w in res.warnings if "Implausible change fraction" in str(w))
        st.error(
            f"⚠️ **Implausible Change Fraction Warning**: {warn_text}\n\n"
            "All detected polygons have been assigned to severity tier **Review** (reason: `implausible_change_fraction`).\n\n"
            "*Note for Analysts*: Genuine severe regional events (e.g. extensive flooding or wildfires) can exceed the threshold. "
            "Analysts can configure `implausible_change_fraction_emergency` or `implausible_change_fraction_enforcement` in `config/default.yaml` to raise it."
        )

    for w in res.warnings:
        st.warning(w)

    # Detector & Routing Status
    st.markdown("#### Detector & Routing Status")
    st_c1, st_c2 = st.columns(2)
    st_c1.write(f"**Detector Used:** `{res.detection.detector}`")
    st_c1.write(f"**Policy:** `{res.routing.policy}`")
    st_c2.write(f"**Output Granularity:** `{res.routing.output_granularity}`")

    b_png, a_png, psize, fsize = render_pair_previews(before, after)
    with rasterio.open(before) as src:
        tr = src.transform

    heatmap_png = None
    scores_tif = res.run_dir / "change_scores.tif"
    if scores_tif.is_file():
        heatmap_png = render_heatmap_preview(scores_tif, psize)

    components.html(
        build_split_view_html(
            b_png, a_png, res.polygons, tr, psize, fsize,
            heatmap_png=heatmap_png,
        ),
        height=740,
        scrolling=True,
    )

    df = res.polygons.drop(columns=["geometry", "label_id"], errors="ignore")
    st.markdown("#### Filter and Inspect Change Polygons")
    if df.empty:
        st.info("No polygons produced with the current parameters.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        available_tiers = sorted(df["severity_tier"].unique()) if "severity_tier" in df else []
        sel_tiers = fc1.multiselect("Filter by Severity Tier", available_tiers, default=available_tiers)

        available_tags = sorted(df["type_tag"].unique()) if "type_tag" in df else []
        sel_tags = fc2.multiselect("Filter by Type Tag", available_tags, default=available_tags)

        only_review = fc3.checkbox("Show only Review / Quality-flagged polygons", value=False)

        view = df.copy()
        if sel_tiers:
            view = view[view["severity_tier"].isin(sel_tiers)]
        if sel_tags:
            view = view[view["type_tag"].isin(sel_tags)]
        if only_review and "quality_flag" in view:
            view = view[view["quality_flag"] != "nominal"]

        st.dataframe(view, hide_index=True)

    d = res.run_dir
    st.markdown("#### Downloads")
    cols = st.columns(5)
    dl_items = [
        ("GeoPackage", "change_polygons.gpkg", "application/geopackage+sqlite3"),
        ("GeoJSON", "change_polygons.geojson", "application/geo+json"),
        ("CSV Summary", "change_summary.csv", "text/csv"),
        ("Provenance", "provenance.json", "application/json"),
        ("Config Snapshot", "config_snapshot.yaml", "application/x-yaml"),
    ]
    for col, (label, fn, mime) in zip(cols, dl_items):
        if (d / fn).is_file():
            col.download_button(label, (d / fn).read_bytes(), file_name=fn, mime=mime)

    st.caption(f"All outputs and run logs are preserved locally at: {d}")
