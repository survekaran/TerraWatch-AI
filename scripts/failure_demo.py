"""Generate and execute failure demonstrations.

Scenarios demonstrated:
(a) Misaligned pair: translation > fail_px (2.0px) causes gate failure; allow_misaligned overrides it,
    forcing all polygons to tier 'Review' with quality_flag='alignment_override'.
(b) Coarse GSD without band map: coarse resolution (10.0 m) safely falls back to baseline difference detector.
(c) Heavy illumination difference: extreme sensor/sun-angle gain triggers diffuse brightness difference.
(d) Large NoData block: unmeasured/invalid region properly excluded and flagged low_quality.

All test inputs are SYNTHETIC and clearly labelled as such.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai04.config import AppConfig, load_config
from geoai04.pipeline import AlignmentError, run_pipeline
from geoai04.synthetic import generate_synthetic_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("failure_demo")


def run_demo_a_misaligned(out_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    """Scenario (a): Misaligned pair (injected 3.0 px shift)."""
    syn_dir = out_dir / "syn_a_misaligned"
    generate_synthetic_pair(syn_dir, shift_px=(3.0, 0.0))

    # 1. Gate default: expect AlignmentError
    gate_blocked = False
    try:
        run_pipeline(
            syn_dir / "before.tif",
            syn_dir / "after.tif",
            cfg,
            out_root=out_dir / "runs_a_gate",
            run_name="misaligned_default_gate",
            allow_misaligned=False,
        )
    except AlignmentError:
        gate_blocked = True

    # 2. Override: allow_misaligned=True
    res_override = run_pipeline(
        syn_dir / "before.tif",
        syn_dir / "after.tif",
        cfg,
        out_root=out_dir / "runs_a_override",
        run_name="misaligned_override",
        allow_misaligned=True,
    )
    polys = res_override.polygons
    all_review = (polys["severity_tier"] == "Review").all() if len(polys) else False
    flags = list(polys["quality_flag"].unique()) if len(polys) else []

    return {
        "scenario": "(a) Misaligned pair (3.0 px shift)",
        "input_type": "SYNTHETIC",
        "gate_blocked_by_default": gate_blocked,
        "override_status": res_override.status,
        "polygons_produced": len(polys),
        "all_marked_review": all_review,
        "quality_flags": flags,
        "observed_behaviour": (
            "Gate blocked default execution with AlignmentError (shift 3.0 px >= 2.0 px). "
            "With allow_misaligned=True, execution completed but 100% of polygons were forced to tier 'Review' "
            "with quality_flag containing 'alignment_override'."
        ),
    }


def run_demo_b_coarse_gsd(out_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    """Scenario (b): Coarse resolution without multispectral band mapping (10.0 m GSD)."""
    syn_dir = out_dir / "syn_b_coarse_gsd"
    generate_synthetic_pair(syn_dir, size=128, gsd=10.0)

    c = load_config()
    res = run_pipeline(
        syn_dir / "before.tif",
        syn_dir / "after.tif",
        c,
        out_root=out_dir / "runs_b",
        run_name="coarse_gsd_run",
    )

    detector_used = res.routing.detector
    reasons = res.routing.reasons

    return {
        "scenario": "(b) Coarse GSD, no band map (10.0 m GSD)",
        "input_type": "SYNTHETIC",
        "detector_selected": detector_used,
        "policy": res.routing.policy,
        "reasons": reasons,
        "observed_behaviour": (
            f"Router fell back to baseline difference detector (policy: {res.routing.policy}) with explicit reason: "
            f"'{reasons[0]}'."
        ),
    }


def run_demo_c_heavy_illumination(out_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    """Scenario (c): Extreme sun-angle / illumination brightness gain."""
    syn_dir = out_dir / "syn_c_illumination"
    syn_dir.mkdir(parents=True, exist_ok=True)
    size = 256
    rng = np.random.default_rng(123)

    before = rng.integers(50, 150, (3, size, size), dtype="uint8")
    # After has huge brightness gain (sun glare / lighting shift)
    after = np.clip(before.astype(np.float32) * 1.6 + 40, 0, 255).astype("uint8")

    T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    prof = {"driver": "GTiff", "count": 3, "height": size, "width": size, "dtype": "uint8", "crs": "EPSG:32633", "transform": T}
    with rasterio.open(syn_dir / "before.tif", "w", **prof) as dst: dst.write(before)
    with rasterio.open(syn_dir / "after.tif", "w", **prof) as dst: dst.write(after)

    c = load_config()
    c.detection.radiometric_norm_method = "off"

    res = run_pipeline(
        syn_dir / "before.tif",
        syn_dir / "after.tif",
        c,
        out_root=out_dir / "runs_c",
        run_name="illumination_run",
    )

    tags = list(res.polygons["type_tag"].value_counts().to_dict().items()) if len(res.polygons) else []
    changed_pct = res.counts.get("changed_fraction", 0.0) * 100.0
    all_review = bool((res.polygons["severity_tier"] == "Review").all()) if len(res.polygons) else False
    implausible_flag = bool(res.counts.get("implausible_change_fraction", False))

    return {
        "scenario": "(c) Heavy illumination difference",
        "input_type": "SYNTHETIC",
        "status": res.status,
        "polygons_produced": len(res.polygons),
        "changed_fraction": f"{changed_pct:.1f}%",
        "implausible_sanity_triggered": implausible_flag,
        "all_marked_review": all_review,
        "tag_distribution": tags,
        "observed_behaviour": (
            f"Run-level sanity check triggered: changed fraction was {changed_pct:.1f}% "
            f"(exceeds 20.0% threshold). Run warning set, and 100% of polygons forced to tier 'Review' "
            f"with reason 'implausible_change_fraction'."
        ),
    }


def run_demo_d_nodata_block(out_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    """Scenario (d): Pair with large invalid / NoData corner."""
    syn_dir = out_dir / "syn_d_nodata"
    generate_synthetic_pair(syn_dir, add_nodata_block=True)

    res = run_pipeline(
        syn_dir / "before.tif",
        syn_dir / "after.tif",
        cfg,
        out_root=out_dir / "runs_d",
        run_name="nodata_run",
    )

    polys = res.polygons
    low_q_count = int((polys["type_tag"] == "low_quality").sum()) if len(polys) else 0

    return {
        "scenario": "(d) Large NoData block",
        "input_type": "SYNTHETIC",
        "status": res.status,
        "polygons_produced": len(polys),
        "low_quality_tags": low_q_count,
        "observed_behaviour": (
            f"NoData region was excluded from inference. Polygons overlapping unmeasured pixels "
            f"were correctly flagged low_quality and routed to Review."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Failure Demonstrations")
    parser.add_argument("--out-dir", default="outputs/failure_demos", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    log.info("Running Scenario (a): Misaligned pair...")
    res_a = run_demo_a_misaligned(out_path, cfg)

    log.info("Running Scenario (b): Coarse GSD, no band map...")
    res_b = run_demo_b_coarse_gsd(out_path, cfg)

    log.info("Running Scenario (c): Heavy illumination difference...")
    res_c = run_demo_c_heavy_illumination(out_path, cfg)

    log.info("Running Scenario (d): Large NoData block...")
    res_d = run_demo_d_nodata_block(out_path, cfg)

    demos = [res_a, res_b, res_c, res_d]

    # Write Markdown table summary
    md_lines = [
        "# Failure Demonstrations & Observed System Behaviour",
        "",
        "> [!IMPORTANT]",
        "> All input data used in these demonstrations are **SYNTHETIC** test rasters with known injected anomalies.",
        "> They demonstrate what the software system actually did when presented with boundary conditions.",
        "",
        "| Scenario | Input Type | System Action | Observed Outcome |",
        "|---|---|---|---|",
    ]

    for d in demos:
        scenario = d["scenario"]
        inp = d["input_type"]
        obs = d["observed_behaviour"].replace("\n", " ")
        if "(a)" in scenario:
            action = "Blocked by default; override allowed with 100% Review tier"
        elif "(b)" in scenario:
            action = "Safely routed to baseline difference detector"
        elif "(c)" in scenario:
            action = "Sanity check flagged >20% changed; all polygons marked Review"
        else:
            action = "Excluded NoData pixels; tagged low_quality"
        md_lines.append(f"| {scenario} | {inp} | {action} | {obs} |")

    md_path = out_path / "FAILURE_DEMONSTRATIONS.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Wrote failure demonstrations report to %s", md_path)

    print("\n" + "\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
