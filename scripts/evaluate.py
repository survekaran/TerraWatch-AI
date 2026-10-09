"""Evaluate the change detection pipeline against a reference mask and write measured results.

Usage:
  python scripts/evaluate.py --synthetic
  python scripts/evaluate.py --shift-robustness
  python scripts/evaluate.py --before B.tif --after A.tif --reference R.tif [--out outputs/eval]

Writes evaluation.json/csv and shift_robustness.json/csv.
Report only what it actually measures, labelled SYNTHETIC.
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rasterio  # noqa: E402

from geoai04.alignment import AlignmentError  # noqa: E402
from geoai04.config import load_config  # noqa: E402
from geoai04.metrics import pixel_metrics  # noqa: E402
from geoai04.pipeline import run_pipeline  # noqa: E402
from geoai04.synthetic import generate_synthetic_pair  # noqa: E402


def evaluate(before, after, reference, out_dir, methods):
    rows = []
    with rasterio.open(reference) as ref_src:
        ref = ref_src.read(1) > 0
    for method in methods:
        cfg = load_config(overrides={"detection": {"threshold_method": method}})
        res = run_pipeline(before, after, cfg, out_root=out_dir, run_name=f"eval_{method}")
        m = pixel_metrics(res.clean.mask, ref, res.detection.valid)
        rows.append({
            "threshold_method": method,
            "threshold": res.threshold,
            "polygons_before_area_filter": res.counts["polygons_before_area_filter"],
            "polygons_after_area_filter": res.counts["polygons_after_area_filter"],
            "total_changed_area_m2": res.counts["total_changed_area_m2"],
            "elapsed_s": round(res.elapsed_s, 3),
            **m,
        })
    return rows


def evaluate_shift_robustness(out_dir: Path, shifts=(0.0, 1.0, 2.0, 5.0)):
    """Evaluate pipeline across injected shifts (0, 1, 2, 5 px) under gate, override, and auto-correct."""
    rows = []
    cfg = load_config()

    for s in shifts:
        tmp_d = Path(tempfile.mkdtemp(prefix=f"geoai04_shift_{s}_"))
        meta = generate_synthetic_pair(tmp_d, shift_px=(s, 0.0))
        before = tmp_d / "before.tif"
        after = tmp_d / "after.tif"
        ref_path = tmp_d / "reference_mask.tif"
        with rasterio.open(ref_path) as rs:
            ref = rs.read(1) > 0

        # Condition 1: Gate on (default)
        try:
            res_gate = run_pipeline(
                before, after, cfg, out_root=out_dir, run_name=f"shift_{s}_gate_default",
                allow_misaligned=False, auto_correct=False
            )
            m = pixel_metrics(res_gate.clean.mask, ref, res_gate.detection.valid)
            px_area = res_gate.counts["pixel_area_m2"]
            fa_area = int((res_gate.clean.mask & ~ref).sum()) * px_area
            rows.append({
                "dataset": "SYNTHETIC",
                "injected_shift_px": s,
                "mode": "gate_default",
                "status": "completed",
                "detected_shift_px": res_gate.alignment.magnitude_px if res_gate.alignment else 0.0,
                "polygons": len(res_gate.polygons),
                "total_area_m2": res_gate.counts["total_changed_area_m2"],
                "false_alarm_area_m2": fa_area,
                "iou": round(m["iou"], 4) if m["iou"] is not None else None,
                "f1": round(m["f1"], 4) if m["f1"] is not None else None,
                "precision": round(m["precision"], 4) if m["precision"] is not None else None,
                "recall": round(m["recall"], 4) if m["recall"] is not None else None,
            })
        except AlignmentError as exc:
            rows.append({
                "dataset": "SYNTHETIC",
                "injected_shift_px": s,
                "mode": "gate_default",
                "status": "blocked_by_gate",
                "detected_shift_px": exc.result.magnitude_px,
                "polygons": 0,
                "total_area_m2": 0.0,
                "false_alarm_area_m2": 0.0,
                "iou": 0.0,
                "f1": 0.0,
                "precision": None,
                "recall": 0.0,
            })

        # Condition 2: Gate overridden (allow_misaligned=True)
        res_over = run_pipeline(
            before, after, cfg, out_root=out_dir, run_name=f"shift_{s}_allow_misaligned",
            allow_misaligned=True, auto_correct=False
        )
        m_over = pixel_metrics(res_over.clean.mask, ref, res_over.detection.valid)
        px_area = res_over.counts["pixel_area_m2"]
        fa_area_over = int((res_over.clean.mask & ~ref).sum()) * px_area
        rows.append({
            "dataset": "SYNTHETIC",
            "injected_shift_px": s,
            "mode": "allow_misaligned",
            "status": "completed",
            "detected_shift_px": res_over.alignment.magnitude_px if res_over.alignment else 0.0,
            "polygons": len(res_over.polygons),
            "total_area_m2": res_over.counts["total_changed_area_m2"],
            "false_alarm_area_m2": fa_area_over,
            "iou": round(m_over["iou"], 4) if m_over["iou"] is not None else None,
            "f1": round(m_over["f1"], 4) if m_over["f1"] is not None else None,
            "precision": round(m_over["precision"], 4) if m_over["precision"] is not None else None,
            "recall": round(m_over["recall"], 4) if m_over["recall"] is not None else None,
        })

        # Condition 3: Auto-correct on
        res_corr = run_pipeline(
            before, after, cfg, out_root=out_dir, run_name=f"shift_{s}_auto_correct",
            allow_misaligned=False, auto_correct=True
        )
        m_corr = pixel_metrics(res_corr.clean.mask, ref, res_corr.detection.valid)
        fa_area_corr = int((res_corr.clean.mask & ~ref).sum()) * px_area
        rows.append({
            "dataset": "SYNTHETIC",
            "injected_shift_px": s,
            "mode": "auto_correct",
            "status": "completed",
            "detected_shift_px": res_corr.residual_alignment.magnitude_px if res_corr.residual_alignment else 0.0,
            "polygons": len(res_corr.polygons),
            "total_area_m2": res_corr.counts["total_changed_area_m2"],
            "false_alarm_area_m2": fa_area_corr,
            "iou": round(m_corr["iou"], 4) if m_corr["iou"] is not None else None,
            "f1": round(m_corr["f1"], 4) if m_corr["f1"] is not None else None,
            "precision": round(m_corr["precision"], 4) if m_corr["precision"] is not None else None,
            "recall": round(m_corr["recall"], 4) if m_corr["recall"] is not None else None,
        })

    (out_dir / "shift_robustness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(out_dir / "shift_robustness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--reference")
    ap.add_argument("--synthetic", action="store_true", help="Evaluate baseline and shift robustness on SYNTHETIC pair")
    ap.add_argument("--shift-robustness", action="store_true", help="Run measured shift robustness table")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--methods", default="otsu,fixed,percentile")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.synthetic or a.shift_robustness:
        d = Path(tempfile.mkdtemp(prefix="geoai04_syn_"))
        generate_synthetic_pair(d)
        a.before, a.after, a.reference = str(d / "before.tif"), str(d / "after.tif"), str(d / "reference_mask.tif")
        note = "SYNTHETIC data: a sanity check of the pipeline, not a performance claim."
    elif not (a.before and a.after and a.reference):
        ap.error("Provide --before, --after and --reference, or use --synthetic / --shift-robustness")
    else:
        note = "Real inputs. Prefer scene- or geography-level holdouts when comparing methods."

    if not a.shift_robustness or a.synthetic:
        rows = evaluate(a.before, a.after, a.reference, out, a.methods.split(","))
        (out / "evaluation.json").write_text(json.dumps({"note": note, "results": rows}, indent=2), encoding="utf-8")
        with open(out / "evaluation.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(note)
        for r in rows:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("threshold_method", "threshold", "precision", "recall", "f1", "iou", "polygons_after_area_filter")})
        print(f"Wrote {out/'evaluation.json'} and {out/'evaluation.csv'}")

    if a.synthetic or a.shift_robustness:
        print("\n--- MEASURED SHIFT ROBUSTNESS (SYNTHETIC) ---")
        robust_rows = evaluate_shift_robustness(out)
        for r in robust_rows:
            print(f"Shift: {r['injected_shift_px']}px | Mode: {r['mode']:<16} | Status: {r['status']:<15} | Polygons: {r['polygons']:<2} | FA Area: {r['false_alarm_area_m2']:>6.1f}m² | IoU: {r['iou']} | F1: {r['f1']}")
        print(f"Wrote {out/'shift_robustness.json'} and {out/'shift_robustness.csv'}")
