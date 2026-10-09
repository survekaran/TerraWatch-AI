"""Evaluate the baseline detector against a reference mask and write measured results.

Usage:
  python scripts/evaluate.py --synthetic
  python scripts/evaluate.py --before B.tif --after A.tif --reference R.tif [--out outputs/eval]

Writes evaluation.json and evaluation.csv. Every number comes from an actual run.
No result is reported unless a reference mask is provided.
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rasterio  # noqa: E402

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
        rows.append({"threshold_method": method, "threshold": res.threshold,
                     "polygons_before_area_filter": res.counts["polygons_before_area_filter"],
                     "polygons_after_area_filter": res.counts["polygons_after_area_filter"],
                     "total_changed_area_m2": res.counts["total_changed_area_m2"], "elapsed_s": round(res.elapsed_s, 3), **m})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before"); ap.add_argument("--after"); ap.add_argument("--reference")
    ap.add_argument("--synthetic", action="store_true", help="Evaluate on the SYNTHETIC pair (a pipeline sanity check only)")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--methods", default="otsu,fixed,percentile")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        d = Path(tempfile.mkdtemp(prefix="geoai04_syn_"))
        generate_synthetic_pair(d)
        a.before, a.after, a.reference = str(d / "before.tif"), str(d / "after.tif"), str(d / "reference_mask.tif")
        note = "SYNTHETIC data: a sanity check of the pipeline, not a performance claim."
    elif not (a.before and a.after and a.reference):
        ap.error("Provide --before, --after and --reference, or use --synthetic")
    else:
        note = "Real inputs. Prefer scene- or geography-level holdouts when comparing methods."
    rows = evaluate(a.before, a.after, a.reference, out, a.methods.split(","))
    (out / "evaluation.json").write_text(json.dumps({"note": note, "results": rows}, indent=2), encoding="utf-8")
    with open(out / "evaluation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(note)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("threshold_method", "threshold", "precision", "recall", "f1", "iou", "polygons_after_area_filter")})
    print(f"Wrote {out/'evaluation.json'} and {out/'evaluation.csv'}")
