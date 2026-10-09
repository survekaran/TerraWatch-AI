"""Headless CLI runner for the GEOAI 04 change detection pipeline.

Usage:
  python scripts/run_pipeline.py --before B.tif --after A.tif [--mode emergency|enforcement] [--config C.yaml] [--out outputs] [--allow-misaligned] [--auto-correct]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoai04.config import load_config
from geoai04.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="Path to BEFORE GeoTIFF")
    parser.add_argument("--after", required=True, help="Path to AFTER GeoTIFF")
    parser.add_argument("--mode", default="emergency", choices=["emergency", "enforcement"], help="Severity mode")
    parser.add_argument("--config", default=None, help="Path to custom config YAML")
    parser.add_argument("--out", default="outputs", help="Output directory root")
    parser.add_argument("--allow-misaligned", action="store_true", help="Run anyway even if alignment gate fails")
    parser.add_argument("--auto-correct", action="store_true", help="Auto-correct estimated sub-pixel shift on AFTER image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    res = run_pipeline(
        before_path=args.before,
        after_path=args.after,
        cfg=cfg,
        out_root=args.out,
        mode=args.mode,
        allow_misaligned=args.allow_misaligned,
        auto_correct=args.auto_correct,
    )
    print(f"Pipeline completed successfully.")
    print(f"Run directory: {res.run_dir}")
    print(f"Polygons produced: {len(res.polygons)}")
    print(f"Alignment status: {res.alignment.status if res.alignment else 'not_run'}")
    print(f"Routing policy: {res.routing.policy if res.routing else 'not_run'}")
    print(f"Detector used: {res.detection.detector if res.detection else 'none'}")


if __name__ == "__main__":
    main()
