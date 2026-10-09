"""Generate a SYNTHETIC georeferenced before/after pair with a known change.

Usage: python scripts/generate_synthetic_pair.py [--out data/sample]
The output is test data only. It is not satellite imagery.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoai04.synthetic import generate_synthetic_pair  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    meta = generate_synthetic_pair(a.out, a.size, a.gsd, a.seed)
    print(f"Wrote SYNTHETIC pair to {a.out}: before.tif, after.tif, reference_mask.tif")
    print(f"Known total changed area: {meta['total_changed_area_m2']:.1f} m2")
