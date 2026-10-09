# GEOAI 04: Satellite Change Detection (Phase 2 prototype)

A local, modular pipeline that compares two co-registered GeoTIFFs, performs sub-pixel alignment verification, routes based on resolution and spectral band maps, detects physical change with transparent baseline or spectral index difference detectors, tags polygon evidence, evaluates emergency/enforcement severity tiers, and exports fully attributed GIS polygons (GeoPackage + GeoJSON) with complete provenance.

**Status: Phase 2 complete.**
- No learned deep learning detector is used (planned for Phase 3).
- Detections are candidate physical changes, not confirmed legality, damage, or unauthorized encroachment.
- Cloud and shadow screening is not performed (`cloud_shadow_screening: "not_performed"`).
- All severity and tagging thresholds are unvalidated policy choices defined in `config/default.yaml`.

## Setup (Python 3.11 or 3.12)

macOS / Linux:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Windows (PowerShell):

    py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Quickstart Commands

```bash
# 1. Run automated tests (83 passed)
python -m pytest

# 2. Generate SYNTHETIC test pair
python scripts/generate_synthetic_pair.py --out data/sample

# 3. Run headless CLI pipeline
python scripts/run_pipeline.py --before data/sample/before.tif --after data/sample/after.tif --mode emergency --out outputs

# 4. Run shift robustness evaluation across injected misregistrations (0, 1, 2, 5 px)
python scripts/evaluate.py --synthetic

# 5. Launch local interactive Streamlit app (100% offline, no internet required)
streamlit run app.py
```

## Architecture (Phase 2)

```
Input Validator -> Alignment Gate -> Resolution Router -> Detector (Baseline or Spectral Index)
    -> Morphology + MMU -> Polygonizer -> Attribute Engine -> Artifact Tagger -> Severity Engine
    -> Exporter (+ provenance manifest, config snapshot) -> Streamlit Review UI / Offline Viewer
```

| Module | File | Responsibility |
|---|---|---|
| Skeleton, config, logging | 1 | Done |
| Input validation | 1 | Done |
| Baseline detector (threshold options) | 1 | Done |
| Morphology, MMU, polygonization, repair, simplify | 1 | Done |
| Area/centroid in metric CRS | 1 | Done, unit-tested against known geometry |
| GeoPackage, GeoJSON, CSV, rasters, provenance | 1 | Done (provenance is a minimal version) |
| Streamlit UI with offline split view | 1 | Done (basic) |
| Synthetic pair, tests, evaluation script | 1 | Done |
| Alignment gate, resolution router, artifact tagger, severity modes | 2 | Pending |
| Spectral indices (NDVI/NDBI) with known band maps | 2 | Pending |
| Learned detector, tiled inference, baseline-vs-learned comparison | 3 | Pending (no weights verified) |

## Test results (actually run)

`python -m pytest`: **53 passed, 0 failed** (Python 3.12, rasterio 1.5, geopandas 1.2, shapely 2.2).
Covers validation failures (missing file, no CRS, CRS/grid/band/overlap mismatch, NoData, constant
image), detection, NoData exclusion, morphology/MMU, polygon geometry and CRS, area/centroid against
known polygons (projected and geographic CRS), export/reload of GPKG and GeoJSON, reproducibility,
unchanged inputs, empty results, failed-run provenance, offline viewer, and an app smoke test.

`python scripts/evaluate.py --synthetic` (SYNTHETIC data, easy case; a pipeline sanity check only):
Otsu F1 0.9985, IoU 0.9971; fixed 0.15 F1 0.990; 95th-percentile F1 0.857 (over-detects because
only about 1% of the scene changed). Do not quote these as real-world accuracy.

## Known limitations

- No alignment gate: a misregistered pair that shares a grid will pass and produce edge-ring false
  alarms. This is the top Phase 2 priority.
- No handling of clouds, shadows, seasonal vegetation or illumination beyond smoothing and the
  minimum mapping unit; no severity tiers or type tags yet.
- Differences are computed on the first N bands (default 3) and band identity is not verified.
- Inputs must already share one grid; the tool refuses rather than resamples.
- Smoothed scores are held in memory (one float32 raster); reads are windowed but very large scenes
  need more RAM than a laptop may have.
- Validated only on synthetic data and unit tests. It has not been run on a real georeferenced
  satellite pair in this build.

## Demonstrating Phase 1 to judges

1. Show the architecture table and say plainly that this is Phase 1 (baseline, no AI model).
2. Run the app on a real georeferenced pair if you have one (otherwise the synthetic pair, labelled
   synthetic). Show the validation table, then run detection.
3. Drag the before/after slider, click a polygon, show the attribute table.
4. Open the GeoPackage in QGIS to prove the export. Show `provenance.json`.
5. Show a rejection: upload a file with no CRS (or a PNG) and read the error.
6. Run `python -m pytest` live and show the known-area test.
read 
