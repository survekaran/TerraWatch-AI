# GEOAI 04: Satellite Change Detection (Phase 1 prototype)

rakshit logged in

A local, modular pipeline that compares two co-registered GeoTIFFs, detects candidate physical
change with a transparent **baseline image-difference detector**, and exports attributed GIS
polygons (GeoPackage + GeoJSON) with a provenance record. A Streamlit app provides validation,
parameter controls, an offline before/after split view and downloads.

**Status: Phase 1 only.** No AI model is used. Detections are candidate changes, not confirmed
construction, damage or illegal encroachment.

## Setup (Python 3.11 or 3.12)

macOS / Linux:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Windows (PowerShell):

    py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

If `pip install rasterio` fails on your platform, use `conda install -c conda-forge rasterio geopandas`.

## Run

    python scripts/generate_synthetic_pair.py --out data/sample   # SYNTHETIC test data
    streamlit run app.py                                           # local app (no internet needed)
    python -m pytest                                               # automated tests
    python scripts/evaluate.py --synthetic                         # writes evaluation.json/csv

In the app: upload two GeoTIFFs (or paste local paths, or generate the synthetic pair), review the
validation table, adjust parameters in the sidebar, click **Run detection**. Outputs are written to
`outputs/run_<timestamp>_<pairid>/`.

## Architecture (Phase 1)

    Input Validator -> Baseline Detector -> Morphological Filter + MMU -> Polygonizer
        -> Attribute Engine -> Exporter (+ provenance) -> Streamlit review UI

| Module | File | Notes |
|---|---|---|
| Config | `config.py`, `config/default.yaml` | All thresholds live here; unknown keys are rejected |
| Validation | `validator.py` | CRS, grid, overlap, bands, GSD, NoData, empty/constant imagery; never resamples or assigns a CRS |
| Baseline detector | `baseline_detector.py` | Windowed RMS difference over the first N bands, NoData-aware smoothing, Otsu/fixed/percentile threshold |
| Postprocessing | `postprocessing.py` | Opening, closing, connected components, minimum mapping unit in m² |
| Polygonizer | `polygonizer.py` | `rasterio.features.shapes`, geometry repair, conservative simplification |
| Attributes | `attributes.py`, `geometry_utils.py` | Area/centroids measured in a metric CRS before any reprojection |
| Export / provenance | `exporter.py`, `provenance.py` | GPKG (scene CRS), GeoJSON (EPSG:4326), CSV, rasters, provenance.json |
| Viewer | `viewer.py` | Self-contained offline HTML (no CDN, no online tiles) |
| Pipeline | `pipeline.py` | Orchestrates the stages and writes the run folder |

Measurements use the scene CRS if it is a metre-based projected CRS (not Web Mercator); otherwise a
UTM zone chosen from the scene centre. The CRS used is recorded in every polygon and in provenance.

## Sample output structure

    outputs/run_20261009_101500_3f0683a53a17/
        change_mask.tif          uint8: 1 changed, 0 unchanged, 255 NoData
        change_scores.tif        float32 change score 0..1 (-9999 NoData)
        change_polygons.gpkg     layer "change_polygons", scene CRS
        change_polygons.geojson  EPSG:4326
        change_summary.csv       attributes without geometry
        provenance.json          inputs + SHA-256, parameters, counts, warnings, versions
        validation_report.json
        run.log

Polygon attributes: `change_id, area_m2, centroid_lon, centroid_lat, centroid_x, centroid_y,
projected_crs, change_magnitude, confidence, pixel_count, source_pair_id, detector_used`.

- `change_magnitude`: mean change score inside the region (0..1).
- `confidence`: fraction of region pixels at or above the detection threshold. It measures detector
  support, **not** a calibrated probability, and is close to 1.0 for most clean regions in Phase 1.
- `area_m2`: physical size, measured in the metric CRS before simplification.

## Implemented vs pending

| Feature | Phase | Status |
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
