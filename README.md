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
| Config | `config.py`, `config/default.yaml` | All thresholds live here; dataclasses validate inputs and unknown keys are rejected |
| Validation | `validator.py` | CRS, grid, overlap, band count, GSD, NoData, constant image; never resamples silently |
| Alignment Gate | `alignment.py` | Sub-pixel phase cross-correlation on Sobel gradients across windowed reads; auto-correction |
| Router | `resolution_router.py` | High-res RGB vs multispectral coarse vs ambiguous fallback; explicit learned weights check |
| Spectral Indices | `spectral.py` | Safe NDVI, NDBI, MNDWI, NDWI, vegetation_proxy_exg; spectral index difference detector |
| Baseline Detector | `baseline_detector.py` | Windowed RMS difference over first N bands; NoData-aware smoothing; Otsu/fixed/percentile |
| Postprocessing | `postprocessing.py` | Binary opening, closing, connected components, minimum mapping unit (m²) |
| Polygonizer | `polygonizer.py` | `rasterio.features.shapes`, polygon repair, conservative simplification |
| Attributes | `attributes.py`, `geometry_utils.py` | Metric CRS area/centroids; informative margin-based detector confidence |
| Artifact Tagger | `artifact_tagger.py` | Computes explainable evidence; tags `building_like`, `built_up_patch`, `vegetation_conversion`, `vegetation_fluctuation`, `misregistration_suspect`, `low_quality`, `model_disagreement`, `unknown` |
| Severity Engine | `severity.py` | Emergency (recall) vs Enforcement (precision) modes; Low/Medium/Critical tiers; Review override |
| Export & Provenance | `exporter.py`, `provenance.py` | GPKG, GeoJSON, CSV, rasters, config snapshot, and provenance with output SHA-256 manifest |
| Viewer | `viewer.py` | 100% offline self-contained HTML: pan/zoom, slider, heatmap, inspection, review Blob export |
| Pipeline & CLI | `pipeline.py`, `scripts/run_pipeline.py` | Orchestrates complete pipeline, logging, and error handling |

## Sample Output Structure

```
outputs/run_20261009_140000_a8ca8f531a3b/
    change_mask.tif          uint8: 1 changed, 0 unchanged, 255 NoData
    change_scores.tif        float32 change score 0..1 (-9999 NoData)
    change_polygons.gpkg     layer "change_polygons", scene CRS
    change_polygons.geojson  EPSG:4326
    change_summary.csv       attributes without geometry
    config_snapshot.yaml     exact parameters used in the run
    provenance.json          inputs + SHA-256, output manifest SHA-256, software, warnings
    validation_report.json   all validation check records
    run.log                  complete timestamped execution log
```

### Polygon Attributes

- `change_id`: Deterministic identifier (`CHG-0001`, `CHG-0002`, ...)
- `severity_tier`: Primary operational tier (`Low`, `Medium`, `Critical`, or `Review`)
- `base_severity`: Pre-override tier (`Low`, `Medium`, `Critical`)
- `severity_mode`: Operational mode used (`emergency` or `enforcement`)
- `type_tag`: Primary semantic tag from closed vocabulary
- `tags`: Semicolon-separated list of all applicable tags
- `quality_flag`: Quality / review flag (`nominal`, `alignment_override`, `misregistration_suspect`, etc.)
- `review_reasons`: Semicolon-separated triggers that caused Review placement
- `change_magnitude`: Mean change score inside the polygon (0..1)
- `confidence`: Uncalibrated margin-based detector support: mean over region pixels of `clip((score - threshold) / (1 - threshold), 0, 1)`
- `area_m2`: Physical size in m², measured in metric CRS before simplification
- `centroid_lon`, `centroid_lat`: Geographic coordinates (WGS84)
- `tag_evidence`: JSON-encoded geometric, spectral, and contextual evidence

## Test Results (Actually Run)

`python -m pytest`: **83 passed, 0 failed** in 21.24s.
- Validation checks (missing file, no CRS, CRS/grid/band/overlap mismatch, NoData, constant image)
- Sub-pixel alignment recovery (0, 1, 2, 5 px, 0.5 px sub-pixel) and documented sign convention
- Alignment gate failure and provenance recording on shifted pairs
- Alignment auto-correction with byte-identical original inputs and reduced residual shift
- Resolution router policies, GSD derivation from affine transform & CRS, fallback on missing weights
- Spectral indices (NDVI, NDBI, MNDWI) with zero-division safety and band order neutrality
- Index-mode change detection with delta rasters
- Artifact tagger rules, precedence, explicit anti-veto preservation of vegetation-to-built conversion
- Severity engine emergency vs enforcement thresholds, monotonicity, and Review overrides
- Bit-for-bit raster byte reproducibility and polygon attribute determinism
- Fully offline viewer without external URLs, links, or CDNs
- Headless Streamlit AppTest smoke tests

## Measured Shift-Robustness Table (SYNTHETIC)

Measured using `python scripts/evaluate.py --synthetic` across injected translations:

| Injected Shift | Mode | Status | Polygons | False Alarm Area (m²) | IoU | F1 |
|---|---|---|---|---|---|---|
| 0.0 px | `gate_default` | completed | 3 | 0.0 m² | 0.9971 | 0.9985 |
| 0.0 px | `allow_misaligned` | completed | 3 | 0.0 m² | 0.9971 | 0.9985 |
| 0.0 px | `auto_correct` | completed | 3 | 0.0 m² | 0.9971 | 0.9985 |
| 1.0 px | `gate_default` | completed | 3 | 29.0 m² | 0.9739 | 0.9868 |
| 1.0 px | `allow_misaligned` | completed | 3 | 29.0 m² | 0.9739 | 0.9868 |
| 1.0 px | `auto_correct` | completed | 3 | 0.0 m² | 0.9975 | 0.9987 |
| 2.0 px | `gate_default` | **blocked_by_gate** | 0 | 0.0 m² | 0.0 | 0.0 |
| 2.0 px | `allow_misaligned` | completed | 3 | 59.8 m² | 0.9522 | 0.9755 |
| 2.0 px | `auto_correct` | completed | 3 | 0.0 m² | 0.9971 | 0.9985 |
| 5.0 px | `gate_default` | **blocked_by_gate** | 0 | 0.0 m² | 0.0 | 0.0 |
| 5.0 px | `allow_misaligned` | completed | 5 | 304.5 m² | 0.8367 | 0.9111 |
| 5.0 px | `auto_correct` | completed | 3 | 0.0 m² | 0.9971 | 0.9985 |

*Note: All data in this table is measured on synthetic test pairs with exact ground truth; not a real-world accuracy claim.*

## Known Limitations

- No deep-learning detector in Phase 2: uses transparent baseline and spectral index detectors.
- Cloud, haze, and shadow screening is not performed; false alarms may occur in overcast imagery.
- Inputs must share the same pixel grid (same origin, GSD, and dimensions); the system refuses rather than silently reprojecting.
- Band order is never assumed; multispectral index mode requires explicit band mapping in config.
- Memory: scores and masks are held as float32 in RAM during smoothing.
