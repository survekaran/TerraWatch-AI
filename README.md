# TerraWatch: Satellite Change Detection

A local, modular bi-temporal change detection pipeline comparing co-registered GeoTIFFs, performing sub-pixel alignment verification, routing based on resolution and band availability, detecting physical change with transparent baseline (RMS differencing) and spectral index detectors, tagging polygon evidence, evaluating emergency/enforcement severity tiers, and exporting fully attributed GIS polygons (GeoPackage + GeoJSON) with complete provenance.

**Operational Highlights:**
- **Analysis Methodology**: Transparent, deterministic baseline analysis (Root-Mean-Square image differencing and multi-spectral index differencing). No neural network checkpoints or GPU dependencies.
- **Sub-Pixel Alignment Gate**: Phase cross-correlation registration verification on Sobel gradients; optional sub-pixel auto-correction.
- **Resolution-Aware Routing**: Routes high-res optical imagery (GSD <= 2.0 m) to baseline differencing (`individual_buildings`), and multispectral imagery to spectral index differencing (`built-up / land-cover change patches`).
- **Explainable Evidence**: Geometric, spectral, and contextual features calculated per change region without destructive filtering.
- **Offline & Local**: 100% self-contained offline HTML viewer and interactive Streamlit application. No cloud dependencies or internet access required.

## Setup (Python 3.11 or 3.12)

macOS / Linux:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Windows (PowerShell):

    py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Quickstart Commands

```bash
# 1. Run all automated tests
python -m pytest

# 2. Run boundary condition and failure demonstrations
python scripts/failure_demo.py --out-dir outputs/failure_demos

# 3. Run headless CLI pipeline
python scripts/run_pipeline.py --before data/sample/before.tif --after data/sample/after.tif --mode emergency --out outputs

# 4. Launch local interactive Streamlit app (100% offline)
streamlit run app.py
```

## Architecture

```
Input Validator -> Alignment Gate -> Resolution Router -> Detector (Baseline / Spectral Index)
    -> Morphology + MMU -> Polygonizer -> Attribute Engine -> Artifact Tagger -> Severity Engine
    -> Exporter (+ provenance manifest, config snapshot) -> Streamlit Review UI / Offline Viewer
```

| Module | File | Responsibility |
|---|---|---|
| Config | `config.py`, `config/default.yaml` | All thresholds live here; dataclasses validate inputs and unknown keys are rejected |
| Validation | `validator.py` | CRS, grid, overlap, band count, GSD, NoData, constant image; never resamples silently |
| Alignment Gate | `alignment.py` | Sub-pixel phase cross-correlation on Sobel gradients across windowed reads; auto-correction |
| Router | `resolution_router.py` | High-res RGB vs multispectral coarse vs ambiguous fallback |
| Spectral Indices | `spectral.py` | Safe NDVI, NDBI, MNDWI, NDWI, vegetation_proxy_exg; spectral index difference detector |
| Baseline Detector | `baseline_detector.py` | Windowed RMS difference over first N bands; NoData-aware smoothing; Otsu/fixed/percentile |
| Postprocessing | `postprocessing.py` | Binary opening, closing, connected components, minimum mapping unit (m²) |
| Polygonizer | `polygonizer.py` | `rasterio.features.shapes`, polygon repair, conservative simplification |
| Attributes | `attributes.py`, `geometry_utils.py` | Metric CRS area/centroids; informative margin-based detector confidence |
| Artifact Tagger | `artifact_tagger.py` | Computes explainable evidence; tags `building_like`, `built_up_patch`, `vegetation_conversion`, `vegetation_fluctuation`, `misregistration_suspect`, `low_quality`, `unknown` |
| Severity Engine | `severity.py` | Emergency (recall) vs Enforcement (precision) modes; Low/Medium/Critical tiers; Review override |
| Export & Provenance | `exporter.py`, `provenance.py` | GPKG, GeoJSON, CSV, rasters, config snapshot, and provenance with output SHA-256 manifest |
| Viewer | `viewer.py` | 100% offline self-contained HTML: pan/zoom, slider, heatmap, inspection, review Blob export |
| Pipeline & CLI | `pipeline.py`, `scripts/run_pipeline.py` | Orchestrates complete pipeline, logging, and error handling |

## Output Structure

```
outputs/run_<timestamp>_<pair_id>/
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
