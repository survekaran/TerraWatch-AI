# Architecture & Stage Contracts (Phase 2)

## 1. System Overview

TerraWatch is a local, reproducible bi-temporal satellite change detection pipeline.
It processes two co-registered GeoTIFF rasters (`before.tif` and `after.tif`), evaluates their alignment, selects an appropriate detection policy, computes change scores, extracts polygons, tags artifact evidence, and assigns severity tiers.

```
Input Validator -> Alignment Gate -> Resolution Router -> Detector (Baseline / Spectral Index)
    -> Morphology + MMU -> Polygonizer -> Attribute Engine -> Artifact Tagger -> Severity Engine
    -> Exporter (+ provenance manifest, config snapshot) -> Streamlit Review UI / Offline Viewer
```

---

## 2. Stage Contracts & Data Flow

### Stage 1: Input Validator (`validator.py`)
- **Inputs**: File paths to `before` and `after` GeoTIFFs, `ValidationConfig`.
- **Contracts**:
  - Never resamples, reprojects, or assigns a CRS silently.
  - Verifies CRS existence and identity, grid alignment within pixel tolerance, bounding box overlap fraction, band count match, NoData fraction, and non-constant variance.
- **Output**: `ValidationReport` containing `RasterInfo` for each input.
  - `RasterInfo` includes: `crs`, `transform`, `width`, `height`, `count`, `dtype`, `nodata`, `valid_fraction`, `gsd_x_m`, `gsd_y_m`, `gsd_method`.

### Stage 2: Alignment Gate (`alignment.py`)
- **Inputs**: `before` and `after` paths, `AlignmentConfig`, GSD in metres.
- **Contracts**:
  - Memory-bounded: samples candidate windows using windowed reads; filters by valid fraction and Sobel gradient texture threshold.
  - Computes sub-pixel translation vector using `skimage.registration.phase_cross_correlation` on Sobel gradients.
  - Takes median across valid windows and measures inter-window spread.
  - **Sign Convention**: Returns translation vector `(shift_row_px, shift_col_px)` such that applying `scipy.ndimage.shift(after, (shift_row_px, shift_col_px))` aligns `after` with `before`.
  - On `status == "fail"`, raises `AlignmentError` by default, stopping the pipeline and writing failed provenance unless `allow_misaligned=True`.
  - Optional auto-correction: shifts `after` array, masks boundary-shifted pixels as NoData, writes `after_aligned.tif`, and re-measures residual shift without modifying original input bytes.
- **Output**: `AlignmentResult`.

### Stage 3: Resolution Router (`resolution_router.py`)
- **Inputs**: `RasterInfo` for both dates, `AppConfig`.
- **Contracts**:
  - Derives GSD strictly from affine transform and metric CRS via `estimate_gsd_m` (never from image dimensions).
  - Routes to:
    1. `high_res_rgb`: GSD <= 2.0m, count >= 3. Checks for learned weights via `resolve_learned_weights()`. When weights are absent, falls back to baseline detector with explicit reason `"learned detector unavailable"`.
    2. `multispectral_coarse`: 5m <= GSD <= 30m with required bands mapped. Routes to spectral index detector. Output granularity: `"built-up / land-cover change patches"`.
    3. `unsupported_or_ambiguous`: Single band, unknown GSD, or unmapped bands. Falls back to baseline detector with explicit warnings.
- **Output**: `RoutingDecision` (`policy`, `detector`, `reasons`, `warnings`, `output_granularity`).

### Stage 4: Change Detector (`baseline_detector.py` / `spectral.py`)
- **Inputs**: Raster inputs, `DetectionConfig` or `BandsConfig`.
- **Contracts**:
  - `baseline_difference`: Windowed root-mean-square difference over first N bands on normalised pixel values in [0, 1].
  - `spectral_index_difference`: Computes available physical indices (NDVI, NDBI, MNDWI) using explicit `band_map`. Never assumes band order. Combines absolute changes via root-mean-square. Generates signed delta rasters (`delta_ndvi`, `delta_ndbi`, `delta_exg`).
  - NoData-aware Gaussian smoothing via normalised convolution (`smooth_scores`).
  - Score thresholding (`apply_threshold`) using Otsu, fixed, or percentile thresholds.
- **Output**: `DetectionResult` (`scores`, `valid`, `mask`, `threshold`, `threshold_method`, `band_indices`, `detector`).

### Stage 5: Morphology & MMU (`postprocessing.py`)
- **Inputs**: Raw mask, valid mask, `PostprocessingConfig`, pixel area in m².
- **Contracts**: Binary opening (radius from config), binary closing, connected components, and removal of connected components smaller than MMU in square metres.
- **Output**: `CleanResult` (`mask`, `labels`, component counts).

### Stage 6: Polygonizer (`polygonizer.py`)
- **Inputs**: Region labels, affine transform, CRS.
- **Contracts**: Uses `rasterio.features.shapes`, shapely geometry repair (`make_valid`), and topology-preserving conservative simplification.
- **Output**: GeoDataFrame with repaired polygons in native scene CRS.

### Stage 7: Attributes Engine (`attributes.py` & `geometry_utils.py`)
- **Inputs**: Raw GeoDataFrame, region labels, change scores, detection threshold.
- **Contracts**:
  - Physical area and centroids are measured in metric projected CRS (scene CRS if metric, else local UTM zone) before reprojection.
  - `confidence`: Defined as margin-based detector support: `mean(clip((score - threshold) / (1 - threshold), 0, 1))` over region pixels. Uncalibrated support, not probability.
  - `change_magnitude`: Mean change score inside region.
- **Output**: GeoDataFrame with standard attribute schema.

### Stage 8: Artifact Tagger (`artifact_tagger.py`)
- **Inputs**: Attributed GeoDataFrame, region labels, change scores, input rasters, `TaggingConfig`, `AlignmentResult`, `RoutingDecision`, delta rasters.
- **Contracts**:
  - Extracts explainable geometric (compactness, extent, MRR width/length, elongation), spectral (delta luminance, chromatic change, delta vegetation index, delta built-up index), and contextual (NoData fraction, adjacent edge fraction) metrics.
  - Stores all metrics in `tag_evidence` JSON string.
  - Assigns primary `type_tag` and multi-tag string `tags` from closed vocabulary.
  - **Critical Rule**: Tags NEVER delete or discard polygons. There is no NDVI veto.
- **Output**: GeoDataFrame with `type_tag`, `tags`, and `tag_evidence`.

### Stage 9: Severity Engine (`severity.py`)
- **Inputs**: Tagged GeoDataFrame, `SeverityConfig`, alignment status, alignment override flag.
- **Contracts**:
  - Evaluates under `emergency` (recall-weighted) or `enforcement` (precision-weighted) modes.
  - Base tiers: `Low`, `Medium`, `Critical`.
  - Review override: any polygon with `misregistration_suspect`, `low_quality`, `model_disagreement`, or run-level alignment warning/failure/override is assigned tier `Review`.
  - Retains pre-override tier in `base_severity` and trigger reasons in `review_reasons`.
  - Assigns `quality_flag`.
- **Output**: GeoDataFrame with `severity_tier`, `base_severity`, `severity_mode`, `review_reasons`, `quality_flag`.

### Stage 10: Exporter & Provenance (`exporter.py` & `provenance.py`)
- **Inputs**: Run directory, GeoDataFrame, raster arrays, config, run metadata.
- **Contracts**:
  - GeoPackage (`change_polygons.gpkg`) in scene CRS.
  - GeoJSON (`change_polygons.geojson`) in EPSG:4326.
  - CSV summary (`change_summary.csv`).
  - GeoTIFF rasters (`change_scores.tif`, `change_mask.tif`).
  - Configuration snapshot (`config_snapshot.yaml`).
  - Complete provenance record (`provenance.json`) including input SHA-256, output SHA-256 manifest, software versions, parameter dictionaries, alignment details, routing decisions, and warnings.

---

## 3. Configuration Schema

Configuration is structured in `config/default.yaml` and validated by dataclasses in `src/geoai04/config.py`:

```yaml
detection:
  max_bands: 3
  uint8_scale: 255.0
  uint16_scale: 10000.0
  smoothing_sigma_px: 1.0
  threshold_method: otsu        # otsu | fixed | percentile
  fixed_threshold: 0.15
  percentile: 95.0
  otsu_floor: 0.05
  tile_size: 1024

validation:
  min_valid_fraction: 0.01
  min_overlap_fraction: 0.99
  grid_tolerance_px: 0.001
  gsd_rel_tolerance: 0.01

postprocessing:
  opening_radius_px: 1
  closing_radius_px: 2
  min_mapping_unit_m2: 25.0

polygons:
  connectivity: 8
  simplify_tolerance_px: 0.5

alignment:
  warn_px: 0.5
  fail_px: 2.0
  upsample_factor: 20
  n_windows: 9
  window_size_px: 128
  texture_threshold: 0.02
  min_windows: 2
  spread_warn_px: 1.0
  block_on_fail: true
  auto_correct: false
  interpolation: linear

routing:
  high_res_max_gsd_m: 2.0
  coarse_min_gsd_m: 5.0
  coarse_max_gsd_m: 30.0
  weights_path: null
  fallback_policy: baseline

bands:
  band_map: {}
  vegetation_proxy_exg: true

tagging:
  min_compactness_building: 0.35
  min_building_area_m2: 20.0
  max_building_area_m2: 2500.0
  veg_drop_threshold: 0.15
  built_rise_threshold: 0.08
  brightness_rise_threshold: 0.06
  fluctuation_min_area_m2: 200.0
  max_compactness_diffuse: 0.40
  misregistration_edge_threshold: 0.20
  misregistration_min_elongation: 2.5
  misregistration_width_factor_px: 2.5
  low_quality_nodata_fraction: 0.15
  disagreement_iou_threshold: 0.40

severity:
  mode: emergency
  area_rule: absolute
  min_polygons_for_percentile: 10
  percentile_critical: 90.0
  percentile_medium: 60.0
  emergency_crit_conf: 0.60
  emergency_med_conf: 0.50
  emergency_critical_area_m2: 500.0
  emergency_medium_area_m2: 100.0
  enforcement_crit_conf: 0.85
  enforcement_med_conf: 0.75
  enforcement_critical_area_m2: 1000.0
  enforcement_medium_area_m2: 200.0
  context_layers: []
```
