# Alignment Gate, Artifact Tagging & Severity Rules

## 1. Alignment Gate

### Mathematical Basis
The alignment gate evaluates sub-pixel relative translation between BEFORE (T1) and AFTER (T2) images using phase cross-correlation (`skimage.registration.phase_cross_correlation`) on Sobel gradient representations. The Sobel gradient isolates high-frequency spatial texture and suppresses broad illumination or atmospheric differences.

### Windowed Sampling & Texture Gate
To maintain a bounded memory footprint, rasters are never loaded in bulk for registration:
1. A spatial grid of up to $N$ windows (`n_windows: 9`, size `window_size_px: 128`) is sampled across the scene.
2. Windows must have valid data fraction $\ge 0.85$.
3. Gradients must exceed `texture_threshold: 0.02` to filter out flat, homogeneous areas (water bodies, bare sand, cloud-free sky).
4. Sub-pixel shift is estimated per window; the scene-level estimate is taken as the **median** shift across all textured windows.
5. Inter-window spread (standard deviation of window shifts from the median) is measured. If spread exceeds `spread_warn_px: 1.0`, a warning is logged that genuine physical change or parallax may contaminate the estimate.
6. If fewer than `min_windows: 2` textured windows exist, status is returned as `"unknown"` with an explicit reason rather than inventing a shift number.

### Sign Convention
The estimated translation vector `(shift_row_px, shift_col_px)` represents the translation vector to apply to the AFTER image to register it with the BEFORE image:
$$\text{registered\_after}(r, c) = \text{after}(r - \text{shift\_row\_px}, c - \text{shift\_col\_px})$$

- If physical features in AFTER appear displaced by $(+\Delta r, +\Delta c)$ pixels relative to BEFORE, the returned correction vector is $(-\Delta r, -\Delta c)$.
- Applying `scipy.ndimage.shift(after, (shift_row_px, shift_col_px))` aligns AFTER with BEFORE.

### Thresholds & Actions
- **Pass** ($\text{magnitude} < 0.5\text{ px}$): Pipeline proceeds normally.
- **Warn** ($0.5\text{ px} \le \text{magnitude} < 2.0\text{ px}$ or spread $> 1.0\text{ px}$): Pipeline proceeds with warnings; polygons flagged for Review.
- **Fail** ($\text{magnitude} \ge 2.0\text{ px}$): `AlignmentError` is raised by default; run stops; failed provenance is written.
- **Override**: Passing `allow_misaligned=True` (or checking the UI override) allows execution, marking all polygons as `Review` with `quality_flag="alignment_override"`.
- **Auto-Correction**: When enabled (`auto_correct=True`), applies sub-pixel interpolation to AFTER, writes `after_aligned.tif`, masks edge-shifted pixels as NoData, re-measures residual shift, and leaves the original input byte-identical (verified by SHA-256).

---

## 2. Artifact Tagger

The tagger computes explainable geometric, spectral, and contextual evidence for every candidate polygon and assigns a single primary `type_tag` and a semicolon-separated multi-tag `tags` string.

### Vocabulary (Strictly Enforced)
- `building_like`: Compact polygon within building area bounds on high-resolution imagery.
- `built_up_patch`: Coarse multispectral patch exhibiting built-up or brightness increases.
- `vegetation_conversion`: Real encroachment: vegetation evidence decreased while built-up evidence or brightness increased, or a compact structure appeared.
- `vegetation_fluctuation`: Diffuse seasonal change: large area or low compactness without brightness/built-up increase.
- `misregistration_suspect`: Thin, elongated polygon along strong pre-existing edges, or associated with alignment warnings.
- `low_quality`: High NoData fraction or flagged sensor data.
- `model_disagreement`: Overlap with an alternative detector mask below the IoU threshold.
- `unknown`: Insufficient evidence to justify a specific tag.

### Evidence Metrics (stored in `tag_evidence` JSON)
- `compactness`: Isoperimetric quotient $4\pi A / P^2 \in [0, 1]$.
- `extent`: Ratio of polygon area to minimum rotated rectangle area $\in [0, 1]$.
- `min_rot_rect_width_m`, `min_rot_rect_length_m`: Minimum rotated bounding box dimensions in metres.
- `elongation`: Length / width ratio.
- `mean_change_score`: Mean detector score within the polygon.
- `brightness_change`: Mean luminance change (T2 - T1).
- `chromatic_change`: Mean colour difference vector norm.
- `delta_vegetation_index`: Change in NDVI (or proxy ExG).
- `delta_built_up_index`: Change in NDBI.
- `nodata_fraction`: Fraction of invalid pixels inside polygon.
- `adjacent_edge_fraction`: Fraction of pixels coinciding with pre-existing edges in T1.
- `other_mask_iou`: Spatial IoU against secondary detector mask (`"not_evaluated"` if none supplied).

### Tag Precedence Order (for primary `type_tag`)
1. `low_quality`
2. `misregistration_suspect`
3. `model_disagreement`
4. `vegetation_fluctuation`
5. `vegetation_conversion`
6. `building_like`
7. `built_up_patch`
8. `unknown`

### What the Tagger CANNOT Do
- **NO Cloud or Shadow Screening**: No cloud detector exists in this data pipeline; `cloud_shadow_screening: "not_performed"` is recorded in provenance.
- **NO Legal Claims**: Detections are candidate changes, never confirmed violations or illegal encroachments.
- **NO NDVI Veto**: Candidate polygons are NEVER deleted due to vegetation indices. Construction on green land causes an NDVI drop that must remain visible.

---

## 3. Severity Engine

The severity engine translates detections into operational priorities. All thresholds are unvalidated policy parameters defined in `config/default.yaml`.

### Modes
- **Emergency mode**: Recall-weighted (lower confidence and area thresholds).
  - Critical: `confidence >= 0.60` and `area_m2 >= 500.0`
  - Medium: `confidence >= 0.50` and `area_m2 >= 100.0`
  - Low: All other detections above MMU
- **Enforcement mode**: Precision-weighted (higher confidence and area thresholds).
  - Critical: `confidence >= 0.85` and `area_m2 >= 1000.0`
  - Medium: `confidence >= 0.75` and `area_m2 >= 200.0`
  - Low: All other detections above MMU

### Review Override
Regardless of base tier, any polygon triggered by:
- `misregistration_suspect`
- `low_quality`
- `model_disagreement`
- Run-level alignment warning, failure, or override

is assigned `severity_tier: "Review"`. The pre-override tier is preserved in `base_severity` and reasons are listed in `review_reasons`.
