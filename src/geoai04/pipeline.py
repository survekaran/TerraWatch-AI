"""End-to-end Phase 2 change detection pipeline.

Pipeline stage order:
validate -> alignment gate -> router -> detector (baseline or spectral index) ->
clean_mask -> polygonize -> build_attributes -> artifact tagger -> severity engine ->
filter/simplify -> export (+ provenance, config snapshot, run log).
"""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml

from . import __version__
from .alignment import (
    AlignmentError,
    AlignmentResult,
    apply_alignment_correction,
    estimate_alignment,
)
from .artifact_tagger import tag_polygons
from .attributes import build_attributes, file_sha256, filter_by_area, pair_id
from .baseline_detector import DETECTOR_NAME, DetectionResult, run_baseline
from .config import AppConfig, load_config
from .exporter import write_rasters, write_vectors
from .geometry_utils import pixel_area_m2
from .polygonizer import polygonize, simplify
from .postprocessing import CleanResult, clean_mask
from .provenance import write_provenance
from .resolution_router import RoutingDecision, route_inputs
from .severity import evaluate_severity
from .spectral import SPECTRAL_DETECTOR_NAME, run_spectral_detector
from .validator import InputValidationError, ValidationReport, validate_pair

log = logging.getLogger("geoai04.pipeline")

BASELINE_NOT_IMPLEMENTED = [
    "Cloud and shadow screening is not implemented; cloud_shadow_screening='not_performed'.",
]
PHASE3_NOT_IMPLEMENTED = BASELINE_NOT_IMPLEMENTED
PHASE2_NOT_IMPLEMENTED = BASELINE_NOT_IMPLEMENTED
PHASE1_NOT_IMPLEMENTED = BASELINE_NOT_IMPLEMENTED


@dataclass
class RunResult:
    run_dir: Path
    status: str
    validation: ValidationReport
    counts: dict
    threshold: float
    polygons: gpd.GeoDataFrame
    outputs: dict
    warnings: list[str] = field(default_factory=list)
    detection: DetectionResult | None = None
    clean: CleanResult | None = None
    alignment: AlignmentResult | None = None
    routing: RoutingDecision | None = None
    residual_alignment: AlignmentResult | None = None
    elapsed_s: float = 0.0


def _file_meta(path: Path) -> dict:
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _manifest(run_dir: Path) -> dict[str, dict]:
    manifest = {}
    for p in sorted(run_dir.glob("*")):
        if p.is_file() and p.name != "provenance.json":
            manifest[p.name] = {
                "size_bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
    return manifest


def run_pipeline(
    before_path: str | Path,
    after_path: str | Path,
    cfg: AppConfig | None = None,
    out_root: str | Path = "outputs",
    run_name: str | None = None,
    input_notes: str | None = None,
    mode: str = "emergency",
    allow_misaligned: bool = False,
    auto_correct: bool | None = None,
    band_map: dict | None = None,
    other_mask: Any = None,
) -> RunResult:
    """Run the Phase 2 pipeline and write all outputs into a fresh run directory."""
    t0 = time.perf_counter()
    cfg = copy.deepcopy(cfg) if cfg is not None else load_config()
    if band_map is not None:
        cfg.bands.band_map = band_map
    if mode:
        cfg.severity.mode = mode

    before_path, after_path = Path(before_path), Path(after_path)
    try:
        pid = pair_id(before_path, after_path)
    except OSError:
        pid = "unavailable"

    run_dir = Path(out_root) / (run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pid}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration snapshot
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    config_yaml_str = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    config_snapshot_path.write_text(config_yaml_str, encoding="utf-8")
    config_hash = file_sha256(config_snapshot_path)

    logger = logging.getLogger("geoai04")
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        log.info("Phase 2 run started: before=%s after=%s pair_id=%s mode=%s", before_path, after_path, pid, mode)

        # STAGE 1: Input Validation
        report = validate_pair(before_path, after_path, cfg.validation)
        (run_dir / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        base_record = {
            "tool": "GEOAI 04 change-intelligence prototype",
            "phase": 2,
            "source_pair_id": pid,
            "input_notes": input_notes,
            "severity_mode": mode,
            "cloud_shadow_screening": "not_performed",
            "config_hash_sha256": config_hash,
            "parameters": cfg.to_dict(),
            "not_implemented": PHASE2_NOT_IMPLEMENTED,
            "validation": report.to_dict(),
            "inputs": {
                "before": _file_meta(before_path) if before_path.is_file() else None,
                "after": _file_meta(after_path) if after_path.is_file() else None,
            },
        }

        if not report.ok:
            log.error("Validation failed: %s", report.errors)
            write_provenance(
                run_dir / "provenance.json",
                {
                    **base_record,
                    "status": "failed_validation",
                    "warnings": report.warnings,
                    "errors": report.errors,
                },
            )
            raise InputValidationError(report, run_dir)

        # Extract GSD from validator's RasterInfo (derived from affine transform and CRS)
        gsd_x_m = report.before.gsd_x_m if report.before else None
        gsd_y_m = report.before.gsd_y_m if report.before else None

        # STAGE 2: Alignment Gate
        align_cfg = cfg.alignment
        align_res = estimate_alignment(
            before_path, after_path, cfg=align_cfg, gsd_x_m=gsd_x_m, gsd_y_m=gsd_y_m
        )

        effective_after_path = after_path
        residual_alignment: AlignmentResult | None = None
        should_auto_correct = auto_correct if auto_correct is not None else align_cfg.auto_correct

        if should_auto_correct and align_res.shift_row_px is not None:
            corrected_path = run_dir / "after_aligned.tif"
            apply_alignment_correction(
                after_path,
                corrected_path,
                align_res.shift_row_px,
                align_res.shift_col_px,
                interpolation=align_cfg.interpolation,
            )
            effective_after_path = corrected_path
            residual_alignment = estimate_alignment(
                before_path, effective_after_path, cfg=align_cfg, gsd_x_m=gsd_x_m, gsd_y_m=gsd_y_m
            )
            log.info(
                "Auto-correction performed: initial shift=%.2f px, residual shift=%.2f px",
                align_res.magnitude_px or 0.0,
                residual_alignment.magnitude_px or 0.0,
            )

        active_align = residual_alignment if residual_alignment is not None else align_res

        # Check for alignment failure
        if active_align.status == "fail":
            if align_cfg.block_on_fail and not allow_misaligned:
                log.error("Alignment gate failed: %s", active_align.reasons)
                write_provenance(
                    run_dir / "provenance.json",
                    {
                        **base_record,
                        "status": "failed_alignment",
                        "alignment": {
                            **active_align.to_dict(),
                            "allow_misaligned": False,
                            "auto_corrected": should_auto_correct,
                        },
                        "warnings": report.warnings + active_align.reasons,
                        "errors": active_align.reasons,
                    },
                )
                raise AlignmentError(active_align, run_dir)
            else:
                log.warning("Misaligned pair override: proceeding with allow_misaligned=True")

        # STAGE 3: Resolution Router
        routing_dec = route_inputs(report.before, report.after, cfg)
        log.info("Routing decision: policy=%s detector=%s", routing_dec.policy, routing_dec.detector)

        # STAGE 4: Detector (Baseline or Spectral Index)
        with rasterio.open(before_path) as src:
            crs, transform, width, height = src.crs, src.transform, src.width, src.height
        px_area = pixel_area_m2(crs, transform, width, height)

        delta_rasters = None
        if routing_dec.detector == SPECTRAL_DETECTOR_NAME:
            det, delta_rasters = run_spectral_detector(before_path, effective_after_path, cfg)
        else:
            det = run_baseline(before_path, effective_after_path, cfg.detection)

        computed_other_mask = other_mask

        # STAGE 5: Morphology and Connected Components
        clean = clean_mask(
            det.mask, det.valid, cfg.postprocessing, px_area, cfg.polygons.connectivity
        )

        # STAGE 6: Polygonize
        raw = polygonize(clean.labels, transform, crs, cfg.polygons.connectivity)

        # STAGE 7: Attributes
        attributed = build_attributes(
            raw, clean.labels, det.scores, det.threshold, transform, width, height, pid, det.detector
        )

        # STAGE 8: Artifact Tagger
        tagged = tag_polygons(
            attributed,
            clean.labels,
            det.scores,
            before_path,
            effective_after_path,
            cfg=cfg.tagging,
            alignment_result=active_align,
            routing_decision=routing_dec,
            delta_rasters=delta_rasters,
            other_mask=computed_other_mask,
        )

        # Run-level sanity check: implausible change fraction
        valid_px = int(det.valid.sum())
        changed_px = int(clean.mask.sum())
        changed_fraction = float(changed_px / max(1, valid_px))

        if mode == "enforcement":
            implausible_threshold = getattr(
                cfg.severity, "implausible_change_fraction_enforcement", cfg.severity.implausible_change_fraction
            )
        else:
            implausible_threshold = getattr(
                cfg.severity, "implausible_change_fraction_emergency", cfg.severity.implausible_change_fraction
            )
        is_implausible = changed_fraction > implausible_threshold

        # STAGE 9: Severity Engine
        with_severity = evaluate_severity(
            tagged,
            cfg=cfg.severity,
            alignment_result=active_align,
            alignment_override=allow_misaligned,
            is_implausible_change=is_implausible,
        )

        # STAGE 10: MMU Area Filter and Simplification
        n_before_filter = len(with_severity)
        final = filter_by_area(with_severity, cfg.postprocessing.min_mapping_unit_m2)
        final = simplify(final, cfg.polygons.simplify_tolerance_px * abs(transform.a))

        # STAGE 11: Raster and Vector Exports
        outputs = {}
        outputs.update(write_rasters(run_dir, det.scores, clean.mask, det.valid, crs, transform))
        outputs.update(write_vectors(run_dir, final))
        outputs["config_snapshot"] = "config_snapshot.yaml"
        outputs["provenance"] = "provenance.json"
        outputs["run_log"] = "run.log"
        outputs["validation_report"] = "validation_report.json"
        if effective_after_path != after_path:
            outputs["after_aligned"] = effective_after_path.name

        counts = {
            "valid_pixel_fraction": float(det.valid.mean()),
            "changed_fraction": round(changed_fraction, 4),
            "implausible_change_fraction": is_implausible,
            "threshold": det.threshold,
            "threshold_method": det.threshold_method,
            "pixel_area_m2": px_area,
            "mmu_pixels": clean.mmu_pixels,
            "components_raw": clean.components_raw,
            "components_after_morphology": clean.components_after_morphology,
            "components_after_mmu": clean.components_after_mmu,
            "polygons_before_area_filter": n_before_filter,
            "polygons_after_area_filter": len(final),
            "total_changed_area_m2": float(final["area_m2"].sum()) if len(final) else 0.0,
        }

        warnings = list(report.warnings) + list(routing_dec.warnings)
        if is_implausible:
            warn_msg = (
                f"Implausible change fraction ({mode} mode): {changed_fraction:.1%} of valid scene area is flagged as changed "
                f"(threshold: {implausible_threshold:.1%}). Flagging all detections as Review."
            )
            log.warning(warn_msg)
            warnings.append(warn_msg)
        if allow_misaligned and active_align.status in {"warn", "fail"}:
            warnings.append("Misaligned pair override: results will be unreliable.")
        if len(final) == 0:
            warnings.append("No change polygons were produced with the current parameters.")

        elapsed = time.perf_counter() - t0

        rad_norm_active = bool(
            getattr(cfg.detection, "radiometric_normalization", False)
            and getattr(cfg.detection, "radiometric_norm_method", "mean_std") != "off"
        )
        norm_method_recorded = (
            getattr(cfg.detection, "radiometric_norm_method", "mean_std")
            if rad_norm_active
            else "off"
        )

        # STAGE 12: Provenance record with output SHA-256 manifest
        record = {
            **base_record,
            "status": "completed",
            "warnings": warnings,
            "alignment": {
                **active_align.to_dict(),
                "initial_alignment": align_res.to_dict() if residual_alignment else None,
                "allow_misaligned": allow_misaligned,
                "auto_correct_performed": should_auto_correct,
            },
            "routing": routing_dec.to_dict(),
            "detector": {
                "name": det.detector,
                "bands_used": det.band_indices,
                "indices_used": list(delta_rasters.keys()) if delta_rasters else None,
                "radiometric_normalization": rad_norm_active,
                "radiometric_norm_method": norm_method_recorded,
                "description": (
                    "Spectral index difference"
                    if det.detector == SPECTRAL_DETECTOR_NAME
                    else "Thresholded image difference; RMS baseline analysis."
                ),
            },
            "projected_crs_for_measurements": final["projected_crs"].iloc[0] if len(final) else None,
            "counts": counts,
            "outputs": outputs,
            "manifest_sha256": _manifest(run_dir),
            "elapsed_seconds": round(elapsed, 3),
            "version": __version__,
        }
        write_provenance(run_dir / "provenance.json", record)

        log.info(
            "Pipeline run completed in %.2fs: %d polygon(s), %.1f m2",
            elapsed, len(final), counts["total_changed_area_m2"]
        )

        return RunResult(
            run_dir=run_dir,
            status="completed",
            validation=report,
            counts=counts,
            threshold=det.threshold,
            polygons=final,
            outputs=outputs,
            warnings=warnings,
            detection=det,
            clean=clean,
            alignment=align_res,
            routing=routing_dec,
            residual_alignment=residual_alignment,
            elapsed_s=elapsed,
        )

    except (InputValidationError, AlignmentError):
        raise
    except Exception:
        log.exception("Run failed")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(prev_level)
