"""End-to-end Phase 1 pipeline: validate, detect (baseline), clean, polygonize, attribute, export.

Phase 1 does NOT include the alignment gate, resolution router, artifact tagger,
severity tiers or a learned detector. The provenance file states this explicitly.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import rasterio

from . import __version__
from .attributes import build_attributes, file_sha256, filter_by_area, pair_id
from .baseline_detector import DETECTOR_NAME, DetectionResult, run_baseline
from .config import AppConfig, load_config
from .exporter import write_rasters, write_vectors
from .geometry_utils import pixel_area_m2
from .polygonizer import polygonize, simplify
from .postprocessing import CleanResult, clean_mask
from .provenance import write_provenance
from .validator import InputValidationError, ValidationReport, validate_pair

log = logging.getLogger("geoai04.pipeline")

PHASE1_NOT_IMPLEMENTED = [
    "Alignment gate (phase correlation) is not implemented in Phase 1; co-registration is only checked at the metadata/grid level.",
    "Resolution router is not implemented in Phase 1; the baseline detector is always used.",
    "Learned detector is not available; no AI inference was performed.",
    "Artifact tagging and severity tiers are not implemented in Phase 1.",
]


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
    elapsed_s: float = 0.0


def _file_meta(path: Path) -> dict:
    return {"filename": path.name, "path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def run_pipeline(before_path: str | Path, after_path: str | Path, cfg: AppConfig | None = None,
                 out_root: str | Path = "outputs", run_name: str | None = None,
                 input_notes: str | None = None) -> RunResult:
    """Run the Phase 1 pipeline and write all outputs into a fresh run directory."""
    t0 = time.perf_counter()
    cfg = cfg or load_config()
    before_path, after_path = Path(before_path), Path(after_path)
    try:
        pid = pair_id(before_path, after_path)
    except OSError:
        pid = "unavailable"
    run_dir = Path(out_root) / (run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pid}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("geoai04")
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        log.info("Run started: before=%s after=%s pair_id=%s", before_path, after_path, pid)
        report = validate_pair(before_path, after_path, cfg.validation)
        (run_dir / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        base_record = {
            "tool": "GEOAI 04 change-intelligence prototype", "phase": 1, "source_pair_id": pid,
            "input_notes": input_notes,
            "parameters": cfg.to_dict(),
            "not_implemented": PHASE1_NOT_IMPLEMENTED,
            "validation": report.to_dict(),
        }
        if not report.ok:
            log.error("Validation failed: %s", report.errors)
            write_provenance(run_dir / "provenance.json", {**base_record, "status": "failed_validation", "warnings": report.warnings, "errors": report.errors})
            raise InputValidationError(report, run_dir)

        with rasterio.open(before_path) as src:
            crs, transform, width, height = src.crs, src.transform, src.width, src.height
        px_area = pixel_area_m2(crs, transform, width, height)

        det = run_baseline(before_path, after_path, cfg.detection)
        clean = clean_mask(det.mask, det.valid, cfg.postprocessing, px_area, cfg.polygons.connectivity)
        raw = polygonize(clean.labels, transform, crs, cfg.polygons.connectivity)
        attributed = build_attributes(raw, clean.labels, det.scores, det.threshold, transform, width, height, pid, DETECTOR_NAME)
        n_before_filter = len(attributed)
        final = filter_by_area(attributed, cfg.postprocessing.min_mapping_unit_m2)
        final = simplify(final, cfg.polygons.simplify_tolerance_px * abs(transform.a))

        outputs = {}
        outputs.update(write_rasters(run_dir, det.scores, clean.mask, det.valid, crs, transform))
        outputs.update(write_vectors(run_dir, final))

        counts = {
            "valid_pixel_fraction": float(det.valid.mean()),
            "threshold": det.threshold, "threshold_method": det.threshold_method,
            "pixel_area_m2": px_area, "mmu_pixels": clean.mmu_pixels,
            "components_raw": clean.components_raw,
            "components_after_morphology": clean.components_after_morphology,
            "components_after_mmu": clean.components_after_mmu,
            "polygons_before_area_filter": n_before_filter,
            "polygons_after_area_filter": len(final),
            "total_changed_area_m2": float(final["area_m2"].sum()) if len(final) else 0.0,
        }
        warnings = list(report.warnings)
        if len(final) == 0:
            warnings.append("No change polygons were produced with the current parameters.")
        elapsed = time.perf_counter() - t0
        outputs["provenance"] = "provenance.json"
        outputs["run_log"] = "run.log"
        record = {
            **base_record, "status": "completed", "warnings": warnings,
            "inputs": {"before": _file_meta(before_path), "after": _file_meta(after_path)},
            "alignment": {"status": "not_run", "note": "Alignment gate arrives in Phase 2."},
            "routing": {"policy": "phase1_baseline_only"},
            "detector": {"name": DETECTOR_NAME, "learned_model_used": False, "weights": None,
                         "bands_used": det.band_indices, "description": "Thresholded image difference; not AI inference."},
            "projected_crs_for_measurements": final["projected_crs"].iloc[0] if len(final) else None,
            "counts": counts, "outputs": outputs, "elapsed_seconds": round(elapsed, 3),
            "version": __version__,
        }
        write_provenance(run_dir / "provenance.json", record)
        log.info("Run completed in %.2fs: %d polygon(s), %.1f m2", elapsed, len(final), counts["total_changed_area_m2"])
        return RunResult(run_dir=run_dir, status="completed", validation=report, counts=counts, threshold=det.threshold,
                         polygons=final, outputs=outputs, warnings=warnings, detection=det, clean=clean, elapsed_s=elapsed)
    except InputValidationError:
        raise
    except Exception:
        log.exception("Run failed")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(prev_level)
