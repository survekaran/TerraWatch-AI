"""Severity evaluation engine with emergency and enforcement operational modes.

Assigns severity tiers (Low, Medium, Critical, Review) based on policy-driven thresholds.
NOTE: Thresholds are an unvalidated policy choice and do not confirm legality,
damage, or physical impact.

Modes:
  - emergency: recall-weighted, lower confidence & area thresholds.
  - enforcement: precision-weighted, higher confidence & area thresholds.

Review override:
  Any polygon with tags misregistration_suspect, low_quality, model_disagreement,
  or a run-level alignment warning/failure/override is assigned tier 'Review'.
  Original base tier is retained in 'base_severity' and reasons in 'review_reasons'.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from .alignment import AlignmentResult
from .config import SeverityConfig

log = logging.getLogger(__name__)

SEVERITY_TIERS = ("Low", "Medium", "Critical", "Review")


def evaluate_severity(
    polygons: gpd.GeoDataFrame,
    cfg: SeverityConfig,
    alignment_result: AlignmentResult | None = None,
    alignment_override: bool = False,
) -> gpd.GeoDataFrame:
    """Assign severity_tier, base_severity, severity_mode, review_reasons, and quality_flag."""
    if polygons.empty:
        out = polygons.copy()
        out["severity_tier"] = []
        out["base_severity"] = []
        out["severity_mode"] = []
        out["review_reasons"] = []
        out["quality_flag"] = []
        out["context_hits"] = []
        return out

    out = polygons.copy()
    mode = cfg.mode.lower()
    if mode not in {"emergency", "enforcement"}:
        raise ValueError(f"Unknown severity mode {mode!r}; must be emergency or enforcement")

    # Select base thresholds
    if mode == "emergency":
        crit_conf = cfg.emergency_crit_conf
        med_conf = cfg.emergency_med_conf
        crit_area = cfg.emergency_critical_area_m2
        med_area = cfg.emergency_medium_area_m2
    else:
        crit_conf = cfg.enforcement_crit_conf
        med_conf = cfg.enforcement_med_conf
        crit_area = cfg.enforcement_critical_area_m2
        med_area = cfg.enforcement_medium_area_m2

    areas = out["area_m2"].to_numpy()
    confidences = out["confidence"].to_numpy()

    # Percentile area rule handling
    if cfg.area_rule == "percentile":
        if len(out) >= cfg.min_polygons_for_percentile:
            crit_area = float(np.percentile(areas, cfg.percentile_critical))
            med_area = float(np.percentile(areas, cfg.percentile_medium))
            log.info("Severity area rule percentile: med=%.1f m2, crit=%.1f m2", med_area, crit_area)
        else:
            log.info(
                "Fewer than %d polygons (%d); falling back from percentile to absolute area thresholds",
                cfg.min_polygons_for_percentile, len(out)
            )

    # Context layers (stretch feature)
    context_hits_per_poly = [[] for _ in range(len(out))]
    if cfg.context_layers:
        for layer_path_str in cfg.context_layers:
            lp = Path(layer_path_str)
            if lp.is_file():
                try:
                    ctx_gdf = gpd.read_file(lp)
                    if ctx_gdf.crs != out.crs:
                        ctx_gdf = ctx_gdf.to_crs(out.crs)
                    inter = gpd.sjoin(out, ctx_gdf, how="inner", predicate="intersects")
                    for poly_idx in inter.index.unique():
                        context_hits_per_poly[poly_idx].append(lp.name)
                except Exception as exc:
                    log.warning("Could not process context layer %s: %exc", lp, exc)

    base_severities = []
    final_severities = []
    review_reasons_list = []
    quality_flags = []
    context_hits_strings = []

    run_align_status = alignment_result.status if alignment_result else "pass"
    run_has_align_issue = run_align_status in {"warn", "fail"}

    for idx, row in out.iterrows():
        area = float(areas[idx])
        conf = float(confidences[idx])
        tags_str = str(row.get("tags", ""))
        tags = set(tags_str.split(";")) if tags_str else set()

        # Compute base severity
        if conf >= crit_conf and area >= crit_area:
            tier = "Critical"
        elif conf >= med_conf and area >= med_area:
            tier = "Medium"
        else:
            tier = "Low"

        # Context layer tier bump (never above Critical)
        hits = context_hits_per_poly[idx]
        if hits and tier == "Low":
            tier = "Medium"
        elif hits and tier == "Medium":
            tier = "Critical"

        base_severities.append(tier)
        context_hits_strings.append(";".join(hits) if hits else "")

        # Review override triggers
        triggers = []
        flags = []

        if alignment_override:
            triggers.append("run_alignment_override")
            flags.append("alignment_override")
        elif run_has_align_issue:
            triggers.append(f"run_alignment_{run_align_status}")
            flags.append(f"alignment_{run_align_status}")

        if "misregistration_suspect" in tags:
            triggers.append("tag_misregistration_suspect")
            flags.append("misregistration_suspect")
        if "low_quality" in tags:
            triggers.append("tag_low_quality")
            flags.append("low_quality")
        if "model_disagreement" in tags:
            triggers.append("tag_model_disagreement")
            flags.append("model_disagreement")

        if triggers:
            final_tier = "Review"
            review_reasons_list.append(";".join(triggers))
        else:
            final_tier = tier
            review_reasons_list.append("")

        quality_flags.append(";".join(flags) if flags else "nominal")
        final_severities.append(final_tier)

    out["severity_tier"] = final_severities
    out["base_severity"] = base_severities
    out["severity_mode"] = mode
    out["review_reasons"] = review_reasons_list
    out["quality_flag"] = quality_flags
    out["context_hits"] = context_hits_strings

    log.info(
        "Severity evaluation (%s mode): Low=%d, Medium=%d, Critical=%d, Review=%d",
        mode,
        final_severities.count("Low"),
        final_severities.count("Medium"),
        final_severities.count("Critical"),
        final_severities.count("Review"),
    )
    return out
