"""Resolution-aware routing engine for bi-temporal change detection.

Determines the processing policy and detector based on Ground Sample Distance (GSD),
band counts, and configured band maps.

GSD is derived strictly from the affine transform and the metric projected CRS
(via estimate_gsd_m in geometry_utils), NEVER from raw raster dimensions.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from .config import AppConfig
from .validator import RasterInfo

log = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    policy: str  # "high_res_rgb" | "multispectral_coarse" | "unsupported_or_ambiguous"
    detector: str  # "baseline_difference" | "spectral_index_difference"
    reasons: list[str]
    warnings: list[str]
    output_granularity: str  # "individual_buildings" | "built-up / land-cover change patches" | "candidate_change_regions"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def route_inputs(
    before_info: RasterInfo,
    after_info: RasterInfo,
    cfg: AppConfig | None = None,
) -> RoutingDecision:
    """Determine the change detection route and detector.

    Policy selection:
    1. high_res_rgb: GSD <= high_res_max_gsd_m and >= 3 bands.
       Routes to baseline difference detector.
       Granularity: "individual_buildings".
    2. multispectral_coarse: coarse_min_gsd_m <= GSD <= coarse_max_gsd_m and required bands in band_map.
       Routes to spectral index detector.
       Granularity: "built-up / land-cover change patches" (never claims individual buildings).
    3. unsupported_or_ambiguous: mid-resolution, missing band map, single-band, or unknown GSD.
       Falls back safely to baseline detector with explicit warnings.
    """
    cfg = cfg or AppConfig()
    r_cfg = cfg.routing
    band_map = cfg.bands.band_map

    # Ground sample distance in metres from affine transform & CRS
    gsd_m = before_info.gsd_x_m
    bands = min(before_info.count, after_info.count)

    # 1. High-resolution RGB regime
    if gsd_m is not None and gsd_m <= r_cfg.high_res_max_gsd_m and bands >= 3:
        return RoutingDecision(
            policy="high_res_rgb",
            detector="baseline_difference",
            reasons=[
                f"High-resolution RGB imagery ({gsd_m:.2f} m GSD, {bands} bands). "
                "Routed to baseline difference detector."
            ],
            warnings=[],
            output_granularity="individual_buildings",
        )

    # Check if multispectral bands are mapped (e.g., nir + red for NDVI, or swir1 + nir for NDBI)
    has_ms_indices = False
    if band_map:
        lower_map = {k.lower(): v for k, v in band_map.items()}
        has_ndvi = "nir" in lower_map and "red" in lower_map
        has_ndbi = "swir1" in lower_map and "nir" in lower_map
        has_ms_indices = has_ndvi or has_ndbi

    # 2. Coarse multispectral regime
    if gsd_m is not None and r_cfg.coarse_min_gsd_m <= gsd_m <= r_cfg.coarse_max_gsd_m:
        if has_ms_indices:
            return RoutingDecision(
                policy="multispectral_coarse",
                detector="spectral_index_difference",
                reasons=[
                    f"Coarse-resolution imagery ({gsd_m:.1f} m GSD) with configured multispectral "
                    f"band map ({sorted(band_map.keys())}). Routed to spectral index difference detector."
                ],
                warnings=[
                    f"Resolution is coarse ({gsd_m:.1f} m GSD): detections represent "
                    "built-up / land-cover change patches, NOT individual buildings."
                ],
                output_granularity="built-up / land-cover change patches",
            )
        return RoutingDecision(
            policy="unsupported_or_ambiguous",
            detector="baseline_difference",
            reasons=[
                f"Coarse resolution ({gsd_m:.1f} m GSD) but no required multispectral bands mapped. "
                "Band order is never assumed; falling back to baseline difference detector."
            ],
            warnings=[
                "Band map is unset or incomplete for multispectral indices. "
                "Detections are built-up / land-cover change patches, not individual buildings."
            ],
            output_granularity="built-up / land-cover change patches",
        )

    # 3. Ambiguous / unsupported fallback
    reasons = []
    if gsd_m is None:
        reasons.append("Ground Sample Distance (GSD) could not be determined.")
    elif bands < 3:
        reasons.append(f"Image has only {bands} band(s).")
    else:
        reasons.append(
            f"GSD ({gsd_m:.2f} m) is outside high-res (<= {r_cfg.high_res_max_gsd_m} m) "
            f"and coarse ({r_cfg.coarse_min_gsd_m}-{r_cfg.coarse_max_gsd_m} m) regimes."
        )
    reasons.append("Falling back to safe baseline difference detector.")

    return RoutingDecision(
        policy="unsupported_or_ambiguous",
        detector="baseline_difference",
        reasons=reasons,
        warnings=[
            "Ambiguous or unsupported input configuration; results computed with baseline difference."
        ],
        output_granularity="candidate_change_regions",
    )
