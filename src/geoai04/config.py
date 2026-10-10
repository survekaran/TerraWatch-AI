"""Configuration loading and validation. All tunable values live here or in YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


@dataclass
class DetectionConfig:
    max_bands: int = 3
    uint8_scale: float = 255.0
    uint16_scale: float = 10000.0
    smoothing_sigma_px: float = 1.0
    threshold_method: str = "otsu"
    fixed_threshold: float = 0.15
    percentile: float = 95.0
    otsu_floor: float = 0.05
    tile_size: int = 1024
    radiometric_normalization: bool = False
    radiometric_norm_method: str = "mean_std"


@dataclass
class ValidationConfig:
    min_valid_fraction: float = 0.01
    min_overlap_fraction: float = 0.99
    grid_tolerance_px: float = 0.001
    gsd_rel_tolerance: float = 0.01


@dataclass
class PostprocessingConfig:
    opening_radius_px: int = 1
    closing_radius_px: int = 2
    min_mapping_unit_m2: float = 25.0


@dataclass
class PolygonConfig:
    connectivity: int = 8
    simplify_tolerance_px: float = 0.5


@dataclass
class AlignmentConfig:
    warn_px: float = 0.5
    fail_px: float = 2.0
    upsample_factor: int = 20
    n_windows: int = 9
    window_size_px: int = 128
    texture_threshold: float = 0.02
    min_windows: int = 2
    spread_warn_px: float = 1.0
    block_on_fail: bool = True
    auto_correct: bool = False
    interpolation: str = "linear"


@dataclass
class RoutingConfig:
    high_res_max_gsd_m: float = 2.0
    coarse_min_gsd_m: float = 5.0
    coarse_max_gsd_m: float = 30.0


@dataclass
class BandsConfig:
    band_map: dict[str, int] = field(default_factory=dict)
    vegetation_proxy_exg: bool = True


@dataclass
class TaggingConfig:
    min_compactness_building: float = 0.35
    min_building_area_m2: float = 20.0
    max_building_area_m2: float = 2500.0
    veg_drop_threshold: float = 0.15
    built_rise_threshold: float = 0.08
    brightness_rise_threshold: float = 0.06
    fluctuation_min_area_m2: float = 200.0
    max_compactness_diffuse: float = 0.40
    misregistration_edge_threshold: float = 0.20
    misregistration_min_elongation: float = 2.5
    misregistration_width_factor_px: float = 2.5
    low_quality_nodata_fraction: float = 0.15
    disagreement_iou_threshold: float = 0.40


@dataclass
class SeverityConfig:
    mode: str = "emergency"
    area_rule: str = "absolute"
    min_polygons_for_percentile: int = 10
    percentile_critical: float = 90.0
    percentile_medium: float = 60.0
    emergency_crit_conf: float = 0.60
    emergency_med_conf: float = 0.50
    emergency_critical_area_m2: float = 500.0
    emergency_medium_area_m2: float = 100.0
    enforcement_crit_conf: float = 0.85
    enforcement_med_conf: float = 0.75
    enforcement_critical_area_m2: float = 1000.0
    enforcement_medium_area_m2: float = 200.0
    implausible_change_fraction: float = 0.20
    implausible_change_fraction_emergency: float = 0.20
    implausible_change_fraction_enforcement: float = 0.20
    context_layers: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    polygons: PolygonConfig = field(default_factory=PolygonConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    bands: BandsConfig = field(default_factory=BandsConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> "AppConfig":
        d, v, p, g = self.detection, self.validation, self.postprocessing, self.polygons
        a, r, s = self.alignment, self.routing, self.severity
        if d.threshold_method not in {"otsu", "fixed", "percentile"}:
            raise ConfigError(f"detection.threshold_method must be otsu|fixed|percentile, got {d.threshold_method!r}")
        if not 0 < d.fixed_threshold < 1:
            raise ConfigError("detection.fixed_threshold must be between 0 and 1")
        if not 0 < d.percentile < 100:
            raise ConfigError("detection.percentile must be between 0 and 100")
        if d.max_bands < 1 or d.tile_size < 16:
            raise ConfigError("detection.max_bands must be >= 1 and tile_size >= 16")
        if d.smoothing_sigma_px < 0 or d.otsu_floor < 0:
            raise ConfigError("detection.smoothing_sigma_px and otsu_floor must be >= 0")
        if p.opening_radius_px < 0 or p.closing_radius_px < 0 or p.min_mapping_unit_m2 < 0:
            raise ConfigError("postprocessing radii and min_mapping_unit_m2 must be >= 0")
        if g.connectivity not in (4, 8) or g.simplify_tolerance_px < 0:
            raise ConfigError("polygons.connectivity must be 4 or 8 and simplify_tolerance_px >= 0")
        if not 0 < v.min_overlap_fraction <= 1 or not 0 <= v.min_valid_fraction < 1:
            raise ConfigError("validation fractions out of range")
        if a.warn_px < 0 or a.fail_px < a.warn_px:
            raise ConfigError("alignment.fail_px must be >= alignment.warn_px >= 0")
        if a.upsample_factor < 1 or a.n_windows < 1 or a.window_size_px < 16:
            raise ConfigError("alignment parameters out of range")
        if a.interpolation not in {"nearest", "linear", "spline"}:
            raise ConfigError("alignment.interpolation must be nearest|linear|spline")
        if r.high_res_max_gsd_m <= 0 or r.coarse_min_gsd_m <= 0 or r.coarse_max_gsd_m < r.coarse_min_gsd_m:
            raise ConfigError("routing GSD thresholds out of range")
        if s.mode not in {"emergency", "enforcement"}:
            raise ConfigError("severity.mode must be emergency|enforcement")
        if s.area_rule not in {"absolute", "percentile"}:
            raise ConfigError("severity.area_rule must be absolute|percentile")
        if s.emergency_crit_conf < s.emergency_med_conf:
            raise ConfigError("emergency critical confidence must be >= medium confidence")
        if s.enforcement_crit_conf < s.enforcement_med_conf:
            raise ConfigError("enforcement critical confidence must be >= medium confidence")
        return self


def _section(cls, data: dict | None, name: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config section {name!r} must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown keys in section {name!r}: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: dict | None) -> AppConfig:
    """Build and validate an AppConfig from a nested dictionary."""
    data = data or {}
    unknown = set(data) - {
        "detection", "validation", "postprocessing", "polygons",
        "alignment", "routing", "bands", "tagging", "severity"
    }
    if unknown:
        raise ConfigError(f"Unknown top-level config sections: {sorted(unknown)}")
    cfg = AppConfig(
        detection=_section(DetectionConfig, data.get("detection"), "detection"),
        validation=_section(ValidationConfig, data.get("validation"), "validation"),
        postprocessing=_section(PostprocessingConfig, data.get("postprocessing"), "postprocessing"),
        polygons=_section(PolygonConfig, data.get("polygons"), "polygons"),
        alignment=_section(AlignmentConfig, data.get("alignment"), "alignment"),
        routing=_section(RoutingConfig, data.get("routing"), "routing"),
        bands=_section(BandsConfig, data.get("bands"), "bands"),
        tagging=_section(TaggingConfig, data.get("tagging"), "tagging"),
        severity=_section(SeverityConfig, data.get("severity"), "severity"),
    )
    return cfg.validate()


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> AppConfig:
    """Load YAML config (default: config/default.yaml) and apply nested overrides."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif path != DEFAULT_CONFIG_PATH:
        raise ConfigError(f"Config file not found: {path}")
    for section, values in (overrides or {}).items():
        if not isinstance(values, dict):
            raise ConfigError(f"Override for {section!r} must be a mapping")
        data.setdefault(section, {}).update(values)
    return config_from_dict(data)
