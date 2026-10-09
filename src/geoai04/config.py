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
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    polygons: PolygonConfig = field(default_factory=PolygonConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> "AppConfig":
        d, v, p, g = self.detection, self.validation, self.postprocessing, self.polygons
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
    unknown = set(data) - {"detection", "validation", "postprocessing", "polygons"}
    if unknown:
        raise ConfigError(f"Unknown top-level config sections: {sorted(unknown)}")
    cfg = AppConfig(
        detection=_section(DetectionConfig, data.get("detection"), "detection"),
        validation=_section(ValidationConfig, data.get("validation"), "validation"),
        postprocessing=_section(PostprocessingConfig, data.get("postprocessing"), "postprocessing"),
        polygons=_section(PolygonConfig, data.get("polygons"), "polygons"),
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
