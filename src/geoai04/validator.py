"""Module A: input validation for a bi-temporal raster pair.

The validator never reprojects, resamples, or assigns a CRS. If the two inputs are not
already on an identical grid it reports a failure and says why.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.transform import Affine
from rasterio.windows import Window

from .config import ValidationConfig
from .geometry_utils import estimate_gsd_m, scene_bounds

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail
    message: str


@dataclass
class RasterInfo:
    path: str
    name: str
    crs: str | None
    width: int
    height: int
    count: int
    dtype: str
    transform: tuple[float, ...]
    bounds: tuple[float, float, float, float]
    gsd_x_m: float | None
    gsd_y_m: float | None
    gsd_method: str | None
    nodata: float | None
    valid_fraction: float
    has_nonfinite: bool


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    before: RasterInfo | None = None
    after: RasterInfo | None = None

    @property
    def errors(self) -> list[str]:
        return [f"{c.name}: {c.message}" for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[str]:
        return [f"{c.name}: {c.message}" for c in self.checks if c.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": [asdict(c) for c in self.checks],
            "before": asdict(self.before) if self.before else None,
            "after": asdict(self.after) if self.after else None,
        }


class InputValidationError(Exception):
    """Raised by the pipeline when validation fails."""

    def __init__(self, report: ValidationReport, run_dir: Path | None = None):
        self.report = report
        self.run_dir = run_dir
        super().__init__("Input validation failed: " + "; ".join(report.errors))


def _windows(width: int, height: int, tile: int):
    for r in range(0, height, tile):
        for c in range(0, width, tile):
            yield Window(c, r, min(tile, width - c), min(tile, height - r))


def _inspect(path: str | Path, label: str, cfg: ValidationConfig, checks: list[Check], tile: int = 1024) -> RasterInfo | None:
    p = Path(path)
    if not p.is_file():
        checks.append(Check(f"{label} file", "fail", f"File not found: {p}"))
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            src = rasterio.open(p)
    except RasterioIOError as exc:
        checks.append(Check(f"{label} file", "fail", f"Cannot open as a raster: {exc}"))
        return None

    with src:
        checks.append(Check(f"{label} file", "pass", f"Opened {p.name} ({src.width} x {src.height} px, {src.count} band(s), {src.dtypes[0]})"))
        crs = src.crs
        transform = src.transform
        georeferenced = crs is not None and transform != Affine.identity()
        if crs is None:
            checks.append(Check(f"{label} CRS", "fail", "No CRS defined. A CRS is never assigned silently; georeference the file first."))
        else:
            checks.append(Check(f"{label} CRS", "pass", crs.to_string()))
        if crs is not None and transform == Affine.identity():
            checks.append(Check(f"{label} transform", "fail", "Identity transform: the file is not georeferenced."))
        elif transform.b != 0 or transform.d != 0:
            checks.append(Check(f"{label} transform", "warn", "Rotated or sheared transform; results are still computed on the native grid."))

        gx = gy = None
        method = None
        if georeferenced:
            gx, gy, method = estimate_gsd_m(crs, transform, src.width, src.height)
            checks.append(Check(f"{label} GSD", "pass", f"{gx:.3f} x {gy:.3f} m/px; {method}"))
            if abs(gx - gy) / max(gx, gy) > 0.05:
                checks.append(Check(f"{label} pixel shape", "warn", "Pixels are not square (more than 5% difference between x and y GSD)."))

        valid = total = 0
        for w in _windows(src.width, src.height, tile):
            m = src.dataset_mask(window=w)
            valid += int((m > 0).sum())
            total += m.size
        valid_fraction = valid / total if total else 0.0
        if valid_fraction < cfg.min_valid_fraction:
            checks.append(Check(f"{label} valid data", "fail", f"Only {valid_fraction:.2%} of pixels are valid (nodata/mask): image is empty or unusable."))
        elif valid_fraction < 1.0:
            checks.append(Check(f"{label} valid data", "warn", f"{1 - valid_fraction:.2%} of pixels are NoData and will be excluded."))
        else:
            checks.append(Check(f"{label} valid data", "pass", "No NoData pixels."))

        scale = max(1.0, max(src.width, src.height) / float(tile))
        out_shape = (src.count, max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        sample = src.read(out_shape=out_shape, resampling=Resampling.nearest).astype(np.float64)
        mask = src.read_masks(1, out_shape=out_shape[1:], resampling=Resampling.nearest) > 0
        nonfinite = bool(not np.isfinite(sample).all())
        if nonfinite:
            checks.append(Check(f"{label} values", "warn", "Contains NaN/inf values; they are treated as invalid pixels."))
        finite = np.isfinite(sample).all(axis=0) & mask
        if finite.any():
            std = float(sample[:, finite].std())
            if std < 1e-9:
                checks.append(Check(f"{label} content", "fail", "Image is constant (no variation): blank or empty imagery."))
            else:
                checks.append(Check(f"{label} content", "pass", "Image has spatial/spectral variation (checked on a decimated read)."))
        else:
            checks.append(Check(f"{label} content", "fail", "No finite valid pixels found."))

        return RasterInfo(
            path=str(p), name=p.name, crs=crs.to_string() if crs else None,
            width=src.width, height=src.height, count=src.count, dtype=src.dtypes[0],
            transform=tuple(transform)[:6], bounds=tuple(scene_bounds(transform, src.width, src.height)),
            gsd_x_m=gx, gsd_y_m=gy, gsd_method=method,
            nodata=src.nodata, valid_fraction=valid_fraction, has_nonfinite=nonfinite,
        )


def _same_grid(a: RasterInfo, b: RasterInfo, tol_px: float) -> bool:
    """True if size, pixel size and origin agree (origin within tol_px of a pixel)."""
    ta, tb = a.transform, b.transform
    px, py = abs(ta[0]) or 1.0, abs(ta[4]) or 1.0
    return (
        a.width == b.width and a.height == b.height
        and abs(ta[0] - tb[0]) <= 1e-6 * px and abs(ta[4] - tb[4]) <= 1e-6 * py
        and abs(ta[1] - tb[1]) <= 1e-9 * px and abs(ta[3] - tb[3]) <= 1e-9 * py
        and abs(ta[2] - tb[2]) <= tol_px * px and abs(ta[5] - tb[5]) <= tol_px * py
    )


def _overlap_fraction(a: RasterInfo, b: RasterInfo) -> float:
    w0, s0, e0, n0 = a.bounds
    w1, s1, e1, n1 = b.bounds
    iw, ih = min(e0, e1) - max(w0, w1), min(n0, n1) - max(s0, s1)
    if iw <= 0 or ih <= 0:
        return 0.0
    area = lambda w, s, e, n: (e - w) * (n - s)
    return float(iw * ih / min(area(w0, s0, e0, n0), area(w1, s1, e1, n1)))


def validate_pair(before_path: str | Path, after_path: str | Path, cfg: ValidationConfig | None = None) -> ValidationReport:
    """Validate two rasters for bi-temporal comparison. Never raises for bad inputs."""
    cfg = cfg or ValidationConfig()
    report = ValidationReport()
    report.before = _inspect(before_path, "Before", cfg, report.checks)
    report.after = _inspect(after_path, "After", cfg, report.checks)
    a, b = report.before, report.after
    if a is None or b is None:
        return report
    add = report.checks.append

    if a.crs is None or b.crs is None:
        add(Check("Pair CRS", "fail", "At least one input has no CRS; the pair cannot be compared."))
        return report
    if a.crs != b.crs:
        add(Check("Pair CRS", "fail", f"CRS mismatch ({a.crs} vs {b.crs}). Reprojection is not performed automatically; reproject one file explicitly."))
        return report
    add(Check("Pair CRS", "pass", f"Both inputs use {a.crs}."))

    overlap = _overlap_fraction(a, b)
    if overlap < cfg.min_overlap_fraction:
        add(Check("Pair overlap", "fail", f"Bounding boxes overlap by only {overlap:.1%} (need at least {cfg.min_overlap_fraction:.0%})."))
    else:
        add(Check("Pair overlap", "pass", f"Bounding boxes overlap by {overlap:.1%}."))

    if a.count != b.count:
        add(Check("Pair bands", "fail", f"Band count differs ({a.count} vs {b.count}); band correspondence cannot be assumed."))
    else:
        add(Check("Pair bands", "pass", f"Both inputs have {a.count} band(s). Band identity is not verified beyond count."))

    if a.dtype != b.dtype:
        add(Check("Pair dtype", "warn", f"Data types differ ({a.dtype} vs {b.dtype}); each is normalised by its own type."))

    if a.gsd_x_m and b.gsd_x_m:
        rel = abs(a.gsd_x_m - b.gsd_x_m) / max(a.gsd_x_m, b.gsd_x_m)
        if rel > cfg.gsd_rel_tolerance:
            add(Check("Pair GSD", "fail", f"Ground resolution differs by {rel:.1%} ({a.gsd_x_m:.3f} vs {b.gsd_x_m:.3f} m/px). Resampling is not performed in Phase 1."))
        else:
            add(Check("Pair GSD", "pass", "Resolutions are compatible."))

    if _same_grid(a, b, cfg.grid_tolerance_px):
        add(Check("Pair grid", "pass", "Both inputs share an identical pixel grid."))
    else:
        add(Check("Pair grid", "fail", "Inputs overlap but are not on the same pixel grid (size or offset differs). Phase 1 does not resample or shift; clip/align them externally (for example with gdalwarp)."))

    for label, info in (("Before", a), ("After", b)):
        if info.gsd_x_m is not None and info.gsd_x_m > 5.0:
            add(Check(f"{label} resolution note", "warn", f"GSD about {info.gsd_x_m:.1f} m: detections will be built-up/land-cover patches, not reliable individual buildings."))
    log.info("Validation finished: ok=%s, %d error(s), %d warning(s)", report.ok, len(report.errors), len(report.warnings))
    return report
