"""Module I (minimal): provenance record for every run, including failed ones."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def software_versions() -> dict[str, str]:
    import geopandas, numpy, pyproj, rasterio, scipy, shapely, skimage
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "geoai04": __version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-image": skimage.__version__,
        "rasterio": rasterio.__version__, "gdal": getattr(rasterio, "__gdal_version__", "unknown"),
        "shapely": shapely.__version__, "pyproj": pyproj.__version__, "geopandas": geopandas.__version__,
    }


def write_provenance(path: str | Path, record: dict) -> Path:
    record = dict(record)
    record.setdefault("generated_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    record.setdefault("software", software_versions())
    path = Path(path)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
