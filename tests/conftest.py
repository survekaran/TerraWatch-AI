import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoai04.synthetic import generate_synthetic_pair  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    meta = generate_synthetic_pair(d)
    return {"dir": d, "before": d / "before.tif", "after": d / "after.tif", "reference": d / "reference_mask.tif", "meta": meta}


def write_tif(path, data, crs="EPSG:32633", origin=(500000.0, 4500000.0), gsd=0.5, nodata=None, dtype=None):
    """Helper: write a small GeoTIFF (data shape: bands, rows, cols). crs=None writes no CRS."""
    data = np.asarray(data)
    if dtype:
        data = data.astype(dtype)
    kw = dict(driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype=data.dtype, nodata=nodata)
    if crs:
        kw.update(crs=crs, transform=from_origin(origin[0], origin[1], gsd, gsd))
    with rasterio.open(path, "w", **kw) as d:
        d.write(data)
    return path


@pytest.fixture
def tif():
    return write_tif
