import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geoai04.polygonizer import polygonize, simplify

T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
CRS = "EPSG:32633"


def labels_rect(shape=(100, 100), r0=10, r1=30, c0=20, c1=60):
    lab = np.zeros(shape, np.int32)
    lab[r0:r1, c0:c1] = 1
    return lab


def test_rectangle_becomes_exact_polygon_with_correct_bounds():
    gdf = polygonize(labels_rect(), T, CRS)
    assert len(gdf) == 1 and gdf.crs.to_epsg() == 32633
    expected = box(500000 + 20 * 0.5, 4500000 - 30 * 0.5, 500000 + 60 * 0.5, 4500000 - 10 * 0.5)
    assert gdf.geometry.iloc[0].equals(expected)


def test_holes_are_preserved():
    lab = labels_rect()
    lab[15:20, 30:40] = 0
    g = polygonize(lab, T, CRS).geometry.iloc[0]
    assert len(g.interiors) == 1
    assert abs(g.area - (20 * 40 - 50) * 0.25) < 1e-9


def test_multiple_regions_and_validity():
    lab = labels_rect()
    lab[70:90, 70:90] = 2
    gdf = polygonize(lab, T, CRS)
    assert len(gdf) == 2 and gdf.geometry.is_valid.all()


def test_diagonal_pixels_connectivity():
    lab = np.zeros((10, 10), np.int32)
    lab[2, 2] = 1
    lab[3, 3] = 1  # touches diagonally
    assert len(polygonize(lab, T, CRS, connectivity=8)) == 1


def test_simplify_is_conservative_and_valid():
    lab = labels_rect()
    gdf = polygonize(lab, T, CRS)
    s = simplify(gdf, tolerance=0.25)
    assert s.geometry.is_valid.all()
    assert abs(s.geometry.iloc[0].area - gdf.geometry.iloc[0].area) < 1e-6  # straight edges unchanged


def test_empty_labels_give_empty_frame():
    gdf = polygonize(np.zeros((10, 10), np.int32), T, CRS)
    assert gdf.empty and gdf.crs.to_epsg() == 32633


def test_polygon_overlays_mask_raster(tmp_path):
    """Vector and raster stay spatially consistent: rasterising the polygon reproduces the mask."""
    from rasterio import features
    lab = labels_rect()
    gdf = polygonize(lab, T, CRS)
    back = features.rasterize([(gdf.geometry.iloc[0], 1)], out_shape=lab.shape, transform=T)
    assert (back.astype(bool) == (lab > 0)).all()
