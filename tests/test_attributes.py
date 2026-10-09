import numpy as np
import pytest
from pyproj import Geod
from rasterio.transform import from_origin

from geoai04.attributes import build_attributes, filter_by_area
from geoai04.geometry_utils import estimate_gsd_m, pixel_area_m2, select_metric_crs
from geoai04.polygonizer import polygonize


def make(crs, transform, shape=(100, 100)):
    lab = np.zeros(shape, np.int32)
    lab[10:30, 20:60] = 1           # 20 x 40 px = 800 px
    scores = np.where(lab > 0, 0.6, 0.02).astype("float32")
    raw = polygonize(lab, transform, crs)
    return build_attributes(raw, lab, scores, 0.2, transform, shape[1], shape[0], "pair123", "baseline_difference")


def test_area_and_centroid_known_polygon_projected():
    T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    gdf = make("EPSG:32633", T)
    r = gdf.iloc[0]
    assert abs(r.area_m2 - 800 * 0.25) < 1e-6          # 200 m2
    assert abs(r.centroid_x - (500000 + 40 * 0.5)) < 1e-6
    assert abs(r.centroid_y - (4500000 - 20 * 0.5)) < 1e-6
    assert r.projected_crs == "EPSG:32633"
    assert abs(r.change_magnitude - 0.6) < 1e-6 and r.confidence == 1.0
    assert 14.9 < r.centroid_lon < 15.1 and 40.0 < r.centroid_lat < 41.5


def test_area_is_square_metres_not_degrees_for_geographic_crs():
    px = 1e-5  # degrees, roughly 1.1 m at this latitude
    T = from_origin(77.0, 18.0, px, px)
    gdf = make("EPSG:4326", T)
    ring = gdf.geometry.iloc[0]
    geod_area = abs(Geod(ellps="WGS84").geometry_area_perimeter(ring)[0])
    assert gdf.area_m2.iloc[0] == pytest.approx(geod_area, rel=0.005)
    assert 500 < gdf.area_m2.iloc[0] < 1500  # nowhere near a degrees-squared number
    assert gdf.projected_crs.iloc[0] == "EPSG:32643"  # UTM zone from scene centre
    assert gdf.crs.to_epsg() == 4326  # geometries stay in the scene CRS


def test_metric_crs_selection():
    T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    assert select_metric_crs("EPSG:32633", T, 100, 100).to_epsg() == 32633
    Tm = from_origin(8_000_000.0, 2_000_000.0, 0.5, 0.5)
    assert select_metric_crs("EPSG:3857", Tm, 100, 100).to_epsg() != 3857  # Web Mercator distorts area
    assert pixel_area_m2("EPSG:32633", T, 100, 100) == pytest.approx(0.25)
    gx, gy, method = estimate_gsd_m("EPSG:32633", T, 100, 100)
    assert gx == pytest.approx(0.5) and gy == pytest.approx(0.5) and "projected" in method


def test_required_columns_and_ids():
    T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    gdf = make("EPSG:32633", T)
    for c in ["change_id", "area_m2", "centroid_lon", "centroid_lat", "centroid_x", "centroid_y",
              "change_magnitude", "confidence", "source_pair_id", "detector_used"]:
        assert c in gdf.columns
    assert gdf.change_id.iloc[0] == "CHG-0001" and gdf.detector_used.iloc[0] == "baseline_difference"


def test_area_filter_uses_square_metres():
    T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    gdf = make("EPSG:32633", T)
    assert len(filter_by_area(gdf, 100.0)) == 1
    assert len(filter_by_area(gdf, 250.0)) == 0


def test_empty_input_yields_empty_attributed_frame():
    T = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    lab = np.zeros((10, 10), np.int32)
    raw = polygonize(lab, T, "EPSG:32633")
    out = build_attributes(raw, lab, np.zeros((10, 10), "float32"), 0.2, T, 10, 10, "x", "baseline_difference")
    assert out.empty and "area_m2" in out.columns
