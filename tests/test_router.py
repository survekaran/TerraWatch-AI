from geoai04.config import AppConfig
from geoai04.resolution_router import route_inputs
from geoai04.validator import RasterInfo


def _make_info(gsd_m=0.5, count=3, width=1024, height=1024):
    return RasterInfo(
        path="test.tif",
        name="test.tif",
        crs="EPSG:32633",
        width=width,
        height=height,
        count=count,
        dtype="uint8",
        transform=(0.5, 0.0, 500000.0, 0.0, -0.5, 4500000.0),
        bounds=(500000.0, 4499488.0, 500512.0, 4500000.0),
        gsd_x_m=gsd_m,
        gsd_y_m=gsd_m,
        gsd_method="affine transform in projected metric CRS",
        nodata=None,
        valid_fraction=1.0,
        has_nonfinite=False,
    )


def test_high_res_rgb_routes_to_baseline_difference():
    info = _make_info(gsd_m=0.5, count=3)
    cfg = AppConfig()

    dec = route_inputs(info, info, cfg)
    assert dec.policy == "high_res_rgb"
    assert dec.detector == "baseline_difference"
    assert dec.output_granularity == "individual_buildings"


def test_multispectral_coarse_with_band_map():
    info = _make_info(gsd_m=10.0, count=4)
    cfg = AppConfig()
    cfg.bands.band_map = {"blue": 1, "green": 2, "red": 3, "nir": 4}

    dec = route_inputs(info, info, cfg)
    assert dec.policy == "multispectral_coarse"
    assert dec.detector == "spectral_index_difference"
    assert "built-up / land-cover change patches" in dec.output_granularity
    assert "individual buildings" not in dec.output_granularity
    assert any("built-up / land-cover change patches, NOT individual buildings" in w for w in dec.warnings)


def test_multispectral_coarse_without_band_map_falls_back():
    info = _make_info(gsd_m=10.0, count=4)
    cfg = AppConfig()
    cfg.bands.band_map = {}  # Empty: band order is never assumed

    dec = route_inputs(info, info, cfg)
    assert dec.policy == "unsupported_or_ambiguous"
    assert dec.detector == "baseline_difference"
    assert any("Band order is never assumed" in r for r in dec.reasons)


def test_ambiguous_and_single_band_input():
    # Single band high-res
    single = _make_info(gsd_m=0.5, count=1)
    cfg = AppConfig()
    dec = route_inputs(single, single, cfg)
    assert dec.policy == "unsupported_or_ambiguous"
    assert dec.detector == "baseline_difference"

    # Unknown GSD
    no_gsd = _make_info(gsd_m=None, count=3)
    dec2 = route_inputs(no_gsd, no_gsd, cfg)
    assert dec2.policy == "unsupported_or_ambiguous"
    assert any("could not be determined" in r for r in dec2.reasons)


def test_gsd_derived_from_transform_not_dimensions():
    # A small image with small pixel size has high resolution
    small_highres = _make_info(gsd_m=0.2, count=3, width=64, height=64)
    cfg = AppConfig()
    dec = route_inputs(small_highres, small_highres, cfg)
    assert dec.policy == "high_res_rgb"

    # A huge image with large pixel size is coarse resolution
    big_coarse = _make_info(gsd_m=20.0, count=4, width=10000, height=10000)
    cfg.bands.band_map = {"nir": 4, "red": 3}
    dec2 = route_inputs(big_coarse, big_coarse, cfg)
    assert dec2.policy == "multispectral_coarse"
