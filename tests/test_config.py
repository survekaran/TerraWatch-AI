import pytest

from geoai04.config import ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.detection.threshold_method == "otsu"
    assert cfg.postprocessing.min_mapping_unit_m2 == 25.0


def test_overrides_apply():
    cfg = load_config(overrides={"detection": {"threshold_method": "fixed", "fixed_threshold": 0.3}})
    assert cfg.detection.fixed_threshold == 0.3


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"detection": {"nope": 1}})


def test_invalid_value_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"detection": {"threshold_method": "magic"}})
    with pytest.raises(ConfigError):
        load_config(overrides={"polygons": {"connectivity": 6}})
