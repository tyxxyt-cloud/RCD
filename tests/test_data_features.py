from __future__ import annotations

import math

from crcdm.data.build_dataset import (
    STATIC_FEATURE_NAMES,
    make_static_feature,
    parse_lanes,
    parse_oneway,
    resolve_road_attributes,
)


FEATURE_CFG = {
    "lane_defaults": {"motorway": 3, "primary": 2, "default": 1},
    "oneway_defaults": {"motorway": True, "default": False},
}


def test_real_road_attributes_override_defaults() -> None:
    resolved = resolve_road_attributes(
        {"highway": "primary", "lanes": "4", "oneway": "yes"},
        FEATURE_CFG,
    )
    assert resolved == {
        "highway": "primary",
        "lanes": 4.0,
        "oneway": True,
        "lanes_inferred": False,
        "oneway_inferred": False,
    }


def test_missing_road_attributes_use_highway_defaults() -> None:
    resolved = resolve_road_attributes(
        {"highway": "motorway", "lanes": None, "oneway": None},
        FEATURE_CFG,
    )
    assert resolved["lanes"] == 3.0
    assert resolved["oneway"] is True
    assert resolved["lanes_inferred"] is True
    assert resolved["oneway_inferred"] is True


def test_osm_style_parsers_and_static_vector() -> None:
    assert parse_lanes("2;3") == 3.0
    assert parse_lanes("unknown") is None
    assert parse_oneway("-1") is True
    assert parse_oneway("no") is False
    meta = {
        7: {
            "length_m": 99.0,
            "bearing_deg": 90.0,
            "lanes": 2.0,
            "oneway": True,
            "lanes_inferred": False,
            "oneway_inferred": True,
        }
    }
    features = make_static_feature(7, meta, math.log(100.0), 1.0, 2.0, 1.0)
    assert len(features) == len(STATIC_FEATURE_NAMES)
    assert abs(features[0]) < 1e-6
    assert abs(features[1]) < 1e-6
    assert abs(features[2] - 1.0) < 1e-6
    assert abs(features[3]) < 1e-6
    assert features[4:] == [1.0, 0.0, 1.0]
