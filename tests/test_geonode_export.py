from __future__ import annotations

import sys
import xml.dom.minidom
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import geonode_export as ge  # noqa: E402


def _group_gdf() -> gpd.GeoDataFrame:
    n = 200
    return gpd.GeoDataFrame(
        {
            "importance_max": [1 + i % 5 for i in range(n)],
            "index_owa": [1 + (i * 7) % 100 for i in range(n)],
            "asset_groups_total": [1 + i % 9 for i in range(n)],
            "assets_overlap_total": [1 + (i % 50) ** 2 for i in range(n)],
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(n)],
        crs=4326,
    )


def _covered_once(values: pd.Series, classes) -> bool:
    for v in values:
        hits = sum(1 for lo, hi, _, _ in classes
                   if (lo is None or v >= lo) and (hi is None or v < hi))
        if hits != 1:
            return False
    return True


def test_result_style_classes_cover_every_value_exactly_once() -> None:
    gdf = _group_gdf()
    classes = ge.result_style_classes(gdf, str(ROOT / "config.ini"))
    for style in ge.RESULT_STYLES[1:]:
        assert style["key"] in classes
        assert _covered_once(gdf[style["field"]], classes[style["key"]]), style["key"]


def test_result_style_classes_skip_missing_columns() -> None:
    gdf = _group_gdf().drop(columns=["index_owa"])
    classes = ge.result_style_classes(gdf, None)
    assert "index_owa" not in classes
    assert "importance_max" in classes


def test_build_class_sld_is_valid_xml_with_one_rule_per_class() -> None:
    classes = [(None, 10, "#ffffff", "< 10"), (10, None, "#000000", "≥ 10 & more")]
    sld = ge.build_class_sld("mesa_x_index_owa", "OWA index", "index_owa", classes)
    doc = xml.dom.minidom.parseString(sld.encode("utf-8"))
    assert len(doc.getElementsByTagName("Rule")) == 2


def test_group_map_shows_only_the_first_style_and_keeps_it_on_top() -> None:
    styles = [("mesa_x", "Sensitivity"), ("mesa_x_index_owa", "OWA index"),
              ("mesa_x_importance_max", "Importance (max)")]
    data, maplayers = ge.build_group_map_config(
        "http://gn/", "geonode:mesa_x", 42, styles, [10.0, 59.0, 11.0, 60.0])
    wms = [l for l in data["map"]["layers"] if l["type"] == "wms"]
    assert [l["title"] for l in wms] == ["Importance (max)", "OWA index", "Sensitivity"]
    assert [l["visibility"] for l in wms] == [False, False, True]
    assert {l["style"] for l in wms} == {f"geonode:{n}" for n, _ in styles}
    assert all(l["url"] == "http://gn/geoserver/ows" for l in wms)
    assert {m["extra_params"]["msId"] for m in maplayers} == {l["id"] for l in wms}
    # GeoNode's MapLayer.dataset is a pk relation; anything else is an HTTP 500.
    assert all(m["dataset"] == 42 for m in maplayers)
    assert 1 <= data["map"]["zoom"] <= 16
