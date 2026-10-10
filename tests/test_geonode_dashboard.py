from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import geonode_dashboard as gd  # noqa: E402


SPECS = [
    {"type": "text", "title": "About", "text": "<p>x</p>"},
    {"type": "counter", "title": "Area", "field": "area_km2", "op": "sum"},
    {"type": "counter", "title": "Cells", "field": "code", "op": "count"},
    {"type": "map"},
    {"type": "chart", "chart": "bar", "title": "By class", "group_by": "sensitivity_code_max",
     "field": "area_km2", "op": "sum", "colors": {"A": "#bd0026", "B": "#f03b20"}},
    {"type": "chart", "chart": "pie", "title": "By importance", "group_by": "importance_max",
     "field": "area_km2", "op": "avg", "colors": {1: "#111111"}},
    {"type": "chart", "chart": "bar", "title": "Plain", "group_by": "g", "field": "f", "op": "sum"},
    {"type": "table", "title": "Cells", "columns": ["code", "area_km2"]},
]
STYLES = [("mesa_x", "Sensitivity"), ("mesa_x_index_owa", "OWA index")]


def _data():
    return gd.build_dashboard_data("http://gn/", "geonode:mesa_x", 42, STYLES,
                                   [10.0, 59.0, 11.0, 60.0], SPECS, geom="geom")


def test_dashboard_has_one_widget_per_spec_and_is_json() -> None:
    data = _data()
    assert [w["widgetType"] for w in data["widgets"]] == [
        "text", "counter", "counter", "map", "chart", "chart", "chart", "table"]
    json.dumps(data)   # the payload goes over the wire as JSON
    assert {l["i"] for l in data["layouts"]["md"]} == {w["id"] for w in data["widgets"]}


def test_widgets_follow_the_map_widget() -> None:
    data = _data()
    mw = next(w for w in data["widgets"] if w["widgetType"] == "map")
    mid = mw["selectedMapId"]
    assert mw["maps"][0]["mapId"] == mid
    for w in data["widgets"]:
        if w["widgetType"] in ("chart", "counter", "table"):
            assert w["mapSync"] is True
            assert w["dependenciesMap"]["viewport"] == f"widgets[{mw['id']}].maps[{mid}].viewport"


def test_chart_uses_server_aggregation_and_typed_classes() -> None:
    charts = [w for w in _data()["widgets"] if w["widgetType"] == "chart"]
    t0 = charts[0]["charts"][0]["traces"][0]
    assert t0["options"] == {"groupByAttributes": "sensitivity_code_max",
                             "aggregationAttribute": "area_km2", "aggregateFunction": "Sum"}
    cls = t0["style"]["msClassification"]["classes"]
    assert [(c["unique"], c["color"]) for c in cls] == [("A", "#bd0026"), ("B", "#f03b20")]
    t1 = charts[1]["charts"][0]["traces"][0]
    assert t1["type"] == "pie" and t1["options"]["aggregateFunction"] == "Average"
    # MapStore compares class values with ===, so numeric fields keep numeric values.
    assert t1["style"]["msClassification"]["classes"][0]["unique"] == 1
    assert charts[2]["charts"][0]["traces"][0]["style"]["msMode"] == "simple"
    assert charts[0]["selectedTraceId"] == t0["id"]
    # A widget-level layer makes MapStore convert the chart as legacy and drop its
    # traces, which shows "No data in the current viewport" (seen on GeoNode 5).
    assert all("layer" not in c for c in charts)
    assert all(c["charts"][0]["traces"][0]["layer"]["name"] == "geonode:mesa_x" for c in charts)
    assert all(c["charts"][0]["traces"][0]["geomProp"] == "geom" for c in charts)


def test_layout_fits_the_six_column_grid_without_overlap() -> None:
    cells = set()
    for w in _data()["widgets"]:
        g = w["dataGrid"]
        assert 0 <= g["x"] and g["x"] + g["w"] <= 6
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                assert (x, y) not in cells, w["title"]
                cells.add((x, y))


def test_create_dashboard_posts_a_named_geoapp_with_marker(monkeypatch) -> None:
    posted = {}

    class R:
        def __init__(self, status, body=None):
            self.status_code, self._b, self.text = status, body or {}, ""

        def json(self):
            return self._b

    def fake_get(url, params=None, **k):
        if "geoapps" in url:
            return R(200, {"geoapps": []})
        return R(200, {"featureTypes": [{"properties": [
            {"name": "code", "type": "xsd:string"}, {"name": "the_geom", "type": "gml:MultiPolygon"}]}]})

    def fake_post(url, json=None, **k):
        posted.update(url=url, body=json)
        return R(201, {"geoapp": {"pk": 9}})

    monkeypatch.setattr(gd.requests, "get", fake_get)
    monkeypatch.setattr(gd.requests, "post", fake_post)
    url = gd.create_dashboard("http://gn/", "u", "p", "My dash", "geonode:mesa_x", 42,
                              STYLES, None, SPECS, lambda m: None)
    assert url == "http://gn/catalogue/#/dashboard/9"
    body = posted["body"]
    assert posted["url"] == "http://gn/api/v2/geoapps/"
    assert body["resource_type"] == "dashboard" and body["owner"] == "u"
    assert body["name"].startswith("My dash ") and body["name"] != "My dash"
    assert gd.DASHBOARD_MARKER in body["abstract"]
    assert all(w.get("geomProp") == "the_geom" for w in body["data"]["widgets"]
               if w["widgetType"] in ("counter", "table"))
    assert all(w["charts"][0]["traces"][0]["geomProp"] == "the_geom" for w in body["data"]["widgets"]
               if w["widgetType"] == "chart")
