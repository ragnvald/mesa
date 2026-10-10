from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import geopandas as gpd
import pytest
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import qgis_results_project as qrp  # noqa: E402


def _read_project(path: Path) -> ET.Element:
    with zipfile.ZipFile(path) as project:
        qgs_name = next(name for name in project.namelist() if name.endswith(".qgs"))
        return ET.fromstring(project.read(qgs_name))


def test_selected_3d_groups_keeps_every_group_and_tags_its_family() -> None:
    groups = ["basic_mosaic", "H3_R8", "h3_r11", "QDGC_L12", "wards"]

    assert qrp._selected_3d_groups(groups) == [
        ("MOSAIC", "basic_mosaic"),
        ("H3", "H3_R8"),
        ("H3", "h3_r11"),
        ("QDGC", "QDGC_L12"),
        # An uploaded polygon layer has no grid level to measure against.
        ("OTHER", "wards"),
    ]


def test_landscape_height_is_scaled_to_six_percent_with_half_sized_limits() -> None:
    assert qrp._landscape_max_height(100) == 25
    assert qrp._landscape_max_height(1000) == 60
    assert qrp._landscape_max_height(100_000) == 2500


def test_landscape_height_is_capped_by_the_grid_cell_width() -> None:
    # A 28.9 km scene gives every grid the same 1733 m columns on span alone,
    # which is 32x the width of a QDGC L11 cell.
    span = 28_884.0
    grids = [("H3", "H3_R8"), ("H3", "H3_R10"), ("QDGC", "QDGC_L9"), ("QDGC", "QDGC_L11")]
    heights = {}
    for family, group in grids:
        width = qrp._grid_cell_width_m(family, group, -8.0)
        height = qrp._landscape_max_height(span, width)
        heights[group] = height
        assert height <= width * qrp._LANDSCAPE_CELL_ASPECT

    # The coarsest grid is wide enough that the span still governs it.
    assert heights["H3_R8"] == qrp._landscape_max_height(span)
    assert heights["QDGC_L11"] < heights["H3_R10"] < heights["H3_R8"]
    assert heights["QDGC_L11"] < heights["QDGC_L9"] < heights["H3_R8"]


def test_outsized_cells_get_a_glass_variant_of_each_rule() -> None:
    rules = [("\"sensitivity_code_max\" = 'A'", "Very High", (215, 48, 39), "A")]
    layer = ET.Element("maplayer")
    qrp._add_3d_renderer(layer, "id", rules, "100", area_threshold=121_500.0)

    variants = layer.findall("./renderer-3d/rules/rule")
    assert len(variants) == 2
    solid, glass = variants
    assert 'coalesce("area_m2", 0) <= 121500.0' in solid.get("filter")
    assert solid.find("./symbol/material").get("opacity") == qrp._LANDSCAPE_OPACITY
    assert '"area_m2" > 121500.0' in glass.get("filter")
    assert glass.find("./symbol/material").get("opacity") == qrp._OUTSIZED_OPACITY
    assert glass.get("description").endswith("outsized cell")
    # Same colour and the same height — only the transparency differs.
    assert (solid.find("./symbol/material").get("diffuse")
            == glass.find("./symbol/material").get("diffuse"))
    assert {rule.find('.//Option[@name="expression"]').get("value")
            for rule in variants} == {"100"}

    # Without a threshold — a regular grid, or no area column — nothing splits.
    plain = ET.Element("maplayer")
    qrp._add_3d_renderer(plain, "id", rules, "100")
    assert len(plain.findall("./renderer-3d/rules/rule")) == 1


def test_median_ignores_an_empty_or_missing_series() -> None:
    from array import array

    assert qrp._median(None) is None
    assert qrp._median(array("d")) is None
    assert qrp._median(array("d", [3.0, 1.0, 2.0])) == 2.0
    assert qrp._median(array("d", [4.0, 1.0, 3.0, 2.0])) == 2.5


def test_grid_cell_width_follows_the_level_and_narrows_with_latitude() -> None:
    assert qrp._grid_cell_width_m("H3", "H3_R10", 0.0) == pytest.approx(131.8, abs=0.5)
    assert qrp._grid_cell_width_m("H3", "h3_r8", 60.0) == pytest.approx(922.7, abs=0.5)
    assert qrp._grid_cell_width_m("QDGC", "QDGC_L11", -8.0) == pytest.approx(53.8, abs=0.5)
    assert qrp._grid_cell_width_m("QDGC", "QDGC_L9", 0.0) == pytest.approx(217.4, abs=0.5)
    assert qrp._grid_cell_width_m("H3", "basic_mosaic", 0.0) is None


def test_results_project_adds_interactive_landscapes_for_finest_grids(tmp_path: Path) -> None:
    qgis_dir = tmp_path / "qgis"
    output_dir = tmp_path / "output" / "geoparquet"
    qgis_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    shutil.copyfile(ROOT / "qgis" / "mesa.qgz", qgis_dir / "mesa.qgz")

    groups = [
        "basic_mosaic", "H3_R8", "H3_R10", "H3_R10",
        "QDGC_L9", "QDGC_L11", "QDGC_L11",
    ]
    flat = gpd.GeoDataFrame(
        {
            "name_gis_geocodegroup": groups,
            "assets_overlap_total": [1, 3, 2, 8, 4, 5, 10],
            "sensitivity_code_max": ["E", "D", "C", "A", "B", "C", "A"],
        },
        geometry=[box(10.70 + i * 0.01, 59.90, 10.705 + i * 0.01, 59.905)
                  for i in range(len(groups))],
        crs="EPSG:4326",
    )
    flat.to_parquet(output_dir / "tbl_flat.parquet")
    coverage_bounds = (10.71, 59.901, 10.735, 59.904)
    coverage = gpd.GeoDataFrame(
        {"name": ["data_extent"]}, geometry=[box(*coverage_bounds)], crs="EPSG:4326"
    )
    coverage.to_parquet(output_dir / "tbl_data_extent.parquet")

    messages: list[str] = []
    result = qrp.build_results_project(tmp_path, log=messages.append)

    assert result == qgis_dir / "mesa_results.qgz"
    root = _read_project(result)
    views = root.findall("./mapViewDocks3D/view")
    assert [view.get("name") for view in views] == [
        "3D asset overlap — basic_mosaic",
        "3D sensitivity — basic_mosaic",
        "3D asset overlap — H3_R10",
        "3D sensitivity — H3_R10",
        "3D asset overlap — H3_R8",
        "3D sensitivity — H3_R8",
        "3D asset overlap — QDGC_L11",
        "3D sensitivity — QDGC_L11",
        "3D asset overlap — QDGC_L9",
        "3D sensitivity — QDGC_L9",
    ]
    assert all(view.get("isOpen") == "0" for view in views)
    assert all((view.findtext("./qgis3d/crs/spatialrefsys/authid") or "").startswith("EPSG:326")
               for view in views)

    canvas_extent = root.find("./mapcanvas/extent")
    actual_canvas_bounds = tuple(float(canvas_extent.findtext(tag)) for tag in (
        "xmin", "ymin", "xmax", "ymax"
    ))
    assert all(math.isclose(actual, expected, abs_tol=1e-9)
               for actual, expected in zip(actual_canvas_bounds, coverage_bounds))

    for view in views:
        animation = view.find("animation3d")
        assert animation is not None
        assert animation.get("widget-visible") == "1"
        keyframes = animation.findall("./keyframes/keyframe")
        assert [float(frame.get("time")) for frame in keyframes] == [
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
        ]
        orbit = keyframes[3:]
        for attribute in ("x", "y", "z", "dist", "pitch"):
            assert len({frame.get(attribute) for frame in orbit}) == 1
        orbit_yaws = [float(frame.get("yaw")) for frame in orbit]
        assert [right - left for left, right in zip(orbit_yaws, orbit_yaws[1:])] == [
            20, 40, 60, 60, 60, 60, 60,
        ]
        assert math.isclose(orbit_yaws[-1] - orbit_yaws[0], 360.0)

    landscapes = {
        layer.findtext("layername"): layer
        for layer in root.findall("./projectlayers/maplayer")
        if layer.find("renderer-3d") is not None
    }
    assert set(landscapes) == {
        f"{group} — {label} landscape"
        for group in ("basic_mosaic", "H3_R10", "H3_R8", "QDGC_L11", "QDGC_L9")
        for label in ("3D asset overlap", "3D sensitivity")
    }

    expected_maxima = {
        "basic_mosaic": 1, "H3_R10": 8, "H3_R8": 3, "QDGC_L11": 10, "QDGC_L9": 4,
    }
    for group, maximum in expected_maxima.items():
        layer = landscapes[f"{group} — 3D asset overlap landscape"]
        assert f"'{group}'" in layer.findtext("datasource")
        assert '"assets_overlap_total" > 0' in layer.findtext("datasource")
        renderer = layer.find("renderer-3d")
        assert renderer.get("type") == "rulebased"
        tiling = renderer.find("vector-layer-3d-tiling")
        assert tiling.get("zoom-levels-count") == "3"
        assert tiling.get("max-chunk-features") == "25000"
        assert tiling.get("show-bounding-boxes") == "0"
        assert len(renderer.findall("./rules/rule")) == 5
        expected_colors = qrp._SENSITIVITY_3D_COLORS
        for rule in renderer.findall("./rules/rule"):
            code = re.search(r"'([A-E])'", rule.get("filter")).group(1)
            material = rule.find("./symbol/material")
            diffuse = tuple(int(part) for part in material.get("diffuse").split(",")[:3])
            assert diffuse == expected_colors[code]
            assert material.get("opacity") == qrp._LANDSCAPE_OPACITY
            assert material.get("shininess") == qrp._LANDSCAPE_SHININESS
            assert material.get("ks") == qrp._LANDSCAPE_SPECULAR_K
            assert rule.find("./symbol/edges").get("enabled") == "0"
        expressions = {
            option.get("value")
            for option in renderer.findall('.//Option[@name="expression"]')
        }
        assert len(expressions) == 1
        expression = expressions.pop()
        assert f'/ {maximum} *' in expression
        assert 'coalesce("assets_overlap_total", 0)' in expression
        height_match = re.search(r"\* ([0-9.]+)$", expression)
        assert height_match is not None
        _groups, _maxima, geo_metadata, stats = qrp._scan_flat(
            output_dir / "tbl_flat.parquet"
        )
        _, scene_extent = qrp._scene_spatial_context(geo_metadata)
        span = max(scene_extent[2] - scene_extent[0], scene_extent[3] - scene_extent[1])
        # Each group is capped by its own cell width, so the layers differ.
        latitude = qrp._centre_lonlat(geo_metadata)[1]
        if group == "basic_mosaic":
            # No grid level to read a cell size from; the count stands in.
            cell_width = qrp._counted_cell_width_m(scene_extent, stats[group]["cells"])
        else:
            family = "H3" if group.startswith("H3") else "QDGC"
            cell_width = qrp._grid_cell_width_m(family, group, latitude)
        assert cell_width is not None
        assert math.isclose(
            float(height_match.group(1)),
            qrp._landscape_max_height(span, cell_width),
            abs_tol=1e-6,
        )

        layer_id = layer.findtext("id")
        matching_view = next(view for view in views
                             if view.get("name") == f"3D asset overlap — {group}")
        scene_layer_ids = {
            scene_layer.get("id")
            for scene_layer in matching_view.findall("./qgis3d/terrain/layers/layer")
        }
        assert layer_id in scene_layer_ids

    # The second category: height comes from the A-E code, tallest at A, and the
    # colours are the seed's own sensitivity ramp rather than the recoloured one.
    seed_colors = {
        code: rgb
        for _filter, _label, rgb, code in qrp._sensitivity_3d_rules(
            qrp._find_maplayer(root, qrp._SENS_REF), seed_colors=True
        )
    }
    for group in expected_maxima:
        layer = landscapes[f"{group} — 3D sensitivity landscape"]
        renderer = layer.find("renderer-3d")
        heights = {}
        for rule in renderer.findall("./rules/rule"):
            code = re.search(r"'([A-E])'", rule.get("filter")).group(1)
            diffuse = tuple(int(part) for part in
                            rule.find("./symbol/material").get("diffuse").split(",")[:3])
            assert diffuse == seed_colors[code]
            expression = rule.find('.//Option[@name="expression"]').get("value")
            heights[code] = float(expression)
        assert heights["A"] > heights["B"] > heights["C"] > heights["D"] > heights["E"] > 0
        overlap_layer = landscapes[f"{group} — 3D asset overlap landscape"]
        overlap_expression = overlap_layer.find(
            './/Option[@name="expression"]'
        ).get("value")
        # Both categories top out at the same cell-capped ceiling.
        assert math.isclose(
            heights["A"], float(re.search(r"\* ([0-9.]+)$", overlap_expression).group(1)),
            abs_tol=1e-6,
        )

    assert any("10 interactive 3D view(s)" in message for message in messages)
