from __future__ import annotations

import sys
import configparser
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import processing_internal as pi  # noqa: E402


def _class_ranges() -> dict[str, range]:
    return {
        "A": range(21, 26),
        "B": range(16, 21),
        "C": range(11, 16),
        "D": range(6, 11),
        "E": range(1, 6),
    }


def test_mesa_version_label_sanitizes_config_values() -> None:
    cfg = configparser.ConfigParser()
    cfg["DEFAULT"] = {"mesa_version": "5.0 beta"}

    assert pi._mesa_version_label(cfg) == "5.0_beta"


def test_regular_grid_fast_path_matches_legacy_join_for_single_cell_geometries() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "code": ["g1", "g2", "g3", "g4"],
            "geometry": [
                box(0.10, 0.10, 0.20, 0.20),
                box(1.20, 0.10, 1.30, 0.20),
                box(0.10, 1.20, 0.20, 1.30),
                Point(1.25, 1.25),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    cell_size_deg = 1.0
    grid_cells = pi.create_grid(gdf, cell_size_deg)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    x_edges = pi.np.arange(xmin, xmax + cell_size_deg, cell_size_deg)
    y_edges = pi.np.arange(ymin, ymax + cell_size_deg, cell_size_deg)
    nx = max(1, len(x_edges) - 1)
    ny = max(1, len(y_edges) - 1)

    fast = pi._assign_geocodes_to_regular_grid(
        gdf,
        xmin=float(x_edges[0]),
        ymin=float(y_edges[0]),
        cell_size_deg=cell_size_deg,
        nx=nx,
        ny=ny,
    )
    fast = pi._dedupe_tagged_geocodes(fast)

    grid_gdf = gpd.GeoDataFrame(
        {
            "grid_cell": range(len(grid_cells)),
            "geometry": [box(x0, y0, x1, y1) for (x0, y0, x1, y1) in grid_cells],
        },
        geometry="geometry",
        crs=gdf.crs,
    )
    legacy = gpd.sjoin(gdf, grid_gdf, how="left", predicate="intersects")
    legacy = legacy.drop(columns=["index_right"], errors="ignore")
    legacy = pi._dedupe_tagged_geocodes(legacy)

    fast_map = fast.set_index("code")["grid_cell"].to_dict()
    legacy_map = legacy.set_index("code")["grid_cell"].to_dict()
    assert fast_map == legacy_map


def test_regular_grid_fast_path_assigns_single_cell_to_multi_cell_geometry() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "code": ["wide"],
            "geometry": [box(0.10, 0.10, 2.90, 0.90)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    fast = pi._assign_geocodes_to_regular_grid(
        gdf,
        xmin=0.10,
        ymin=0.10,
        cell_size_deg=1.0,
        nx=3,
        ny=1,
    )

    assert len(fast) == 1
    assert int(fast.iloc[0]["grid_cell"]) == 1


def test_flatten_worker_preserves_unique_asset_lists_and_scores(tmp_path: Path, monkeypatch) -> None:
    # _flatten_worker backfills from tbl_asset_group via _get_asset_group_lookup(),
    # which reads gpq_dir()/tbl_asset_group.parquet — resolved from the working
    # directory. Run from the repo root, this test therefore read whatever project
    # happened to be sitting in output/geoparquet, and the ref_asset_group values
    # below collide with real ids there. Prime the module cache with an empty frame
    # so the fixture below is the only input.
    monkeypatch.setattr(pi, "_ASSET_GROUP_LOOKUP", pd.DataFrame(), raising=False)

    gdf = gpd.GeoDataFrame(
        {
            "code": ["A", "A", "A", "B"],
            "ref_geocodegroup": [1, 1, 1, 2],
            "name_gis_geocodegroup": ["basic_mosaic", "basic_mosaic", "basic_mosaic", "basic_mosaic"],
            "ref_asset_group": [10, 10, 20, 30],
            "name_gis_assetgroup": ["roads", "roads", "water", "power"],
            "importance": [1, 5, 5, 2],
            "sensitivity": [1, 25, 25, 4],
            "susceptibility": [1, 5, 5, 2],
            "geometry": [
                box(0, 0, 1, 1),
                box(0, 0, 1, 1),
                box(0, 0, 1, 1),
                box(1, 1, 2, 2),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    part_path = tmp_path / "part.parquet"
    gdf.to_parquet(part_path, index=False)

    result = pi._flatten_worker((part_path, _class_ranges(), {}))
    assert result is not None
    assert set(result["code"]) == {"A", "B"}

    row_a = result.loc[result["code"] == "A"].iloc[0]
    assert int(row_a["assets_overlap_total"]) == 3
    assert set(row_a["ref_asset_group"]) == {10, 20}
    assert set(row_a["name_gis_assetgroup"]) == {"roads", "water"}
    assert int(row_a["importance_min"]) == 1
    assert int(row_a["importance_max"]) == 5
    # The *_raw_score columns this test used to assert on were dropped from the
    # flatten output; min/max plus the owa_n* histogram carry the same content.
    assert int(row_a["sensitivity_min"]) == 1
    assert int(row_a["sensitivity_max"]) == 25
    # 25 is the top of A (21-25). The expectation used to read "B", which no
    # reading of the ranges above supports.
    assert row_a["sensitivity_code_max"] == "A"
    assert row_a["sensitivity_code_min"] == "E"
    # importance/susceptibility are single factors and get their own 1-5 bands:
    # 5 -> A, 1 -> E. These were constant "E" before derive_value_class_ranges.
    assert row_a["importance_code_max"] == "A"
    assert row_a["importance_code_min"] == "E"
    assert int(row_a["owa_n25"]) == 2

    row_b = result.loc[result["code"] == "B"].iloc[0]
    assert int(row_b["assets_overlap_total"]) == 1
    assert set(row_b["ref_asset_group"]) == {30}

    # Flatten no longer carries the cell geometry out of the partition; the
    # parent joins it from tbl_geocode_object. See learning.md
    # "tbl_stacked carries no geometry".
    assert "geometry" not in result.columns


def _stacked_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["A", "A", "B"],
            "ref_geocodegroup": [1, 1, 1],
            "name_gis_geocodegroup": ["basic_mosaic"] * 3,
            "ref_asset_group": [10, 20, 10],
            "name_gis_assetgroup": ["roads", "water", "roads"],
            "importance": [5, 3, 5],
            "sensitivity": [25, 6, 25],
            "susceptibility": [5, 2, 5],
        }
    )


def test_flatten_worker_reads_a_partition_that_has_no_geometry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pi, "_ASSET_GROUP_LOOKUP", pd.DataFrame(), raising=False)

    part_path = tmp_path / "part.parquet"
    _stacked_rows().to_parquet(part_path, index=False)

    result = pi._flatten_worker((part_path, _class_ranges(), {}))
    assert result is not None
    assert set(result["code"]) == {"A", "B"}
    assert int(result.loc[result["code"] == "A", "assets_overlap_total"].iloc[0]) == 2
    assert "geometry" not in result.columns


def test_stacked_contract_drops_cell_and_asset_group_grained_columns() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "code": ["A"],
            "name_gis_geocodegroup": ["basic_mosaic"],
            "ref_asset_group": [10],
            "sensitivity": [25],
            # cell-grained / merge artefacts that must not be repeated per row
            "geometry": [box(0, 0, 1, 1)],
            "id_x": [1],
            "id_y": [10],
            "attributes": ["{'a': 1}"],
            "grid_cell": [0],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    out = pi._to_stacked_contract(gdf)

    assert list(out.columns) == ["code", "name_gis_geocodegroup", "ref_asset_group", "sensitivity"]
    assert not isinstance(out, gpd.GeoDataFrame)


def test_read_stacked_partition_never_returns_geometry(tmp_path: Path) -> None:
    # An older package still carries the column; reading it back is exactly the
    # waste the format change removes, so the reader must skip it either way.
    gdf = gpd.GeoDataFrame(
        _stacked_rows().assign(geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1), box(1, 1, 2, 2)]),
        geometry="geometry",
        crs="EPSG:4326",
    )
    legacy = tmp_path / "legacy.parquet"
    gdf.to_parquet(legacy, index=False)

    out = pi._read_stacked_partition(legacy)
    assert "geometry" not in out.columns
    assert len(out) == 3

    projected = pi._read_stacked_partition(legacy, ["code", "sensitivity", "geometry", "nope"])
    assert list(projected.columns) == ["code", "sensitivity"]


def test_attach_geocode_geometry_keys_on_group_and_code(tmp_path: Path, monkeypatch) -> None:
    # The same `code` in two geocode groups must not cross-join: codes are only
    # unique per import batch, and H3/QDGC/mosaic are generated on separate paths.
    geocodes = gpd.GeoDataFrame(
        {
            "code": ["c1", "c1", "c2"],
            "name_gis_geocodegroup": ["basic_mosaic", "H3_R8", "basic_mosaic"],
            "geometry": [box(0, 0, 1, 1), box(5, 5, 6, 6), box(1, 1, 2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    geocodes.to_parquet(tmp_path / "tbl_geocode_object.parquet", index=False)
    monkeypatch.setattr(pi, "gpq_dir", lambda: tmp_path)

    flat = pd.DataFrame({"code": ["c1", "c2"], "name_gis_geocodegroup": ["basic_mosaic", "basic_mosaic"]})
    out = pi._attach_geocode_geometry(flat, "4326")

    assert len(out) == 2, "a bare `code` key would have duplicated c1 across groups"
    assert out.loc[out["code"] == "c1", "geometry"].iloc[0].equals(box(0, 0, 1, 1))
    assert out.loc[out["code"] == "c2", "geometry"].iloc[0].equals(box(1, 1, 2, 2))


def test_flatten_preflight_relaxes_amplification_when_there_is_no_geometry(tmp_path: Path) -> None:
    cfg = configparser.ConfigParser()
    cfg["DEFAULT"] = {"flatten_max_workers": "4", "flatten_small_max_workers": "4"}

    flat_part = tmp_path / "flat.parquet"
    _stacked_rows().to_parquet(flat_part, index=False)
    geo_part = tmp_path / "geo.parquet"
    gpd.GeoDataFrame(
        _stacked_rows().assign(geometry=[box(0, 0, 1, 1)] * 3), geometry="geometry", crs="EPSG:4326"
    ).to_parquet(geo_part, index=False)

    _, flat_debug = pi._estimate_flatten_peak_gb([flat_part], cfg)
    _, geo_debug = pi._estimate_flatten_peak_gb([geo_part], cfg)

    assert "k_amp=3.0 (no geometry)" in flat_debug
    assert "k_amp=10.0 (with geometry)" in geo_debug


def test_running_chunk_ids_skips_chunks_that_returned_out_of_order() -> None:
    # imap_unordered: chunk 3 came back before 1 and 2.
    assert pi._running_chunk_ids({3}, total_chunks=6, max_workers=3) == [1, 2, 4]
    assert pi._running_chunk_ids({1, 2, 3, 4, 5}, total_chunks=6, max_workers=3) == [6]
    assert pi._running_chunk_ids(set(range(1, 7)), total_chunks=6, max_workers=3) == []


def test_flatten_minimap_tracker_marks_a_chunk_done_only_when_all_its_parts_are() -> None:
    status = {
        "cells": [
            {"id": 10, "state": "intersected"},
            {"id": 11, "state": "intersected"},
            {"id": 20, "state": "intersected"},
        ],
        # chunk 1 wrote two parts (split write), chunk 2 one.
        "part_chunks": {"part_a.parquet": 1, "part_b.parquet": 1, "part_c.parquet": 2},
        "chunk_cells": {"1": [10, 11], "2": [20]},
    }
    tracker = pi._FlattenMinimapTracker(status)
    assert tracker.active

    # Flatten reports full paths under tbl_stacked/, intersect recorded basenames.
    assert tracker.partition_done("/x/tbl_stacked/part_c.parquet") is True
    assert tracker.partition_done("/x/tbl_stacked/part_a.parquet") is False
    assert [c["state"] for c in status["cells"]] == ["intersected", "intersected", "done"]

    assert tracker.partition_done("/x/tbl_stacked/part_b.parquet") is True
    assert [c["state"] for c in status["cells"]] == ["done", "done", "done"]
    # A repeat (soft-throttle resubmit) or an unknown file changes nothing.
    assert tracker.partition_done("/x/tbl_stacked/part_b.parquet") is False
    assert tracker.partition_done("/x/tbl_stacked/other.parquet") is False


def test_flatten_minimap_tracker_is_inert_without_intersect_maps() -> None:
    tracker = pi._FlattenMinimapTracker({"cells": [{"id": 1, "state": "done"}]})
    assert not tracker.active
    assert tracker.partition_done("part.parquet") is False
