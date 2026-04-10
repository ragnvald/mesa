from __future__ import annotations

import sys
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


def test_flatten_worker_preserves_unique_asset_lists_and_scores(tmp_path: Path) -> None:
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

    result = pi._flatten_worker((part_path, _class_ranges(), {}, pi.INDEX_WEIGHT_DEFAULTS))
    assert result is not None
    assert set(result["code"]) == {"A", "B"}

    row_a = result.loc[result["code"] == "A"].iloc[0]
    assert int(row_a["assets_overlap_total"]) == 3
    assert set(row_a["ref_asset_group"]) == {10, 20}
    assert set(row_a["name_gis_assetgroup"]) == {"roads", "water"}
    assert int(row_a["importance_min"]) == 1
    assert int(row_a["importance_max"]) == 5
    assert int(row_a["importance_raw_score"]) == 7
    assert int(row_a["sensitivity_raw_score"]) == 30
    assert int(row_a["owa_n25"]) == 2

    row_b = result.loc[result["code"] == "B"].iloc[0]
    assert int(row_b["assets_overlap_total"]) == 1
    assert set(row_b["ref_asset_group"]) == {30}
