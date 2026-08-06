"""Merging geocode objects must never lose the groups it is not touching.

The regression these cover: adding H3/QDGC levels to a project whose
basic_mosaic holds 17.9 million faces wrote a table containing only the new
grids. The old merge read the whole of tbl_geocode_object through geopandas,
and a failed read was swallowed into an empty frame that was then written back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import geocode_manage as gm  # noqa: E402


def _objects(group: str, ref: int, count: int, x0: float = 0.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "code": [f"{group}_{i:06d}" for i in range(count)],
            "name_gis_geocodegroup": [group] * count,
            "ref_geocodegroup": [ref] * count,
        },
        geometry=[box(x0 + i, 0, x0 + i + 1, 1) for i in range(count)],
        crs="EPSG:4326",
    )


def _groups(rows: list[tuple[int, str]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "name_gis_geocodegroup": [r[1] for r in rows],
            "geocode_origin": [r[1] if r[1] == "basic_mosaic" else "generated" for r in rows],
            "title_user": [r[1] for r in rows],
            "description": [""] * len(rows),
        },
        geometry=[box(0, 0, 1, 1)] * len(rows),
        crs="EPSG:4326",
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project holding one mosaic group, written the way MESA writes it.

    gpq_dir() caches the resolved directory in a module global on first call,
    which is right for the app (one base_dir per process) and wrong for a test
    session: without the reset every test after the first writes into the first
    one's tmp_path. See learning.md "A test that reads the repo's live project
    data is not a test".
    """
    gm._PARQUET_OVERRIDE = None
    gpq = tmp_path / "output" / "geoparquet"
    gpq.mkdir(parents=True)
    _groups([(1, "basic_mosaic")]).to_parquet(gpq / "tbl_geocode_group.parquet", index=False)
    _objects("basic_mosaic", 1, 50).to_parquet(gpq / "tbl_geocode_object.parquet", index=False)
    return tmp_path


def _object_counts(base: Path) -> dict[str, int]:
    table = pq.read_table(base / "output" / "geoparquet" / "tbl_geocode_object.parquet",
                          columns=["name_gis_geocodegroup"])
    return pd.Series(table.column(0).to_pylist()).value_counts().to_dict()


def test_adding_a_grid_keeps_the_mosaic(project: Path) -> None:
    added_g, added_o, total_g, total_o = gm._merge_and_write_geocodes(
        project, _groups([(0, "H3_R8")]), _objects("H3_R8", 0, 7, x0=100.0), []
    )

    assert (added_g, added_o) == (1, 7)
    assert (total_g, total_o) == (2, 57)
    assert _object_counts(project) == {"basic_mosaic": 50, "H3_R8": 7}

    groups = gpd.read_parquet(project / "output" / "geoparquet" / "tbl_geocode_group.parquet")
    assert list(groups["name_gis_geocodegroup"]) == ["basic_mosaic", "H3_R8"]
    # The new group is numbered past the mosaic, and its objects point at it.
    new_id = int(groups.loc[groups["name_gis_geocodegroup"] == "H3_R8", "id"].iloc[0])
    assert new_id == 2
    objects = pq.read_table(project / "output" / "geoparquet" / "tbl_geocode_object.parquet")
    refs = dict(zip(objects.column("name_gis_geocodegroup").to_pylist(),
                    objects.column("ref_geocodegroup").to_pylist()))
    assert refs == {"basic_mosaic": 1, "H3_R8": 2}


def test_refreshing_a_group_replaces_only_that_group(project: Path) -> None:
    gm._merge_and_write_geocodes(project, _groups([(0, "H3_R8")]),
                                 _objects("H3_R8", 0, 7, x0=100.0), [])
    gm._merge_and_write_geocodes(project, _groups([(0, "H3_R8")]),
                                 _objects("H3_R8", 0, 3, x0=200.0), ["H3_R8"])

    assert _object_counts(project) == {"basic_mosaic": 50, "H3_R8": 3}


def test_clearing_a_group_leaves_the_others(project: Path) -> None:
    gm._merge_and_write_geocodes(project, _groups([(0, "H3_R8")]),
                                 _objects("H3_R8", 0, 7, x0=100.0), [])
    gm._clear_geocode_groups(project, ["H3_R8"])

    assert _object_counts(project) == {"basic_mosaic": 50}
    groups = gpd.read_parquet(project / "output" / "geoparquet" / "tbl_geocode_group.parquet")
    assert list(groups["name_gis_geocodegroup"]) == ["basic_mosaic"]


def test_an_unreadable_object_table_raises_instead_of_deleting(project: Path) -> None:
    objects_path = project / "output" / "geoparquet" / "tbl_geocode_object.parquet"
    original = objects_path.read_bytes()
    objects_path.write_bytes(b"not a parquet file")

    with pytest.raises(Exception):
        gm._merge_and_write_geocodes(project, _groups([(0, "H3_R8")]),
                                     _objects("H3_R8", 0, 7, x0=100.0), [])

    # The damaged table is left exactly as found - not replaced by the new grid,
    # which is what the swallowed read used to do.
    assert objects_path.read_bytes() == b"not a parquet file"
    assert original.startswith(b"PAR1")
    groups = gpd.read_parquet(project / "output" / "geoparquet" / "tbl_geocode_group.parquet")
    assert list(groups["name_gis_geocodegroup"]) == ["basic_mosaic"]


def test_existing_objects_are_never_read_through_geopandas(project: Path, monkeypatch) -> None:
    """The point of the rewrite: existing geometry stays WKB on the way through.

    Decoding it is what costs 1.3 kB a row, so refuse the call outright and the
    merge must still finish. Group tables are small and keep reading normally.
    """
    real_read_parquet = gm.gpd.read_parquet

    def _guarded(path, *args, **kwargs):
        if "tbl_geocode_object" in str(path):
            raise AssertionError("the merge decoded the existing object geometry")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(gm.gpd, "read_parquet", _guarded)

    gm._merge_and_write_geocodes(project, _groups([(0, "QDGC_L8")]),
                                 _objects("QDGC_L8", 0, 4, x0=300.0), [])

    assert _object_counts(project) == {"basic_mosaic": 50, "QDGC_L8": 4}


def test_bbox_metadata_grows_to_cover_the_new_objects(project: Path) -> None:
    gm._merge_and_write_geocodes(project, _groups([(0, "H3_R8")]),
                                 _objects("H3_R8", 0, 7, x0=100.0), [])

    import json
    meta = pq.ParquetFile(
        project / "output" / "geoparquet" / "tbl_geocode_object.parquet"
    ).metadata.metadata
    geo = json.loads(meta[b"geo"])
    bbox = geo["columns"][geo["primary_column"]]["bbox"]
    assert bbox[0] == 0.0 and bbox[2] == 107.0
