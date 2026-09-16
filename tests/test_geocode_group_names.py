"""Generated geocode group names are zero-padded, and older projects are migrated.

H3_R10 used to sort ahead of H3_R7, which put the finest grid first in every list
and made choose_primary_geocode pick the most expensive fallback. MESA 5.7 names
levels H3_R07 / QDGC_L06; projects written by 5.6 and earlier are renamed in place.
"""
from __future__ import annotations

import sqlite3
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

import mesa_shared as ms  # noqa: E402


@pytest.mark.parametrize("res, expected", [(0, "H3_R00"), (7, "H3_R07"), (10, "H3_R10"), (15, "H3_R15")])
def test_h3_group_name_is_padded(res, expected):
    assert ms.h3_group_name(res) == expected


@pytest.mark.parametrize("level, expected", [(0, "QDGC_L00"), (6, "QDGC_L06"), (12, "QDGC_L12")])
def test_qdgc_group_name_is_padded(level, expected):
    assert ms.qdgc_group_name(level) == expected


def test_padded_names_sort_in_level_order():
    names = [ms.h3_group_name(r) for r in (10, 7, 9, 8)]
    assert sorted(names) == ["H3_R07", "H3_R08", "H3_R09", "H3_R10"]


def test_primary_geocode_fallback_is_the_coarsest_grid():
    groups = [ms.h3_group_name(r) for r in (10, 6, 8)]
    assert ms.choose_primary_geocode(groups) == "H3_R06"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("H3_R7", "H3_R07"),
        ("QDGC_L6", "QDGC_L06"),
        ("H3_R07", "H3_R07"),
        ("H3_R10", "H3_R10"),
        ("QDGC_L11", "QDGC_L11"),
        ("basic_mosaic", "basic_mosaic"),
        ("wards_H3_R7", "wards_H3_R7"),  # only whole generated names are renamed
        ("h3_r7", "h3_r7"),              # data names are exact; imported sets keep their case
    ],
)
def test_canonical_name(name, expected):
    assert ms.canonical_geocode_group_name(name) == expected


def _write_project(base: Path, groups: list[str]) -> Path:
    gpq = base / "output" / "geoparquet"
    gpq.mkdir(parents=True)
    grp = gpd.GeoDataFrame(
        {"id": list(range(len(groups))), "name": groups, "name_gis_geocodegroup": groups},
        geometry=[box(i, 0, i + 1, 1) for i in range(len(groups))],
        crs="EPSG:4326",
    )
    grp.to_parquet(gpq / "tbl_geocode_group.parquet")
    obj = gpd.GeoDataFrame(
        {
            "code": [f"c{i}" for i in range(len(groups) * 3)],
            "name_gis_geocodegroup": [g for g in groups for _ in range(3)],
            "note": ["H3_R7 mentioned in free text"] * (len(groups) * 3),
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(len(groups) * 3)],
        crs="EPSG:4326",
    )
    obj.to_parquet(gpq / "tbl_geocode_object.parquet")
    stacked = gpq / "tbl_stacked"
    stacked.mkdir()
    pd.DataFrame({"code": ["c0", "c1"], "name_gis_geocodegroup": [groups[0], groups[-1]], "v": [1, 2]}).to_parquet(
        stacked / "part_00000.parquet"
    )
    seg = gpq / "tbl_segmentation"
    seg.mkdir()
    pd.DataFrame({"name_gis_geocodegroup": ["H3_R7"], "segment": [1]}).to_parquet(seg / "H3_R7.parquet")
    mb = base / "output" / "mbtiles"
    mb.mkdir(parents=True)
    for name in ("H3_R7_index_owa", "QDGC_L10_index_owa", "basic_mosaic_index_owa"):
        con = sqlite3.connect(mb / f"{name}.mbtiles")
        con.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
        con.executemany("INSERT INTO metadata VALUES (?, ?)", [("name", name), ("description", name), ("minzoom", "6")])
        con.commit()
        con.close()
    (base / "config.ini").write_bytes(
        b"[DEFAULT]\r\n"
        b"#   list -> e.g. basic_mosaic, H3_R9\r\n"
        b"segment_geocode_layer = basic_mosaic, h3_r7, QDGC_L10\r\n"
        b"segmv_geocode_layer = QDGC_L6\r\n"
        b"parquet_folder = output/geoparquet\r\n"
    )
    return gpq


def test_migration_renames_tables_files_and_config(tmp_path):
    gpq = _write_project(tmp_path, ["basic_mosaic", "H3_R7", "H3_R10", "QDGC_L6"])
    geo_before = pq.ParquetFile(gpq / "tbl_geocode_object.parquet").schema_arrow.metadata[b"geo"]

    summary = ms.migrate_geocode_group_names(tmp_path)

    assert list(pd.read_parquet(gpq / "tbl_geocode_group.parquet")["name_gis_geocodegroup"]) == [
        "basic_mosaic", "H3_R07", "H3_R10", "QDGC_L06",
    ]
    assert list(pd.read_parquet(gpq / "tbl_geocode_group.parquet")["name"]) == [
        "basic_mosaic", "H3_R07", "H3_R10", "QDGC_L06",
    ]
    obj = gpd.read_parquet(gpq / "tbl_geocode_object.parquet")
    assert sorted(set(obj["name_gis_geocodegroup"])) == ["H3_R07", "H3_R10", "QDGC_L06", "basic_mosaic"]
    assert set(obj["note"]) == {"H3_R7 mentioned in free text"}  # not a whole name: untouched
    assert obj.crs is not None and len(obj) == 12
    assert pq.ParquetFile(gpq / "tbl_geocode_object.parquet").schema_arrow.metadata[b"geo"] == geo_before
    assert list(pd.read_parquet(gpq / "tbl_stacked" / "part_00000.parquet")["name_gis_geocodegroup"]) == [
        "basic_mosaic", "QDGC_L06",
    ]

    assert (gpq / "tbl_segmentation" / "H3_R07.parquet").is_file()
    assert not (gpq / "tbl_segmentation" / "H3_R7.parquet").exists()
    mb = tmp_path / "output" / "mbtiles"
    assert sorted(p.name for p in mb.iterdir()) == [
        "H3_R07_index_owa.mbtiles", "QDGC_L10_index_owa.mbtiles", "basic_mosaic_index_owa.mbtiles",
    ]
    con = sqlite3.connect(mb / "H3_R07_index_owa.mbtiles")
    assert dict(con.execute("SELECT name, value FROM metadata")) == {
        "name": "H3_R07_index_owa", "description": "H3_R07_index_owa", "minzoom": "6",
    }
    con.close()

    cfg = (tmp_path / "config.ini").read_bytes()
    assert b"segment_geocode_layer = basic_mosaic, H3_R07, QDGC_L10\r\n" in cfg
    assert b"segmv_geocode_layer = QDGC_L06\r\n" in cfg
    assert b"#   list -> e.g. basic_mosaic, H3_R9\r\n" in cfg  # comments are left alone
    assert b"\n" not in cfg.replace(b"\r\n", b"")  # line endings preserved

    assert summary["files"] == 2 and summary["config"] == 2 and summary["tables"] >= 4


def test_migration_is_idempotent_and_noop_when_current(tmp_path):
    gpq = _write_project(tmp_path, ["basic_mosaic", "H3_R7"])
    ms.migrate_geocode_group_names(tmp_path)
    stamp = {p: p.stat().st_mtime_ns for p in gpq.rglob("*.parquet")}
    assert ms.migrate_geocode_group_names(tmp_path) == {"tables": 0, "files": 0, "config": 0}
    assert {p: p.stat().st_mtime_ns for p in gpq.rglob("*.parquet")} == stamp


def test_existing_padded_file_wins_over_legacy_copy(tmp_path):
    _write_project(tmp_path, ["H3_R7"])
    mb = tmp_path / "output" / "mbtiles"
    newer = mb / "H3_R07_index_owa.mbtiles"
    newer.write_bytes((mb / "H3_R7_index_owa.mbtiles").read_bytes())
    ms.migrate_geocode_group_names(tmp_path)
    assert newer.is_file()
    assert not (mb / "H3_R7_index_owa.mbtiles").exists()


def test_interrupted_migration_is_completed_on_next_run(tmp_path):
    gpq = _write_project(tmp_path, ["basic_mosaic", "H3_R7"])
    # Simulate a crash after the derived tables but before the group table (the marker).
    ms._rewrite_parquet_legacy_names(gpq / "tbl_geocode_object.parquet")
    (gpq / "tbl_geocode_object.parquet.renaming").replace(gpq / "tbl_geocode_object.parquet")
    assert ms._legacy_geocode_names_pending(tmp_path)
    ms.migrate_geocode_group_names(tmp_path)
    assert not ms._legacy_geocode_names_pending(tmp_path)
    assert "H3_R07" in set(pd.read_parquet(gpq / "tbl_geocode_group.parquet")["name_gis_geocodegroup"])
