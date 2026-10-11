"""AnalysisStorage stages edits in memory and writes only on save().

The Analysis stage and the report read tbl_analysis_group / tbl_analysis_polygons
from disk, so what these tests pin down is what those readers get to see.
See devtools/docs/SPECIAL_FOCUS_REDESIGN_PLAN.md.
"""
from __future__ import annotations

import configparser
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import analysis_setup as aset  # noqa: E402

aset._load_runtime_data_stack()
gpd = aset.gpd
pd = aset.pd

SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
OTHER = {"type": "Polygon", "coordinates": [[[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]]]}


def _cfg() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["DEFAULT"] = {"parquet_folder": "output/geoparquet"}
    return cfg


def _storage(tmp_path: Path) -> "aset.AnalysisStorage":
    return aset.AnalysisStorage(tmp_path, _cfg())


def _feature(geometry: dict, title: str | None = None) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": ({"title": title} if title else {})}


def test_nothing_is_written_until_save(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.add_group("North")
    storage.add_polygon(_feature(SQUARE))

    assert not storage.group_path.exists()
    assert not storage.polygon_path.exists()


def test_save_writes_both_tables_even_when_only_a_polygon_was_added(tmp_path: Path) -> None:
    # With the polygon file alone on disk the Analysis stage reports itself
    # unavailable and the report offers no study areas.
    storage = _storage(tmp_path)
    storage.add_polygon(_feature(SQUARE))
    storage.save()

    groups = pd.read_parquet(storage.group_path)
    polygons = gpd.read_parquet(storage.polygon_path)
    assert list(groups["id"]) == ["group_default"]
    assert list(groups.columns) == aset.AnalysisStorage.GROUP_COLUMNS
    assert list(polygons.columns) == aset.AnalysisStorage.POLYGON_COLUMNS
    assert set(polygons["group_id"]) <= set(groups["id"])
    assert polygons.crs is not None and polygons.crs.to_epsg() == 4326
    assert not list(storage.polygon_path.parent.glob("*.tmp"))


def test_discard_returns_to_what_is_on_disk(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    kept = storage.add_polygon(_feature(SQUARE, "Kept"))
    storage.save()

    storage.add_polygon(_feature(OTHER, "Dropped"))
    storage.delete(kept.identifier, None)
    storage.add_group("Temporary")
    storage.discard()

    assert [r.title for r in storage.list_records()] == ["Kept"]
    assert [g.identifier for g in storage.list_groups()] == ["group_default"]


def test_a_polygon_never_gets_an_empty_title(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    first = storage.add_polygon(_feature(SQUARE))
    second = storage.add_polygon(_feature(OTHER))
    assert (first.title, second.title) == ("Polygon 1", "Polygon 2")

    storage.update_properties(first.identifier, None, "Wetland", "note")
    storage.update_properties(first.identifier, None, "   ", "note 2")
    record = storage.get_records(None, [first.identifier])[0]
    assert (record.title, record.notes) == ("Wetland", "note 2")


def test_deleting_a_group_takes_its_polygons_and_the_last_group_stays(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    north = storage.add_group("North")
    storage.add_polygon(_feature(SQUARE), north.identifier)
    storage.delete_group(north.identifier)

    assert [g.identifier for g in storage.list_groups()] == ["group_default"]
    assert storage.list_records() == []
    with pytest.raises(ValueError):
        storage.delete_group("group_default")


def _write_results(storage, rows: list[tuple[str, str]]) -> None:
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {
            "analysis_group_id": [g for g, _ in rows],
            "analysis_polygon_id": [p for _, p in rows],
            "analysis_area_m2": [1.0] * len(rows),
        },
        geometry=[box(0, 0, 1, 1)] * len(rows),
        crs="EPSG:4326",
    )
    for path in (aset.analysis_flat_path(storage.base_dir, storage.cfg),
                 aset.analysis_stacked_path(storage.base_dir, storage.cfg)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def test_save_drops_results_for_deleted_polygons_and_groups(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    north = storage.add_group("North")
    stays = storage.add_polygon(_feature(SQUARE), "group_default")
    goes = storage.add_polygon(_feature(OTHER), "group_default")
    north_poly = storage.add_polygon(_feature(SQUARE), north.identifier)
    storage.save()
    _write_results(storage, [("group_default", stays.identifier), ("group_default", goes.identifier),
                             (north.identifier, north_poly.identifier)])

    storage.delete(goes.identifier, "group_default")
    storage.delete_group(north.identifier)
    removed = storage.save()

    assert removed == 4  # two rows in each of the two result tables
    for path in (aset.analysis_flat_path(tmp_path, storage.cfg), aset.analysis_stacked_path(tmp_path, storage.cfg)):
        left = gpd.read_parquet(path)
        assert list(left["analysis_polygon_id"]) == [stays.identifier]
        assert left.crs is not None


def test_save_leaves_results_alone_when_nothing_was_deleted(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    poly = storage.add_polygon(_feature(SQUARE))
    storage.save()
    _write_results(storage, [("group_default", poly.identifier)])
    flat = aset.analysis_flat_path(tmp_path, storage.cfg)
    stamp = flat.stat().st_mtime_ns

    storage.update_properties(poly.identifier, None, "Renamed", "")
    assert storage.save() == 0
    assert flat.stat().st_mtime_ns == stamp


def test_edited_since_run_flags_only_groups_edited_after_their_results(tmp_path: Path) -> None:
    import mesa_shared

    storage = _storage(tmp_path)
    north = storage.add_group("North")
    old_poly = storage.add_polygon(_feature(SQUARE), "group_default")
    new_poly = storage.add_polygon(_feature(OTHER), north.identifier)
    storage.save()
    gpq = storage.polygon_path.parent
    assert mesa_shared.analysis_groups_edited_since_run(gpq) == {}  # no results yet

    # Results stamped later than every polygon edit: nothing is out of date.
    _write_results(storage, [("group_default", old_poly.identifier), (north.identifier, new_poly.identifier)])
    for path in (aset.analysis_flat_path(tmp_path, storage.cfg),):
        frame = gpd.read_parquet(path)
        frame["analysis_timestamp"] = "2999-01-01T00:00:00"
        frame.to_parquet(path, index=False)
    assert mesa_shared.analysis_groups_edited_since_run(gpq) == {}

    # An older run for one group: that group, and only that one, is flagged with the run time.
    frame = gpd.read_parquet(aset.analysis_flat_path(tmp_path, storage.cfg))
    frame.loc[frame["analysis_group_id"] == north.identifier, "analysis_timestamp"] = "2001-02-03T04:05:00"
    frame.to_parquet(aset.analysis_flat_path(tmp_path, storage.cfg), index=False)
    assert mesa_shared.analysis_groups_edited_since_run(gpq) == {north.identifier: "2001-02-03 04:05"}
