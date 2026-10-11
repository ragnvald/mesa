"""The Lines editor stages edits in memory; commit() is the only writer of tbl_lines.

The Lines stage and the report read tbl_lines from disk, so these tests pin
down what they get. See devtools/docs/SPECIAL_FOCUS_REDESIGN_PLAN.md.
"""
from __future__ import annotations

import configparser
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import geopandas as gpd  # noqa: E402

import line_manage as lm  # noqa: E402
import mesa_shared  # noqa: E402

LINE = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}, "properties": {}}


def _api(tmp_path: Path) -> "lm.Api":
    cfg = configparser.ConfigParser()
    cfg["DEFAULT"] = {"parquet_folder": "output/geoparquet", "segment_length": "1000", "segment_width": "600"}
    return lm.Api(str(tmp_path), cfg)


def _lines_file(tmp_path: Path) -> Path:
    return Path(lm.lines_parquet_path(str(tmp_path)))


def test_a_new_line_is_staged_until_commit(tmp_path: Path) -> None:
    api = _api(tmp_path)
    assert api.has_unsaved() is False

    added = api.add_line(LINE)
    assert added["ok"] and api.has_unsaved() is True
    assert not _lines_file(tmp_path).exists()

    saved = api.commit()
    assert saved["ok"] and api.has_unsaved() is False
    on_disk = gpd.read_parquet(_lines_file(tmp_path))
    assert list(on_disk["name_gis"]) == [added["record"]["name_gis"]]
    assert not list(_lines_file(tmp_path).parent.glob("*.tmp"))


def test_commit_fills_missing_segment_values_with_the_defaults(tmp_path: Path) -> None:
    # The Lines stage skips a line whose segment length or width is empty.
    api = _api(tmp_path)
    name = api.add_line(LINE)["record"]["name_gis"]
    api.update_attribute(name, "segment_length", "")

    saved = api.commit()
    row = gpd.read_parquet(_lines_file(tmp_path)).iloc[0]
    assert saved["defaults_filled"] >= 1
    assert (int(row["segment_length"]), int(row["segment_width"])) == (1000, 600)


def test_discard_goes_back_to_the_saved_lines(tmp_path: Path) -> None:
    api = _api(tmp_path)
    kept = api.add_line(LINE)["record"]["name_gis"]
    api.commit()

    api.add_line(LINE)
    api.delete_line(kept)
    assert api.has_unsaved() is True
    assert api.discard()["ok"] and api.has_unsaved() is False
    assert [r["name_gis"] for r in api.get_state()["records"]] == [kept]


class _FakeEvent:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _FakeWindow:
    def __init__(self, answer: bool) -> None:
        self.events = type("E", (), {})()
        self.events.closing = _FakeEvent()
        self.answer = answer
        self.asked = []

    def create_confirmation_dialog(self, title: str, message: str) -> bool:
        self.asked.append(message)
        return self.answer


def test_close_guard_only_asks_when_something_is_unsaved() -> None:
    unsaved: list[str] = []
    window = _FakeWindow(answer=False)
    mesa_shared.guard_unsaved_on_close(window, lambda: unsaved)
    closing = window.events.closing.handlers[0]

    assert closing() is None and window.asked == []          # nothing unsaved: closes silently

    unsaved[:] = ["Lines", "Analysis"]
    assert closing() is False                                  # Cancel vetoes the close
    assert "Lines, Analysis" in window.asked[0]

    window.answer = True
    assert closing() is None                                   # OK lets it close
