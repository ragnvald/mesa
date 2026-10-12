"""The launcher's own data tools: what Clear output and Delete all data leave behind."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "code"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mesa  # noqa: E402


def test_delete_all_data_keeps_the_note_in_each_source_folder(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    (asset / "sub").mkdir(parents=True)
    (asset / "readme.txt").write_text("what goes here")
    (asset / "roads.gpkg").write_text("x")
    (asset / "sub" / "more.shp").write_text("x")
    (asset / ".gitkeep").write_text("")

    removed, kept, failed = mesa.MesaMainWindow._purge_directory(str(asset), keep=("readme.txt",))

    assert removed == 2 and not failed
    assert sorted(os.listdir(asset)) == [".gitkeep", "readme.txt"]
    assert sorted(kept) == [".gitkeep", "readme.txt"]


def test_clear_output_removes_everything_in_the_folder(tmp_path: Path) -> None:
    out = tmp_path / "output"
    (out / "geoparquet").mkdir(parents=True)
    (out / "readme.txt").write_text("not a note we keep here")
    removed, _kept, failed = mesa.MesaMainWindow._purge_directory(str(out))
    assert removed == 2 and not failed and os.listdir(out) == []
