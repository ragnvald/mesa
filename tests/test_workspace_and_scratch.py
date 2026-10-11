"""Where a packaged MESA keeps a project, and where the tile stage puts its scratch file."""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import mesa_shared as ms  # noqa: E402
import tiles_create_raster as tcr  # noqa: E402


def test_a_folder_inside_onedrive_is_recognised(tmp_path: Path) -> None:
    onedrive = tmp_path / "OneDrive - Some Agency"
    env = {"OneDriveCommercial": str(onedrive)}
    assert ms.cloud_sync_provider(onedrive / "Documents" / "MESA", env) == "OneDrive"
    assert ms.cloud_sync_provider(tmp_path / "MESA", env) == ""
    # recognised from the path as well, when the environment says nothing
    assert ms.cloud_sync_provider(onedrive / "Documents" / "MESA", {}) == "OneDrive"
    assert ms.cloud_sync_provider(tmp_path / "Dropbox" / "MESA", {}) == "Dropbox"
    assert ms.cloud_sync_provider(tmp_path / "onedriveway" / "MESA", {}) == ""


def test_a_new_workspace_is_proposed_outside_a_synced_documents_folder(tmp_path: Path) -> None:
    home = tmp_path / "home"
    plain = home / "Documents"
    synced = home / "OneDrive - Some Agency" / "Documents"
    assert ms.suggest_workspace_dir(plain, home, {}) == str(plain / "MESA")
    assert ms.suggest_workspace_dir(synced, home, {}) == str(home / "MESA")


def test_a_picked_folder_with_other_files_gets_a_mesa_subfolder(tmp_path: Path) -> None:
    empty = tmp_path / "empty"; empty.mkdir()
    busy = tmp_path / "busy"; busy.mkdir(); (busy / "notes.docx").write_text("x")
    workspace = tmp_path / "ws"; workspace.mkdir(); (workspace / "config.ini").write_text("[DEFAULT]\n")
    (workspace / "log.txt").write_text("x")

    assert ms.resolve_workspace_choice(empty) == str(empty)
    assert ms.resolve_workspace_choice(tmp_path / "not_there_yet") == str(tmp_path / "not_there_yet")
    assert ms.resolve_workspace_choice(busy) == str(busy / "MESA")
    assert ms.resolve_workspace_choice(workspace) == str(workspace)      # an existing workspace is used as it is


def test_the_version_a_workspace_was_last_used_with(tmp_path: Path) -> None:
    assert ms.workspace_version_seen(tmp_path) == ""                     # not a workspace
    (tmp_path / "config.ini").write_text("[DEFAULT]\n# comment\nmesa_version                    = 5.6.0\n", encoding="utf-8")
    assert ms.workspace_version_seen(tmp_path) == "5.6.0"                # from config.ini until MESA has marked it
    ms.write_workspace_marker(tmp_path, "5.7.0")
    assert ms.workspace_version_seen(tmp_path) == "5.7.0"


def test_setting_a_workspace_aside_renames_it_and_deletes_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "MESA"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.parquet").write_text("data")
    when = datetime.datetime(2026, 10, 12, 9, 30)

    moved = ms.archive_workspace(workspace, "5.6.0", now=when)

    assert moved == str(tmp_path / "MESA_5.6.0_2026-10-12_0930")
    assert not workspace.exists()
    assert (Path(moved) / "output" / "result.parquet").read_text() == "data"
    # a second one in the same minute gets its own name
    workspace.mkdir()
    assert ms.archive_workspace(workspace, "5.6.0", now=when).endswith("_0930_2")


def _geoms():
    return np.array([box(0, 0, 1, 1), box(1, 1, 2, 2)], dtype=object)


def test_tile_scratch_file_goes_beside_the_tiles_in_an_ordinary_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "cloud_sync_provider", lambda path, env=None: "")
    path, offsets = tcr.spill_wkb_scratch(_geoms(), tmp_path, "grid")
    assert path.parent == tmp_path and path.exists()
    assert str(os.getpid()) in path.name
    assert len(offsets) == 3 and offsets[-1] == path.stat().st_size


def test_tile_scratch_file_avoids_a_synced_project_folder(tmp_path: Path, monkeypatch) -> None:
    scratch = tmp_path / "temp"; scratch.mkdir()
    project = tmp_path / "OneDrive - Some Agency" / "MESA" / "output" / "mbtiles"; project.mkdir(parents=True)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(scratch))
    notes: list[str] = []

    path, _offsets = tcr.spill_wkb_scratch(_geoms(), project, "grid", notes.append)

    assert path.parent == scratch and not list(project.iterdir())
    assert notes and "OneDrive" in notes[0]


def test_tile_scratch_file_falls_back_when_the_first_place_refuses_it(tmp_path: Path, monkeypatch) -> None:
    scratch = tmp_path / "temp"; scratch.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(scratch))
    monkeypatch.setattr(ms, "cloud_sync_provider", lambda path, env=None: "")
    real = tcr.spill_wkb

    def refusing(geoms, out_path, chunk=500_000):
        if Path(out_path).parent == tmp_path:
            raise OSError(22, "Invalid argument", str(out_path))
        return real(geoms, out_path, chunk)

    monkeypatch.setattr(tcr, "spill_wkb", refusing)
    path, offsets = tcr.spill_wkb_scratch(_geoms(), tmp_path, "grid")
    assert path.parent == scratch and len(offsets) == 3
