from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

import processing_pipeline_run as pipeline  # noqa: E402


def _empty_plan(*, run_qgis: bool) -> pipeline.ProcessPlan:
    return pipeline.ProcessPlan(
        run_prep=False,
        run_intersect=False,
        run_flatten=False,
        run_segment=False,
        explode_flat_multipolygons=False,
        cleanup_slivers=True,
        run_classification=False,
        run_tiles=False,
        run_qgis=run_qgis,
        run_lines=False,
        run_analysis=False,
    )


def test_qgis_stage_runs_without_tiles_when_tbl_flat_exists(
    tmp_path: Path, monkeypatch
) -> None:
    flat = tmp_path / "output" / "geoparquet" / "tbl_flat.parquet"
    flat.parent.mkdir(parents=True)
    flat.touch()
    calls: list[Path] = []

    monkeypatch.setattr(
        pipeline,
        "run_qgis_process",
        lambda base_dir, cfg, log_fn, progress_fn: (
            calls.append(base_dir), progress_fn(100.0)
        ),
    )
    monkeypatch.setattr(pipeline, "_append_stage_runtime", lambda *args, **kwargs: None)
    logs: list[str] = []
    progress: list[float] = []

    pipeline.run_selected(
        tmp_path,
        configparser.ConfigParser(),
        _empty_plan(run_qgis=True),
        logs.append,
        progress.append,
    )

    assert calls == [tmp_path]
    assert any("Progress range QGIS" in line for line in logs)
    assert progress[-1] == 100.0


def test_advanced_ui_has_enabled_qgis_checkbox_for_existing_tbl_flat(tmp_path: Path) -> None:
    (tmp_path / "qgis").mkdir(parents=True)
    (tmp_path / "qgis" / "mesa.qgz").touch()
    flat = tmp_path / "output" / "geoparquet" / "tbl_flat.parquet"
    flat.parent.mkdir(parents=True)
    flat.touch()

    app = QApplication.instance() or QApplication([])
    window = pipeline.ProcessRunnerWindow(tmp_path, configparser.ConfigParser())
    try:
        # Numbered 7 since the Backfill stage was removed and the list closed up.
        assert window._cb_qgis.text().strip() == "7. QGIS project (mesa_results.qgz)"
        assert window._cb_qgis.isEnabled()
        assert window._cb_qgis.isChecked()
        assert window._cb_qgis in window.findChildren(type(window._cb_qgis))
    finally:
        window.close()
        app.processEvents()
