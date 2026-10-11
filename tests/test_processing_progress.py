"""The processing runner's progress bar: nothing from an earlier run may reach it."""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pytest  # noqa: E402

import processing_pipeline_run as ppr  # noqa: E402


def test_the_data_stage_stops_reporting_when_it_fails(tmp_path: Path, monkeypatch) -> None:
    # A cancelled or failed data stage used to leave its progress pulse running; the
    # pulse then pushed its own percentage into the bar of the next run.
    def boom(*_args, **_kwargs):
        time.sleep(0.3)
        raise RuntimeError("cancelled")

    monkeypatch.setitem(sys.modules, "processing_internal", types.SimpleNamespace(run_headless=boom))
    real_pulse = ppr._start_progress_pulse
    monkeypatch.setattr(ppr, "_start_progress_pulse",
                        lambda fn, **kw: real_pulse(fn, **{**kw, "interval_s": 0.2}))
    seen: list[float] = []

    with pytest.raises(RuntimeError):
        ppr.run_data_process(tmp_path, lambda _line: None, seen.append)

    count = len(seen)
    time.sleep(0.8)                      # four pulse intervals
    assert len(seen) == count
    assert seen == sorted(seen)          # and what was reported only went up
