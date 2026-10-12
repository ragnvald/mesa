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


def test_numeric_extremes_survive_when_a_code_is_split_over_partitions() -> None:
    # tbl_flat went out with importance_max, sensitivity_max and their like empty
    # whenever a geocode's rows were spread over several partition files: the value
    # column was in both tables and the merge renamed the copies to _x and _y.
    import pandas as pd
    import processing_internal as dpi

    per_code = pd.DataFrame({"code": ["a", "b"], "assets_overlap_total": [3, 1],
                             "importance_max": pd.array([4, 2], dtype="Int64"),
                             "importance_min": pd.array([1, 2], dtype="Int64")})
    winners_max = pd.DataFrame({"code": ["a", "b"], "importance_max": pd.array([4, 2], dtype="Int64"),
                                "importance_code_max": ["B", "D"]})
    winners_min = pd.DataFrame({"code": ["a", "b"], "importance_min": pd.array([1, 2], dtype="Int64"),
                                "importance_code_min": ["E", "D"]})

    out = dpi._merge_code_extremes(per_code, [winners_min, winners_max])

    assert not [c for c in out.columns if c.endswith(("_x", "_y"))]
    assert out["importance_max"].tolist() == [4, 2] and out["importance_min"].tolist() == [1, 2]
    assert out["importance_code_max"].tolist() == ["B", "D"]
    assert out["assets_overlap_total"].tolist() == [3, 1]
