#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmentation_setup.py — Classification (setup / config UI).

PURPOSE
    Configure and launch the Classification clustering (the "what kind of
    sensitivity pattern is this place part of?" view that complements the A–E
    "how sensitive is this place?" classification). Cells are clustered by the
    shape of their (importance, susceptibility) histogram. Lets the operator
    choose the geocode layer, the k range searched by BIC, the histogram
    transform (Hellinger/CLR), the coverage-feature weight, and a minimum-area
    sliver filter; persists the choices to config.ini (segmv_* keys) and
    optionally runs the heavy helper.

INPUTS
    output/geoparquet/tbl_geocode_group, tbl_geocode_object  (layer list)
    output/geoparquet/tbl_stacked                            (pressure detection)
    config.ini                                               (segmv_* defaults)

OUTPUTS
    config.ini segmv_* keys (comment-preserving write). On "Run now", spawns
    segmentation_run as a subprocess (see its OUTPUTS).

CALLED BY
    mesa.exe launcher (Workflows tab → "Sensitivity generalisation"), or
    standalone:  python code/segmentation_setup.py --original_working_directory <dir>

CALLS
    code/segmentation_run.py (params_from_config, detect_pressure_columns, the
    subprocess run), code/segmentation.py (list_geocode_layers),
    code/mesa_shared.py (find_base_dir, read_config, parquet_dir),
    ui_style.apply_shared_stylesheet (GIS-free shared Qt look-and-feel).

NOTES
    MESA v5+ feature. Lightweight — imports no heavy compute libs (sklearn etc.);
    those live only in segmentation_run, which is launched as a separate process.
    Writes only segmv_* config keys + the new tbl_seg_mv* tables via the run
    helper; never touches the shipped tbl_segmentation* namespace.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mesa_shared

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QSpinBox, QMessageBox, QProgressBar, QPlainTextEdit,
)
from PySide6.QtCore import Qt, QThread, Signal

try:
    from ui_style import apply_shared_stylesheet
except Exception:  # keep standalone-launchable even if the shared sheet is unavailable
    def apply_shared_stylesheet(_app):  # type: ignore
        return None

import segmentation as _seg
import segmentation_run as _run


def _shared_window_icon(base_dir: Path):
    from PySide6.QtGui import QIcon
    for candidate in (base_dir / "system_resources" / "icon.png",
                      base_dir / "system_resources" / "mesa.ico"):
        try:
            if candidate.exists():
                icon = QIcon(str(candidate))
                if not icon.isNull():
                    return icon
        except Exception:
            pass
    return QIcon()


def _update_config(cfg_path: str, **kwargs) -> None:
    """Comment-preserving write of [DEFAULT] keys (copy of processing_setup's
    update_config_with_values — kept local so this helper stays lightweight)."""
    if not os.path.isabs(cfg_path):
        raise ValueError("cfg_path must be absolute")
    if not os.path.isfile(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("[DEFAULT]\n")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not any(line.strip().startswith("[") for line in lines):
        lines.insert(0, "[DEFAULT]\n")
    for key, value in kwargs.items():
        found = False
        key_norm = key.strip().casefold()
        for i, line in enumerate(lines):
            left, sep, _right = line.partition("=")
            if sep == "" or left.strip().casefold() != key_norm:
                continue
            indent = left[: len(left) - len(left.lstrip())]
            lines[i] = f"{indent}{key} = {value}\n"
            found = True
            break
        if not found:
            # Insert into the [DEFAULT] section, not EOF: appending at EOF can land
            # inside a later section (e.g. [VALID_VALUES]), where the key is invisible
            # to cfg["DEFAULT"].get(). Find the end of the DEFAULT block.
            insert_at, in_default = len(lines), False
            for i, line in enumerate(lines):
                s = line.strip()
                if s.startswith("["):
                    if in_default:
                        insert_at = i
                        break
                    if s.lower() == "[default]":
                        in_default = True
            lines.insert(insert_at, f"{key} = {value}\n")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


class _ProcReader(QThread):
    """Reads a subprocess' merged stdout line-by-line off the GUI thread and
    relays each line; emits the exit code when the stream closes."""

    line = Signal(str)
    done = Signal(int)

    def __init__(self, popen):
        super().__init__()
        self._p = popen

    def run(self):
        out = self._p.stdout
        try:
            if out is not None:
                for raw in iter(out.readline, b""):
                    s = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if s:
                        self.line.emit(s)
        except Exception:
            pass
        finally:
            try:
                if out is not None:
                    out.close()
            except Exception:
                pass
        try:
            rc = self._p.wait()
        except Exception:
            rc = -1
        self.done.emit(int(rc if rc is not None else -1))


class SegmentationSetupWindow(QMainWindow):
    def __init__(self, base_dir: Path):
        super().__init__()
        self._base_dir = Path(base_dir)
        self._config_file = str(self._base_dir / "config.ini")
        self._cfg = mesa_shared.read_config(self._base_dir)
        self._gpq = mesa_shared.parquet_dir(self._base_dir, self._cfg)
        self._params = _run.params_from_config(self._cfg)
        self._popen = None    # the running segmentation_run subprocess (Popen)
        self._reader = None   # background stdout reader thread

        self.setWindowTitle("MESA – Classification (setup)")
        self.resize(900, 700)
        icon = _shared_window_icon(self._base_dir)
        if not icon.isNull():
            self.setWindowIcon(icon)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Run output up top: a compact live log, the progress bar right under it,
        # then the settings below.
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(84)
        self._log_view.setMaximumHeight(120)
        self._log_view.setPlaceholderText("Run output appears here…")
        layout.addWidget(self._log_view)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        intro = QLabel(
            "Group polygons into <b>types</b> of sensitivity pattern (composition / "
            "character) from their full stacked per-asset profile — complementary to the "
            "A–E sensitivity classes. The classification answers <i>“how sensitive is this "
            "place?”</i>; this answers <i>“what kind of sensitivity pattern is this place "
            "part of?”</i>")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #5c4a2f; font-size: 9pt;")
        layout.addWidget(intro)

        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        layout.addLayout(form)
        r = 0

        # Geocode layer + Number of classes — side by side on one row.
        form.addWidget(QLabel("Geocode layer:"), r, 0, Qt.AlignRight)
        self._cmb_layer = QComboBox()
        try:
            layers = _seg.list_geocode_layers(self._gpq)
        except Exception:
            layers = []
        self._cmb_layer.addItems(layers or ["basic_mosaic"])
        want = self._params.layer or "basic_mosaic"
        idx = self._cmb_layer.findText(want)
        self._cmb_layer.setCurrentIndex(idx if idx >= 0 else 0)
        form.addWidget(self._cmb_layer, r, 1)

        form.addWidget(QLabel("k range (BIC search):"), r, 2, Qt.AlignRight)
        self._txt_k = QLineEdit(f"{self._params.k_min}-{self._params.k_max}")
        self._txt_k.setPlaceholderText("min-max, e.g. 2-15")
        self._txt_k.setToolTip("A Gaussian Mixture is fit for every k in this range; "
                               "the best is chosen automatically by minimum BIC.")
        form.addWidget(self._txt_k, r, 3); r += 1

        # The single-control rows below span the field columns (1–3) so they fill
        # the width set by the two-pair header row above.
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        # Histogram transform + coverage weight on one row.
        form.addWidget(QLabel("Histogram transform:"), r, 0, Qt.AlignRight)
        self._cmb_transform = QComboBox()
        self._cmb_transform.addItem("Hellinger (recommended)", userData="hellinger")
        self._cmb_transform.addItem("CLR (centred log-ratio)", userData="clr")
        ti = self._cmb_transform.findData((self._params.transform or "hellinger"))
        self._cmb_transform.setCurrentIndex(ti if ti >= 0 else 0)
        self._cmb_transform.setToolTip(
            "How the (importance, susceptibility) proportions are transformed before "
            "clustering. Hellinger (√p) handles zeros without pseudocounts; CLR is the "
            "compositional alternative.")
        form.addWidget(self._cmb_transform, r, 1)

        form.addWidget(QLabel("Coverage weight:"), r, 2, Qt.AlignRight)
        self._txt_cov = QLineEdit(str(self._params.coverage_weight))
        self._txt_cov.setPlaceholderText("0 disables; 1.0 default")
        self._txt_cov.setToolTip(
            "Weight of the standardised coverage/intensity feature (overlap depth) "
            "appended to the histogram. 0 clusters on histogram shape only.")
        form.addWidget(self._txt_cov, r, 3); r += 1

        # Plain-language explanation of the method.
        method_help = QLabel(
            "Cells are grouped by the <b>shape</b> of their (importance, susceptibility) "
            "histogram, so a place dominated by high-importance / low-susceptibility "
            "assets separates from a high-susceptibility / low-importance one even when "
            "their A–E sensitivity code is identical. The number of types is chosen "
            "automatically by BIC over the k range; per-cell certainty (p_max, entropy) "
            "is kept for the map.")
        method_help.setWordWrap(True)
        method_help.setStyleSheet("color: #6a5533; font-size: 9pt;")
        form.addWidget(method_help, r, 1, 1, 3); r += 1

        # Pressure filter — only shown when tbl_stacked actually has pressure
        # columns. With just the aggregate option there's nothing to choose, so the
        # combo is kept (as a data holder for _collect_values) but the row is not
        # added to the form. The canonical MESA stack has no pressure column, so
        # this row is usually absent.
        self._cmb_pressure = QComboBox()
        self._cmb_pressure.addItem("All pressures (aggregate)", userData="")
        try:
            press = _run.detect_pressure_columns(self._gpq)
        except Exception:
            press = []
        for pc in press:
            self._cmb_pressure.addItem(pc, userData=pc)
        if press:
            form.addWidget(QLabel("Pressure filter:"), r, 0, Qt.AlignRight)
            form.addWidget(self._cmb_pressure, r, 1, 1, 3); r += 1
        else:
            self._cmb_pressure.hide()

        # Min area
        form.addWidget(QLabel("Min polygon area (m²):"), r, 0, Qt.AlignRight)
        self._spn_area = QSpinBox()
        self._spn_area.setRange(0, 1_000_000_000)
        self._spn_area.setSingleStep(100)
        self._spn_area.setGroupSeparatorShown(True)
        self._spn_area.setValue(int(self._params.min_area_m2 or 0))
        self._spn_area.setToolTip("Drop slivers below this area before clustering (0 = keep all).")
        form.addWidget(self._spn_area, r, 1, 1, 3); r += 1

        # AI toggle
        self._cb_ai = QCheckBox("Generate AI plain-language descriptions per class (optional, off by default)")
        self._cb_ai.setChecked(bool(self._params.ai_enabled))
        self._cb_ai.setToolTip(
            "Calls a local Ollama model (or a configured OpenAI key) to describe each "
            "class. Requires Ollama running at localhost:11434, or OPENAI_API_KEY.")
        layout.addWidget(self._cb_ai)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setProperty("role", "muted")
        layout.addWidget(self._status)

        layout.addStretch()

        # Buttons: Run now + Save settings to the far left, Exit bottom-right.
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run now")
        self._run_btn.setProperty("role", "primary")
        self._run_btn.clicked.connect(self._on_run_now)
        btn_row.addWidget(self._run_btn)
        self._save_btn = QPushButton("Save settings")
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        btn_row.addWidget(exit_btn)
        layout.addLayout(btn_row)

    # -- helpers ---------------------------------------------------------
    def _collect(self) -> dict:
        k_raw = self._txt_k.text().strip() or "2-15"
        cov = (self._txt_cov.text().strip() or "1.0")
        return {
            "segmv_geocode_layer": self._cmb_layer.currentText(),
            "segmv_k_range": k_raw,
            "segmv_transform": self._cmb_transform.currentData() or "hellinger",
            "segmv_coverage_weight": cov,
            "segmv_pressure": self._cmb_pressure.currentData() or "",
            "segmv_min_area_m2": int(self._spn_area.value()),
            "segmv_ai_enabled": "1" if self._cb_ai.isChecked() else "0",
        }

    def _validate_k(self) -> bool:
        raw = self._txt_k.text().strip().replace(":", "-")
        parts = [p for p in raw.split("-") if p.strip()]
        ok = len(parts) in (1, 2) and all(p.strip().isdigit() and int(p) >= 2 for p in parts)
        if ok and len(parts) == 2 and int(parts[0]) > int(parts[1]):
            ok = False
        if not ok:
            QMessageBox.warning(self, "Invalid k range",
                                "Enter a range like 2-15 (or a single integer ≥ 2).")
            return False
        try:
            float(self._txt_cov.text().strip() or "1.0")
        except ValueError:
            QMessageBox.warning(self, "Invalid coverage weight",
                                "Coverage weight must be a number (e.g. 0, 1.0, 2.5).")
            return False
        return True

    def _on_save(self) -> bool:
        if not self._validate_k():
            return False
        try:
            _update_config(self._config_file, **self._collect())
            self._status.setText("Settings saved to config.ini.")
            return True
        except Exception as exc:
            self._status.setText(f"Could not save settings: {exc}")
            return False

    def _on_run_now(self):
        if self._reader is not None and self._reader.isRunning():
            return  # a run is already in flight
        if not self._on_save():
            return
        vals = self._collect()
        args = [
            "--original_working_directory", str(self._base_dir),
            "--layer", vals["segmv_geocode_layer"],
            "--k-range", str(vals["segmv_k_range"]),
            "--transform", vals["segmv_transform"],
            "--coverage-weight", str(vals["segmv_coverage_weight"]),
            "--min-area-m2", str(vals["segmv_min_area_m2"]),
        ]
        if vals["segmv_pressure"]:
            args += ["--pressure", vals["segmv_pressure"]]
        if vals["segmv_ai_enabled"] == "1":
            args += ["--ai"]
        self._start_process(args)

    # -- run as a monitored subprocess (progress bar + live log) ----------
    def _run_command(self, args: list[str]):
        """(program, arguments) for segmentation_run — the frozen exe when packaged,
        else this Python interpreter on the script. Heavy compute (sklearn etc.)
        stays in that separate process; this window only watches it."""
        if getattr(sys, "frozen", False):
            exe = Path(sys.executable).resolve().parent / "segmentation_run.exe"
            return str(exe), list(args)
        script = str(Path(__file__).resolve().parent / "segmentation_run.py")
        return (sys.executable or "python"), [script, *args]

    def _start_process(self, args: list[str]) -> None:
        program, full_args = self._run_command(args)
        cmd = [program, *full_args]
        # CREATE_NO_WINDOW stops a console window from flashing up for the helper:
        # a windowed parent launching this console-mode child otherwise gets a
        # fresh console allocated (redirecting stdout to a pipe doesn't prevent it,
        # and PySide6 lacks QProcess's CreateProcess-flags modifier). Popen + a
        # reader thread gives us that flag and keeps the live log working.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress.setRange(0, 0)   # busy until the first progress marker arrives
        self._progress.setValue(0)
        self._log_view.clear()
        self._status.setText("Running classification…")

        try:
            self._popen = subprocess.Popen(
                cmd, cwd=str(self._base_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=creationflags)
        except Exception as exc:
            self._popen = None
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._status.setText(f"Could not start the classification helper: {exc}")
            self._run_btn.setEnabled(True)
            self._save_btn.setEnabled(True)
            return

        self._reader = _ProcReader(self._popen)
        self._reader.line.connect(self._handle_line)
        self._reader.done.connect(self._on_proc_done)
        self._reader.start()

    _PROGRESS_TOKEN = "@@SEGMV_PROGRESS"

    def _handle_line(self, line: str) -> None:
        if self._PROGRESS_TOKEN in line:
            self._apply_progress(line)
        elif line.strip():
            self._append_log(line)

    def _apply_progress(self, line: str) -> None:
        try:
            payload = line.split(self._PROGRESS_TOKEN, 1)[1].strip()
            parts = payload.split(None, 2)
            done, total = int(parts[0]), max(1, int(parts[1]))
            label = parts[2] if len(parts) > 2 else ""
            self._progress.setRange(0, total)
            self._progress.setValue(min(done, total))
            pct = int(round(done * 100 / total))
            self._status.setText(f"Running classification… {pct}%" + (f" — {label}" if label else ""))
        except Exception:
            pass  # malformed marker: ignore, the bar just stays where it was

    def _append_log(self, line: str) -> None:
        self._log_view.appendPlainText(line)
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_proc_done(self, code: int) -> None:
        if code == 0:
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._status.setText(
                "Done — results saved to output/segmentation_mv/. "
                "Open Maps → Classifications to view them.")
        else:
            self._progress.setRange(0, 100)
            self._status.setText(f"Run failed (exit code {code}). See the log above.")
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    def closeEvent(self, event):
        try:
            if self._popen is not None and self._popen.poll() is None:
                self._popen.kill()
        except Exception:
            pass
        try:
            if self._reader is not None and self._reader.isRunning():
                self._reader.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)


def run(base_dir: str, master=None):
    bd = Path(mesa_shared.find_base_dir(base_dir))
    app = QApplication.instance()
    own = False
    if app is None:
        app = QApplication([])
        apply_shared_stylesheet(app)
        own = True
    win = SegmentationSetupWindow(bd)
    win.show()
    if own:
        app.exec()
    return win


def main(argv=None):
    ap = argparse.ArgumentParser(description="MESA — Sensitivity generalisation (setup)")
    ap.add_argument("--original_working_directory", required=False, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    run(args.original_working_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
