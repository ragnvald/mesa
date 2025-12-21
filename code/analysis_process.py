# -*- coding: utf-8 -*-
"""analysis_process.py — Run study area analysis processing.

This tool is intentionally minimal: it provides a single button that runs the
analysis processing for the configured study areas.

Inputs (created via data_analysis_setup.py):
- tbl_analysis_group.parquet
- tbl_analysis_polygons.parquet

Outputs (consumed by data_analysis_presentation.py):
- tbl_analysis_flat.parquet
- tbl_analysis_stacked.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from typing import Callable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run study area analysis processing")
    parser.add_argument(
        "--original_working_directory",
        default=None,
        help="Base directory for resolving config.ini and output paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Reuse the implementation from data_analysis_setup without duplicating the GIS logic.
    # (This module contains AnalysisStorage + AssetAnalyzer + path resolution.)
    from data_analysis_setup import (
        AnalysisStorage,
        AssetAnalyzer,
        debug_log,
        read_config,
        resolve_base_dir,
    )

    base_dir = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base_dir)

    root = tk.Tk()
    root.title("Analysis processing")
    root.geometry("780x520")

    log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
    status_queue: queue.SimpleQueue[str] = queue.SimpleQueue()

    def _timestamp() -> str:
        return dt.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

    def log_line(message: str) -> None:
        formatted = f"{_timestamp()} - {message}"
        log_queue.put(formatted)
        try:
            with open(base_dir / "log.txt", "a", encoding="utf-8") as fh:
                fh.write(formatted + "\n")
        except Exception:
            pass
        try:
            debug_log(base_dir, message)
        except Exception:
            pass

    def set_status(text: str) -> None:
        status_queue.put(text)

    status = tk.StringVar(value="Ready.")

    def run_processing() -> None:
        try:
            set_status("Loading study areas...")
            log_line("Starting analysis processing.")
            log_line("This run processes ALL analysis groups and ALL polygons within each group.")

            storage = AnalysisStorage(base_dir, cfg)
            analyzer = AssetAnalyzer(base_dir, cfg)

            groups = storage.list_groups()
            total_groups = len(groups)
            total_polygons = 0
            for g in groups:
                try:
                    total_polygons += len(storage.list_records(g.identifier))
                except Exception:
                    pass
            log_line(f"Found {total_groups} analysis group(s) with {total_polygons} polygon(s) total.")

            ran_any = False
            processed_groups = 0
            skipped_groups = 0

            for group in groups:
                records = storage.list_records(group.identifier)
                if not records:
                    skipped_groups += 1
                    log_line(f"Skipping group '{group.name}' ({group.identifier}): no polygons.")
                    continue

                ran_any = True
                set_status(f"Processing group '{group.name}' ({processed_groups + skipped_groups + 1}/{total_groups})...")
                log_line(f"Processing group '{group.name}' ({group.identifier}) with {len(records)} polygon(s)...")
                result = analyzer.run_group_analysis(group, records, geocode=group.default_geocode)
                flat_rows = result.get("flat_rows")
                stacked_rows = result.get("stacked_rows")
                log_line(
                    f"Completed group '{group.name}': flat_rows={flat_rows} stacked_rows={stacked_rows} "
                    f"flat_path={result.get('flat_path')} stacked_path={result.get('stacked_path')}"
                )
                processed_groups += 1

            if not ran_any:
                set_status("No polygons found.")
                messagebox.showwarning(
                    "Nothing to process",
                    "No study area polygons were found. Use the 'Set up analysis' tool first.",
                )
                log_line("No study area polygons found; nothing to do.")
                return

            set_status("Done.")
            log_line(
                f"Processing complete. Processed {processed_groups} group(s); skipped {skipped_groups} empty group(s)."
            )
            messagebox.showinfo(
                "Analysis processing complete",
                f"Processed {processed_groups} group(s). Skipped {skipped_groups} empty group(s).\n\n"
                "Outputs written to output/geoparquet (tbl_analysis_flat.parquet, tbl_analysis_stacked.parquet).",
            )
        except Exception as exc:
            set_status("Error.")
            log_line(f"ERROR: {exc}")
            messagebox.showerror("Analysis processing failed", str(exc))
        finally:
            try:
                root.after(0, lambda: run_btn.config(state=tk.NORMAL))
            except Exception:
                pass

    def on_click() -> None:
        run_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=run_processing, daemon=True)
        thread.start()

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    title = tk.Label(frame, text="Run analysis processing", font=("Segoe UI", 12, "bold"))
    title.pack(anchor="w")

    desc = tk.Label(
        frame,
        text=(
            "Runs the study area analysis for configured polygons and writes the analysis tables.\n"
            "This processes all analysis groups and all polygons (sub-polygons) in each group.\n"
            "Use 'Set up analysis' first to create groups and polygons."
        ),
        justify="left",
        wraplength=740,
    )
    desc.pack(anchor="w", pady=(6, 12))

    log_box = scrolledtext.ScrolledText(frame, height=18, wrap=tk.WORD)
    log_box.pack(fill="both", expand=True, pady=(0, 12))
    log_box.insert(tk.END, f"{_timestamp()} - Ready.\n")
    log_box.see(tk.END)

    run_btn = tk.Button(frame, text="Start analysis processing", command=on_click, width=24)
    run_btn.pack(anchor="w")

    status_lbl = tk.Label(frame, textvariable=status, fg="#334155")
    status_lbl.pack(anchor="w", pady=(12, 0))

    def _pump_queues() -> None:
        try:
            while True:
                line = log_queue.get_nowait()
                log_box.insert(tk.END, line + "\n")
                log_box.see(tk.END)
        except Exception:
            pass

        try:
            while True:
                st = status_queue.get_nowait()
                status.set(st)
        except Exception:
            pass

        root.after(100, _pump_queues)

    _pump_queues()

    root.mainloop()


if __name__ == "__main__":
    main()
