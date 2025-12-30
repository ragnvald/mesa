#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil
import zipfile
import datetime
import threading
import queue
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.ttk as ttk_native

try:
    import ttkbootstrap as tb
except Exception:
    tb = None


def _normalize_base_dir(path: str | None) -> str:
    """Resolve the MESA base folder.

    When launched from inside the `code/` folder (dev) or from frozen builds,
    we normalize to the project root that contains `input/` and `output/`.
    """

    candidates: list[Path] = []
    try:
        env_hint = os.environ.get("MESA_BASE_DIR")
        if env_hint:
            candidates.append(Path(env_hint))
    except Exception:
        pass

    if path:
        candidates.append(Path(path))

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    else:
        candidates.append(Path(__file__).resolve().parent)

    candidates.append(Path(os.getcwd()).resolve())

    def normalize(p: Path) -> Path:
        p = p.resolve()
        if p.name.lower() in {"tools", "system", "code"}:
            p = p.parent
        q = p
        for _ in range(5):
            if (q / "input").exists() and (q / "output").exists():
                return q
            if (q / "config.ini").exists() and ((q / "input").exists() or (q / "output").exists()):
                return q
            q = q.parent
        return p

    for c in candidates:
        root = normalize(c)
        if (root / "input").exists() or (root / "output").exists() or (root / "config.ini").exists():
            return str(root)

    return str(normalize(candidates[0]))


def _safe_relpath(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    rel_str = rel.as_posix().lstrip("/")
    return rel_str


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__"}]
        for fn in filenames:
            yield Path(dirpath) / fn


def create_backup_zip(
    base_dir: str,
    dest_folder: str,
    progress_cb=None,
) -> str:
    base = Path(base_dir)
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)

    config_path = base / "config.ini"
    input_dir = base / "input"
    output_dir = base / "output"

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_path = dest / f"mesa_backup_{ts}.zip"

    files_to_add: list[tuple[Path, str]] = []
    if config_path.is_file():
        files_to_add.append((config_path, "config.ini"))

    if input_dir.is_dir():
        for file_path in _iter_files(input_dir):
            if file_path.is_file():
                files_to_add.append((file_path, _safe_relpath(file_path, base)))

    if output_dir.is_dir():
        for file_path in _iter_files(output_dir):
            if file_path.is_file():
                files_to_add.append((file_path, _safe_relpath(file_path, base)))

    total = max(1, len(files_to_add))
    if progress_cb:
        progress_cb(0, total, "Preparing backup…")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, (file_path, arcname) in enumerate(files_to_add, start=1):
            zf.write(file_path, arcname=arcname)
            if progress_cb:
                progress_cb(idx, total, f"Adding: {arcname}")

    return str(zip_path)


def _zip_members_safe(members: list[str]) -> list[str]:
    safe = []
    for m in members:
        name = str(m).replace("\\", "/")
        if name.startswith("/"):
            continue
        parts = [p for p in name.split("/") if p]
        if any(p == ".." for p in parts):
            continue
        safe.append("/".join(parts))
    return safe


def _safe_extract_member(zf: zipfile.ZipFile, member: str, base: Path) -> None:
    name = str(member).replace("\\", "/")
    if name.startswith("/"):
        raise ValueError(f"Unsafe ZIP member (absolute path): {member}")
    parts = [p for p in name.split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError(f"Unsafe ZIP member (path traversal): {member}")

    # Directory entry
    if name.endswith("/"):
        target_dir = (base / Path(*parts)).resolve()
        base_resolved = base.resolve()
        if target_dir != base_resolved and base_resolved not in target_dir.parents:
            raise ValueError(f"Unsafe ZIP member (outside base): {member}")
        target_dir.mkdir(parents=True, exist_ok=True)
        return

    target_path = (base / Path(*parts)).resolve()
    base_resolved = base.resolve()
    if target_path != base_resolved and base_resolved not in target_path.parents:
        raise ValueError(f"Unsafe ZIP member (outside base): {member}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, open(target_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def restore_from_zip(base_dir: str, zip_path: str, progress_cb=None) -> None:
    base = Path(base_dir)
    zip_path_p = Path(zip_path)
    if not zip_path_p.is_file():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path_p, mode="r") as zf:
        members = _zip_members_safe(zf.namelist())

        to_extract = [m for m in members if m == "config.ini" or m.startswith("input/") or m.startswith("output/")]
        total = max(1, len(to_extract))
        if progress_cb:
            progress_cb(0, total, "Preparing restore…")

        # Remove existing targets first (prevents mixing old/new files)
        if progress_cb:
            progress_cb(0, total, "Deleting current input/ and output/…")
        for folder in (base / "input", base / "output"):
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)

        cfg = base / "config.ini"
        if cfg.exists() and cfg.is_file():
            try:
                cfg.unlink()
            except Exception:
                pass

        # Extract (manual + safe)
        for idx, member in enumerate(to_extract, start=1):
            _safe_extract_member(zf, member, base)
            if progress_cb:
                progress_cb(idx, total, f"Extracting: {member}")


def _format_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    for unit in ("KB", "MB", "GB", "TB"):
        num /= 1024.0
        if num < 1024.0:
            return f"{num:.1f} {unit}"
    return f"{num:.1f} PB"


def _list_zip_files(folder: str) -> list[tuple[str, int]]:
    p = Path(folder)
    if not p.is_dir():
        return []
    out: list[tuple[str, int]] = []
    for f in sorted(p.glob("*.zip")):
        try:
            out.append((str(f), int(f.stat().st_size)))
        except Exception:
            out.append((str(f), 0))
    return out


def launch_gui(base_dir: str):
    theme = "flatly"

    if tb:
        root = tb.Window(themename=theme)
        ttk = tb
    else:
        root = tk.Tk()
        ttk = ttk_native

    root.title("MESA – Backup / Restore")

    container = ttk.Frame(root, padding=12)
    container.pack(fill="both", expand=True)

    # ---------------- Warning ----------------
    warning_text = (
        "IMPORTANT:\n"
        "- Close any open MESA files before backup/restore (reports, DOCX, Excel, etc.).\n"
        "- Restoring a backup will DELETE and replace your current input/, output/ and config.ini.\n"
        "  Make sure you have a backup of the current project first."
    )
    if tb:
        warning = tb.Label(container, text=warning_text, justify="left", bootstyle="danger")
    else:
        warning = tk.Label(container, text=warning_text, justify="left", fg="red")
    warning.pack(fill="x", pady=(0, 10))

    # ---------------- Progress ----------------
    # Keep the same visual layout as other MESA tools:
    # progress bar (bootstyle 'info') + percent label, plus a separate status line.
    op_status_var = tk.StringVar(value="Idle")
    progress_var = tk.DoubleVar(value=0.0)

    prog = ttk.Frame(container)
    prog.pack(fill="x", pady=(0, 12))
    ttk.Label(prog, textvariable=op_status_var, justify="left").pack(anchor="w")

    progress_frame = tk.Frame(prog)
    # Center the bar+percent group (matches other MESA tools)
    progress_frame.pack(pady=(4, 0))

    if tb:
        progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=260,
            mode="determinate",
            maximum=100,
            variable=progress_var,
            bootstyle="info",
        )
    else:
        progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=260,
            mode="determinate",
            maximum=100,
            variable=progress_var,
        )
    progress_bar.pack(side=tk.LEFT)

    progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
    progress_label.pack(side=tk.LEFT, padx=8)

    work_q: queue.Queue = queue.Queue()
    work_running = {"active": False}

    def _set_progress(current: int, total: int, msg: str):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        pct = (100.0 * float(current)) / float(total)
        progress_var.set(pct)
        try:
            progress_label.config(text=f"{int(pct)}%")
        except Exception:
            pass
        op_status_var.set(f"{msg}  ({current}/{total})")
        root.update_idletasks()

    def _poll_work_queue(on_done=None):
        try:
            while True:
                kind, payload = work_q.get_nowait()
                if kind == "progress":
                    cur, total, msg = payload
                    _set_progress(cur, total, msg)
                elif kind == "done":
                    work_running["active"] = False
                    if on_done:
                        on_done(payload)
                elif kind == "error":
                    work_running["active"] = False
                    messagebox.showerror("Operation failed", str(payload))
        except queue.Empty:
            pass

        if work_running["active"]:
            root.after(100, lambda: _poll_work_queue(on_done=on_done))

    def _run_background(worker_fn, on_done=None):
        if work_running["active"]:
            return
        work_running["active"] = True

        def _progress_cb(cur, total, msg):
            work_q.put(("progress", (cur, total, msg)))

        def _thread_body():
            try:
                result = worker_fn(_progress_cb)
                work_q.put(("done", result))
            except Exception as exc:
                work_q.put(("error", exc))

        threading.Thread(target=_thread_body, daemon=True).start()
        _poll_work_queue(on_done=on_done)

    # ---------------- Backup ----------------
    if tb:
        backup_box = tb.LabelFrame(container, text="1) Create backup ZIP", bootstyle="secondary")
    else:
        backup_box = tk.LabelFrame(container, text="1) Create backup ZIP")
    backup_box.pack(fill="x", pady=(0, 12))

    dest_var = tk.StringVar(value="")

    row = ttk.Frame(backup_box)
    row.pack(fill="x", padx=8, pady=8)

    ttk.Label(row, text="Destination folder:").pack(side="left")
    dest_entry = ttk.Entry(row, textvariable=dest_var, width=60)
    dest_entry.pack(side="left", padx=(8, 8), fill="x", expand=True)

    def _browse_dest():
        folder = filedialog.askdirectory(title="Choose folder for backup ZIP")
        if folder:
            dest_var.set(folder)

    ttk.Button(row, text="Browse…", command=_browse_dest).pack(side="left")

    status_var = tk.StringVar(value=f"Base folder: {base_dir}")
    ttk.Label(backup_box, textvariable=status_var, justify="left").pack(anchor="w", padx=8, pady=(0, 8))

    def _do_backup():
        folder = (dest_var.get() or "").strip()
        if not folder:
            messagebox.showwarning("Missing destination", "Choose a destination folder first.")
            return

        def _worker(progress_cb):
            return create_backup_zip(base_dir, folder, progress_cb=progress_cb)

        def _done(out_zip):
            _set_progress(1, 1, "Backup completed")
            status_var.set(f"Backup created: {out_zip}")
            for w in (backup_btn, restore_btn):
                try:
                    w.configure(state="normal")
                except Exception:
                    pass

        for w in (backup_btn, restore_btn):
            try:
                w.configure(state="disabled")
            except Exception:
                pass
        _run_background(_worker, on_done=_done)

    backup_btn = ttk.Button(backup_box, text="Create backup", command=_do_backup)
    backup_btn.pack(anchor="w", padx=8, pady=(0, 10))

    # ---------------- Restore ----------------
    if tb:
        restore_box = tb.LabelFrame(container, text="2) Restore from backup ZIP", bootstyle="secondary")
    else:
        restore_box = tk.LabelFrame(container, text="2) Restore from backup ZIP")
    restore_box.pack(fill="both", expand=True)

    src_folder_var = tk.StringVar(value="")

    top = ttk.Frame(restore_box)
    top.pack(fill="x", padx=8, pady=(8, 6))

    ttk.Label(top, text="Folder with ZIP files:").pack(side="left")
    src_entry = ttk.Entry(top, textvariable=src_folder_var, width=60)
    src_entry.pack(side="left", padx=(8, 8), fill="x", expand=True)

    def _browse_src():
        folder = filedialog.askdirectory(title="Choose folder containing backup ZIPs")
        if folder:
            src_folder_var.set(folder)
            _refresh_list()

    ttk.Button(top, text="Browse…", command=_browse_src).pack(side="left")

    list_frame = ttk.Frame(restore_box)
    list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    lb = tk.Listbox(list_frame, height=10)
    sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    lb.pack(side="left", fill="both", expand=True)
    sb.pack(side="left", fill="y")

    zip_index: list[tuple[str, int]] = []

    def _refresh_list():
        nonlocal zip_index
        folder = (src_folder_var.get() or "").strip()
        lb.delete(0, tk.END)
        zip_index = []
        for path, size in _list_zip_files(folder):
            zip_index.append((path, size))
            lb.insert(tk.END, f"{Path(path).name}   ({_format_bytes(size)})")

    btns = ttk.Frame(restore_box)
    btns.pack(fill="x", padx=8, pady=(0, 10))

    ttk.Button(btns, text="Refresh list", command=_refresh_list).pack(side="left")

    def _do_restore():
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("No file selected", "Select a ZIP file from the list first.")
            return
        idx = int(sel[0])
        zip_path = zip_index[idx][0]

        confirm = messagebox.askyesno(
            "Restore backup",
            "This will replace your current input/, output/ and config.ini in the current folder. Continue?"
        )
        if not confirm:
            return

        def _worker(progress_cb):
            restore_from_zip(base_dir, zip_path, progress_cb=progress_cb)
            return zip_path

        def _done(restored_from):
            _set_progress(1, 1, "Restore completed")
            status_var.set(f"Restore completed from: {restored_from}")
            for w in (backup_btn, restore_btn):
                try:
                    w.configure(state="normal")
                except Exception:
                    pass

        for w in (backup_btn, restore_btn):
            try:
                w.configure(state="disabled")
            except Exception:
                pass
        _run_background(_worker, on_done=_done)

    restore_btn = ttk.Button(btns, text="Restore selected", command=_do_restore)
    restore_btn.pack(side="left", padx=(8, 0))

    root.minsize(820, 520)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="MESA backup/restore tool (zip input/output + config.ini)")
    parser.add_argument("--original_working_directory", required=False, help="Base folder for MESA data")
    args = parser.parse_args()

    base_dir = _normalize_base_dir(args.original_working_directory)
    launch_gui(base_dir)


if __name__ == "__main__":
    main()
