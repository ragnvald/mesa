#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil
import zipfile
import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, filedialog

try:
    import ttkbootstrap as tb
except Exception:
    tb = None


def _normalize_base_dir(path: str | None) -> str:
    if path:
        return os.path.abspath(path)
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _safe_relpath(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    rel_str = rel.as_posix().lstrip("/")
    return rel_str


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__"}]
        for fn in filenames:
            yield Path(dirpath) / fn


def create_backup_zip(base_dir: str, dest_folder: str) -> str:
    base = Path(base_dir)
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)

    config_path = base / "config.ini"
    input_dir = base / "input"
    output_dir = base / "output"

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_path = dest / f"mesa_backup_{ts}.zip"

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        if config_path.is_file():
            zf.write(config_path, arcname="config.ini")

        if input_dir.is_dir():
            for file_path in _iter_files(input_dir):
                if file_path.is_file():
                    zf.write(file_path, arcname=_safe_relpath(file_path, base))

        if output_dir.is_dir():
            for file_path in _iter_files(output_dir):
                if file_path.is_file():
                    zf.write(file_path, arcname=_safe_relpath(file_path, base))

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


def restore_from_zip(base_dir: str, zip_path: str) -> None:
    base = Path(base_dir)
    zip_path_p = Path(zip_path)
    if not zip_path_p.is_file():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path_p, mode="r") as zf:
        members = _zip_members_safe(zf.namelist())

        # Only restore these top-level paths
        allowed_prefixes = ("config.ini", "input/", "output/")
        to_extract = [m for m in members if m == "config.ini" or m.startswith("input/") or m.startswith("output/")]

        # Remove existing targets first (prevents mixing old/new files)
        for folder in (base / "input", base / "output"):
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)

        cfg = base / "config.ini"
        if cfg.exists() and cfg.is_file():
            try:
                cfg.unlink()
            except Exception:
                pass

        # Extract
        for member in to_extract:
            zf.extract(member, path=base)


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
        ttk = tk

    root.title("MESA – Backup / Restore")

    container = ttk.Frame(root, padding=12)
    container.pack(fill="both", expand=True)

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
        try:
            out_zip = create_backup_zip(base_dir, folder)
            status_var.set(f"Backup created: {out_zip}")
        except Exception as exc:
            messagebox.showerror("Backup failed", str(exc))

    ttk.Button(backup_box, text="Create backup", command=_do_backup).pack(anchor="w", padx=8, pady=(0, 10))

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

        try:
            restore_from_zip(base_dir, zip_path)
            status_var.set(f"Restore completed from: {zip_path}")
        except Exception as exc:
            messagebox.showerror("Restore failed", str(exc))

    ttk.Button(btns, text="Restore selected", command=_do_restore).pack(side="left", padx=(8, 0))

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
