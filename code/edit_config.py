#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI helper for editing config.ini with simple section/key management."""

import locale
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass

import argparse
import configparser
import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

try:
    import ttkbootstrap as tb
    from ttkbootstrap import ttk as ttkb
except Exception:
    tb = None
    from tkinter import ttk as ttkb  # type: ignore


def _style_kwargs(style: str | None) -> dict:
    if tb and style:
        return {"bootstyle": style}
    return {}

START_CWD = Path.cwd()
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def has_project_markers(path: Path) -> bool:
    if not path:
        return False
    try:
        if (path / "config.ini").exists():
            return True
        if (path / "system" / "config.ini").exists():
            return True
        if (path / "code" / "config.ini").exists():
            return True
        if (path / "output" / "geoparquet").exists():
            return True
    except Exception:
        return False
    return False


def find_base_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        candidate = Path(cli_arg).expanduser().resolve()
        if has_project_markers(candidate):
            return candidate
    env_dir = os.environ.get("MESA_BASE_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir).expanduser().resolve()
        if has_project_markers(candidate):
            return candidate
    candidates: list[Path] = [START_CWD, APP_DIR]
    for root in (START_CWD, APP_DIR):
        parents = [root.parent, root.parent.parent, root.parent.parent.parent]
        for parent in parents:
            if parent and parent != parent.parent:
                candidates.append(parent)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    for cand in ordered:
        if has_project_markers(cand):
            return cand
    return (START_CWD if START_CWD.exists() else APP_DIR).resolve()


def resolve_config_path(base_dir: Path) -> Path:
    candidates = [
        base_dir / "config.ini",
        base_dir / "system" / "config.ini",
        base_dir / "code" / "config.ini",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate config.ini relative to base directory")


def load_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cfg.optionxform = str  # preserve key casing
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        cfg.read(path, encoding="latin-1")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg


class ConfigEditorApp:
    def __init__(self, master: tk.Misc, config_path: Path, cfg: configparser.ConfigParser):
        self.master = master
        self.master.title(f"MESA Config Editor - {config_path.name}")
        self.config_path = config_path
        self.cfg = cfg

        width = 920
        height = 520
        try:
            screen_w = self.master.winfo_screenwidth()
            screen_h = self.master.winfo_screenheight()
            x_pos = max(0, int((screen_w - width) / 2))
            y_pos = max(0, int((screen_h - height) / 3))
            self.master.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        except Exception:
            pass

        self.section_list: tk.Listbox
        self.key_tree: ttkb.Treeview
        self.key_var = tk.StringVar()
        self.value_var = tk.StringVar()
        self.new_section_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self.refresh_sections()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root_frame = ttkb.Frame(self.master, padding=12)
        root_frame.pack(fill="both", expand=True)

        # Left: sections
        sections_frame = ttkb.LabelFrame(root_frame, text="Sections", padding=10)
        sections_frame.pack(side="left", fill="y", expand=False, padx=(0, 12))

        self.section_list = tk.Listbox(sections_frame, height=18)
        self.section_list.pack(fill="y", expand=True)
        self.section_list.bind("<<ListboxSelect>>", self.on_section_select)

        section_controls = ttkb.Frame(sections_frame)
        section_controls.pack(fill="x", pady=(8, 0))
        ttkb.Entry(section_controls, textvariable=self.new_section_var).pack(fill="x", pady=(0, 4))
        ttkb.Button(section_controls, text="Add section", command=self.add_section, **_style_kwargs("success")).pack(fill="x", pady=2)
        ttkb.Button(section_controls, text="Delete section", command=self.delete_section, **_style_kwargs("danger")).pack(fill="x", pady=2)

        # Right: keys + editing panel
        main_panel = ttkb.Frame(root_frame)
        main_panel.pack(side="left", fill="both", expand=True)

        keys_frame = ttkb.LabelFrame(main_panel, text="Keys", padding=10)
        keys_frame.pack(fill="both", expand=True)

        columns = ("key", "value")
        self.key_tree = ttkb.Treeview(keys_frame, columns=columns, show="headings")
        self.key_tree.heading("key", text="Key")
        self.key_tree.heading("value", text="Value")
        self.key_tree.column("key", width=200, anchor="w")
        self.key_tree.column("value", width=400, anchor="w")
        self.key_tree.pack(fill="both", expand=True)
        self.key_tree.bind("<<TreeviewSelect>>", self.on_key_select)

        editor_frame = ttkb.LabelFrame(main_panel, text="Edit entry", padding=10)
        editor_frame.pack(fill="x", expand=False, pady=(10, 0))

        ttkb.Label(editor_frame, text="Key:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttkb.Entry(editor_frame, textvariable=self.key_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttkb.Label(editor_frame, text="Value:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttkb.Entry(editor_frame, textvariable=self.value_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        editor_frame.columnconfigure(1, weight=1)

        buttons_frame = ttkb.Frame(editor_frame)
        buttons_frame.grid(row=0, column=2, rowspan=2, padx=4, pady=4, sticky="ns")
        ttkb.Button(buttons_frame, text="Add / Update", command=self.add_or_update_key, width=14, **_style_kwargs("primary")).pack(pady=2)
        ttkb.Button(buttons_frame, text="Delete key", command=self.delete_key, width=14, **_style_kwargs("danger")).pack(pady=2)

        actions_frame = ttkb.Frame(main_panel)
        actions_frame.pack(fill="x", pady=(10, 0))
        ttkb.Button(actions_frame, text="Reload from disk", command=self.reload_config).pack(side="left", padx=4)
        ttkb.Button(actions_frame, text="Save changes", command=self.save_config, **_style_kwargs("success")).pack(side="left", padx=4)
        ttkb.Button(actions_frame, text="Close", command=self.master.destroy).pack(side="right", padx=4)

        self.status_var.set(f"Loaded {self.config_path}")
        ttkb.Label(self.master, textvariable=self.status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 8))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def refresh_sections(self) -> None:
        sections = ["DEFAULT", *self.cfg.sections()]
        self.section_list.delete(0, tk.END)
        for name in sections:
            self.section_list.insert(tk.END, name)
        if sections:
            self.section_list.selection_set(0)
            self.on_section_select(None)

    def current_section(self) -> str | None:
        selection = self.section_list.curselection()
        if not selection:
            return None
        return self.section_list.get(selection[0])

    def populate_keys(self, section: str) -> None:
        self.key_tree.delete(*self.key_tree.get_children())
        if section == "DEFAULT":
            items = self.cfg["DEFAULT"].items()
        else:
            if not self.cfg.has_section(section):
                return
            items = self.cfg[section].items()
        for key, value in sorted(items, key=lambda kv: kv[0].lower()):
            self.key_tree.insert("", tk.END, values=(key, value))
        self.key_var.set("")
        self.value_var.set("")

    def on_section_select(self, _event) -> None:
        section = self.current_section()
        if section:
            self.populate_keys(section)
            self.status_var.set(f"Viewing section [{section}] from {self.config_path.name}")

    def on_key_select(self, _event) -> None:
        selection = self.key_tree.selection()
        if not selection:
            return
        item = self.key_tree.item(selection[0])
        values = item.get("values", [])
        if len(values) >= 2:
            self.key_var.set(values[0])
            self.value_var.set(values[1])

    def add_section(self) -> None:
        name = self.new_section_var.get().strip()
        if not name:
            messagebox.showwarning("Add section", "Please enter a section name.")
            return
        if name.upper() == "DEFAULT":
            messagebox.showwarning("Add section", "DEFAULT already exists.")
            return
        if self.cfg.has_section(name):
            messagebox.showinfo("Add section", f"Section [{name}] already exists.")
            return
        self.cfg.add_section(name)
        self.new_section_var.set("")
        self.refresh_sections()
        self.status_var.set(f"Added section [{name}]")

    def delete_section(self) -> None:
        section = self.current_section()
        if not section:
            return
        if section == "DEFAULT":
            messagebox.showwarning("Delete section", "The DEFAULT section cannot be removed.")
            return
        if not self.cfg.has_section(section):
            return
        if not messagebox.askyesno("Delete section", f"Remove section [{section}] and all its keys?"):
            return
        self.cfg.remove_section(section)
        self.refresh_sections()
        self.status_var.set(f"Deleted section [{section}]")

    def add_or_update_key(self) -> None:
        section = self.current_section()
        if not section:
            messagebox.showwarning("Update key", "Select a section first.")
            return
        key = self.key_var.get().strip()
        if not key:
            messagebox.showwarning("Update key", "Enter a key name.")
            return
        value = self.value_var.get()
        if section == "DEFAULT":
            target = self.cfg["DEFAULT"]
        else:
            if not self.cfg.has_section(section):
                self.cfg.add_section(section)
            target = self.cfg[section]
        target[key] = value
        self.populate_keys(section)
        self.status_var.set(f"Updated [{section}] {key}")

    def delete_key(self) -> None:
        section = self.current_section()
        if not section:
            return
        key = self.key_var.get().strip()
        if not key:
            messagebox.showwarning("Delete key", "Enter/select a key to remove.")
            return
        target = self.cfg["DEFAULT"] if section == "DEFAULT" else self.cfg[section]
        if key not in target:
            messagebox.showinfo("Delete key", f"Key '{key}' was not found in section [{section}].")
            return
        if not messagebox.askyesno("Delete key", f"Delete key '{key}' from section [{section}]?"):
            return
        target.pop(key, None)
        self.populate_keys(section)
        self.status_var.set(f"Deleted key {key} from [{section}]")

    def reload_config(self) -> None:
        try:
            self.cfg = load_config(self.config_path)
            self.refresh_sections()
            self.status_var.set("Reloaded configuration from disk")
        except Exception as exc:
            messagebox.showerror("Reload failed", f"Could not reload config: {exc}")

    def save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as handle:
                self.cfg.write(handle)
            self.status_var.set(f"Saved changes to {self.config_path}")
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save config: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit MESA config.ini")
    parser.add_argument("--original_working_directory", required=False, help="Override project base directory")
    args = parser.parse_args()

    base_dir = find_base_dir(args.original_working_directory)
    config_path = resolve_config_path(base_dir)
    cfg = load_config(config_path)

    theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly")
    if tb:
        root = tb.Window(themename=theme)
    else:
        root = tk.Tk()
    root.title("MESA Config Editor")

    app = ConfigEditorApp(root, config_path, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
