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
import re
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


def _strip_comment_prefix(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("#") or stripped.startswith(";"):
        stripped = stripped[1:]
    return stripped.strip()


def load_ini_help_index(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Load help text for sections/options from preceding/inline comments.

    This is a best-effort parser that reads the ini file as text so we can
    surface user-facing descriptions in the GUI.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = path.read_text(encoding="latin-1").splitlines()

    section_help: dict[str, str] = {}
    option_help: dict[tuple[str, str], str] = {}

    current_section: str | None = None
    pending: list[str] = []

    section_header_re = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")
    key_re = re.compile(r"^\s*(?P<key>[^=\s][^=]*)=(?P<rest>.*)$")

    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()

        if not stripped:
            pending = []
            continue

        if stripped.startswith("#") or stripped.startswith(";"):
            pending.append(_strip_comment_prefix(stripped))
            continue

        match = section_header_re.match(line)
        if match:
            name = match.group("name").strip()
            if pending:
                section_help[name] = "\n".join(pending).strip()
            pending = []
            current_section = name
            continue

        match = key_re.match(line)
        if match and current_section:
            key = match.group("key").strip()
            rest = match.group("rest").strip()

            inline = ""
            for marker in (";", "#"):
                idx = rest.find(marker)
                if idx != -1:
                    inline = _strip_comment_prefix(rest[idx:])
                    break

            parts: list[str] = []
            if pending:
                parts.append("\n".join(pending).strip())
            if inline:
                parts.append(inline)
            if parts:
                option_help[(current_section, key)] = "\n".join([p for p in parts if p]).strip()
            pending = []
            continue

        pending = []

    if "DEFAULT" in section_help:
        return section_help, option_help
    if "DEFAULT" not in section_help and "DEFAULT" in {k for (k, _v) in option_help.keys()}:
        section_help.setdefault("DEFAULT", "")
    return section_help, option_help


def _parse_ini_structure(lines: list[str]) -> tuple[dict[str, tuple[int, int]], dict[tuple[str, str], list[int]]]:
    """Parse section ranges and key line indices from raw ini lines."""

    section_ranges: dict[str, tuple[int, int]] = {}
    key_lines: dict[tuple[str, str], list[int]] = {}

    section_header_re = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")

    current_section: str | None = None
    section_start: int | None = None

    for idx, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        match = section_header_re.match(line)
        if match:
            if current_section is not None and section_start is not None:
                section_ranges[current_section] = (section_start, idx)
            current_section = match.group("name").strip()
            section_start = idx
            continue

        if current_section is None:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        if "=" not in line:
            continue

        left = line.split("=", 1)[0]
        key = left.strip()
        if not key:
            continue
        key_lines.setdefault((current_section, key), []).append(idx)

    if current_section is not None and section_start is not None:
        section_ranges[current_section] = (section_start, len(lines))
    return section_ranges, key_lines


def save_config_preserving_comments(path: Path, cfg: configparser.ConfigParser) -> None:
    """Save config updates while preserving comments and most formatting."""

    try:
        original_text = path.read_text(encoding="utf-8")
    except Exception:
        original_text = path.read_text(encoding="latin-1")

    newline = "\r\n" if "\r\n" in original_text else "\n"
    lines = original_text.splitlines(keepends=True)
    section_ranges, key_lines = _parse_ini_structure([ln.rstrip("\r\n") for ln in lines])

    desired_sections: list[str] = ["DEFAULT", *cfg.sections()]
    desired_section_set = set(desired_sections)

    desired_items: dict[str, dict[str, str]] = {}
    desired_items["DEFAULT"] = dict(cfg["DEFAULT"].items())
    for sec in cfg.sections():
        desired_items[sec] = dict(cfg[sec].items())

    written_keys: dict[str, set[str]] = {sec: set() for sec in desired_section_set}

    insert_at: dict[int, list[str]] = {}
    for sec, (start, end) in section_ranges.items():
        if sec not in desired_section_set:
            continue
        existing_keys = {k for (s, k) in key_lines.keys() if s == sec}
        missing = [k for k in desired_items.get(sec, {}).keys() if k not in existing_keys]
        if not missing:
            continue
        additions: list[str] = []
        for k in sorted(missing, key=lambda x: x.lower()):
            additions.append(f"{k} = {desired_items[sec][k]}{newline}")
            written_keys[sec].add(k)
        insert_at.setdefault(end, []).extend(additions)

    new_lines: list[str] = []
    i = 0
    current_section: str | None = None
    skipping_section = False
    section_header_re = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")

    while i < len(lines):
        raw = lines[i]
        line_no_nl = raw.rstrip("\r\n")

        if i in insert_at and current_section and not skipping_section:
            new_lines.extend(insert_at[i])

        match = section_header_re.match(line_no_nl)
        if match:
            sec = match.group("name").strip()
            current_section = sec
            skipping_section = sec not in desired_section_set
            if not skipping_section:
                new_lines.append(raw)
            i += 1
            continue

        if skipping_section:
            i += 1
            continue

        if current_section and current_section in desired_section_set:
            stripped = line_no_nl.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith(";") and "=" in line_no_nl:
                left, rest = line_no_nl.split("=", 1)
                key = left.strip()
                if key:
                    if key not in desired_items.get(current_section, {}):
                        i += 1
                        continue

                    desired_value = desired_items[current_section][key]

                    rest_stripped = rest.rstrip()
                    inline_comment = ""
                    for marker in (";", "#"):
                        idx = rest_stripped.find(marker)
                        if idx != -1:
                            inline_comment = rest_stripped[idx:]
                            rest_stripped = rest_stripped[:idx].rstrip()
                            break
                    leading_ws = ""
                    for ch in rest:
                        if ch in (" ", "\t"):
                            leading_ws += ch
                        else:
                            break

                    rebuilt = f"{left}={leading_ws}{desired_value}"
                    if inline_comment:
                        rebuilt += f" {inline_comment.strip()}"
                    if raw.endswith("\r\n"):
                        rebuilt += "\r\n"
                    elif raw.endswith("\n"):
                        rebuilt += "\n"

                    new_lines.append(rebuilt)
                    written_keys.setdefault(current_section, set()).add(key)
                    i += 1
                    continue

        new_lines.append(raw)
        i += 1

    eof_inserts = insert_at.get(len(lines), [])
    if eof_inserts:
        if new_lines and new_lines[-1].strip():
            new_lines.append(newline)
        new_lines.extend(eof_inserts)

    existing_sections = set(section_ranges.keys())
    for sec in desired_sections:
        if sec in existing_sections:
            continue
        if new_lines and new_lines[-1].strip():
            new_lines.append(newline)
        new_lines.append(f"[{sec}]{newline}")
        items = desired_items.get(sec, {})
        for k in sorted(items.keys(), key=lambda x: x.lower()):
            new_lines.append(f"{k} = {items[k]}{newline}")

    path.write_text("".join(new_lines), encoding="utf-8")


class ConfigEditorApp:
    def __init__(
        self,
        master: tk.Misc,
        config_path: Path,
        cfg: configparser.ConfigParser,
        section_help: dict[str, str],
        option_help: dict[tuple[str, str], str],
    ):
        self.master = master
        self.master.title(f"MESA Config Editor - {config_path.name}")
        self.config_path = config_path
        self.cfg = cfg
        self.section_help = section_help
        self.option_help = option_help

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
        self.filter_var = tk.StringVar()

        self._section_display_to_name: dict[str, str] = {}
        self._section_name_to_display: dict[str, str] = {}
        self._current_section_name: str | None = None

        self._build_ui()
        self.refresh_sections()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root_frame = ttkb.Frame(self.master, padding=12)
        root_frame.pack(fill="both", expand=True)

        # Left: sections
        sections_frame = ttkb.LabelFrame(root_frame, text="Config areas", padding=10)
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

        keys_frame = ttkb.LabelFrame(main_panel, text="Settings", padding=10)
        keys_frame.pack(fill="both", expand=True)

        filter_row = ttkb.Frame(keys_frame)
        filter_row.pack(fill="x", pady=(0, 6))
        ttkb.Label(filter_row, text="Filter:").pack(side="left")
        filter_entry = ttkb.Entry(filter_row, textvariable=self.filter_var)
        filter_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.filter_var.trace_add("write", lambda *_: self._refresh_keys_for_current_section())

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

        ttkb.Label(editor_frame, text="Info:").grid(row=2, column=0, sticky="nw", padx=4, pady=(8, 4))
        self.help_text = tk.Text(editor_frame, height=5, wrap="word")
        self.help_text.grid(row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=(8, 4))
        self.help_text.configure(state="disabled")

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
        self._section_display_to_name.clear()
        self._section_name_to_display.clear()

        self.section_list.delete(0, tk.END)
        for name in sections:
            display = self._section_display_name(name)
            self._section_display_to_name[display] = name
            self._section_name_to_display[name] = display
            self.section_list.insert(tk.END, display)
        if sections:
            self.section_list.selection_set(0)
            self.on_section_select(None)

    def current_section(self) -> str | None:
        selection = self.section_list.curselection()
        if not selection:
            return None
        display = self.section_list.get(selection[0])
        return self._section_display_to_name.get(display, display)

    def _section_display_name(self, section: str) -> str:
        if section == "DEFAULT":
            return "General (DEFAULT)"
        if section in {"A", "B", "C", "D", "E"}:
            desc = ""
            try:
                if self.cfg.has_section(section):
                    desc = self.cfg[section].get("description", "").strip()
            except Exception:
                desc = ""
            return f"Category {section}" + (f" — {desc}" if desc else "")
        if section == "VALID_VALUES":
            return "Valid values (internal)"
        return section

    def _set_help_text(self, text: str) -> None:
        self.help_text.configure(state="normal")
        self.help_text.delete("1.0", tk.END)
        if text:
            self.help_text.insert(tk.END, text)
        self.help_text.configure(state="disabled")

    def _refresh_keys_for_current_section(self) -> None:
        section = self._current_section_name
        if section:
            self.populate_keys(section)

    def populate_keys(self, section: str) -> None:
        self.key_tree.delete(*self.key_tree.get_children())
        self._current_section_name = section
        needle = self.filter_var.get().strip().lower()

        if section == "DEFAULT":
            items = self.cfg["DEFAULT"].items()
        else:
            if not self.cfg.has_section(section):
                return
            items = self.cfg[section].items()

        for key, value in sorted(items, key=lambda kv: kv[0].lower()):
            if needle and needle not in key.lower() and needle not in str(value).lower():
                continue
            self.key_tree.insert("", tk.END, values=(key, value))
        self.key_var.set("")
        self.value_var.set("")
        self._set_help_text(self._help_for_section(section))

    def on_section_select(self, _event) -> None:
        section = self.current_section()
        if section:
            self.populate_keys(section)
            display = self._section_name_to_display.get(section, section)
            self.status_var.set(f"Viewing {display} from {self.config_path.name}")

    def on_key_select(self, _event) -> None:
        selection = self.key_tree.selection()
        if not selection:
            return
        item = self.key_tree.item(selection[0])
        values = item.get("values", [])
        if len(values) >= 2:
            self.key_var.set(values[0])
            self.value_var.set(values[1])
            section = self._current_section_name
            if section:
                self._set_help_text(self._help_for_option(section, str(values[0])))

    def _help_for_section(self, section: str) -> str:
        raw = self.section_help.get(section, "").strip()
        if section in {"A", "B", "C", "D", "E"} and self.cfg.has_section(section):
            desc = self.cfg[section].get("description", "").strip()
            rng = self.cfg[section].get("range", "").strip()
            color = self.cfg[section].get("category_colour", "").strip()
            parts = []
            if desc:
                parts.append(f"Description: {desc}")
            if rng:
                parts.append(f"Range: {rng}")
            if color:
                parts.append(f"Colour: {color}")
            extra = "\n".join(parts).strip()
            if raw and extra:
                return raw + "\n\n" + extra
            return raw or extra
        return raw

    def _help_for_option(self, section: str, key: str) -> str:
        text = self.option_help.get((section, key), "").strip()
        if not text:
            text = self._help_for_section(section)
        return text

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
            self.section_help, self.option_help = load_ini_help_index(self.config_path)
            self.refresh_sections()
            self.status_var.set("Reloaded configuration from disk")
        except Exception as exc:
            messagebox.showerror("Reload failed", f"Could not reload config: {exc}")

    def save_config(self) -> None:
        try:
            save_config_preserving_comments(self.config_path, self.cfg)
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
    section_help, option_help = load_ini_help_index(config_path)

    theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly")
    if tb:
        root = tb.Window(themename=theme)
    else:
        root = tk.Tk()
    root.title("MESA Config Editor")

    app = ConfigEditorApp(root, config_path, cfg, section_help, option_help)
    root.mainloop()


if __name__ == "__main__":
    main()
