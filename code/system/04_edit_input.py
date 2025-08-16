# -*- coding: utf-8 -*-
# Redigeringsvindu for assosiering av importance/susceptibility
# med robust kvalitetssikring, Excel-inn/ut og stabil klassifisering.

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import argparse
import configparser
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

import tkinter as tk
from tkinter import messagebox, filedialog

# Bruk ttkbootstrap for tema + ttk-widgets
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap import ttk as ttkb  # themed ttk widgets

# -------------------------------
# Globale variabler / standarder
# -------------------------------
column_widths = [35, 13, 13, 13, 13, 30]
valid_input_values = []
classification = {}
entries = []  # rad-kontroller i UI
FALLBACK = 3  # default, kan overstyres fra config

# -------------------------------
# Konfig / klassifisering
# -------------------------------
def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    return cfg

def read_config_classification(file_name: str) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    classification.clear()
    for section in cfg.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:
            range_str = cfg[section].get('range', '').strip()
            description = cfg[section].get('description', '')
            if '-' in range_str:
                start, end = map(int, range_str.split('-'))
                classification[section] = {
                    'range': range(start, end + 1),
                    'description': description
                }
    return classification

def determine_category(sensitivity: int) -> tuple[str, str]:
    for category, info in classification.items():
        if sensitivity in info['range']:
            return category, info['description']
    return '', ''  # utenfor konfig-områdene

def get_valid_values(cfg) -> list[int]:
    try:
        vals = [int(x.strip()) for x in cfg['VALID_VALUES']['valid_input'].split(',')]
        vals = [v for v in vals if 0 <= v <= 9999]
        return sorted(set(vals)) or [1, 2, 3, 4, 5]
    except Exception:
        return [1, 2, 3, 4, 5]

def get_fallback_value(cfg, valid_vals: list[int]) -> int:
    try:
        v = int(cfg['DEFAULT'].get('default_fallback_value', '3'))
        if v in valid_vals:
            return v
    except Exception:
        pass
    return int(np.median(valid_vals))

# -------------------------------
# Logging
# -------------------------------
def log_to_file(message: str) -> None:
    ts = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    line = f"{ts} - {message}"
    try:
        dest = os.path.join(original_working_directory, "log.txt")
        with open(dest, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# -------------------------------
# Datavask / sanering
# -------------------------------
def sanitize_scores(df: pd.DataFrame, valid_vals: list[int], fallback: int) -> pd.DataFrame:
    """Tving importance/susceptibility til gyldige heltall; beregn sensitivity + A–E for alle rader."""
    df = df.copy()

    for col in ['importance', 'susceptibility']:
        if col not in df.columns:
            df[col] = np.nan
        s = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')

        # tom/NaN -> fallback
        s = s.where(s.notna(), fallback)

        # klipp til min/max i valid_vals
        s = s.clip(min(valid_vals), max(valid_vals))

        # tving til nærmeste gyldige verdi (om valid_vals er ikke-sammenhengende)
        s = s.apply(lambda x: min(valid_vals, key=lambda vv: abs(int(x) - vv)))
        df[col] = s.astype(int)

    df['sensitivity'] = (pd.to_numeric(df['importance']) * pd.to_numeric(df['susceptibility'])).astype(int)

    # Klassifiser
    def _det(sens):
        code, desc = determine_category(int(sens))
        if not code:  # utenfor klassene, gi laveste kategori for å unngå "Unknown"
            code, desc = determine_category(max(1, int(sens)))
        return pd.Series([code, desc], index=['sensitivity_code', 'sensitivity_description'])

    klass = df['sensitivity'].apply(_det)
    df['sensitivity_code'] = klass['sensitivity_code']
    df['sensitivity_description'] = klass['sensitivity_description']

    return df

def best_join_key(df: pd.DataFrame) -> str:
    return 'id' if 'id' in df.columns else 'name_original'

# -------------------------------
# GUI-validator for inndata
# -------------------------------
def _entry_validator(P: str) -> bool:
    if P == "":
        return True
    return P.isdigit() and int(P) in valid_input_values

# -------------------------------
# Kjernefunksjoner (UI→DF)
# -------------------------------
def calculate_sensitivity(entry_importance, entry_susceptibility, index, entries, df_assetgroup):
    try:
        imp_txt = entry_importance.get().strip()
        sus_txt = entry_susceptibility.get().strip()

        importance = int(imp_txt) if imp_txt else FALLBACK
        susceptibility = int(sus_txt) if sus_txt else FALLBACK

        if importance not in valid_input_values:
            importance = FALLBACK
        if susceptibility not in valid_input_values:
            susceptibility = FALLBACK

        sensitivity = importance * susceptibility
        sensitivity_code, sensitivity_description = determine_category(sensitivity)
        if not sensitivity_code:
            sensitivity_code, sensitivity_description = determine_category(max(1, sensitivity))

        entries[index]['sensitivity']['text'] = str(sensitivity)
        entries[index]['sensitivity_code']['text'] = sensitivity_code
        entries[index]['sensitivity_description']['text'] = sensitivity_description

        df_assetgroup.at[index, 'importance'] = importance
        df_assetgroup.at[index, 'susceptibility'] = susceptibility
        df_assetgroup.at[index, 'sensitivity'] = sensitivity
        df_assetgroup.at[index, 'sensitivity_code'] = sensitivity_code
        df_assetgroup.at[index, 'sensitivity_description'] = sensitivity_description

    except Exception as e:
        log_to_file(f"Input Error: {e}")

def update_all_rows_immediately(entries, df_assetgroup):
    """Gå gjennom alle rader i UI og tving gyldige verdier + oppdater DF."""
    for entry in entries:
        try:
            imp_txt = entry['importance'].get().strip()
            sus_txt = entry['susceptibility'].get().strip()

            importance = int(imp_txt) if imp_txt else FALLBACK
            susceptibility = int(sus_txt) if sus_txt else FALLBACK

            if importance not in valid_input_values:
                importance = FALLBACK
            if susceptibility not in valid_input_values:
                susceptibility = FALLBACK

            sensitivity = importance * susceptibility
            sensitivity_code, sensitivity_description = determine_category(sensitivity)
            if not sensitivity_code:
                sensitivity_code, sensitivity_description = determine_category(max(1, sensitivity))

            idx = entry['row_index']
            df_assetgroup.at[idx, 'importance'] = importance
            df_assetgroup.at[idx, 'susceptibility'] = susceptibility
            df_assetgroup.at[idx, 'sensitivity'] = sensitivity
            df_assetgroup.at[idx, 'sensitivity_code'] = sensitivity_code
            df_assetgroup.at[idx, 'sensitivity_description'] = sensitivity_description

            if 'geom' in entry:
                df_assetgroup.at[idx, 'geom'] = entry['geom']

            entry['sensitivity']['text'] = str(sensitivity)
            entry['sensitivity_code']['text'] = sensitivity_code
            entry['sensitivity_description']['text'] = sensitivity_description

        except Exception as e:
            log_to_file(f"Input Error (update_all_rows): {e}")
            continue

# -------------------------------
# I/O (GPKG / Excel / Parquet)
# -------------------------------
def load_data(gpkg_file: str):
    try:
        layer_name = "tbl_asset_group"
        df = gpd.read_file(gpkg_file, layer=layer_name)

        # Sikre korrekt geometri-kolonne
        if 'geom' in df.columns and df.geometry.name != 'geom':
            df.set_geometry('geom', inplace=True)

        # Saner alltid ved innlasting (unngå 0/NaN → Unknown)
        df = sanitize_scores(df, valid_input_values, FALLBACK)

        if df.geometry.isnull().any():
            log_to_file("Some geometries failed to load or are invalid.")

        return df
    except Exception as e:
        log_to_file(f"Failed to load data: {e}")
        return None

def save_to_gpkg(df_assetgroup: gpd.GeoDataFrame, gpkg_file: str):
    try:
        if df_assetgroup is None or df_assetgroup.empty:
            log_to_file("GeoDataFrame is empty.")
            return

        # Saner før lagring
        df_assetgroup = sanitize_scores(df_assetgroup, valid_input_values, FALLBACK)

        # Finn geometri-kolonner
        geom_cols = [c for c in df_assetgroup.columns if str(df_assetgroup.dtypes.get(c)) == 'geometry']
        if geom_cols:
            main_geom_col = geom_cols[0]
        else:
            # fallback til aktiv geometri
            main_geom_col = df_assetgroup.geometry.name if hasattr(df_assetgroup, "geometry") else None
            if not main_geom_col:
                raise ValueError("No geometry column found in the DataFrame.")

        # Kast eventuelle ekstra geometri-kolonner
        for col in geom_cols:
            if col != main_geom_col:
                df_assetgroup.drop(columns=[col], inplace=True, errors='ignore')

        if df_assetgroup.geometry.name != main_geom_col:
            df_assetgroup.set_geometry(main_geom_col, inplace=True)

        if df_assetgroup.crs is None:
            df_assetgroup.set_crs(epsg=int(workingprojection_epsg), inplace=True)

        df_assetgroup.to_file(filename=gpkg_file, layer='tbl_asset_group', driver='GPKG')
        log_to_file("Data saved successfully to GeoPackage.")
    except Exception as e:
        log_to_file(f"Failed to save GeoDataFrame: {e}")

def save_to_excel(df_assetgroup: pd.DataFrame, excel_file: str):
    try:
        cols = ['id', 'name_original', 'susceptibility', 'importance']
        for c in cols:
            if c not in df_assetgroup.columns:
                df_assetgroup[c] = None
        df_assetgroup[cols].to_excel(excel_file, index=False)
        log_to_file("Selected data saved successfully to Excel.")
    except Exception as e:
        log_to_file(f"Failed to save data to Excel: {e}")

def load_from_excel(excel_file: str, df_assetgroup: pd.DataFrame) -> pd.DataFrame:
    try:
        df_x = pd.read_excel(excel_file)
        log_to_file("Data loaded successfully from Excel.")

        key_df = best_join_key(df_assetgroup)
        key_x = 'id' if 'id' in df_x.columns else 'name_original'

        # Saner Excel-verdier
        for col in ['importance', 'susceptibility']:
            if col in df_x.columns:
                s = pd.to_numeric(df_x[col], errors='coerce').round().astype('Int64')
                s = s.where(s.notna(), FALLBACK)
                s = s.clip(min(valid_input_values), max(valid_input_values))
                s = s.apply(lambda v: min(valid_input_values, key=lambda vv: abs(int(v) - vv)))
                df_x[col] = s.astype(int)

        upd = df_x[[key_x] + [c for c in ['importance', 'susceptibility'] if c in df_x.columns]].drop_duplicates(subset=[key_x])

        # Venstreslåing på databasen: kun oppdater eksisterende grupper
        merged = df_assetgroup.merge(
            upd, left_on=key_df, right_on=key_x, how='left', suffixes=('', '_xlsx')
        )

        for col in ['importance', 'susceptibility']:
            if f"{col}_xlsx" in merged.columns:
                merged[col] = merged[f"{col}_xlsx"].where(merged[f"{col}_xlsx"].notna(), merged[col])
                merged.drop(columns=[f"{col}_xlsx"], inplace=True, errors='ignore')

        # Saner + klassifiser alt etter oppdatering
        merged = sanitize_scores(merged, valid_input_values, FALLBACK)

        # Logg fremmede rader i Excel (finnes ikke i DB)
        try:
            set_x = set(df_x[key_x].dropna().astype(str))
            set_db = set(df_assetgroup[key_df].dropna().astype(str))
            unknown = sorted(set_x - set_db)
            if unknown:
                sample = ', '.join(unknown[:10])
                more = ' …' if len(unknown) > 10 else ''
                log_to_file(f"Excel rows not matched to database (ignored): {sample}{more}")
        except Exception:
            pass

        return merged
    except Exception as e:
        log_to_file(f"Failed to load data from Excel: {e}")
        return df_assetgroup

def save_to_geoparquet(gdf: gpd.GeoDataFrame, file_path: str):
    try:
        gdf.to_parquet(file_path, index=False)
        log_to_file(f"Saved to GeoParquet: {file_path}")
    except Exception as e:
        log_to_file(f"Error saving to GeoParquet: {e}")

def save_assets_to_geoparquet(df_assetgroup: gpd.GeoDataFrame, original_working_directory: str):
    try:
        out = os.path.join(original_working_directory, "output", "geoparquet", "assets_groups.parquet")
        save_to_geoparquet(df_assetgroup, out)
    except Exception as e:
        log_to_file(f"Error saving assets to GeoParquet: {e}")

# -------------------------------
# UI-bygging
# -------------------------------
def setup_headers(frame, column_widths):
    headers = ["Dataset", "Importance", "Susceptibility", "Sensitivity", "Code", "Description"]
    for idx, header in enumerate(headers):
        label = ttkb.Label(frame, text=header, anchor='w', width=column_widths[idx])
        label.grid(row=0, column=idx, padx=5, pady=5, sticky='ew')
    return frame

def add_data_row(index, row, frame, column_widths, entries, df_assetgroup):
    vcmd = (frame.register(_entry_validator), "%P")

    entry_importance = ttkb.Entry(frame, width=column_widths[1], validate="key", validatecommand=vcmd)
    entry_importance.insert(0, str(getattr(row, 'importance', FALLBACK)))
    entry_importance.grid(row=index, column=1, padx=5)

    entry_susceptibility = ttkb.Entry(frame, width=column_widths[2], validate="key", validatecommand=vcmd)
    entry_susceptibility.insert(0, str(getattr(row, 'susceptibility', FALLBACK)))
    entry_susceptibility.grid(row=index, column=2, padx=5)

    entry_importance.bind(
        '<KeyRelease>',
        lambda event, imp=entry_importance, sus=entry_susceptibility, idx=index-1:
            calculate_sensitivity(imp, sus, idx, entries, df_assetgroup)
    )
    entry_susceptibility.bind(
        '<KeyRelease>',
        lambda event, imp=entry_importance, sus=entry_susceptibility, idx=index-1:
            calculate_sensitivity(imp, sus, idx, entries, df_assetgroup)
    )

    geom = getattr(row, 'geom', None)

    label_name = ttkb.Label(frame, text=getattr(row, 'name_original', ''), anchor='w', width=column_widths[0])
    label_name.grid(row=index, column=0, padx=5, sticky='ew')

    label_sensitivity = ttkb.Label(frame, text=str(getattr(row, 'sensitivity', '')), anchor='w', width=column_widths[3])
    label_sensitivity.grid(row=index, column=3, padx=5, sticky='ew')

    label_code = ttkb.Label(frame, text=str(getattr(row, 'sensitivity_code', '')), anchor='w', width=column_widths[4])
    label_code.grid(row=index, column=4, padx=5, sticky='ew')

    label_description = ttkb.Label(frame, text=str(getattr(row, 'sensitivity_description', '')), anchor='w', width=column_widths[5])
    label_description.grid(row=index, column=5, padx=5, sticky='ew')

    entries.append({
        'row_index': index - 1,
        'name': label_name,
        'importance': entry_importance,
        'susceptibility': entry_susceptibility,
        'sensitivity': label_sensitivity,
        'sensitivity_code': label_code,
        'sensitivity_description': label_description,
        'geom': geom
    })

def create_scrollable_area(root):
    scrollable_frame = ttkb.Frame(root)
    canvas = tk.Canvas(scrollable_frame, highlightthickness=0)
    scrollbar_y = ttkb.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar_y.set)
    canvas.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.pack(side=tk.LEFT, fill="both", expand=True)
    scrollbar_y.pack(side=tk.RIGHT, fill="y")
    scrollable_frame.pack(side=tk.TOP, fill="both", expand=True)
    frame = ttkb.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    return canvas, frame

def setup_ui_elements(root, df_assetgroup, column_widths):
    canvas, frame = create_scrollable_area(root)
    entries = []
    setup_headers(frame, column_widths)

    if df_assetgroup is not None and not df_assetgroup.empty:
        for i, row in enumerate(df_assetgroup.itertuples(), start=1):
            add_data_row(i, row, frame, column_widths, entries, df_assetgroup)
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    else:
        log_to_file("No data to display.")

    return canvas, frame, entries

# -------------------------------
# Diverse hjelpefunksjoner
# -------------------------------
def update_df_assetgroup(entries):
    for entry in entries:
        idx = entry['row_index']
        df_assetgroup.at[idx, 'importance'] = entry['importance'].get()
        df_assetgroup.at[idx, 'susceptibility'] = entry['susceptibility'].get()
        df_assetgroup.at[idx, 'sensitivity'] = entry['sensitivity']['text']
        df_assetgroup.at[idx, 'sensitivity_code'] = entry['sensitivity_code']['text']
        df_assetgroup.at[idx, 'sensitivity_description'] = entry['sensitivity_description']['text']
        df_assetgroup.at[idx, 'geom'] = entry['geom']

def close_application():
    save_to_gpkg(df_assetgroup, gpkg_file)
    save_assets_to_geoparquet(df_assetgroup, original_working_directory)
    root.destroy()

def handle_load_from_excel():
    global df_assetgroup
    input_folder = os.path.join(original_working_directory, "input")
    excel_file = filedialog.askopenfilename(
        title="Select Excel File",
        initialdir=input_folder,
        filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
    )
    if excel_file:
        df_assetgroup = load_from_excel(excel_file, df_assetgroup)
        # oppdater UI med sanerte/oppdaterte verdier
        for entry in entries:
            name_original = entry['name'].cget("text")
            key_df = best_join_key(df_assetgroup)
            if key_df == 'id' and 'id' in df_assetgroup.columns:
                # prøv å finne rad ved navn (fallback)
                mask = (df_assetgroup['name_original'] == name_original)
            else:
                mask = (df_assetgroup['name_original'] == name_original)

            if mask.any():
                idx = df_assetgroup[mask].index[0]

                entry['importance'].delete(0, tk.END)
                entry['importance'].insert(0, str(int(df_assetgroup.at[idx, 'importance'])))

                entry['susceptibility'].delete(0, tk.END)
                entry['susceptibility'].insert(0, str(int(df_assetgroup.at[idx, 'susceptibility'])))

                entry['sensitivity']['text'] = str(int(df_assetgroup.at[idx, 'sensitivity']))
                entry['sensitivity_code']['text'] = str(df_assetgroup.at[idx, 'sensitivity_code'])
                entry['sensitivity_description']['text'] = str(df_assetgroup.at[idx, 'sensitivity_description'])
            else:
                log_to_file(f"Warning: {name_original} not found in the Excel mapping view.")

        root.update_idletasks()
        log_to_file("Data loaded and UI updated from Excel.")

def handle_save_to_excel():
    global df_assetgroup
    input_folder = os.path.join(original_working_directory, "input")
    excel_file = filedialog.asksaveasfilename(
        title="Save Excel File",
        initialdir=input_folder,
        defaultextension=".xlsx",
        filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
    )
    if excel_file:
        save_to_excel(df_assetgroup, excel_file)

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Asset group editor with QC')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory

    if not original_working_directory:
        original_working_directory = os.getcwd()
        if "system" in os.path.basename(original_working_directory).lower():
            original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

    config_file = os.path.join(original_working_directory, "system", "config.ini")
    gpkg_file = os.path.join(original_working_directory, "output", "mesa.gpkg")
    excel_file = os.path.join(original_working_directory, "input", "settings.xlsx")

    config = read_config(config_file)
    classification = read_config_classification(config_file)
    valid_input_values = get_valid_values(config)
    FALLBACK = get_fallback_value(config, valid_input_values)

    ttk_bootstrap_theme = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
    workingprojection_epsg = config['DEFAULT'].get('workingprojection_epsg', '4326')

    # GUI-root
    root = tb.Window(themename=ttk_bootstrap_theme)
    root.title("Set up processing (QC)")
    try:
        icon_path = os.path.join(original_working_directory, "system_resources", "mesa.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass
    root.geometry("900x800")

    # Last data
    df_assetgroup = load_data(gpkg_file)
    if df_assetgroup is None:
        log_to_file("Failed to load the GeoDataFrame. Check the GeoPackage file and the data integrity.")
        sys.exit(1)

    # Bygg UI
    canvas, frame, entries = setup_ui_elements(root, df_assetgroup, column_widths)
    update_all_rows_immediately(entries, df_assetgroup)

    info_text = ("Her registrerer du verdier for importance og susceptibility. "
                 "Alle verdier tvinges til gyldige intervaller (config.ini). "
                 "Sensitivity og kode beregnes automatisk.")
    info_label = ttkb.Label(root, text=info_text, wraplength=600, justify="center")
    info_label.pack(padx=10, pady=10)

    btn_frame = ttkb.Frame(root)
    btn_frame.pack(side='bottom', fill='x', padx=10, pady=10)

    close_button = ttkb.Button(btn_frame, text="Exit", command=close_application, bootstyle=WARNING)
    close_button.pack(side='right', padx=5)

    save_button = ttkb.Button(btn_frame, text="Save", command=lambda: save_to_gpkg(df_assetgroup, gpkg_file), bootstyle=SUCCESS)
    save_button.pack(side='right', padx=5)

    save_excel_button = ttkb.Button(btn_frame, text="Save to Excel", command=handle_save_to_excel, bootstyle=INFO)
    save_excel_button.pack(side='right', padx=5)

    load_excel_button = ttkb.Button(btn_frame, text="Load from Excel", command=handle_load_from_excel, bootstyle=PRIMARY)
    load_excel_button.pack(side='right', padx=5)

    root.mainloop()
