"""One-shot diagnostic: check whether sensitivity values and sensitivity codes
are consistent in tbl_stacked (per-asset) and tbl_flat (per-cell).

Bins (from config.ini):
    A: 21..25, B: 16..20, C: 11..15, D: 6..10, E: 1..5
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "output" / "geoparquet"

BINS = [
    ("A", 21, 25),
    ("B", 16, 20),
    ("C", 11, 15),
    ("D", 6, 10),
    ("E", 1, 5),
]


def bin_value(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return ""
    try:
        iv = int(round(float(v)))
    except Exception:
        return ""
    for label, lo, hi in BINS:
        if lo <= iv <= hi:
            return label
    return ""


def report_mismatches(df: pd.DataFrame, val_col: str, code_col: str, label: str) -> None:
    if val_col not in df.columns or code_col not in df.columns:
        print(f"[{label}] missing column(s): need {val_col} and {code_col}")
        return
    sub = df[[val_col, code_col]].copy()
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    sub[code_col] = sub[code_col].astype("string").str.strip().str.upper()
    sub = sub[sub[val_col].notna() & sub[code_col].notna() & (sub[code_col] != "")]
    if sub.empty:
        print(f"[{label}] no rows with both columns populated")
        return
    sub["expected"] = sub[val_col].apply(bin_value)
    mismatch = sub[sub["expected"] != sub[code_col]]
    n_total = len(sub)
    n_mis = len(mismatch)
    pct = (100.0 * n_mis / n_total) if n_total else 0.0
    print(f"[{label}] {n_mis:,} / {n_total:,} rows mismatch ({pct:.3f}%)")
    if not mismatch.empty:
        # Per-cat breakdown of what's miscategorised:
        cross = (
            mismatch.groupby([code_col, "expected"]).size().unstack(fill_value=0)
        )
        print(f"[{label}] cross-tab (rows: code stored, cols: code derived from value):")
        print(cross)
        print(f"[{label}] sample mismatched rows:")
        print(mismatch.head(10).to_string(index=False))


def main() -> None:
    print("=" * 72)
    print("CHECK 1 — tbl_stacked: per-asset sensitivity vs sensitivity_code")
    print("=" * 72)
    stacked_root = GP / "tbl_stacked"
    if stacked_root.exists() and stacked_root.is_dir():
        parts = sorted(stacked_root.glob("*.parquet"))
        if not parts:
            print("[tbl_stacked] no part files found")
        else:
            print(f"[tbl_stacked] reading {len(parts)} partition file(s)")
            frames = []
            for p in parts:
                try:
                    frames.append(
                        pd.read_parquet(p, columns=["sensitivity", "sensitivity_code"])
                    )
                except Exception as exc:
                    print(f"  skip {p.name}: {exc}")
            if frames:
                stacked = pd.concat(frames, ignore_index=True)
                report_mismatches(stacked, "sensitivity", "sensitivity_code", "tbl_stacked")
    else:
        print("[tbl_stacked] folder not found")

    print()
    print("=" * 72)
    print("CHECK 2 — tbl_flat: per-cell sensitivity_max vs sensitivity_code_max")
    print("=" * 72)
    flat_p = GP / "tbl_flat.parquet"
    if not flat_p.exists():
        print("[tbl_flat] file not found")
        return
    flat = pd.read_parquet(flat_p, columns=["sensitivity_max", "sensitivity_code_max", "name_gis_geocodegroup"])
    print(f"[tbl_flat] {len(flat):,} rows")
    report_mismatches(flat, "sensitivity_max", "sensitivity_code_max", "tbl_flat (all groups)")

    if "name_gis_geocodegroup" in flat.columns:
        print()
        print("Per-geocode-group breakdown:")
        for grp, sub in flat.groupby("name_gis_geocodegroup"):
            report_mismatches(sub, "sensitivity_max", "sensitivity_code_max", f"tbl_flat[{grp}]")


if __name__ == "__main__":
    main()
