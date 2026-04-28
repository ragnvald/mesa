# -*- coding: utf-8 -*-
"""Heuristic auto-tuning for the MESA processing pipeline.

Centralises the rules that derive worker counts, partition thresholds, and
similar performance knobs from observed hardware + data fingerprint. Called
once at the top of run_processing_pipeline so the rest of the pipeline can
read its config keys as if the user had set them all explicitly.

Design rule: respect explicit user overrides. If a key is set to a non-zero
/ non-default value in config.ini, auto_tune leaves it alone. The user
opts into the tuning by leaving 0 / "auto" / blank.

The function logs a single [auto-tune] block summarising every decision and
its rationale, so the operator can audit what was picked without grepping
through scattered log lines.
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Callable, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # type: ignore


_LogFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Hardware + data probing
# ---------------------------------------------------------------------------

def _probe_hardware() -> dict:
    """Hardware fingerprint: total RAM, currently-available RAM, CPU count.

    Returns conservative fallbacks if psutil is unavailable - we'd rather
    under-tune than OOM.
    """
    import os
    cpu_count = max(1, os.cpu_count() or 4)
    ram_total_gb = 8.0
    ram_avail_gb = 4.0
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            ram_total_gb = float(vm.total) / (1024 ** 3)
            ram_avail_gb = float(vm.available) / (1024 ** 3)
        except Exception:
            pass
    return {
        "cpu_count": cpu_count,
        "ram_total_gb": ram_total_gb,
        "ram_avail_gb": ram_avail_gb,
    }


def _parquet_dir(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet").strip()
    p = Path(sub)
    if not p.is_absolute():
        p = base_dir / p
    return p


def _parquet_rows(path: Path) -> int:
    """Cheap row count via parquet metadata (no full read). 0 if unavailable."""
    if pq is None or not path.exists():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def _file_size_mb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024 ** 2)
    except Exception:
        return 0.0


def _probe_data(base_dir: Path, cfg: configparser.ConfigParser) -> dict:
    """Data fingerprint: row counts and file sizes for the heavy tables that
    feed intersect / flatten. Cheap - reads parquet metadata only.

    Missing tables are reported as 0 / 0.0; callers should treat 0 as
    "unknown / no data yet" and fall back to default heuristics.
    """
    gpq = _parquet_dir(base_dir, cfg)
    asset_path     = gpq / "tbl_asset_object.parquet"
    geocode_path   = gpq / "tbl_geocode_object.parquet"
    flat_path      = gpq / "tbl_flat.parquet"

    return {
        "asset_rows":      _parquet_rows(asset_path),
        "asset_size_mb":   _file_size_mb(asset_path),
        "geocode_rows":    _parquet_rows(geocode_path),
        "geocode_size_mb": _file_size_mb(geocode_path),
        "flat_rows":       _parquet_rows(flat_path),
        "flat_size_mb":    _file_size_mb(flat_path),
    }


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def _explicit_int(cfg: configparser.ConfigParser, key: str) -> Optional[int]:
    """Return user's explicit value if config.ini has a non-zero int for
    `key`, else None to signal "auto-derive please"."""
    raw = cfg["DEFAULT"].get(key, "").strip()
    if not raw:
        return None
    try:
        v = int(float(raw))
    except Exception:
        return None
    return v if v != 0 else None


def _explicit_float(cfg: configparser.ConfigParser, key: str,
                    treat_default_as_unset: float | None = None) -> Optional[float]:
    """Floats are trickier than ints because there's no natural "0 = auto"
    sentinel. Caller passes treat_default_as_unset so we can recognise
    "still at the shipped default" as overridable."""
    raw = cfg["DEFAULT"].get(key, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    if treat_default_as_unset is not None and abs(v - treat_default_as_unset) < 1e-9:
        return None
    return v


def _derive_max_workers(hw: dict, cfg: configparser.ConfigParser) -> tuple[int, str]:
    """Intersect-stage worker count from RAM budget and CPU count."""
    cpu_cap = hw["cpu_count"]
    avail_gb = hw["ram_avail_gb"]
    try:
        per_worker_gb = float(cfg["DEFAULT"].get("approx_gb_per_worker", "4.0"))
    except Exception:
        per_worker_gb = 4.0
    try:
        mem_target = float(cfg["DEFAULT"].get("mem_target_frac", "0.7"))
    except Exception:
        mem_target = 0.7
    mem_target = max(0.3, min(0.95, mem_target))
    by_ram = max(1, int((avail_gb * mem_target) / max(1.0, per_worker_gb)))
    n = max(1, min(by_ram, cpu_cap))
    reason = (f"avail {avail_gb:.1f} GB × {mem_target:.0%} / "
              f"{per_worker_gb:.1f} GB per-worker = {by_ram}, "
              f"capped to CPU={cpu_cap}")
    return n, reason


def _derive_flatten_huge_partition_mb(hw: dict) -> tuple[int, str]:
    """Scale the huge-partition threshold linearly with total RAM, anchored
    to 200 MB at 64 GB. Clamped to a defensible range so we don't pick
    nonsense on tiny or absurdly large boxes."""
    ram_gb = hw["ram_total_gb"]
    raw = int(round(200 * (ram_gb / 64.0)))
    n = max(100, min(raw, 1000))
    reason = f"RAM {ram_gb:.0f} GB × (200 / 64) = {raw}, clamped [100, 1000]"
    return n, reason


def _derive_flatten_max_workers(hw: dict, cfg: configparser.ConfigParser) -> tuple[int, str]:
    """Large-flatten worker count: per-worker peak is ~3× intersect's, so
    budget is much tighter. Cap to half of CPU count as well so we don't
    starve the OS during long pandas groupby/merge passes."""
    avail_gb = hw["ram_avail_gb"]
    try:
        per_intersect_gb = float(cfg["DEFAULT"].get("approx_gb_per_worker", "4.0"))
    except Exception:
        per_intersect_gb = 4.0
    flatten_per_worker_gb = max(2.0, per_intersect_gb * 3.0)
    try:
        mem_target = float(cfg["DEFAULT"].get("mem_target_frac", "0.7"))
    except Exception:
        mem_target = 0.7
    by_ram = max(1, int((avail_gb * mem_target) / flatten_per_worker_gb))
    by_cpu = max(1, hw["cpu_count"] // 2)
    n = max(1, min(by_ram, by_cpu))
    reason = (f"avail {avail_gb:.1f} GB × {mem_target:.0%} / "
              f"{flatten_per_worker_gb:.1f} GB per-flatten-worker = {by_ram}, "
              f"capped to CPU/2 = {by_cpu}")
    return n, reason


def _derive_flatten_small_max_workers(hw: dict, cfg: configparser.ConfigParser) -> tuple[int, str]:
    """Small-flatten: per-worker peak ~1/4 of large, so we can saturate
    cores without RAM headache."""
    avail_gb = hw["ram_avail_gb"]
    try:
        per_intersect_gb = float(cfg["DEFAULT"].get("approx_gb_per_worker", "4.0"))
    except Exception:
        per_intersect_gb = 4.0
    flatten_small_gb = max(1.0, (per_intersect_gb * 3.0) / 4.0)
    try:
        mem_target = float(cfg["DEFAULT"].get("mem_target_frac", "0.7"))
    except Exception:
        mem_target = 0.7
    by_ram = max(1, int((avail_gb * mem_target) / flatten_small_gb))
    n = max(1, min(by_ram, hw["cpu_count"]))
    reason = (f"avail {avail_gb:.1f} GB × {mem_target:.0%} / "
              f"{flatten_small_gb:.1f} GB per-small-worker = {by_ram}, "
              f"capped to CPU={hw['cpu_count']}")
    return n, reason


def _derive_tiles_max_workers(hw: dict, cfg: configparser.ConfigParser) -> tuple[int, str]:
    """Tiles: each worker holds a pickled per-group geometry list. Memory
    cost scales with the basic_mosaic / tbl_flat geometry payload, which
    we approximate from tbl_flat's on-disk size."""
    avail_gb = hw["ram_avail_gb"]
    try:
        per_tile_gb = float(cfg["DEFAULT"].get("tiles_approx_gb_per_worker", "3.0"))
    except Exception:
        per_tile_gb = 3.0
    try:
        mem_target = float(cfg["DEFAULT"].get("mem_target_frac", "0.7"))
    except Exception:
        mem_target = 0.7
    by_ram = max(1, int((avail_gb * mem_target) / max(0.5, per_tile_gb)))
    n = max(2, min(by_ram, hw["cpu_count"], 8))   # tiles also has spawn overhead
    reason = (f"avail {avail_gb:.1f} GB × {mem_target:.0%} / "
              f"{per_tile_gb:.1f} GB per-tile-worker = {by_ram}, "
              f"capped to min(CPU={hw['cpu_count']}, 8), floor 2")
    return n, reason


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def auto_tune_in_place(
    cfg: configparser.ConfigParser,
    base_dir: Path,
    log_fn: Optional[_LogFn] = None,
) -> dict:
    """Mutate cfg["DEFAULT"] in place, filling in derived values for any
    config key the user has at 0 / missing. Explicit non-zero settings in
    config.ini are preserved as-is.

    Returns a dict of {key: (value, source, reason)} where source is
    "user" (came from config.ini) or "auto" (derived here). Caller uses
    this only for logging and diagnostics; pipeline code reads from cfg.

    Logs a single [auto-tune] block via log_fn so operators see exactly
    what was picked. log_fn defaults to a no-op so this function is
    safe to call from contexts without a log widget.
    """
    log = log_fn if callable(log_fn) else (lambda s: None)
    hw = _probe_hardware()
    data = _probe_data(base_dir, cfg)

    decisions: dict[str, tuple] = {}

    def _apply_int(key: str, derive: Callable[[], tuple[int, str]]) -> None:
        existing = _explicit_int(cfg, key)
        if existing is not None:
            decisions[key] = (existing, "user", "explicit in config.ini")
            return
        try:
            value, reason = derive()
        except Exception as exc:
            decisions[key] = (None, "error", str(exc))
            return
        cfg["DEFAULT"][key] = str(value)
        decisions[key] = (value, "auto", reason)

    _apply_int("max_workers",                 lambda: _derive_max_workers(hw, cfg))
    _apply_int("flatten_huge_partition_mb",   lambda: _derive_flatten_huge_partition_mb(hw))
    _apply_int("flatten_max_workers",         lambda: _derive_flatten_max_workers(hw, cfg))
    _apply_int("flatten_small_max_workers",   lambda: _derive_flatten_small_max_workers(hw, cfg))
    _apply_int("tiles_max_workers",           lambda: _derive_tiles_max_workers(hw, cfg))

    # Single coherent log block so the operator can audit at a glance.
    log("[auto-tune] " + "-" * 60)
    log(f"[auto-tune] Hardware:  RAM total {hw['ram_total_gb']:.1f} GB, "
        f"avail {hw['ram_avail_gb']:.1f} GB, CPU {hw['cpu_count']}")
    if data["asset_rows"] or data["geocode_rows"] or data["flat_rows"]:
        log(f"[auto-tune] Data:      "
            f"asset_object {data['asset_rows']:,} rows / {data['asset_size_mb']:.0f} MB, "
            f"geocode_object {data['geocode_rows']:,} rows / {data['geocode_size_mb']:.0f} MB, "
            f"tbl_flat {data['flat_rows']:,} rows / {data['flat_size_mb']:.0f} MB")
    else:
        log("[auto-tune] Data:      (no input parquet found yet - using "
            "hardware-only heuristics)")
    for key, (value, source, reason) in decisions.items():
        if source == "user":
            log(f"[auto-tune] {key} = {value}  (user-set; auto-tune skipped)")
        elif source == "auto":
            log(f"[auto-tune] {key} = {value}  ({reason})")
        else:
            log(f"[auto-tune] {key}: derivation failed ({reason})")
    log("[auto-tune] " + "-" * 60)

    return decisions
