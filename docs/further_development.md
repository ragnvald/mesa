# Further development — pending work & ideas

Single index of work that is **decided but not yet done** and **ideas worth
considering**. The durable "why/how" lives in `learning.md`; this file is the
roadmap surface so a future session (or the other dev host) can pick something up
without re-deriving it from the code. Keep entries short; link to `learning.md`
or `file:line` for detail. Move an item to "Done" (or delete it) when it lands.

---

## A. Pending — committed direction, not yet implemented

### A1. Settings store — Phase 2 (config.ini → live table)
- **State:** Phase 1 landed (2026-06-26) — `read_config` overlays
  `output/geoparquet/tbl_settings.parquet` on config.ini, with a config.ini
  fallback when the table is absent. No production behaviour change yet.
- **To do:** (1) seed the table from config.ini at mesa.py startup for *missing*
  keys only (never clobber saved tuning); (2) move "Tune processing" Commit to
  `mesa_shared.write_settings(...)` instead of editing config.ini, and load
  current values from the table when the window opens.
- **Refs:** `learning.md` "Settings store: config.ini defaults overlaid by
  tbl_settings (Phase 1)"; `mesa_shared.apply_settings_overlay` / `write_settings`.

### A2. basic_mosaic — Tier 2 (tiled overlay + membership dissolve)
- **State:** Tier 1 landed (spawn fix + snap knob + pre-flight gate). The
  single-threaded `polygonize` peak (~21.7 GB on the 3.53 M-asset run) is only
  *guarded* by the pre-flight gate, not *bounded*.
- **To do:** partition the metric AOI into tiles; node + polygonize per tile in
  parallel; reconcile tile-boundary faces by **asset-membership-signature
  dissolve** (tag each face with the set of covering assets; dissolve adjacent
  faces with identical signature) to remove the moiré seams that retired the old
  tiled path. This parallelises polygonize *and* bounds per-process memory, so the
  largest projects become runnable on small machines instead of just skipped.
- **Refs:** `learning.md` "Mosaic union reduction is spawn-bound";
  `docs/basic_mosaic_capacity.md`. The deleted `_mosaic_tile_worker` (git history)
  is a starting point but needs the membership-dissolve it never had.

### A3. Python 3.14 — finish validation & frozen build
- **State:** 3.14 is the default dev venv; high-risk stages (pandas-3 flatten,
  classification, pyogrio GPKG export) validated. Not yet: a full from-scratch
  headless run (Prep/import + intersect + tiles-to-completion + lines + analysis)
  and a green PyInstaller frozen build on 3.14.
- **To do:** run the from-scratch pipeline on 3.14; run `devtools\setup_venvs.bat`
  + `devtools\compile_win_11.bat` and pin `pyinstaller`/`-hooks-contrib` to the
  cp314-working versions once the build is green.
- **Refs:** `cooperation.md` "Python 3.14 promoted to default on Windows";
  `requirements_compile_win.txt`.

### A4. basic_mosaic — post-Tier-1 measurement
- **State:** Tier 1 should cut the spawn-bound ~87 % reduction dramatically, but
  the actual post-change wall-clock has not been measured.
- **To do:** re-run basic_mosaic on the 3.53 M-asset project and fill the result
  into `docs/basic_mosaic_capacity.md` ("Expected effect" → measured).

---

## B. Ideas — candidates, not yet decided

- **Snap-rounding default.** `mosaic_union_grid_size` is opt-in (0 = off). Once
  validated on a few datasets, consider a small default (e.g. 0.05 m) for the
  extra speed/robustness — it is safe in the metric mosaic CRS.
- **Mosaic memory watchdog.** The two-tier panic watchdog
  (`_start_panic_watchdog`) is not wired into the mosaic pool. Wiring it would let
  the extraction/reduction pools throttle under pressure (complements the
  pre-flight gate, which only fires at start).
- **Settings re-seed / force-defaults.** Once Phase 2 exists, an explicit
  "reset to config.ini defaults" action (re-seed the table) for operators.
- **Geocode import naming.** Import names groups `geocode_001..N` sequentially, so
  importing a *different* source folder can collide with a previous import's
  names (refresh semantics overwrite them). A stable per-source naming scheme
  would let multiple imported sets coexist.
- **Dependabot vulnerabilities (14).** Low priority for a local, air-gapped app;
  GDAL/numpy are the oldest. Revisit if the threat model changes.
- **Progress bar should reflect real work, not flat per-stage weights.** The
  "Process all" progress bar does not track actual progress for large projects —
  e.g. it showed ~90% while the run was still in Stage 1 (the data-extent
  dissolve), i.e. ~5% of the real work. The self-calibrating weights
  (`tbl_stage_runtime.parquet`: mean of the last 5 clean runs per stage —
  data/tiles/lines/analysis) help, but (a) they mix wildly different dataset
  sizes (a tiny run's timing skews a huge run's bar), (b) the whole "data" stage
  is one band, so prep / intersect / flatten / backfill / segment are not shown
  individually, and (c) the dominant cost variable — the intersect chunk count
  (≈ assets × geocode density) — is not used. Fix direction: scale each stage's
  band by the run's concrete work units (intersect chunk count, flatten
  partition count, tile count) rather than flat historical means; split the
  "data" band into its sub-stages; and drive the bar inside the intersect band
  from the already-reported done/total chunks. Goal: a bar that is roughly true
  regardless of dataset size and geocode detail.

---

*Update rule: when an item lands, delete it here and ensure the durable record is
in `learning.md`. Add new pending/idea items to the matching section.*
