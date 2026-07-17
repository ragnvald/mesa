# Plans — pending work & ideas

Single index of work that is **decided but not yet done** and **ideas worth
considering**. The durable "why/how" lives in `learning.md`; this file is the
roadmap surface so a future session (or the other dev host) can pick something up
without re-deriving it from the code. Keep entries short; link to `learning.md`
or `file:line` for detail. Move an item to "Done" (or delete it) when it lands.

Lives at the repo root next to `learning.md` — both are developer notes and neither
ships. Full design documents for individual items live in `devtools/docs/`, which
`build_all.py` strips from the distribution wholesale.

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
  `devtools/docs/basic_mosaic_capacity.md`. The deleted `_mosaic_tile_worker` (git history)
  is a starting point but needs the membership-dissolve it never had.

### A3. Python 3.14 — finish validation
- **State:** the frozen build is **done** (2026-07-17): green PyInstaller build on
  3.14.6, toolchain pinned, `mesa.exe` launches and reports its version. See
  `learning.md` "First green frozen build on Python 3.14".
- **To do:** what remains is the *run*, not the build — a full from-scratch pipeline
  in the **compiled** app: Prep/import + intersect + tiles-to-completion + **lines +
  analysis**. The 3.14 source validation stopped during tiles, so lines and analysis
  have never executed on 3.14 in any form.
- **Refs:** `learning.md` "Python 3.14 validation run"; `requirements_compile_win.txt`.

### A4. basic_mosaic — post-Tier-1 measurement
- **State:** Tier 1 should cut the spawn-bound ~87 % reduction dramatically, but
  the actual post-change wall-clock has not been measured.
- **To do:** re-run basic_mosaic on the 3.53 M-asset project and fill the result
  into `devtools/docs/basic_mosaic_capacity.md` ("Expected effect" → measured).

---

## B. Ideas — candidates, not yet decided

- **Scalable processing — bounded-working-set partitioning & adaptive sizing
  (5.4+).** Generalises A2 (mosaic tiling) into one principle covering the
  **intersect** stage too, plus upgrades the scaling mechanism (`auto_tune.py`)
  from open-loop static guesses to closed-loop calibration + a runtime memory
  watchdog, so peak memory is *bounded* (small machines slow down instead of
  aborting; big machines ramp up without hubris). Also records why GPU (CUDA/Metal)
  is not the cross-platform lever, and a 2026-07 benchmark showing config-only
  mosaic tuning tops out at ~1.0–1.1×. Full design: `devtools/docs/SCALABLE_PROCESSING_PLAN.md`.
- **MESA processing server — hosted heavy compute (version 6).** A **separate
  portfolio component** (not a desktop change): users upload a project as the
  existing backup ZIP, a high-capacity Linux server runs the headless pipeline under
  full environment control (`fork` instead of spawn-bound `spawn`, huge RAM, pinned
  GEOS), and returns an output backup. Additive — **no desktop implications**;
  reuses backup upload/download + headless entry points. Builds on 5.4's scalable
  processing. Full design: `devtools/docs/CLOUD_PROCESSING_SERVER_PLAN.md`.
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
