# Integration plan — wire segmentation into MESA processing & reporting

> **How to use this file.** It is written as a ready-to-run *prompt* for a future
> Claude Code (or developer) session. Paste the “Prompt” block into a fresh session
> at the repo root, or follow the phased plan directly. It is grounded in the actual
> MESA architecture (file paths + symbols verified 2026-06-04). The experimental
> proof of concept lives in `devtools/test_segmentation.py`,
> `devtools/signature_analysis.py`, `devtools/dissolve_clusters.py`, and is described
> in `docs/MESA_Segmentation_PoC.docx`.

---

## 0. The prompt (copy/paste)

```
You are integrating the experimental "segmentation" capability (currently in
devtools/) into MESA as a first-class, optional pipeline stage plus an optional
report section. Read docs/SEGMENTATION_INTEGRATION_PLAN.md and docs/
MESA_Segmentation_PoC.docx first. Then:

1. Add a new optional processing stage "Segment" that, after Backfill, builds a
   per-geocode-layer segmentation from tbl_stacked and writes tbl_segmentation to the
   GeoParquet store. Default OFF. Reuse the proof-of-concept logic in
   devtools/test_segmentation.py and devtools/signature_analysis.py — promote the
   reusable functions into code/, do not shell out to devtools.
2. Make it config-driven ([segmentation] section) and auto-tuned (worker cap).
3. Surface it in the GUI runner as a checkbox + a --no-segment / --segment CLI flag,
   following the exact pattern of the existing Backfill stage.
4. Add an optional "Segmentation" report section to the Word report engine: a zone
   map, the per-zone profile table, and the signature mosaic chart.
5. Respect the parent-side memory rule (CLAUDE.md + learning.md): never materialise
   tbl_stacked/tbl_flat in the parent; read partition-by-partition with a pyarrow
   filter exactly as the devtools already do; do segmentation per geocode layer
   inside a guarded worker, not in the orchestrator.
6. Default behaviour must be unchanged when the stage is OFF (non-regression).

Validate by running mesa.py end-to-end through the GUI on a real project (never a
partial-pipeline shortcut). Update learning.md and cooperation.md when done.
```

---

## 1. Goal & scope

Promote the segmentation proof of concept from a manual devtool into:

- **A) an optional processing stage** that writes a reusable `tbl_segmentation` table, and
- **B) an optional report section** that presents the result.

Both are **OFF by default**. The deterministic *overlap-signature* typology is the
recommended first target — it needs no `k` and no tuning, so it is the lowest-risk,
highest-explainability slice to ship first. Algorithmic clustering (KMeans / HDBSCAN /
Agglomerative-ward) can follow as a second, opt-in mode.

### Non-goals (first iteration)
- Cross-layer / multi-resolution segmentation.
- SKATER (too slow above ~10k polygons; Agglomerative-ward is the practical default).
- Live re-segmentation in the interactive map viewer.

---

## 2. Architecture touch-points (verified)

| Concern | File | Symbol(s) | Line(s) |
|---|---|---|---|
| Pipeline orchestrator | `code/processing_internal.py` | `run_processing_pipeline()` | 3959–4108 |
| Backfill stage (the template to copy) | `code/processing_internal.py` | `backfill_tbl_stacked()` | 3610+ |
| auto_tune hook | `code/processing_internal.py` | call to auto_tune | ~4016 |
| GeoParquet dir + I/O | `code/processing_internal.py` | `gpq_dir()`, `write_parquet()`, `read_parquet_or_empty()` | 350 / 1231 / 1172 |
| Worker-pool pattern + watchdog | `code/processing_internal.py` | `get_context("spawn").Pool`, `_start_panic_watchdog()` | 1635/3396, 541 |
| Auto-tune derivations | `code/auto_tune.py` | `auto_tune_in_place()`, `_derive_backfill_max_workers()` | 316–383, 278 |
| Config | `config.ini` | `[DEFAULT]` worker-cap + mem keys | 64,101,107,138,140 |
| GUI runner window | `code/processing_pipeline_run.py` | `ProcessRunnerWindow`, `_cb_backfill` | 1914+, 2077 |
| GUI plan + runner | `code/processing_pipeline_run.py` | `ProcessPlan`, `run_data_process()` | 93–111, 547–609 |
| CLI flags | `code/processing_pipeline_run.py` | `--no-backfill` etc. | 2700+ |
| Report entry | `code/report_generate.py` | `generate_report()` | 4381+ |
| Report engine | `code/report_generate.py` | `ReportEngine`, `render_geocode_maps()` | 877–912, 914 |
| Doc building API | `code/report_generate.py` | `add_heading/add_picture/add_table` | 4069+ |

> **Rule of thumb:** the **Backfill** stage is the closest structural template for the
> new stage end-to-end (config key → auto-tune cap → orchestrator param → GUI checkbox
> → CLI flag). Mirror it.

---

## 3. Data contract — `tbl_segmentation`

Write one canonical table to the GeoParquet store (`output/geoparquet/`), ZSTD-3 like
every other `to_parquet` site (see learning.md "ZSTD-3 GeoParquet writes"):

| Column | Type | Notes |
|---|---|---|
| `code` | str | geocode/polygon id (join key to `tbl_geocode_object`) |
| `name_gis_geocodegroup` | str | geocode layer name (segmentation is per layer) |
| `signature` | str | deterministic A–E overlap signature, e.g. `B+C+D+E` (empty = no overlap) |
| `n_assets` | int | asset overlaps contributing to the signature |
| `cluster_id` | int | algorithmic-cluster label (nullable; -1 = noise, -999 = unassigned) |
| `cluster_method` | str | e.g. `agglomerative_ward_k6` (nullable if signatures-only) |
| `sens_mean` | float | per-cell mean sensitivity (carried for report colouring) |

Geometry is **not** duplicated here — join to `tbl_geocode_object` on `code` at render
time. Keep `tbl_segmentation` slim so it never becomes a second large table the parent reads.

Also write a small `tbl_segmentation_profiles` (one row per zone: method, cluster_id, size,
mean sensitivity, total area m², top-3 asset groups) — this is the report table, and is
tiny and safe to read in the parent.

---

## 4. Phased implementation

### Phase 1 — Promote reusable logic into `code/`
- Create `code/segmentation.py`. Move (don't shell out to) the proof-of-concept building
  blocks: partitioned `tbl_stacked` read (pyarrow filter), feature-matrix build, the
  deterministic `_sig()` signature, optional clustering, and `tbl_segmentation` /
  `tbl_segmentation_profiles` writers.
- Keep the devtools scripts as-is for ad-hoc exploration; have them import from
  `code/segmentation.py` so logic lives in one place.
- **Memory discipline (mandatory):** read `tbl_stacked` partition-by-partition with a
  `pyarrow.dataset` filter on `name_gis_geocodegroup`; never call
  `read_parquet_or_empty("tbl_stacked")`. Do the per-layer work inside a spawned worker
  guarded by `_start_panic_watchdog()`, not in the orchestrator. See CLAUDE.md and
  learning.md "Parent-side memory in the pipeline".

### Phase 2 — Wire the optional stage
- Add `segment_tbl_stacked(base_dir, cfg, ...)` to `code/processing_internal.py`,
  structurally mirroring `backfill_tbl_stacked()` (its own `Pool`, its own progress
  via `set_progress_stage()` / `update_progress()`, its own watchdog).
- Add `run_segment: bool = False` to `run_processing_pipeline()` (line 3959) and call the
  stage after Backfill, before Tiles. Gate every entry with `_check_pause_or_cancel()`
  so the new stage honours the mid-stage Cancel added in 5.1.

### Phase 3 — Config + auto-tune
- `config.ini`: add a `[segmentation]` section (or `[DEFAULT]` keys to match house style):
  - `segment_enabled = 0` (off)
  - `segment_mode = signatures` (`signatures` | `clusters` | `both`)
  - `segment_geocode_layer =` (empty = all layers, or a named layer)
  - `segment_n_clusters = 0` (0 = auto {4,8,16}; only used for `clusters`)
  - `segment_max_workers = 0` (0 = auto) + `segment_approx_gb_per_worker = 4.0`
  - Keep a one-line operator comment per key (CLAUDE.md), pointing at learning.md if a
    failure mode is known.
- `code/auto_tune.py`: add `_derive_segment_max_workers(hw, cfg)` next to
  `_derive_backfill_max_workers()` (line 278) and register it in `auto_tune_in_place()`
  so a `0` value is filled from the live host fingerprint and logged in the `[auto-tune]`
  block.

### Phase 4 — GUI runner
In `code/processing_pipeline_run.py`:
- Add `self._cb_segment` checkbox after `_cb_backfill` (line 2077): label
  "5. Segment (build tbl_segmentation)". Default unchecked.
- Add `run_segment` to `ProcessPlan` (line 93–111) and pass it through
  `run_data_process()` (line 547–609) into `dpi.run_headless(...)`.
- Add `--segment` / `--no-segment` CLI flags alongside `--no-backfill` (line 2700+),
  default OFF, mirroring the existing flag wiring.
- Add a short warning/info label like the sliver-cleanup one if a run will overwrite an
  existing `tbl_segmentation`.

### Phase 5 — Report section
In `code/report_generate.py`:
- Add `include_segmentation: bool = False` to `generate_report()` (line 4381+).
- When set and `tbl_segmentation` exists, add a "Segmentation" section via the existing
  `add_heading` / `add_picture` / `add_table` API:
  1. **Zone map** — join `tbl_segmentation` to `tbl_geocode_object` on `code`, render with the
     MESA A–E ramp (reuse `palette_A2E`); follow `render_geocode_maps()` (line 914) for
     map plumbing, legend ribbon, and height clamping.
  2. **Per-zone profile table** — straight from `tbl_segmentation_profiles` (size, mean
     sensitivity, top asset groups, area).
  3. **Signature mosaic** — the Marimekko/“domination overview” chart from
     `signature_analysis.py` (promote its plotting helper into `code/segmentation.py`).
- Keep the section self-contained and skipped cleanly when `tbl_segmentation` is absent.

### Phase 6 — Validate (no shortcuts)
- Run `mesa.py` end-to-end through the GUI on a real project, with the Segment stage ON,
  then generate a report with the Segmentation section ON. **Do not** use
  `run_flatten_only.py` or any partial-pipeline shortcut to validate (per project rule).
- Confirm default-OFF leaves byte-identical behaviour and outputs vs. a baseline run.
- Append a dated entry to `learning.md` and a session note to `cooperation.md`.

---

## 5. Constraints & guardrails (do not skip)

- **Parent-side memory.** The orchestrator must not allocate large datasets between
  pools. All `tbl_stacked` access is partitioned + filtered inside a worker. This is the
  single most-bitten failure mode in this pipeline.
- **Cancel & lock.** New stage must raise `ProcessingCancelled` at its inter-stage gate
  and run under the single-instance `output/.pipeline.lock` like every other stage.
- **ZSTD-3.** All new `to_parquet` calls use `compression="zstd", compression_level=3`.
- **Non-regression.** Stage OFF ⇒ no new files, no behaviour change, no perf change.
- **Explainability first.** Ship signatures before clusters; always write the
  determinants/profile alongside any algorithmic clustering so a zone is never unexplained.

---

## 6. Acceptance criteria

- [x] `tbl_segmentation` + `tbl_segmentation_profiles` written to `output/geoparquet/` (ZSTD-3) when stage ON.
- [x] Stage visible as a GUI checkbox and a CLI flag; default OFF; honours Cancel (inter-stage gate) + lock (runs under the existing pipeline lock).
- [x] `[auto-tune]` log shows a derived `segment_max_workers` when left at 0.
- [~] Report grows a "Segmentation" section — **profile table + signature mosaic shipped**; the categorical **zone map** (cluster geometry render) is the one remaining follow-up.
- [x] Baseline run with stage OFF is unchanged (default OFF in orchestrator/GUI/CLI/config/report).
- [~] `learning.md` + `cooperation.md` updated; **validated via the real `segment_tbl_stacked` spawn path on H3_R5/R6** — the full GUI run on basic_mosaic remains for the operator.

---

## 7. Implementation status (2026-06-05)

**Done** (working tree, not committed):
- `code/segmentation.py` — signatures + KMeans/agglomerative-ward; slim `tbl_segmentation/<layer>.parquet` + `tbl_segmentation_profiles.parquet`; per-layer pyarrow read inside spawned workers; mosaic helper for the report.
- `code/processing_internal.py` — `segment_tbl_stacked()` (Stage 4b) via spawned Pool + hard-panic watchdog; `run_segment` threaded through `run_processing_pipeline` / `run_headless`; `segment_enabled` force-on; `cleanup_outputs()` clears stale segmentation.
- `code/processing_pipeline_run.py` — `ProcessPlan.run_segment`, GUI checkbox "4b. Segment" (OFF, out of master cascade), `--segment`/`--no-segment`, threaded through `run_data_process`.
- `code/auto_tune.py` — `_derive_segment_max_workers` registered.
- `config.ini` — `[DEFAULT]` `segment_*` keys.
- `code/report_generate.py` — optional "Segmentation" section + GUI checkbox + `include_segmentation`.

**Naming note:** uses `tbl_segmentation*`, **not** `tbl_segments` (that is the Lines stage). See learning.md 2026-06-05.

**Refinements landed 2026-06-06:** profile tables sort zones by **total area (km²) desc**
(new `total_area_km2` column, area from per-cell `area_m2`); Segment stage defaults to
**basic_mosaic only** (`segment_geocode_layer` = blank → basic_mosaic, `all`/`*`, or
comma-list); report renders one sub-table **per method** (method in the heading, not a
column); report dialog gained a two-column layout with a **per-level multi-select**
(`generate_report(segmentation_layers=...)`).

**Remaining follow-ups:**
1. Report **zone map** — render cluster geometry with the engine's map machinery (join `tbl_segmentation` → `tbl_geocode_object`, MESA A–E ramp). Section currently ships mosaic + table.
2. Full **GUI end-to-end** run on basic_mosaic (9M) with the stage ON, then a report with the section ON — per the project rule, never a partial-pipeline shortcut.
3. Optional: a CLI `--report-segmentation` flag for the headless report path.

---

_Plan authored 2026-06-04. Architecture references verified against the working tree on
branch `main` (MESA 5.1). If line numbers have drifted, match on the named symbols in the
table in §2._
