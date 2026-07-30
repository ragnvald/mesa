# History

Chronological log of sessions, incidents, deliveries and decisions — what happened and in what
order. The durable rules distilled out of it live in [learning.md](learning.md); the coordination
with sibling projects lives in [cooperation.md](cooperation.md).

Format is defined in [CLAUDE.md](CLAUDE.md). Append at the bottom, newest last. Do not rewrite
earlier entries; if a later one overturns something, say so in the later entry.

---

## 2026-07-26 — workspace relocation broke helper lookup and the Maps window

A field report on an imported mesa-server package: "Classification is greyed out, and segmentation
results never appear on the map." Neither was about the data.

`_find_helper_runner` searched `base_dir/{tools,code,system}` plus `<sys.executable>/<stem>.exe`, but
`tools/` was only ever tried relative to `base_dir` — which used to *be* the install dir. With the
workspace moved to `<Documents>/MESA`, all five candidates missed, so Classification and Tiles both
reported their helper missing and were disabled. That second failure did real damage: with Data run
and Tiles unavailable, the stale-tiles cleanup deleted 15 imported `.mbtiles` the operator had no way
to rebuild. Separately, `open_combined_map()` was the one launcher not passing
`--original_working_directory`, so Maps opened the demo data shipped beside the exe instead of the
operator's project — a wrong map, not an empty one, which is why it read as "results missing".

Fixed in `36422b7`, `bf45e3f`; recorded in `91b3760`. Full build produced, 9m 40s.

## 2026-07-26 — terminology: package vs backup

Three words for the same ZIP in visible text — backup, archive, package — so the file picker offered
to "Select backup ZIP" for what is a mesa-server package, and the success dialog said "Backup
restored" three lines above "This package includes a description". Settled on the vocabulary
mesa-server uses: a **package** is the ZIP holding exactly one project; a **backup** is a package you
made of your own project. "archive" is gone from the UI. Log lines still say archive on purpose —
they mirror `create_backup_archive`/`restore_backup_archive` and should change with those
identifiers, not before them. `c1d8739`.

## 2026-07-26 — builds stop shipping the developer's data

`copy_resources()` copied `input/` and `output/` wholesale, so a distribution carried whatever
project happened to be in the repo when it was built — 43 files at the time, including an imported
`.gpkg`, every parquet table and 24 mbtiles. Decision: distributions ship a skeleton (folder tree +
`readme.txt` placeholders) and users bring their own data by restoring a package, with instructions
supplied separately. `a7b6c6f`. This reverses the earlier practice of shipping a ready-to-explore
project.

## 2026-07-27 to 2026-07-28 — Albertine Graben basic_mosaic, 25 hours

Ran `basic_mosaic` for the Uganda team on 30 of 35 shapefile layers (the five `SpeciesRichness2010_*`
layers excluded on request), because their 32 GB server could not complete it. Extracted from a
264 MB `.rar` with 7-Zip, imported headlessly through `asset_manage`'s own code path, then
`geocode_manage --nogui --mosaic`.

- Import: 20 min, 980,418 polygons, 831 MB of geometry.
- Edge reduction: 2 h. Coverage reduction: 10 h. Polygonize: 10 h. Publish: 25 min.
- **Result: 17,853,399 mosaic faces**, all geometrically valid with nothing repaired, and
  `coverage_area = faces_area` exactly — `diff = 0 m²`. Extent ~183,079 km².

Peak memory **85.65 GB** with free RAM down to 2.35 GB. The pre-flight gate had estimated 6.9 GB and
passed with a factor-of-nine margin — see learning.md. Delivered as a 2.23 GB GeoParquet and a
2.99 GB package; the parameter workbooks for three pressures × two layer sets followed.

Three false alarms from watchdogs written during the run, all self-inflicted, each pulling the
operator into unnecessary alarm. The lesson is in learning.md.

## 2026-07-28 — a source run restored a package into the repo

Restoring `Elvegrisen_260726_1532.zip` from a source run replaced the repo's own `input/` and
`output/`: `_resolve_working_dir()` returns `PROJECT_BASE` when not frozen. Three tracked
`readme.txt` files were recovered with `git restore input/`; untracked sample assets were not, and
never could be — `shutil.rmtree` does not use the recycle bin. The repo's `tbl_flat`/`tbl_stacked`
have been that package's ever since, computed by MESA 5.5.0, which later showed up as a ~2%
difference against fresh output and had to be explained rather than mistaken for a regression.

## 2026-07-28 to 2026-07-29 — parameter workbooks: join on name, not id

mesa-server found that `_apply_vulnerability_from_df` merged on `id`, which is import order and not
an identity. Measured on Albertine Graben: 33 of 35 ids disagreed between a package and a workbook.
The same mechanism had set Coral/Algae to 1/1 instead of 5/5 in the Mombasa project — the most
sensitive habitat in the dataset published as the least.

Desktop mirrored mesa-server's `app/param_io.py`: exact match on `name_original` with case preserved,
`id` only as a legacy fallback, unmatched rows loud in both directions, out-of-scale values refused
rather than clamped, and unassessed groups exported blank instead of `0` — the last of which is
required, since their reader refuses out-of-scale values and our import seeds `0`. `4207c37`.
Verified with their acceptance test: shuffling the `id` column now yields an identical import.

## 2026-07-29 — segmv → classification rename deferred to 6.0

mesa-server proposed renaming `segmv`/`seg_mv` to `classification`, shipped it, then reverted at the
user's request so both sides stay on the old names. Deferred to 6.0. Measured cost on desktop: 108
occurrences across 9 files, three namespaces (tables, config keys, **and** mbtiles filenames — which
their proposal had missed), and a config fallback that fails silently. Vocabulary agreed for 6.0:
A–E are **categories** (decided); the GMM output is likely **clusters**, which is free because
`cluster_id`/`cluster_label` are already the column names.

## 2026-07-30 — full review of everything written into tbl_flat

Prompted by a failing test before the 5.6.0 release, the user chose review over pushing the release
out. All 27 columns audited against real output from two projects rather than read from the code.

Two live defects, both changing published numbers:

- **Eight columns constant.** `importance_code_*` and `susceptibility_code_*` were the letter `E` in
  every row of every project, because the config's sensitivity bands (1..25) were applied to the 1..5
  factors. `154df19`. This overturned a conclusion in learning.md that had attributed the uniform
  class E in the QGIS Importance layer to "a data property, not a renderer fault"; that entry is now
  marked superseded.
- **A fabricated index.** The OWA histogram clipped sensitivity into 1..25, so an unassessed `0`
  became a real lowest-bin count and `index_owa` ranked cells by overlap count while presenting it as
  a sensitivity index — 17..100 across 8,480 cells on a project where every sensitivity is 0.
  `085b033`.

Also: four dead branches removed from the flatten category helpers, verified behaviour-preserving
with 0 differences across 2,723 cells (`ab0f3ac`); and the failing test turned out to be reading the
repo's live project data rather than its own fixture (`7d210b5`). Suite green, 4 passed.

Reported to mesa-server and mesa_demodata. mesa-server had the same banding defect in a different
file — `worker/export_mesa_package.py`, which our check had missed because it only looked at
`build_flat.sql` — and mirrored the fix. They also took the OWA exclusion immediately, as the one
deliberate exception to the 5.5.0 parity window running to 2026-08-04. mesa_demodata's two golden
`tbl_flat` files carried the constant `E` and are being regenerated; their nine distributed packages
are input-only and unaffected.

Open at the end of the session: whether to reject a `valid_input` containing `0`, which is coherent
with neither the sensitivity bands nor the "0 means not assessed" convention. mesa-server has said
they will follow whichever way desktop decides.

## 2026-07-27 to 2026-07-28 — Albertine Graben on the Mac: tiling the 35-layer mosaic

Ran the same Uganda dataset the Windows box ran, but the full 35 asset layers and through the
processing pipeline rather than the mosaic build, on an M4 Max (16 cores, 64 GB).

The 2026-07-26 run had died during Tiles. The log stopped at `building basic_mosaic_sensitivity_max`
while the tiles child kept going and finished at 15:00 — what died was the *parent* that streamed
stdout into `log.txt`, so the failure was invisible in the record. Two casualties only surfaced by
counting tiles afterwards: `basic_mosaic_segmv` held **0 tiles**, and its `_cert` companion and both
`_latest` aliases were never written. The join keys were fine (verified: both sides use
`basic_mosaic_NNNNNN`, 2,015,450 of 18,466,569 cells carry a `cluster_id`) — the block's broad
`except … log("skipped")` had turned an OOM into silence.

Cause, measured rather than reasoned: `tiles_create_raster.py` shipped its per-feature payload
through `Pool(initargs=…)` under spawn, so every worker unpickled a private copy. **15.43 GB per
worker** on basic_mosaic's 17,590,032 features; at the helper's own `cpu//2` default that is
**123 GB of worker memory on a 64 GB host**, on top of a parent holding all of `tbl_flat` (~18 GB)
plus a second copy of the group slice. Rewritten to a WKB blob + int64 offsets with numpy value and
colour arrays: **2.58 GB per worker**, peak tree RSS >64 GB → **39.2 GB**, swap → 0. `EXIT 0` in
1 h 53 m. Output byte-identical across 21 layers on unchanged input; segmv went 0 → 1,123 tiles.

Also found: `run_tiles_process` never passed `--procs`, so the RAM-aware sizing in
`processing_internal._tiles_procs_from_config()` was dead on the live path and `tiles_max_workers`
was an inert knob — the string "worker process(es) for MBTiles" appears **0 times in 62 MB of log**
before 2026-07-28. Wired through; it now logs the count it chose.

The 2026-07-28 re-run then deadlocked in Stage 3b: `backfill_max_workers = 0` auto-picked **16**
workers, RAM hit 81 %, the panic watchdog killed the pool, and the parent sat waiting on workers
that no longer existed — 75 minutes at 0 % CPU with no output. The key's own comment already said
*"Pinned at 4"*; the value had drifted to `0`. Set to 4: the same 801 partitions completed in
**7 minutes** with no throttle and no panic — fewer workers was both safer and faster, the phase
being I/O-bound. Resumed with `--no-prep --no-intersect --no-flatten`, which saved the 21 h of
intersect and flatten already on disk, and the pipeline completed clean.

Packaged as `basic_mosaic_35_assets.zip` (6.12 GB, 1,207 members, restore round-trip verified against
`mesa.restore_backup_archive`). **Withdrawn on 2026-07-30**: the 35-layer run was superseded once the
Windows box had delivered the 30-layer mosaic the Uganda team actually asked for, so the package was
deleted and never reached Drive. The memory findings below stand on their own and are what the run
was worth keeping for.

One measurement worth keeping for comparison with the Windows run: rebuilding `tbl_flat` changes the
tiles even though no value changes. 248 of 1,123 tiles differed byte-wise, but only **0.11 % of
pixels**, and **no tile gained or lost a colour**. The 801 intersect partitions are written by a
parallel pool and complete in non-deterministic order, so row order in `tbl_flat` differs between
runs; at ~530 polygons per pixel at z6 the last one painted wins. Byte-identity is only expected
when the input `tbl_flat` is identical.

Not exercised here: the mosaic build itself. `tbl_geocode_object` was reused from 2026-07-25
throughout, so nothing on this side tested the pre-flight gate that Windows found underestimating by
an order of magnitude.
