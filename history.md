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

## 2026-07-31 — vocabulary: backup becomes store, and the 5.6.0 release checklist

The Mac side landed five commits during the day — tiles shipping geometry as a WKB blob rather than
per-worker Python objects, `--procs` finally reaching `tiles_create_raster` so its worker caps apply
at all, and `backfill_max_workers` pinned to 4 after auto-sizing picked 16 on a 16-core/64 GB host
and tripped the panic watchdog into a deadlock. A push from this side collided with them; rebased
after checking that neither touched the same file and that their `config.ini` hunk sat far from the
uncommitted one here.

Two UI additions from live use of the app:

- **H3/QDGC level picker** (`5e38c6a`). Suggest wrote its matching levels into a read-only label and
  Generate built all of them — seven levels for the default 50 m to 50 km range. It is now one
  checkbox per level, all ticked, and Generate reads the ticks. The capability had always existed in
  the CLI (`--h3-levels`); only the GUI hardcoded "all". Never a regression — no level selector has
  ever been in that file, checked against history rather than memory.
- **Asset names in the map popup.** The cell popup showed counts but not which assets. First attempt
  listed `layer_002, layer_003` — `tbl_flat.asset_group_names` carries the synthetic GIS name, not
  the readable one — so it now looks up `title_fromuser` from `tbl_asset_group`, falling back to
  `name_original` and then the raw name.

**The vocabulary change.** "Backup" is out of the user-facing Manage-data flow, in the app and in the
wiki. The rule: the verb is *store* / *restore*, the noun is *MESA package*. `Create backup` →
**Store data**, `Backup and restore` → **Store and restore data**, `Backup options` → **Options**,
and the default filename from `mesa_backup_<ts>.zip` to `mesa_package_<ts>.zip` — accepted as
harmless with roughly two users today. 19 visible strings in `mesa.py`, five wiki pages.

Two things deliberately left alone. `Mesa-changelog-overview.md` keeps its seven occurrences: it
records what earlier versions were called, and rewriting it would falsify that. Log lines and the
internal identifiers they mirror (`create_backup_archive`, `restore_backup_archive`) also stay —
they move together, in an identifier pass, not before it. This supersedes the vocabulary settled on
2026-07-26, which kept "backup" for a package you made of your own project.

A question about `In MESA 5.1` in Troubleshooting turned out to be a provenance marker, not a stale
version — the wiki uses that form throughout (`added in MESA 5.1`, `From MESA 5.5 the template…`).
Tightened to `From MESA 5.1` to match the dominant phrasing. But it surfaced three markers that
genuinely are stale.

**Open for the 5.6.0 release.** Version is already bumped in `config.ini`; there is no 5.6.0 tag.

1. **Full build against HEAD.** `D:\dist\mesa` is from 2026-07-26 and is now eleven commits behind,
   including both `tbl_flat` value fixes, the parameter-workbook join key, and the skeleton filter
   for `input/`/`output/`. None of it has been seen in a frozen build.
2. **Smoke-test the frozen fixes**: Maps opens on the workspace rather than the install folder;
   Classification and Tiles are not greyed out; the distribution's `input/`/`output/` contain only
   the folder tree and `readme.txt`.
3. **Run the test suite** — `pytest tests/ -q`, green at 4 passed. It is not in CI, so it only
   protects a release if somebody runs it.
4. **Wiki version markers.** `Home.md:5`, `Home.md:14` and `Technical.md:6` still say MESA 5.5 is the
   current release line. Move them to 5.6.0 *when the tag exists*, not before — a wiki announcing an
   unreleased version is worse than one a step behind.
5. **Screenshots.** `images/ui_backup_restore.png` shows buttons that no longer exist after the
   vocabulary change. Needs the GUI and `devtools/capture_ui_active_batch.py`; the 5.2 screenshot
   refresh is also still outstanding. Picked up as its own pass, not as part of the release.
6. **Tag 5.6.0.**

## 2026-08-01 — MESA 5.6.0 released

The release checklist opened on 2026-07-31 is closed. Screenshots were refreshed first, against a
new demonstration project with continuous coverage across the Kampala area — 16 asset layers, 5
geocode layers, 26,975 cells in `basic_mosaic`. The earlier demo left half the map empty, which made
the Overview and Segmentation captures read as broken rather than sparse.

The refresh found a real defect in `devtools/capture_ui_active_batch.py`. `ui_status.png` came out
showing the Workflows tab: the Status tab reads project statistics synchronously and blocked Qt's
paint loop for longer than the tool's fixed 1.2 s wait, so the grab returned the previous tab still
on screen, tab bar included. Waiting for the pixels to settle would not have caught it either — a
frozen window is perfectly stable. The tool now requires a change from the pre-keystroke frame first,
then stability, and logs `STALE` when the timeout hits (050f7ab). The H3 level picker still needs a
one-off script, since the batch never clicks inside a helper and the checkboxes only appear after
*Suggest*.

`config.ini` arrived from the demodata side carrying a generated-file header pointing at
`tools/make_demo_config.py`, a script that does not exist in this repo. The header was dropped and
the two intended tuning changes kept (`backfill_max_workers = 0`, `segmv_k_range = 5`). Two things
were caught in the same diff: the incoming comment said "Pinned at 4" above a value of 0, and
`segmv_ai_enabled` had been flipped to 1, which would have shipped AI cluster descriptions on by
default against a comment two lines above stating they are opt-in so Process never depends on a
running Ollama. Reverted to 0 by the user's decision.

`docs/readme_demodata.txt` also appeared in the tree and was *not* committed. It is a member of a
demo package, written by mesa_demodata and read by MESA on restore; `mesa.py` deliberately omits it
when storing a package and deletes any stale copy before extracting, so the description exists
exactly while it is true. Tracking it would ship one package's measured overlap as the description
of every project.

Published to Zenodo as 10.5281/zenodo.21745481 — `MESA_560_win11_2026_08_01.zip` (1.5 GB) and
`MESA_560_macos_2026_08_01.dmg` (469.6 MB), the first release with a macOS build. The Windows build
is unsigned and the macOS build is Developer-ID signed and notarised, which the release notes and
the Zenodo description both state. Demonstration data moved out of the software record into two of
its own: 10.5281/zenodo.21744273 (real open data, East Africa) and 10.5281/zenodo.21744040
(synthetic). The wiki was swept for Zenodo references in the same pass, which surfaced two claims
that had been wrong since 5.6 changed the workspace layout: `Data.md` and `Advanced.md` both still
said the workspace sits next to `mesa.exe`.

`pytest tests/ -q` green at 4 passed before tagging.

## 2026-08-05 — Two 3D landscape categories in the generated QGIS project

The interactive 3D views in `qgis/mesa_results.qgz` were reported as hexagons rendering "walls
without a roof, more like pipes stretching upward". They were pipes. Column height came from the
dataset span alone, computed once and reused for every view, while cell width varies twentyfold
between the grids in one project: QDGC L11's 54 m cells carried 1733 m columns, 32:1, and H3 R10's
156 m cells 11:1. At that aspect the top face never turns toward the camera, and at scene distance
the cell is under a pixel wide, so only the lit side walls register. Height is now capped at four
times the cell width, derived from the grid level (H3 edge = 1107.7 km / sqrt(7)^r, QDGC = 2^-n
degrees) so nothing reads geometry. Only the finest grids move: H3 R8 stayed at 1733 m.

Two mechanisms were ruled out along the way, and one claim had to be withdrawn. The tessellation of
the top face was not failing — the cells are clean closed polygons, 7 vertices for H3 and 5 for
QDGC — and `rendered-facade="3"` is Walls|Roof in both QGIS 3 and 4, verified against the QGIS
source. The claim that `opacity="0.99"` pushed the columns into QGIS' transparent render pass was
written into learning.md as fact before the source was checked; the pass exists and runs with
`QNoDepthMask`, but no code was found routing vector symbol entities into it, and the entry was
corrected the same session.

Each geocode group now gets two landscapes instead of one: the existing asset-overlap view, and a
second extruded by the A–E sensitivity code, A tallest and E short but present, on the seed's own
sensitivity palette so it matches the 2D map. Every group is included, `basic_mosaic` and uploaded
layers among them. Mosaic cells inherit the size distribution of the assets they were cut from, so
a few are enormous: on the Albertine Graben project the largest is 186 km², 153,000 times the
median and 34% of the mosaic's area, and at full height it roofed over the whole scene. Cells past
100 times the group's median area keep their height but drop to 0.3 opacity. Filtering them out was
considered and rejected — 435 of 26,975 cells carry 76% of the area. Shipped as bc74e79 and
e13b224.

## 2026-08-06 — Generating H3/QDGC silently deleted the mosaic

The Albertine Graben project was set up in the development rig from
`2026_07_28_albertinegraben_basic_mosaic_30assets.zip`, a MESA package holding the 30-asset run:
17,853,399 `basic_mosaic` faces, 980,418 asset objects across 30 groups, and the three parameter
workbooks. Processing was then run in the belief that the mosaic was in place. It was not.
`tbl_geocode_object` held 1,568,725 rows across eight generated H3/QDGC layers and not one mosaic
face, while `tbl_geocode_group` still listed `basic_mosaic` at id 1. The project therefore looked
registered and processed without it.

The cause is `_load_existing_geocodes` in `geocode_manage.py`: it
caught every exception from `gpd.read_parquet(tbl_geocode_object)` and returned an empty frame,
which `_merge_and_write_geocodes` then concatenated the new grid objects onto and wrote back. A
failed read and an empty table were indistinguishable, and nothing was logged. The swallow is
older than `geocode_manage.py` itself — it came across from `geocodes_create.py` and predates the
5.0 tag, so every MESA 5 release carries it, which is what the 5.7 changelog states. What was new
was not the code but the data — decoding that table costs 1,269 bytes a row, measured, so 23 GB for
19.4 million rows before `pd.concat` copies it, against 64 GB free while a mosaic build was
running. Earlier projects had mosaics small enough to read.

Both halves are fixed. The group loader no longer swallows, and object merging streams row groups
through pyarrow, leaving geometry as the WKB it already is on disk: the same merge that would have
cost 23 GB adds 250,000 objects to the real 2.3 GB table in 15 s at a peak of 0.50 GB, measured on
a copy. Objects now commit before groups, so a failure leaves both tables as they were rather than
a group listed with no cells. Six tests cover it in `tests/test_geocode_merge.py`; two of them
failed first because `gpq_dir()` caches its directory in a module global, so every test after the
first wrote into the first one's tmp_path — the same isolation trap recorded on 2026-07-30.

The lost mosaic was restored by streaming it back in from the package alongside the eight grids,
19,422,124 rows in 15 s. Processing results from the run without it — `tbl_flat`, `tbl_stacked`,
the segmentation and `tbl_data_extent` — are stale and need a re-run. A mosaic rebuild that had
been started as a workaround was stopped at 96% of round 5 of 6, after 6,221 s, before it wrote
anything.

This one is worth stating plainly in the next release notes: any project that generated H3 or QDGC
layers on top of a large `basic_mosaic` may have lost the mosaic without being told, and any
results computed afterwards exclude it.

`pytest tests/ -q` green at 19 passed.

## 2026-08-08 — tbl_stacked loses its geometry column

mesa-server sent a proposal (`forslag-tbl-stacked-geometri.md`, not archived — see docs/exchange/README.md) to drop the geometry column from
`tbl_stacked`: on a real package the table was 30.41 GB of 27, and 83% of it was the same cell
geometry repeated once per asset that touched the cell. Their reading of our flatten step was
correct to the line — it read 94 million geometries to keep 18 million via
`drop_duplicates(subset=["code"])`.

Verified on our side before answering. On the run sitting in `output/geoparquet/__stacked_parts`
here (142 partitions, 15,517,653 rows, 395 MB, already ZSTD-3) geometry is 71.6% of the bytes and
everything else is 7.2 B/row — which scaled to their 94 million rows gives ~660 MB, matching their
595 MB estimate. 71.6% is the low-water mark: those are grid layers with cheap WKB, and
basic_mosaic is what takes it to their 83%.

We found a third consumer they had missed, and it was the expensive one: the Backfill stage read
every partition with `gpd.read_parquet`, merged one Int64 column, and wrote the whole partition
back with geometry and all — tens of GB of I/O to add 8 bytes a row. The column it wrote,
`area_m2`, is a property of the cell, not of the (cell, asset) pair. And the polygon analysis
loaded the entire stacked dataset into the orchestrator process with one `gpd.read_parquet` before
filtering to basic_mosaic, which on a 30 GB table cannot run at all — the feature was already
broken at that scale, so removing the column cost us nothing and forced the fix.

The classification path (`segmentation.py` / `segmentation_run.py`) already did exactly what the
proposal asks for: pyarrow column projection and predicate pushdown on the layer, geometry joined
from `tbl_geocode_object` at render time, no geometry in its output tables. That stage was the
existing proof, and the rest now follows it.

Shipped in this session: a `STACKED_COLUMNS` contract that also drops the merge artefacts `id_x`
and `id_y`, the unread `attributes` blob and `grid_cell` (~10% on top of the geometry); a
column-projected `_read_stacked_partition` used by flatten; the cell geometry joined into
`tbl_flat` once, keyed on (`name_gis_geocodegroup`, `code`); the analysis rewritten to read stacked
rows scoped to the cells a polygon actually touches; and the flatten pre-flight amplification split
into geometry (10x) and geometry-free (3x) so the gate stops refusing jobs that now fit easily.

The Backfill stage was removed outright rather than left as a no-op — the operator's call, taken
after the first cut had parked it for 6.0. Gone with it: `backfill_max_workers` from config.ini and
`mesa.py`, `_derive_backfill_max_workers` in `auto_tune.py`, the "4. Backfill" checkbox in the
Advanced panel, and the `--no-backfill` CLI flag. A script still passing that flag now fails on an
unknown argument, deliberately: silently accepting a flag that does nothing is worse.

Also noted while working: the repo's `config.ini` had been overwritten at some point by a generated
demo config (`tools/make_demo_config.py` banner, `mem_target_frac` 0.85, `chunk_size` 22000,
`segmv_ai_enabled` 1 and others changed from canonical). It was reset to HEAD, and only the two new
pre-flight keys were added on top.

Non-regression measured, not assumed: HEAD and the new code were run over the same synthetic
project and `tbl_flat` came out identical in all 27 columns including geometry; the analysis
outputs matched on everything except the per-run uuids and timestamp; the retained `tbl_stacked`
columns are row-for-row identical. `pytest tests/ -q` green at 24 passed (5 new).

Answered mesa-server in [`docs/exchange/2026-08-08-tbl-stacked-geometri-svar.md`](docs/exchange/2026-08-08-tbl-stacked-geometri-svar.md): yes to the change, no to a version marker
(the parquet schema is the authority — `mesa_version` describes the writer, not the file), lesson
about `gpd.read_parquet` raising outright on a geometry-free parquet so readers must ship before
their writers do, and two asks back — an agreed column contract rather than just "minus geometry",
and their `overlap_m2`, which we compute per row in `_drop_boundary_only_joins` for the 1 m² rule
and then throw away. It is the one genuinely per-row value in the table and cannot be looked up.

## 2026-08-08 — first run after the tbl_stacked change fails on a missing GeoParquet header

The first real run after the geometry change died in Stage 2 with "The CRS attribute of a
GeoDataFrame without an active geometry column is not defined". Not the tbl_stacked work: the
project's `tbl_geocode_object.parquet` had been written without the schema-level GeoParquet `geo`
header, so `read_parquet_or_empty` fell into its "Missing geo metadata" branch, returned a frame
with `geometry=None`, and the next line's `.crs` raised.

Root cause in `_rewrite_geocode_objects` (geocode_manage.py), the streamed merge added on
2026-08-06. It builds its output schema from `new_table.schema` when there is no existing file, and
`GeoDataFrame.to_arrow()` records the CRS in per-field geoarrow metadata rather than the
schema-level `geo` key. Any project that generated geocodes from scratch since that change has the
same file. The data was never damaged — measured on the affected project, all 29,682 cells and the
EPSG:4326 CRS were present in the field metadata throughout.

One of the new readers had the same fragility, and it was ours: `_attach_geocode_geometry` called
`gpd.read_parquet` bare — exactly the failure mode the reply to mesa-server warns them about.

Two fixes. `mesa_shared.read_geoparquet` tries geopandas first and recovers through Arrow on that
specific error, used now by `read_parquet_or_empty`, `_attach_geocode_geometry`,
`segmentation_run.read_layer_geometry` and `segmentation.py`. `_geoparquet_header` in
geocode_manage.py writes the header when the table is built from scratch, taken from geopandas
itself via an empty frame so the JSON cannot drift from the spec; a following merge keeps it and
widens the bbox.

Verified by running intersect + flatten against a copy of the affected project with the headerless
file untouched: 9 partitions, 53,949 rows, 0.35 MB (~6.5 B/row, matching the 7.2 B/row measured
earlier); flatten 5.7 s to 28,674 rows with geometry and a positive area on every row across all
four geocode groups, no missing-geometry warnings. So existing projects need no regeneration.

Two process notes. The verification harness itself burned twenty minutes first: the script had no
`if __name__ == "__main__"` guard, so on Windows spawn every worker re-ran it, deleted the project
directory and started another pool — visible as cycling worker PIDs, no output and almost no CPU.
And resetting `config.ini` to HEAD earlier in the session reverted this host's tuning
(`geocode_soft_limit` 420 -> 250, `asset_soft_limit` 100000 -> 50000, `chunk_size` 22000 -> 8000,
`mem_target_frac` 0.85 -> 0.60); the run's log is full of "geocode subset size > 250; splitting",
so intersect at 553.9 s is paying for that, not for the format change.

## 2026-08-08 — tile renderer, Landing A and B

mesa-server proposed replacing the tile renderer's per-cell drawing with pixel-centre sampling
against a PostGIS spatial join, to escape a 26.7 GB memory ceiling and speckled coarse tiles. The
assessment found the premise did not hold for desktop: our renderer has none of the hybrid /
centroid-scatter / adaptive-cap machinery theirs has grown, so the speckling cannot occur here, and
the efficiency argument rests on a database desktop does not have. Point sampling was measured
against MESA's own data and rejected — MESA buffers line and point assets to 30 m, which is under
one pixel at every zoom it renders (z6 = 2446 m/px, z12 = 38.2 m/px), and pixel-centre sampling
turns a continuous 0.05 px band into 38.7 of 248 columns where PIL's polygon fill keeps 216.

The operator then asked the better question: render the finest zoom once and aggregate upward.
Measured, every zoom level costs about one full pass over the features (1.00-1.02x of N, because
tile count quadruples as candidates per tile quarter), so the pyramid is worth ~6.9x on
basic_mosaic — and combined with sharing one geometry pass across the seven layers, ~49x. That is
Landing C and is not built yet; it changes coarse-zoom appearance and should be agreed with
mesa-server first.

Landings A and B shipped in this session, on the parts that do not change what a pixel means:

- The WKB payload is spilled to a file and mmapped rather than pickled into every worker. The
  module's own docstring recorded 3.2 GB for 17.6M features, which at eight workers is the 26 GB
  that made a large mosaic unrenderable; it is now one physical copy.
- Tile planning is vectorised bbox arithmetic instead of one `sindex.intersection` per tile in the
  parent, which also removes the ~1.8 GB STRtree.
- Projection is batched over all of a tile's rings in one call.
- Candidates are drawn in ascending value order, so the highest value wins a contested pixel.
- `tbl_geocode_object` is read through `mesa_shared.read_geoparquet`, which had been silently
  costing this project its Classification rasters ("classification tiles skipped: Missing geo
  metadata").

Measured end to end on a real 28,674-cell project, z6-z11, four geocode groups, 20 mbtiles,
195 tiles: **63.3 s -> 37.3 s**, output 360 KB -> 347 KB. The render core alone, timed on one z6
tile with all 26,975 basic_mosaic cells as candidates: 1488 ms -> 485 ms, 3.1x, or 55.2 -> 18.0 us
per cell.

Correctness was established piece by piece rather than by looking at the total. The vectorised
planner returns identical candidate sets on all 41 tiles, nothing lost or gained, in a different
order. The batched projection returns paths identical to the previous implementation over 26,984
rings at three zooms on two groups, 0 mismatches, including which rings are skipped. The 1.32% of
pixels that changed are therefore entirely draw order — which is the point of the ordering change,
and `--no-max-aggregate` restores the old behaviour.

Two things were built and then deleted on the evidence. Palettised PNG looked like a 5x win on a
synthetic dense tile and measured *slower and larger* on real ones. The first attempt at vectorised
projection worked ring by ring and delivered 1.1x, because the average cell is 38 vertices and
numpy call overhead ate it; only batching across the tile made it 3.1x. Both are recorded in
learning.md, because in both cases the benchmark, not the code, was what had been wrong.

Written up for mesa-server in [`docs/exchange/2026-08-08-veiledning-mesa-server.md`](docs/exchange/2026-08-08-veiledning-mesa-server.md): the column contract, the GeoParquet
metadata trap, the tile-renderer changes with their measurements, and a memory section written
against their 32 GB machine specifically. Their 26.7 GB ceiling turns out to be exactly the
per-worker WKB copy — 3.2 GB × 8 workers = 25.6 GB, which their own inherited docstring already
recorded — so the mmap fix is the first item on the checklist. The one place they are told *not*
to copy us is the parent-side geometry join in flatten: measured 1.47 GB per 3 million cells over
three columns, ~9.5 GB extrapolated to 19.4 million, which is fine on a 127 GB desktop and is not
fine on 32 GB. Doing it per geocode group is owed to them, and to us.

Removing Backfill had left the operator-facing numbering with a hole — the Advanced panel read
1, 2, 3, 5, 6, 7… and the log emitted "[Stage 3a]" and "[Stage 3c]" with no 3b. Both were closed
up: the panel is 1-9 contiguous and the data sub-stages log as 1/4 through 4/4. Found by building
`ProcessRunnerWindow` headlessly, which is also the only check that the three removed
`_cb_backfill` references left nothing behind; no test covered window construction before.

Note on process: a verification harness written during this session lacked an
`if __name__ == "__main__"` guard and, separately, resolved `base_dir()` to the mesa repo because
its sandbox project had no `input/` folder — so one baseline run wrote tiles into the real
`output/mbtiles`. Same source data, rendered by the HEAD renderer, so nothing was lost, but the
harness now refuses to report a run that escaped its project directory.

## 2026-08-09 — mesa-server on draw order; an A-E divergence surfaces

mesa-server implemented four of the six checklist items and independently verified the ordering
finding, more strictly than we had: feeding the new renderer the old planner's candidate order gave
0 differences across 18,874,368 pixels. Both sides found the same non-determinism, from opposite
directions, and neither had seen it before something moved the order.

They then asked six questions before adopting max-aggregation, all of which needed answering from
the code rather than from memory. Answered in [`docs/exchange/2026-08-09-tegnerekkefolge-svar.md`](docs/exchange/2026-08-09-tegnerekkefolge-svar.md).

The important one was not the one they thought. They report that on their side E is the most
sensitive class, ascending values mapping to ascending letters. Here A is: `config.ini` gives
`[A] range = 21-25, description = Very high, category_colour = #bd0026`. The function they cite,
`build_code_from_numeric_if_missing`, hard-codes nothing — the whole meaning of A..E comes from
config, so if their report is accurate the two projects assign opposite value ranges to the same
letters. That would make `sensitivity_code_max` mean the opposite thing in a package crossing
between us, and it reaches legends, QGIS styling and any report naming a class. Worse,
`read_sensitivity_palette_from_config` — vendored from us — carries
`defaults = {"A": "#bd0026", ..., "E": "#ffffb2"}`, dark red through pale yellow, so an unswapped
config would paint their most sensitive class pale. They have been asked to check `[A] range` and
`[A] category_colour` and report back rather than us assuming.

The renderer-level answer sidesteps it: we sort on the numeric `sensitivity_max`, not on the
letter, so letter direction cannot flip a map.

Two of our own claims were corrected by measurement before sending. We had assumed sorting would
beat `np.maximum.at` for the scatter path; measured, `fmax.at` is ~10x faster (5M points: 21 ms
against 750 ms) because argsort dominates — their instinct was right and ours was wrong. And the
right primitive is `np.fmax.at`, not `np.maximum.at`: `maximum` propagates NaN, so a single
unassessed cell blanks the whole pixel, which would read as missing data rather than as an
aggregation artefact.

Their question 6 was answered in code rather than prose: `mesa_aggregation` is now written to the
mbtiles metadata of every layer, `max` where a draw-order key exists and `none` where none does,
with `mode` / `mean` / `count` reserved for the value-raster work. Verified on a real run across
all ten layers of a group. One gap admitted on our side: `segmv_cert` is ordinal but rides the
classification pool without an order key, so it is labelled `none`.

## 2026-08-09 — the mesa-server correspondence gets a home

Four letters to mesa-server had accumulated in the repository root — the Classification
specification from 2026-07-26 plus the three written this week — and **three of the four had never
been committed**, so deleting them would have been irreversible rather than a tidy-up. That was
the finding that decided the order of work: commit first, rearrange second, and never delete an
untracked file to reduce clutter.

They were not purely historical either. Four live things lived only in those letters: the
`tbl_stacked` column contract, the `mesa_aggregation` vocabulary, the unresolved A-E direction
question, and the open per-group geometry join. A contract that exists only in correspondence has
to be reconstructed by reading letters in date order, which is not a way to keep an agreement.

`cooperation.md` turned out to be the intended home and to have drifted: its own preamble called it
a dialog between Claude sessions on two developer hosts, while `history.md`'s header said "the
coordination with sibling projects lives in cooperation.md". Neither description clearly claimed the
mesa-server letters, which is most likely why they ended up in the root. The preamble now covers
both counterparts and says so, dated, rather than quietly widening.

What landed:

- A maintained-in-place **Agreed formats** section at the top of `cooperation.md`: the `tbl_stacked`
  columns, the GeoParquet header rule and its readers-before-writers rollout order, the
  `mesa_aggregation` values, and the three open cross-project questions. Marked as an explicit
  exception to the file's append-only rule, since a contract is only useful in its current form.
- `docs/exchange/` holding the four letters verbatim under dated names, with `klassifisering.md`
  moved via `git mv` so its history survives. Root markdown files: 14 -> 10.
- A `README.md` in that folder saying it is the audit trail and not the reference, and recording
  that the archive is **one-sided** — all four documents are outbound, because mesa-server's two
  proposals and their question letter arrived as pasted text in a working session and were never
  saved. They should be retrieved and filed, or the folder reads as a monologue.
- A line in `CLAUDE.md` under "Where to put what", so the next session files a letter instead of
  dropping it in the root and the cleanup does not undo itself.

## 2026-08-09 — what the tbl_stacked change actually bought, measured

A true A/B of the data stages, HEAD against the working tree, same input, same host, both variants
run in sequence:

```
Intersect           498.4s      490.2s       1.02x
Flatten               5.3s        5.3s       0.99x
Backfill              1.4s        0.0s     removed
TOTAL               505.0s      495.5s       1.02x
tbl_stacked on disk  5.64 MB -> 0.41 MB      92.8% smaller
rows identical: stacked 65,541 | flat 31,826
```

**The format change bought disk and memory, not wall clock.** That contradicts what the component
measurements had suggested and what had been implied to the operator: the flatten read is 13.5x
faster per row and the intersect write 17.4x, but on this project flatten is 5.3 s in total,
dominated by process spawn and pool setup, and intersect is dominated by the sjoin and the
boundary-bleed filter, neither of which was touched. Two percent is noise.

Both component figures remain true and both scale with data — backfill on a 94-million-row table
was tens of GB of read-and-rewrite, and the flatten read saves ~0.8 us a row, which is 0.05 s here
and over a minute there. But on the project actually in front of the operator, the honest answer is
that processing takes the same time and the table is 93% smaller.

The changelog was corrected to say so outright rather than let a reader infer a speed-up from a
size reduction. The 92.8% is also a better number than the 72-83% quoted from the geometry share
alone, because it is a measured before/after of the whole table including the other repeated
columns.

This is the third time this session that a component benchmark predicted an end-to-end result it
did not deliver — after the per-ring projection (19x predicted, 1.1x delivered) and the palette PNG
(5x predicted, slower delivered). The pattern is consistent enough to be worth stating as a rule:
a component speed-up is a claim about a component, and becomes a claim about the program only
after the program has been timed.

## 2026-08-09 — Link zoom & pan restored; the stale comment was the clue

The operator reported two things in quick succession: switching layer in the Overview tab reset the
view, and "Link zoom & pan" had stopped working — it used to hold the view across tabs.

The first was a plain bug in `showResLayer`, which called `fitBounds` on every call. Three of its
four call paths do not move the data at all: the layer-type radios, the group dropdown reselecting
the same group, and the master geocode-layer checkbox, which reset the view every time it was
toggled. Fixed by tracking the group the map was last fitted to.

The second was a regression with a name. While fixing the first I noticed that `showClassTiles`
carried a comment — "When linked, keep the synced view from the previous tab; only auto-fit when
unlinked" — above a `fitBounds` that ran unconditionally, and set it aside as a curiosity worth
mentioning. It was not a curiosity. `git log` on the file named `d88d280` (2026-07-23, "always zoom
to layer data, and size the map before the first fit"), which rewrote six sites from
`if(linking){ syncFrom(x) } else { fitBounds() }` to `fitBounds(); if(linking){ syncFrom(x) }`. That
is exactly the reported loss, and it shipped four days after a commit whose subject line was "stop
layer toggles panning when linked".

The commit was not wrong about its own problem: a `fitBounds` against a map the webview has not
sized yet picks the wrong zoom, which is why it added `invalidateSize()` everywhere. Those calls are
kept. What is restored is the condition, now in a single `frameMap(key, bounds, keepView)` that all
seven call sites go through — six copies of a rule is how half of them end up on an old version of
it.

One improvement over the pre-regression behaviour: the first framing always happens, even when
linked. The old code would push Leaflet's default world view to every map if a loader ran before
anything had been framed.

Verified by extracting the real JS from `combined_map.py` and running it in QtWebEngine — the engine
the Maps window actually uses, since there is no node on this host — against a Leaflet stub. Six
cases, all passing: unlinked first show fits, unlinked layer switch holds, unlinked other tab fits,
linked other tab holds and pushes its view onward, and linked-with-nothing-framed still fits.

## 2026-08-09 — Classification becomes an opt-in

Classification joined every Process run as long as a geocode layer was configured, and it is the
most expensive thing MESA does: a Gaussian Mixture fit for every k in the range, over every cell of
the chosen layer, repeated on each run. Nothing told the operator that, and nothing asked.

It is now opt-in behind `segmv_enabled`, default 0. The tick sits at the top of Configure ->
Classification, above the settings it governs, with a plain-language note on what the cost is —
that a 2-15 range means fourteen fits over the whole layer, that on a large `basic_mosaic` that is
a long wait, and that everything else in the pipeline completes without it. The settings below grey
out while it is off rather than disappearing, so a configuration made earlier is visibly still
there.

The gate is in `detect_classification_processing`, not on the runner's checkbox. That matters:
Normal mode builds its plan from availability and never reads checkbox state, so gating at the
checkbox would have left "off" meaning off in Advanced mode only. One place covers both — in
Advanced the stage greys out with the reason as its tooltip, and in Normal it is simply not run.

Deliberately not gated: invoking `segmentation_run` directly. The flag decides whether the stage
joins Process automatically, not whether a deliberate run is allowed.

Verified headlessly against the real windows rather than by reading the code: the setup window
defaults to off and greys its six dependent controls, `_collect()` writes `segmv_enabled` both ways,
`detect_classification_processing` blocks with a reason pointing at the setup window, the Advanced
checkbox is disabled and unchecked with that reason as its tooltip, Normal mode's plan comes out
False, and `params_from_config` round-trips the flag.

## 2026-08-09 — the classification popup names its asset groups

Clicking a cell on the Classifications map gave the type, certainty, overlap count and the top
importance x susceptibility bins — everything about how sensitive the type is, nothing about what it
is made of. `top_asset_groups` already existed, computed per run and shown as a column in the
Classifications table, but never reached the popup.

It is a per-TYPE attribute in `tbl_seg_mv_profile`, not per cell, so both popup paths had to join it
on `cluster_id`. The vector path builds a `cluster_id -> top_asset_groups` map in JS from the profile
rows the same response already carries; the raster path joins it onto the cached cell frame in
`_fi_load_class`, where `_fi_class_info` can list it as another field. `cluster_id` was added to the
profile rows so the join is on an id rather than on `cluster_label`, which is a display string.

While in that code, both reads of `tbl_geocode_object` in `combined_map.py` were switched to
`mesa_shared.read_geoparquet`. They were the last two `gpd.read_parquet` calls on that table outside
the pipeline, and on a project whose geocode table lacks the GeoParquet header — which is every
project that generated geocodes during 5.7 development — the fallback was `gdf = None`, meaning the
Classifications map would have failed to identify anything at all, silently.

Verified against the real project, run 2026-08-09_130819: all 7 profile rows carry both fields, and
a raster cell resolves to "Top asset groups: Minor road corridor (30 m buffer), Shrubland,
Grassland" across 19,150 cells.

## 2026-09-16 — Generated grid names get a two-digit level; demo packages follow

The report's map selection listed `H3_R10` before `H3_R7`, and the maps did the same: group names
sorted as text. The operator asked for `H3_R07` / `QDGC_L06`, on the condition that it did not
compromise the libraries' own naming. It does not — H3 names only resolutions and cell indexes,
QDGC only cell codes; the `H3_R` / `QDGC_L` label is MESA's. The same sort had quietly inverted
`choose_primary_geocode`, which returned the finest H3 grid as the fallback instead of the coarsest.

Shipped in `e0e2412`: names from `mesa_shared.h3_group_name()` / `qdgc_group_name()`, and
`migrate_geocode_group_names()` at startup, after restore and before generation. On a copy of the
live workspace it renamed 4 tables and 40 MBTiles in 0.50 s, with all 24 tables equal before and
after apart from the names. `e0e75a0` committed the two template lines mesa_demodata builds from
(`mesa_version = 5.7.0`, example `H3_R09`); `config.ini` is otherwise not committed.

mesa_demodata was sent a prompt (`docs/exchange/2026-09-16-geokodenavn-ledende-null-prompt.md`).
It first stopped at the template step — correctly, the template was not committed — and also
reported that no commit held padded names, which was wrong: its clone predated the push, and it
searched for the prompt's illustrative spelling `H3_R{r:02d}`. After a fetch it rebuilt all eight
packages as 5.7.0 (reply: `docs/exchange/2026-09-16-geokodenavn-ledende-null-svar.md`). Only
`config.ini` differs from 5.6.0 in each, +51 bytes zipped; zero unpadded names found. The template
also moved `segmv_ai_enabled` from 1 to 0 in the packages — 5.6.0 had shipped a value that differed
from the committed default. Zenodo versions are prepared, not published.

Two observations from the reply, not acted on: QDGC L8 took 18.6 s for 1,817 cells against 0.8 s
for 2,312 H3 R9 cells; and the working `config.ini` here still carries the "GENERATED FILE" banner
from a restored demo package, which `e0e75a0` did not pick up because it was built from HEAD.
mesa-server has not been told about the naming change.

## 2026-09-16 — mesa-server agrees to the two-digit grid names

mesa-server answered the same day (`docs/exchange/2026-09-16-geokodenavn-ledende-null-server-svar.md`)
and accepted the format, so it now stands in `cooperation.md` as agreed by all three projects.

The server writes grid names in one place, `worker/generate_geocodes.py`, and now calls functions
named and behaving like desktop's. It had the same fallback inversion: classification picked "the
first layer" from a PostgreSQL `string_agg(DISTINCT …)`, which is text order. Its fix reads the
level as a number rather than relying on padding, because the server deliberately does not rename
old names in packages it reads — it copies the geocode tables through, and renaming only its own
tables would leave a package disagreeing with itself. Desktop's restore migration covers what comes
back. The change is not deployed: a changed SSH host key on the server is being verified first.

Desktop was asked whether it ever reads a server package without restoring it. Checked: it does
not — outside restore, package zips are opened only to read `config.ini` and list member names.

One consequence for the 5.7.0 release: the server's package contract accepts pre-built output only
from 5.5.0 and 5.6.0, so a 5.7.0 package *with output* is refused until they have confirmed the
schema. Input-only packages are unaffected. Logged as open in `cooperation.md`.

The line numbers the server quoted for `mesa_shared.py` (466–473, 704–725) come from this machine's
working tree, which carries uncommitted changes; in `e0e75a0` the functions sit at 425, 430 and 663.
The code they compared against is the same.

## 2026-10-08 – 2026-10-09 — Button review: Exit placement, roles, safe dialogs, unsaved edits

Started from an operator remark: Exit should sit top-right in every window, and it did not in
Classification setup, Process all and a few others. A survey found six windows with Exit at the
bottom (Classification setup, Process all, Report generator, Publish to GeoNode, Tune processing,
the GeoParquet processing window), and the tan Exit style pasted inline about twelve times in two
drifting variants — which is how Classification setup ended up with neither the position nor the
colour. Fixed in `a0c92f7`: Exit top-right everywhere (tab-bar corner where there are tabs, a header
row otherwise) and one `role="exit"` style. The tan colour was kept on purpose: a shade darker than
ordinary buttons so it is easy to find, but not red, since leaving a window is not destructive.

A wider read-only review of buttons across all windows followed. Of its ranked findings the operator
chose four to fix first, committed one purpose each:

- `896f98b` — disabled buttons with a role looked enabled (stylesheet ordering; see learning.md
  "Button roles need their own :disabled rule"). Found while preparing the next fix; it was
  already wrong in Tune processing.
- `396af6a` — Enter confirmed seven destructive prompts (Clear output, both Restore paths, Restore
  previous tuning, Move workspace, both geocode deletes). No/Cancel is now the default.
- `824c874` — `danger` role on Restore data, Delete token, Delete current atlas and GeoNode's
  Delete selected (which had its own bright-red inline style); new confirmations for Delete token,
  Cancel in Process all (it discards the run) and Load from Excel (it persists at once).
- `891d0d5` — long jobs could be started twice: every Process click in the GeoParquet window
  spawned another worker; Geocodes and Atlas job buttons stayed live. Now locked while running and
  restored afterwards, also when a job fails (Atlas needed a new `job_done` signal because
  `task_finished` fires only on success). Checked by driving both windows with dummy jobs.
- `10f8cde` — one marked primary action per window; the duplicate primaries (Suggest/Generate,
  Create/Import) reduced to one.

Then the largest finding: no editor tracked unsaved edits (learning.md "Helper windows ask before
discarding edits"). Mapping it turned up two things worse than lost edits. Processing setup wrote the
whole `tbl_asset_group` back on every close from the copy loaded at open, so Assets edits made in
between were overwritten without a trace (learning.md "Never write a whole table back from a copy
loaded when the window opened"). And Tune processing's commit and restore called
`self._load_config_editor()`, a method that exists only on the main window, inside a bare
`try/except: pass` — the Config tab never refreshed, so a later "Save all" wrote the pre-tuning
values back over the tuning. The operator accepted the recommendation that Processing setup prompt
like the other windows rather than keep a (fixed) auto-save, so all editors behave the same.

The fix added `DirtyTracker` and `confirm_unsaved` to `ui_style.py` and wired them into Atlas,
Assets, Geocodes, Classification setup, Processing setup and the main window (Config, Welcome, AI
connection as one prompt on close). Tune now pushes only the keys it wrote into the Config tab,
keeping other unsaved edits there. Verified with scripted probes per window (Cancel keeps the edit
and blocks close or navigation, Discard reloads clean, the guards block a geocode import), a merge
probe on a copy of `tbl_asset_group`, and the test suite (68 passed).

Remaining from the review, not started: Delete placed next to Save/Next in the record editors,
inconsistent save wording, no keyboard shortcuts, fixed button widths. Nothing here touched the
sibling projects; nothing reported to mesa-server or mesa_demodata.

## 2026-10-10 – 2026-10-11 — 5.7.0 prepared; Special focus rebuilt around one editing model

**Release preparation (2026-10-10).** The backlog from August and September was committed as fourteen
thematic commits (`dea9169` … `077a5af`) and pushed. Version information was already 5.7.0 in
`config.ini` and the user-guide script; what was missing around it went in locally: the config
template without the dead `backfill_max_workers` and with `segmv_enabled = 0` (`d68582b`), draft
release notes (`4097db7`), the capture tool covering Classification setup (`1cb9642`). The wiki got
the rest of the 5.7 changelog, the interface page, four downscaled figures from the method
manuscript, and all interface screenshots retaken. Taking the screenshots surfaced three interface
texts that no longer matched the behaviour (`3c5209c`) and the operator's GeoNode server address in
the publish popup, which was replaced with example values in both GeoNode images before commit.

**Progress-map legend (2026-10-11).** Reported as "legend written under the map". It had no
z-index and Leaflet's tile pane covered it once tiles loaded (`a541cb8`). The first probe showed
no difference between old and new, because the probe's map had loaded no tiles and so nothing
covered the legend; only a simulated tile over the whole map separated the two.

**Special focus.** The operator reported that adding lines and areas was illogical and easy to
misunderstand. A walk through both editors found the interface saying one thing and doing another:
"Finish: Enter" with no Enter handler, a polygon drawn from the map icon silently dropped, "area"
used for both the group and the polygon, a Save button in Analysis that saved nothing new, default
segment values staged but shown blank. Those were fixed first (`d20178c`, `a6e5413`).

The structural cause was that the two tabs followed different logics — Lines staged edits and
wrote on Save, Analysis wrote on every operation with no way back. The plan agreed with the
operator (`devtools/docs/SPECIAL_FOCUS_REDESIGN_PLAN.md`, `0e312ec`) was to move Analysis to the
Lines model rather than the reverse, since the reverse would take Discard away, and to give both
tabs the same layout.

The operator asked that the change be checked against report generation and the statistics. That
trace mattered more than the layout. It found that drawing a polygon wrote
`tbl_analysis_polygons` but not `tbl_analysis_group`, so the Analysis stage reported itself
unavailable and the report listed no study areas; that result rows for deleted polygons and groups
were never removed and stayed in report totals and in the GeoNode export; that the report mixes
current polygons (map, total area) with figures from the last run and says nothing when they
differ; that the Lines stage skips a line with an empty segment value and writes three random
template lines into an empty table; and that the Status tab counted lines from an import snapshot
that edits never update.

Shipped: Analysis stages edits and Save writes both tables atomically and purges orphan results
(`4e0877c`); Lines gets a list instead of a dropdown, keeps every line visible, stages Import and
fills missing segment values on Save (`bbb14de`); the Special focus window asks about unsaved
edits on either tab for every way out, and the tab is called "Study areas" (`ba6c813`); the report
and Compare study areas say when a study area was edited after its analysis ran (`5530c7d`); the
Status tab counts `tbl_lines` (`39da1a0`).

Verified with twelve new unit tests (80 in all), both pages driven in QtWebEngine against stub
backends, and one end-to-end pass on a copy of the development project: two polygons drawn and
saved, the Analysis stage run in 5.3 s (167 + 7 result rows), the report helpers finding the study
area with four sensitivity classes and 1,587 km², a shape edit flagging it as edited since the
run, a delete removing its 31 result rows on Save, and a re-run clearing the flag. One defect was
caught by the page probe rather than by reading: the first load calls `applyState` directly, not
through the common call path, so the edited-since-run note was never set at start-up.

Not done: the real Special focus window was only started and left running for 25 seconds, not
clicked through, so the native close prompt has not been seen on screen. Left for later and noted
in the plan: `AnalysisStorage` reads `workingprojection_epsg` while config.ini has
`working_projection_epsg`, so it always resolves to 4326 — which is what the Analysis stage
assumes, so correcting only the key would break it.

Nothing here touched the sibling projects, and table and column names are unchanged. Nothing was
reported to mesa-server or mesa_demodata.

## 2026-10-11 — Report review: average area, wider maps, page flow

The user asked for four things on the Word report: average area in the "Geocode groups and object
counts" table, more margin around the maps and wider maps where the page allows, headings kept on
the page of their content, and a test of the result — with the Special focus changes (staged
edits, "Study areas") taken into account. A look through the report engine for other obvious
faults was part of the request.

Tested on a copy of the development project with two study areas added and analysed, one of them
edited afterwards. Each report was opened in a hidden Word instance to read the page of every
paragraph and picture, and exported to PDF for a page-by-page look. The baseline was 60 pages:
21 pictures ran past the right margin, 35 of 38 large maps were narrower than 12 cm, headings sat
alone at the foot of pages, legends and ribbons alone on the next, the "Other maps" and atlas
pages had no legend, the study-area maps were 2.5–4 cm wide, and areas read "4181.8031 km²".

Shipped, one commit per purpose: average and total area in the geocode table, computed on
EPSG:6933, and no more "nan" for a missing title (`031ce63`); area figures with digits that match
their size (`e2908e9`); `_page_map_view()` — 12 % margin and a view widened to the 16 × 12 cm
panel — for every page map (`7ed48f5`); the study-area map frame (`68af228`); line segments
stroked over a white halo (`2b4d4fa`); colour keys under the "Other maps" and atlas tile maps,
from the same class definitions as the tiles (`8c66d60`); page flow — A4 with a 16 cm column,
keep-with-next from heading to legend, page breaks as a paragraph property, table rows that do
not split, repeated figure titles dropped (`22a52ae`); "Study area analysis" headings and no
"No index statistics" message when that section is off (`d9c5f9d`).

The study-area map took two attempts. Taking the axes out of autoscale was a plausible fix and
changed nothing; tracing the bounding box of every artist showed the scale-bar label 4,000 px
below the map, because the frame was passed as (minx, miny, maxx, maxy) to a helper that takes
(west, east, south, north). The fault predates this session.

Result on a report generated from source with everything in: 50 pages, no heading separated from
its content, no figure separated from its introduction, no picture wider than the column, no
table over a page boundary; median width of the large maps 15.8 cm. Six new tests, 86 in all.
The edited-after-analysis note appears under the right study area.

Left as is: atlas tile maps are 10–11 cm wide because the tiles are portrait; they use 16.5 cm of
height. The first baseline run took 78 minutes, nearly all of it basemap tile downloads; with the
cache filled a run takes about five.

Nothing here touched the sibling projects, and no table or column changed. Nothing was reported
to mesa-server or mesa_demodata.

## 2026-10-11 (evening): report progress, type fingerprints, and a build that ships no data

Four smaller requests and one larger one, all the same day as the report layout review above.

Tiles in the Special focus editors stayed grey for seconds after a zoom. Nothing failed: OSM
answered 200, the cache held no broken files and the log had no errors. The local tile proxy
opened a new TLS connection for every tile, about 1.2 s each. With one kept-alive connection per
proxy thread a tile costs about 0.45 s and a screen of 30 uncached tiles 3.3 s instead of 6
(`4db74d0`). The tab bar of the same window was redrawn to match the Qt dialogs (`12719b6`).

The user asked whether a realistic progress indicator for the report was possible. It was: the
report is a countable set of maps, and the unpredictable part, basemap tile downloads, can be
counted in the cache before each map is assembled. `report_progress.py` prices the remaining maps
from measured drawing time and tile time (`5ae6546`). Four full runs on a 36-map test report:
the bar was 3.9 points from true progress on average with no earlier run, 1.5 with one, and 3.4
when 12 % of the tiles had to be downloaded (608 s instead of 267 s). The first version let one
quick map set the price for its kind and ignored work between maps; both were corrected after the
first run. Two runs opened with "about 16 min left" for a 4.5 and a 10 minute run, because 10 % of
the tiles were assumed missing although an earlier run had filled the cache; the assumption is
now 1 % when an earlier run is on record. That last change is unit-tested, not re-measured.

The Classification section's fingerprints were redrawn with percentages, axis labels and an
explanation, and the prose list of types became a table (`abb7d84`). The user then asked for no
em dashes anywhere in the report text, calling them a tell of machine writing; all 28 in
`report_generate.py` were rewritten (`64aa8b4`). Many remain in other windows and in the wiki.

The larger request: 5.7.0 is to be built without data, with the documentation pointing to the
demonstration data on Zenodo. The build already shipped `input/` and `output/` as a skeleton, but
took `config.ini`, the three `readme.txt` files and `qgis/` from the working folder, which at the
time held a restored demo project: a demo `config.ini` with Classification switched on, the
readme files deleted, a generated `qgis/mesa_results.qgz` and an extra `input/` folder. The build
now takes these from git at HEAD, leaves `docs/exchange/` out by the user's decision, and stops if
the distribution holds project data. The copy and the check were run into a temporary folder and
six deliberate faults were each caught; a full `build_all.py` run is the user's to make.

The documentation (wiki Home and Quickstart, the release notes, a new `input/asset/readme.txt`
and the user guides, which were regenerated) now names the two concept DOIs for the
demonstration data, `10.5281/zenodo.21744272` and `10.5281/zenodo.21744039`, taken from
`mesa_demodata`'s own `zenodo/5.7.0/manifest.json`. The user guides got the new section and
nothing else: their "What's new" chapter was not rewritten for 5.7.

Reported to mesa_demodata: an instruction prompt,
`docs/exchange/2026-10-11-demodata-5-7-utgivelse-prompt.md`, at the user's suggestion instead of
testing the packages from here. It points out that the demo `config.ini` is generated from an
older template (it still has `backfill_max_workers` and the old stage numbers), that the guide
still mentions Backfill and "Analysis", and asks for confirmation of the DOIs and a test of each
package in 5.7.0. Nothing was reported to mesa-server.

## 2026-10-11 (late): the user guide brought up to 5.7

The entry above says the user guides got the demonstration-data section and nothing else. On the
user's go-ahead the rest followed the same evening. Chapter 2 of `devtools/build_user_guide.py`
was still "What's new in MESA 5.2" and described Backfill as a stage of its own. It is rewritten
for 5.7 in English and Portuguese from the release notes, with the report changes of the day
added. Backfill is removed from the stage lists, the re-run table and the tuning notes in
chapters 3, 4 and 10; the three mentions left are the ones that say it is gone. All 183 em dashes
in the script were replaced (a colon after a label, commas where they came in pairs), and both
`.docx` files were regenerated: no em dashes, 38 pictures each. The other chapters were not read
through against 5.7. The Portuguese text is my translation and has not been reviewed by a
speaker.

## 2026-10-11 (night): the report's basemap download

The tile proxy entry above noted that the report engine had the same fault and had not been
changed. The user asked for it. `_plot_basemap` now reads the cached tiles first and fetches the
missing ones over two connections that stay open for the whole report; two is the limit OSM's
tile usage policy sets for one client. On single maps with an empty cache: 35 tiles in 7.9 s
instead of 39.3 s, 25 tiles in 5.6 s instead of 28.4 s, about 0.22 s a tile against 1.1 s.

A full 36-map report with 12 % of the cached tiles removed (583 tiles to download) took 333 s.
The comparable run earlier in the day, with 622 tiles to download, took 608 s; the same report
with everything cached takes about 270 to 295 s. So the download part went from roughly 320 s to
roughly 50 s. The run also served as the first full measurement of the progress model after its
starting assumption was changed: it opened with "about 6 min left" for a run of 5.5 minutes, the
time-left figure was off by 6.8 % of the run on average, and the bar by 2.7 points (6.8 at most).

Nothing was reported to the sibling projects.

## 2026-10-12: workflow windows opened behind the launcher

The first 5.7.0 build (427 s, verified clean of project data) was tried by the user, who found
that windows opened from the workflow buttons came up behind the launcher. The launcher never
asked for them to be in front: in-process Qt helpers are deliberately unparented, and the
packaged webview helpers start as detached processes that have lost the right to take the
foreground by the time their window exists.

Fixed in `mesa_shared.py` with two small functions, used by the launcher's in-process path,
`segmentation_setup`, and the four webview windows (Special focus, Maps, and the standalone Lines
and Study areas editors). The webview windows are created "always on top" and made ordinary
again 1.5 s after they appear. The first two ways of releasing the flag were unreliable and are
described in learning.md; the third held in eight consecutive starts of Special focus (in front
throughout, flag cleared) and in one start each of Maps and Special focus followed by minimise,
restore and maximise.

Those checks were made by starting the real helpers about 25 times on the user's desktop while
the user was working, which the user rightly objected to. The in-process Qt part was therefore
not checked on screen at all and is left for the user to try in the next build.

Also asked the same night: where the AI token is kept. It is `secrets/ai_connection.parquet` in
the project folder, which the build does not ship by design.

Nothing was reported to the sibling projects.

## 2026-10-12: the Process bar jumped up and down at the start of a run

Reported by the user from the first 5.7.0 build: at the very start of processing the bar and its
percentage went up and down. Found by reading the code, with no test run on screen. The bar has a
single source, the worker's progress callback, and each stage reports rising values inside its
own slice. The exception was the data stage's progress pulse, a thread that nudges the bar forward
every 0.8 s while the pipeline runs: it was stopped after `run_headless` returned, not in a
`finally`, so a cancelled or failed data stage left it running towards 93 % of its slice. The
window lives across runs, so the next run's 0, 5, 12 % alternated with the old pulse's higher
values until the pulse reached its cap. The other three pulses already stopped in a `finally`.

Fixed twice over: the pulse stops on every way out, and the window gives each run a token so that
progress from an earlier run's worker is dropped. The second part covers a force-stopped run,
whose worker can hang on a killed pool and never reach the `finally`. A test fails on the old
code and passes on the new.

Not confirmed with the user: whether a cancelled or failed run had preceded the one where they
saw it. If it happens on the first run in a freshly opened window, the cause is something else.

## 2026-10-12: the tile stage failed in a project folder inside OneDrive

While trying the first 5.7.0 build the user got an "Unhandled exception in script" dialog from
`tiles_create_raster`: `OSError: [Errno 22] Invalid argument` on
`...\OneDrive - Miljødirektoratet\Documents\MESA\output\mbtiles\__wkb_...`, raised in
`spill_wkb` when it opened its scratch file. log.txt shows two Process runs: the tile stage of the
first completed at 00:37 ("All done"), the second run at 00:39 has no tile lines and ends in
"[Process] FAILED". The folder held no leftover scratch file afterwards.

The scratch file is the WKB blob the tile workers map instead of each receiving a copy. It was
written beside the tiles, which for a packaged MESA with a redirected Documents folder is inside
OneDrive. It now goes to the system temp folder when the project is in a synced folder, carries
the process id in its name, and falls back to the temp folder if the first place refuses it.
The fault was not reproduced; the user suggested the open progress map could be involved, which
cannot be ruled out, and the fallback covers both.

The same session added helper functions to `mesa_shared.py` for the workspace changes that are
under way (recognising a synced folder, proposing a place for a new workspace, setting an old
one aside by renaming it); the tile fix uses the first of them.

## 2026-10-12: the program folder, the first start, and data from an earlier version

After trying the build the user pointed out that the Windows package is confusing: the unzipped
folder has `input`, `output` and `config.ini` beside `mesa.exe`, yet a packaged MESA has kept
its data elsewhere since 5.6 and copies those in on the first start without asking. There was a
notice afterwards, but the lookalike folders stayed, and on this machine the data had gone to
`OneDrive - Miljødirektoratet\Documents\MESA` because Windows redirects Documents there. The
macOS package hides the templates inside the application bundle.

Four measures, agreed with the user, plus one they added while the work was under way:

- The templates (`config.ini`, `docs`, `qgis`, `input`, `system_resources`) moved from beside
  `mesa.exe` into `_internal/`, where `resolve_path()` already looks. Beside the program there is
  now only the program and `READ ME FIRST.txt`, and the build's verification checks exactly that.
- On a first start MESA asks where the data folder should be before anything is written. It
  proposes `Documents\MESA`, or the home folder when Documents is inside a sync service.
- The data folder is shown at the bottom of the main window with an Open folder button and a
  note when it is synchronised.
- The read-me, the release notes and the wiki Quickstart say the same thing.
- Added by the user: when the data folder was last used with another MESA version, MESA asks
  whether to continue with it or start with an empty project. The user asked for a choice between
  deleting the old data and attaching to it; "start empty" renames the old folder instead of
  deleting it, so the choice cannot destroy anything. A marker file in the data folder records
  the version that last opened it.

Two things found on the way: `_has_user_data()` counted a `readme.txt` as user data, and the
tuning backup was written to `output/` beside the program, a folder that no longer exists there.

Checked without a visible window (the user had asked for that the same night): the helper
functions by unit tests, the two dialogs and the decision of when each is shown by driving them
offscreen against temporary folders, the package layout by running the build's copy and
verification steps into a temporary folder. Not checked: the real packaged start, which needs a
new build and a click through by the user.

## 2026-10-12: "Importance (max)" was blank, because tbl_flat had no numbers

From the second 5.7.0 build the user reported that the Importance (max) layer on the Overview tab
of the map window showed nothing. The tile log said why: `importance_max: tiles written 0 /
scheduled 17` for every geocode group. The tiles were empty because the column was: in the
project's `tbl_flat` all of `importance_min/max`, `sensitivity_min/max` and
`susceptibility_min/max` were null in 9,148 of 9,148 rows, while the class codes beside them were
filled. `tbl_stacked` and `tbl_asset_group` held the values.

Reproduced from source on a copy of the project. The flatten workers return correct values per
partition. The loss is in the parent, in the branch that runs when a geocode's rows are spread
over several partition files ("Resolving split geocodes"): the per-code table already has
`importance_max`, the table of (value, class) winners has it too, and the plain merge renamed the
two copies to `importance_max_x` and `importance_max_y`. The table was written without the
column, and the schema step filled it with nulls. The code has been there since 5.0; it shows
only for projects whose codes are split over partitions, which this small demo project's seven
partitions are and the large development project's are not.

Fixed with `_merge_code_extremes()`, which drops the aggregated copy before the merge. The same
flatten on the same copy then gave 9,148 of 9,148 values. A second fault was found beside it and
fixed: the sort key for a missing value was written into a nullable Int64 column, which raises
when a value is actually missing.

What it affected in a project that hit it: the Importance (max) map layer and its tiles, the
numeric min/max columns in the QGIS project and GeoNode export, and anything reading those
numbers from `tbl_flat`. The A to E sensitivity classes, the OWA index and the counts were not
affected. An affected project needs Flatten and Tiles run again with the corrected build.

The first attempt at the reproduction hung for ten minutes: the script called the pipeline
without an `if __name__ == "__main__"` guard, so each spawned worker started the pipeline again.
Nine processes were left and were stopped.

At the same time the four map tabs (Overview, Segmentation, Classifications, Assets) were given
the look of the Qt tabs, as in Special focus. Not seen on screen.

Reported to nobody yet. mesa-server mirrors the flatten step and should be told.

## 2026-10-12: the reply from mesa_demodata, and what it changed here

mesa_demodata answered the task of 2026-10-11
(`docs/exchange/2026-10-12-demodata-5-7-utgivelse-svar.md`). All nine packages were rebuilt with
a `config.ini` generated from the current template and run through MESA 5.7.0 from a copy of
`main` at `a809713`: restore, asset import, parameters, mosaic, Process and a report, without
errors. The two concept DOIs in our documentation are right. `uganda_eacop` is not published.

Three findings concerned MESA.

Our documentation was wrong, and theirs had been since 5.6.0: "restore a package, then run
Process" does not work. After a restore nothing is imported, so Process reports the asset and
geocode tables missing. The assets must be imported, the parameters loaded from
`input/settings.xlsx` and the mosaic built first. Corrected in the wiki (Home with the five steps
and a line per package, Quickstart), the release notes, `READ ME FIRST.txt`,
`input/asset/readme.txt` and the user guide, which was regenerated. The message MESA shows when a
restore finishes now lists the steps when the project has no results.

A report took six to nine minutes on the smallest packages, nearly all of it basemap tiles. They
measured at `a809713`, before the download was made about five times faster (`bf03015`), so the
figure does not describe the current build.

Six of thirteen maps in those reports showed lines the user does not have: Process draws three
example lines in a project without lines, and the report treated them as data. The user decided
that they stay out of the report. They are recognised by the description Process gives them, and
the line editor clears that description when a line is reshaped, so a line the user has made
their own counts again.

One correction to our task letter: it said their overrides switched Classification on. They do
not. The values were in the user's working copy of `config.ini`, saved there by the
Classification dialog. Classification is off in the packages, confirmed by the user.

Also written the same morning: `devtools/ZENODO_5.7.0.md`, the text for the application's Zenodo
record, with the corrections mesa_demodata pointed out in the 5.6.0 record (ORCID
`0000-0002-4184-5224`, nine projects, concept DOIs for the data), and the agreed order of
publication in `cooperation.md`: data first, then the application.

Not reported back to mesa_demodata yet: that the report now leaves example lines out, and that
the importance figures were empty in `volta_delta`-sized projects before `9c1e913`, which their
test at `a809713` would have had too without it showing in what they checked.

## 2026-10-12: the demonstration data for 5.7.0 is published

The user published the 5.7.0 versions of the two data records on Zenodo, and mesa_demodata
updated its reply with the result: open data `10.5281/zenodo.23309462`, synthetic test data
`10.5281/zenodo.23309556`. Both are marked as the latest version, so the concept DOIs that
MESA's documentation uses now lead to 5.7.0 data, and every file on Zenodo has the size and MD5
of the file in `exported/`. The first step of the agreed order is thereby done; the application
record is next. The reply in `docs/exchange/` is the updated one, the version DOIs are noted in
`cooperation.md` and in `devtools/ZENODO_5.7.0.md`, and no documentation text needed changing.

## 2026-10-12: "Import new data to start over", on a tab with no Import button

Trying the fourth 5.7.0 build, the user emptied a project with Delete all data and was told
"The project is empty. Import new data to start over." The tab it stands on has Store data and
Restore data, and no Import. The message also said "Kept: 0 system files" although `config.ini`
and `input/settings.xlsx` are kept; the count was of `.git` entries inside the emptied folders.

The user asked what the best approach was and accepted all four parts of the proposal. The two
words keep their meanings: restore is for packages, import is for one's own files, and "import"
is never used without saying where. The message now says how many items were removed, or that
the project was already empty, names the two files that are kept, and points to both ways on:
Restore data on the same tab, or Workflows, Assets. It has a button for each. Clear output got
the same treatment, and its texts now say that the tables made at import are among what it
removes, so the assets must be imported again before Process.

Found on the way: Delete all data also removed the `readme.txt` in each source folder, the note
that says what belongs there. They are kept now. This is very likely why three of those files
were missing from the user's working copy throughout this release work.

## 2026-10-12: ghost lines in the Lines editor

From the fifth 5.7.0 build the user reported that reshaped and deleted lines stayed on the map
as fainter "ghosts", also after Save, and that it was easy to believe a change was kept when it
had only been made in the editor.

The ghosts were second copies. The page starts loading when the backend is ready, and it learned
of that in two ways at once: the `pywebviewready` event and a polling timer. Inside Special focus
both fired, two loads ran at the same time, each cleared the map before waiting for the data,
and both then drew every line. One copy was tracked, the other was not, and the two lay exactly
on top of each other until one of them was changed. Driven without a screen against a stub
backend that becomes ready the same way: the old page had 6 layers for 3 lines, still 6 after a
reshape and 5 after a delete; the corrected page has 3, 3 and 2.

Three changes, any one of which would have been enough: loading starts once, only the newest
load may draw and it clears the map again right before drawing, and adding a line removes any
earlier layer for the same line. The Study areas editor had the same double start; it drew
correctly because it clears and draws in one step, and now starts once as well.

For the second complaint the staging model stays, as decided on 2026-10-11, but the page says
it plainly. Messages read "Shape changed, not saved yet. Click Save to keep it." instead of
"Geometry staged.", in the warning colour. While there are unsaved changes the Save button is
orange and reads "Save changes", and the badge beside it reads "Not saved yet".

## 2026-10-12: a console window flashed by during processing

Reported from the sixth 5.7.0 build: a command-line window appears and closes during processing.
MESA's helpers are built without a console, and MESA's own calls to PowerShell already ask for
no window. The likely source is a library: joblib, which scikit-learn uses, asks PowerShell for
the number of processor cores the first time it needs it, with a plain `subprocess.run`. In a
program without a console Windows opens one for that child. It runs in the Classification stage,
which is switched on in the user's project.

Not observed directly: no stage was run on screen to catch the window. The fix does not depend on
which program it was. `mesa_shared.hide_child_console_windows()` sets "no window" where a process
is created, for every child, unless the caller asks for a console of its own. It is applied when
`mesa_shared` is imported in a packaged build, which every MESA process does early, the helpers
and the worker processes included. From source nothing changes.
