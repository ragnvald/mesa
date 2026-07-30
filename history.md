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
