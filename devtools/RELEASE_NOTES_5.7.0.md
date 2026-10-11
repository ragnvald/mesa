A release about scale and safety. Projects take far less disk and memory, map tiles render faster, and a data-loss bug that affected every earlier MESA 5 release is fixed. Around that the interface was gone through window by window: MESA now asks before unsaved edits are lost, destructive actions are harder to trigger by accident, and the QGIS project, GeoNode publishing and the asset tools gained functionality.

> Draft, written 2026-10-10 from the commits between `5.6.0` and `main`. The Zenodo record, file names and demonstration-data links are placeholders until the build is published.

### Where can I find it?

- Zenodo record — *to be added* (`MESA_570_win11_<date>.zip`, `MESA_570_macos_<date>.dmg`)
- [MESA project wiki](https://github.com/ragnvald/mesa/wiki), with the full [changelog](https://github.com/ragnvald/mesa/wiki/Mesa-changelog-overview)

### Demonstration data

The download contains no project data: `input/` holds four empty folders and `output/` is empty. Demonstration packages are published as their own records. Download one, load it under **Manage data → Restore data**, and run **Process**; the packages hold input data only.

- [Open data from sites in Uganda, Kenya and Ghana](https://doi.org/10.5281/zenodo.21744272), built from ESA WorldCover, OpenStreetMap and the Allen Coral Atlas
- [Synthetic test data](https://doi.org/10.5281/zenodo.21744039), with invented and deliberately overlapping asset layers

Both links always lead to the newest version of the data. The 5.7.0 versions of the two records are to be confirmed before publication.

## Read this first

### Generating H3 or QDGC could delete your mosaic

If you added an H3 or QDGC level to a project with a large `basic_mosaic`, check that the mosaic is still there. In every MESA 5 release up to and including 5.6 it could be deleted without any message, and processing afterwards silently excluded it. Open **Geocodes** and look at the object count next to `basic_mosaic`: a group listed with **0 objects** was emptied by this bug, and results computed while it was empty should be re-processed once it is restored.

A geocode table that cannot be read now stops the operation instead of being treated as absent, and adding grid levels streams through the table rather than loading it — the merge that needed 23 GB peaks at 0.5 GB and takes 15 seconds on a 19.4-million-cell table.

## What's new

### Projects are much smaller on disk, and the Backfill step is gone

The analysis table stored a full copy of a cell's outline on every row of that cell — 72–83% of the table on real projects. Outlines are now stored once and looked up when needed. The same project processed by both versions gives identical rows and values in a table **92.8% smaller**. Processing time is unchanged (495 s against 505 s); what you gain is disk space and the memory headroom that lets very large projects finish.

- **Backfill is removed.** It copied each cell's area into every row of the table. The stage list is numbered 1–9 without it.
- **Study areas work on large projects.** The analysis reads only the cells a study-area polygon touches instead of loading the whole table.
- Existing projects keep working; re-running Process is what shrinks them.

### Faster map tiles, and overview tiles show the highest value

On a real project the tile stage went from 63.3 s to 37.3 s, with the drawing core 3.1× faster. Cell outlines are shared between worker processes instead of copied into each, which removes the memory ceiling that stopped very large mosaics from rendering.

One visible change: where a zoomed-out pixel covers many cells, **the highest value now wins**. Before, whichever cell was drawn last won, which could differ between two runs of the same data. Tiles record the rule used.

### Classification is opt-in

Classification is the most expensive stage MESA has and used to join every Process run once a layer was configured. **Configure → Classification** now opens with a tick box, off by default: *Run Classification as part of Process*. With it off the stage is skipped in both Normal and Advanced mode; running the classification tool directly is unaffected. If you use Classification regularly, tick it once.

The classification popup on the map now also names the asset groups a type is made of.

### The QGIS project is its own processing stage

`qgis/mesa_results.qgz` was only produced as a by-product of Tiles. It is now **stage 7, QGIS project**, needing only the flattened results, so it can run without Tiles. The project also carries two interactive 3D landscapes per geocode layer — asset overlap and sensitivity — with column heights proportioned to the cell.

### MESA asks before your edits are lost

No window tracked unsaved changes. Every window with editable fields now asks **Save / Discard / Cancel** before edits would be lost — on close, on Reload, on Previous/Next in the Atlas editor, and before a job that rewrites what is being edited. Cancel is the default.

Two related bugs are fixed:

- **Parameters could overwrite edits made in Assets.** It saved the whole asset-group table on every close from the copy loaded when it opened. It now saves only importance and susceptibility, into the table as it is on disk, and no longer saves by itself on close.
- **Tune processing could be undone by "Save all".** The Config tab kept showing pre-tuning values after a commit; it now shows the tuned values at once.

### Lines and study areas are edited the same way

The two tabs of the Special focus window followed different logics: Lines kept changes until you saved, Analysis wrote every change at once with no way back. Both now have **Save** and **Discard**, the same layout (toolbar, list, form, map), and the same way of adding an item — click the points, finish with Enter or a double-click. The second tab is called **Study areas**. If you are used to that tab saving by itself: it no longer does.

Fixed at the same time, because they affected figures: a study area with only a drawn polygon could be invisible to Process and the report; results for deleted polygons stayed in the totals; and nothing said when a study area had been edited after its analysis ran — the tab, the report and Compare study areas now do.

### The program folder and the data folder (Windows)

The folder you unzip now holds only the program: `mesa.exe`, its tools and a `READ ME FIRST.txt`. The `input`, `output` and `config.ini` that used to lie beside `mesa.exe` are gone from there, because nothing put in them was ever read.

- **First start:** MESA asks where to keep your data before it writes anything. It proposes `Documents\MESA`, or a folder outside OneDrive when your Documents are synchronised, and you can choose another.
- **Data from an earlier version:** if the data folder was last used with another MESA version, MESA asks whether to continue with that data or start with an empty project. Starting empty renames the old folder and keeps it. Nothing is deleted.
- **Always in view:** the data folder is shown at the bottom of the main window, with a button that opens it and a note when it is inside a synchronised folder.
- **Windows open in front:** windows opened from the workflow buttons no longer come up behind the main window.

### Buttons and confirmations

- Exit is top-right in every window.
- Destructive buttons are marked as such, and Enter no longer confirms a deletion — the safe answer is the default in the confirmations for Clear output, Restore, Move workspace and deleting geocode groups.
- Deleting the AI token, **Cancel** in the Process runner and **Load from Excel** in Parameters now ask first.
- A running job cannot be started twice; its buttons are locked until it finishes.
- Each window has one highlighted main action, and disabled buttons look disabled.

### Publish to GeoNode

- One map per geocode group with Sensitivity, OWA index, Importance, number of asset groups and number of asset objects as switchable layers, styled as in MESA.
- Assets published in the MESA map's colours; segmentation levels can be published.
- **Dashboards**: a map, key figures and charts per geocode group and segmentation level.
- Editable titles for datasets, maps and dashboards.
- **Manage server** tab to list and delete datasets, maps and dashboards on the server.

### Assets and project data

- **Upload assets** copies `.gpkg`, `.shp` or `.parquet` files from anywhere on disk into the asset folder and offers to import them.
- **Delete asset** on the Edit tab removes one asset group from the project; the source file is left in place.
- **Delete all data** on the Manage data tab empties results and source folders; `config.ini` and `input/settings.xlsx` are kept.

### Grid levels are named with two digits

`H3_R07` instead of `H3_R7`, `QDGC_L06` instead of `QDGC_L6`, so lists sort in level order. Existing projects and restored packages are renamed automatically; only the name changes. Update grid names you refer to outside MESA.

## Fixes

- **Reports:** one report run at a time, one report window; pressing Create report while one is running says so and when it started. Two simultaneous runs could produce a Word file that would not open.
- **Maps:** "Link zoom & pan" holds the view across tabs again, and switching layer on the Overview tab no longer resets the zoom.
- **Progress map legend** is visible again; it was drawn under the map tiles.
- **Status tab** counts the lines in use, not the ones from the last import.
- **Progress map** shows the areas actually computed: light green when intersected, dark green when their results are written.
- **Geocode tables without a GeoParquet header** (written by early 5.7 development builds) are read correctly; they could stop processing or hide the Classification rasters.
- **Assets editor:** saving no longer strips spatial metadata from the asset-group table, which broke atlas generation.
- **macOS:** the Tiles and Classification stages run in the packaged app.

## Upgrading

- Projects from earlier versions open as before. Grid names are migrated on first start.
- Re-run **Process** to get the smaller analysis table.
- Scripts that pass `--no-backfill` to the processing runner must drop it; the flag no longer exists. `backfill_max_workers` in `config.ini` is ignored.
- If you relied on Classification running automatically, tick it under Configure → Classification.

**Full Changelog**: https://github.com/ragnvald/mesa/compare/5.6.0...5.7.0
