# Special focus: one editing model for Lines and Analysis

Status: agreed with the operator 2026-10-11. Steps 1-6 landed the same day (`4e0877c`, `bbb14de`,
`ba6c813`, `5530c7d`, `39da1a0`); step 7 (wiki and screenshots) follows. Open: the native close
prompt has not been clicked through in the real window.
Scope: `code/line_manage.py`, `code/analysis_setup.py`, `code/special_focus.py`, plus the
readers that turn their tables into statistics and report content.

## Why

Operators found adding lines and study areas illogical. The first round of fixes
(`d20178c`, `a6e5413`) removed the outright traps — Enter not finishing a line, a polygon drawn
from the map icon being dropped, "area" meaning two things. What remains is structural: the two
tabs follow different logics.

| | Lines | Analysis |
|---|---|---|
| Persistence | staged in memory, written by "Save all" | every operation writes to disk at once |
| Undo | Discard | none |
| Selecting | dropdown; the other lines are hidden | tick boxes; everything stays visible |
| Toolbar | Home, Reload, Save all, Discard, Import | none |

Someone who has learned one tab does the wrong thing in the other.

## Decisions

1. **Both tabs stage edits in memory and write on Save**, with Discard. Analysis is the one that
   changes. Reasons: deletions and mis-drawn shapes become undoable, and it matches the
   unsaved-changes prompt the rest of MESA got in 5.7. The alternative — autosave in Lines — would
   take Discard away from operators who have it today.
2. **One selected item at a time**, Ctrl-click for several (Analysis; needed for deleting more
   than one). The form always has one clear target.
3. **Same three zones in both tabs**: toolbar on top (Home, Save, Discard, Import, the
   "Unsaved changes" badge), a list of items with New / Edit shape / Delete and the form for the
   selected item on the left, the map on the right. Exit exists only on the Special focus window.
4. **Everything stays visible on the map**; the selected item is highlighted. Lines no longer
   hides the other lines.
5. **Shape editing is finished from the side panel** (Apply / Cancel) in both tabs.

## What the tables feed, and what must hold

Traced 2026-10-11. Process and the report read these tables from disk, so only saved state counts.

| Table | Written by | Read by | Keyed on |
|---|---|---|---|
| `tbl_analysis_group` | Analysis | Analysis stage, report, Compare study areas | `id` |
| `tbl_analysis_polygons` | Analysis | Analysis stage, report, Compare study areas, GeoNode | `id`, `group_id` |
| `tbl_lines` | Lines, Lines stage (template / legacy normalise) | Lines stage, report, Status tab, GeoNode | `name_gis` |
| `tbl_lines_original` | Lines import only | Status tab line count | — |

Results derived from them (`tbl_analysis_flat`, `tbl_analysis_stacked`, `tbl_lines_buffered`,
`tbl_segments`, `tbl_segment_flat`) carry the ids and snapshots of names and titles. Every join,
filter and file name uses the ids; names and titles are display text. So renaming, and the new
default titles, break nothing.

Invariants the rebuild keeps:

- Table names, column sets (`GROUP_COLUMNS`, `POLYGON_COLUMNS`, lines `REQUIRED_COLUMNS`) and id
  formats are unchanged. `name_gis` stays unique and immutable; `group_default` stays.
- Polygons stay in EPSG:4326 with CRS metadata — the Analysis stage assumes it.
- Every polygon's `group_id` points at an existing group.
- A polygon title is never stored empty (default applied in Python, not only in the page).

Faults found in the trace that the rebuild fixes, because they produce wrong or missing statistics:

- **Group file not written.** Drawing a polygon wrote `tbl_analysis_polygons` only. With no
  `tbl_analysis_group` on disk the Analysis stage reports itself unavailable and the report offers
  no study areas. Save now always writes both files.
- **Two separate, non-atomic writes.** A failure between them leaves the tables disagreeing. Save
  writes each through a temp file and `os.replace`.
- **Results outlive what they describe.** A deleted polygon's area stays in the report totals; a
  deleted group's rows are never purged and are exported to GeoNode. Save removes result rows for
  ids that no longer exist.
- **No staleness signal.** The report takes total area from the current polygons and sensitivity
  from the last Analysis run; after an edit the two describe different shapes and nothing says so.
  A shared check compares each study area's polygons with the run that produced its results; the
  editor, the report and Compare study areas say when a study area was edited after its last run.
- **Lines with empty segment length or width are skipped by the Lines stage without a message.**
  Save fills the configured defaults and says so.
- **A saved `tbl_lines` with no rows makes the Lines stage write three random template lines.**
  Save warns before writing an empty table.
- **Status tab line count comes from `tbl_lines_original`**, an import snapshot never updated by
  edits. It should count `tbl_lines`.

Left alone, noted for later: `AnalysisStorage` reads the config key `workingprojection_epsg` while
`config.ini` has `working_projection_epsg`, so it always resolves to 4326. That happens to be what
the Analysis stage assumes; correcting the key without changing the stage would break it.

## Order of work

Each step is committed and checked on its own.

1. Tests for `AnalysisStorage` as it must behave after the change (staging, Save, Discard, both
   files written, atomic write, default title, result purge).
2. Analysis persistence: storage stages in memory; Save / Discard; result purge on Save.
3. Unsaved-changes prompt for both tabs in the Special focus window, on Exit and on the window's X.
4. Staleness check shared by the editor, the report and Compare study areas.
5. Analysis layout: toolbar, single-select list, Apply / Cancel for shape edits, basemap control on
   the map.
6. Lines layout: list instead of dropdown, all lines visible, Reload removed, Import staged, Save
   validation, Status tab count.
7. Wiki text and the two screenshots.

After step 3 nothing can be lost silently, even though the tabs still look different.

## Risks

- **Operators used to Analysis saving by itself** must now press Save. The prompt on close catches
  the ones who forget; it goes in the changelog.
- **Process started with unsaved edits open** does not see them. Both tabs say so in the status
  line while there are unsaved changes.
- **Both editors can be started on their own**, outside Special focus. Save, Discard and the exit
  prompt must work there too.

## Verification

- Unit tests for the storage layer (step 1).
- The add / edit / delete / save / discard flows driven in QtWebEngine against a stub backend, as
  for the first round of fixes.
- One end-to-end pass on the development project: edit a study area, Save, run the Analysis stage,
  generate the report, and compare the totals with the polygons.
