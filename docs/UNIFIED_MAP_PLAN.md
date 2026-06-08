# Unified map (Overview + Segmentation) — build plan & status

One pywebview window, **two tabs** (Overview = Results, and Segmentation), each its own
Leaflet map, with a **Link zoom & pan** toggle in the header (right of the tabs, left of
an Exit button) that keeps the two maps' view in lockstep.

**The Asset map stays a separate window** (scope decision 2026-06-06, memory-driven): the
asset layers are heavy (638 MB asset table; the test dataset is very large), so loading
them alongside Results risks a crash. Keeping Asset standalone keeps it available without
endangering the Results/Segmentation view. This unified window replaces the **Results
map** once it reaches parity; the **Asset map** remains on its own.

File: `code/combined_map.py` · Launched by: `mesa.py open_combined_map` →
`_launch_helper_subprocess("combined_map")` · Button: Workflows → Results →
**"Maps (unified, beta)"**.

## Architecture decision
A *fresh* page hosting three independent Leaflet maps — **not** iframes of the old
viewers. Reason: the linked zoom/pan only works cleanly when all three maps live in one
page/process (cross-iframe Leaflet sync is fragile). The old viewers are 1.3–2.6k-line
monoliths with import-time side effects (asset_map_view reads the 638 MB asset table at
import) and hardcoded single-`#map` assumptions, so we reuse *data*, not their HTML, and
load each tab's data lazily. Cross-platform: no Windows-only `gui="edgechromium"`.

## Increments
- [x] **1 — Frame + Segmentation tab (DONE 2026-06-06).** Two tabs (Overview,
      Segmentation), link zoom/pan toggle (syncs view live while on, and on tab-switch),
      Exit button, OSM-basemap maps. Segmentation tab is **fully live**: level selector
      (with per-level cell counts), Signatures/Clusters toggle, legend, area-sorted zones
      table, click popups — fed by `segmentation.build_overview_geojson` (cached).
      **Big-dataset warning:** selecting a not-yet-cached level above ~1M cells (e.g.
      basic_mosaic at ~9M) prompts a confirm before the heavy dissolve. The Overview tab
      shows the basemap with a "wiring next" note (link-zoom still syncs its view).
- [x] **2 — Overview (Results) tab (DONE 2026-06-07).** The index layers are MBTiles. Either (a) start a
      loopback tile server like `map_overview.start_mbtiles_server()` and serve the
      combined HTML from that same origin (needed so WebView2/Windows doesn't block
      loopback tile requests from an opaque `html=` origin), or (b) use the GeoParquet
      vector fallback for a first cut. — DONE via the same loopback tile server: the Overview
      tab now has a geocode-group selector + a one-active-layer radio (Sensitivity,
      Importance (max), the three indices, # asset groups/objects), reading raster tiles
      from `output/mbtiles/<group>_<kind>.mbtiles`. Both tabs share the server. Detailed
      per-kind legends are a later polish.
- [ ] **3 — Parity & retire Results map.** Once the Overview tab matches map_overview,
      replace the "Results map" button with this one and retire/​thin `map_overview.py`.
      The Asset map button + module stay as-is.

## Segmentation MBTiles (DONE 2026-06-06, Phase A)
`tiles_create_raster.py` now renders two categorical layers per group **automatically
when `tbl_segmentation/<group>.parquet` exists** — `<group>_seg_signatures.mbtiles`
(A–E signature ramp) and `<group>_seg_clusters.mbtiles` (cluster palette, only if
clusters present). It joins `tbl_flat` geometry to the segmentation category on `code`
and reuses the existing multiprocessing tile engine, so basic_mosaic renders as raster
tiles instead of an OOM-prone vector dissolve. **Re-run the Tiles stage** to produce them.

**Phase B (DONE 2026-06-06):** `combined_map.py` now runs a loopback HTTP server
(serves the UI at `/` and seg tiles at `/tiles/<name>/{z}/{x}/{y}.png` from
`output/mbtiles/`, TMS-flip + blank-on-miss) and the window loads from that origin.
The **Segmentation tab consumes the raster `_seg_*` MBTiles** when present (level
selector + Signatures/Clusters toggle), with legend + zones from
`tbl_segmentation_profiles`; it falls back to the guarded vector view for small levels
that have no tiles. So after **Segment → Tiles**, basic_mosaic shows as raster — no
9M-cell vector, no OOM.

## Asset map (kept separate — memory)
Not merged. Mitigations that already exist / apply in `asset_map_view.py`: it keeps one
layer active by default and renders with `preferCanvas`. If we ever revisit merging it,
do so only behind a dataset-size warning and a strict one-active-layer rule.

## Notes / open questions
- The Results MBTiles origin issue (a) is the main wrinkle; resolve before increment 2.
- Segmentation tab has its own right-hand panel; the Overview tab will need its own layer
  controls — decide shared vs per-tab.
- `build_overview_geojson` first build for basic_mosaic (~9M cells) is heavy but cached
  (now gated by the confirm warning); optional pre-build at Segment-stage time remains a
  follow-up.
