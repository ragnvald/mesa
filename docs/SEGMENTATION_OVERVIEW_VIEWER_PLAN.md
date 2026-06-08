# Plan — "Segmentations overview" standalone viewer

> **STATUS: SUPERSEDED / PARKED 2026-06-06.** A standalone window was implemented and
> then **parked** the same day by an explicit design decision (Option A): the spatial
> side of segmentation should be a **layer in the planned unified Asset + Results map
> app**, not a separate map window. So `code/segmentation_overview.py` and the
> "Segmentations overview" button were removed to avoid adding a 4th map surface.
>
> **What was kept** — the valuable renderer: `segmentation.build_overview_geojson()`
> (dissolve each level into one multipolygon per signature/cluster, simplify, colour,
> cache to `output/cache/segmentation_overview/<level>__<mode>.geojson`, rebuilt when the
> source partition is newer) plus `overview_modes()` / `overview_cache_path()`. This is
> the data layer the unified map app will use to draw the segmentation overlay.
>
> **Division of labour going forward:** *spatial* segmentation → unified map app (as a
> toggleable layer with a level selector + signatures/clusters sub-toggle); *analytical*
> segmentation (signature mosaic, area-sorted zone tables) → the Word report section
> (already shipped). The level-selector / legend / zones-table UX sketched below is the
> blueprint for that layer's controls inside the unified map app.
>
> The rest of this document is retained as the design reference for the segmentation
> **layer** in the unified map app.

> **How to use this file.** Written as a ready-to-run *prompt* for a future Claude Code
> (or developer) session. Paste the “Prompt” block at the repo root, or follow the
> phased plan. Grounded in the actual MESA architecture (file paths + symbols verified
> 2026-06-05). Depends on the Segment stage delivered in
> `docs/SEGMENTATION_INTEGRATION_PLAN.md` (it consumes `tbl_segmentation`).

The goal: a **separate viewer window** — exactly like the **Results map**, *not* an
integrated panel — that lets an analyst look at the segmentation across the different
geocode levels (basic_mosaic, H3_R4…R9). It opens when the operator clicks a new
**“Segmentations overview”** button in Workflows → Results.

---

## 0. The prompt (copy/paste)

```
Build a new standalone viewer "Segmentations overview" for MESA, modelled exactly on
the existing "Results map" (code/map_overview.py): a pywebview + Leaflet window launched
as a subprocess helper, NOT an integrated panel. Read docs/
SEGMENTATION_OVERVIEW_VIEWER_PLAN.md first. Then:

1. Create code/segmentation_overview.py mirroring code/map_overview.py: a run(base_dir)
   entry, MESA_BASE_DIR / --original_working_directory base-dir resolution, a pywebview
   window with a Leaflet map + a right-hand side panel, and a Python<->JS Api class.
2. The window shows ONE geocode level at a time with a level selector (basic_mosaic,
   H3_R4..R9) enumerated from output/geoparquet/tbl_segmentation/, and a mode toggle
   Signatures | Clusters (Clusters only offered when cluster_id is present).
3. Render the segmentation as vector polygons coloured by signature (MESA A–E ramp) or
   cluster id (qualitative palette), with a legend, a per-zone stats table (from
   tbl_segmentation_profiles), and click-a-cell popups (signature, cluster, sens_mean,
   n_assets).
4. CRITICAL performance rule: never GeoJSON the raw per-cell polygons — basic_mosaic is
   ~9M polygons and tbl_geocode_object is 1.2 GB. Dissolve to one multipolygon per
   category per (level, mode) first (promote devtools/dissolve_clusters.py logic into
   code/segmentation.py), cache the dissolved GeoJSON under
   output/cache/segmentation_overview/, and build it on demand on first open with a
   loading state. Read tbl_geocode_object one level at a time with a pyarrow filter.
5. Add a "Segmentations overview" button to Workflows → Results in mesa.py and a handler
   that subprocess-launches the new helper (like open_maps_overview).
6. Empty-state: if tbl_segmentation is absent, show a clear "Run the Segment stage (4b)
   first" message instead of an error.

Reuse map_overview's OSM basemap + attribution + tile cache and its colour-config
reading. Keep it self-contained; do not modify the processing pipeline.
```

---

## 1. Goal & scope

- A **dedicated viewer window** (its own page, like Results map), opened by a button —
  not embedded in the launcher.
- **Level switching:** view the segmentation of any one geocode level at a time.
- **Two view modes:** *Signatures* (deterministic A–E overlap typology) and *Clusters*
  (algorithmic zones), toggled in the panel; Clusters shown only when present.
- **Explainable:** legend + per-zone stats table + click-to-inspect popups.

### Non-goals (first iteration)
- Side-by-side comparison of two levels (single-level view first).
- Editing/re-segmenting from the viewer (read-only).
- 3D / time animation.

---

## 2. Architecture touch-points (verified 2026-06-05)

| Concern | File | Symbol / anchor | Line(s) |
|---|---|---|---|
| Results button group (add the button here) | `mesa.py` | `("Results map", open_maps_overview, …)` tuple in "Results (step 4)" | 3360–3369 |
| Handler to mirror | `mesa.py` | `def open_maps_overview()` → `_launch_helper_subprocess("map_overview")` | 793–795 |
| Subprocess launcher | `mesa.py` | `_launch_helper_subprocess(file_name, extra_args)` | 736 |
| In-process helper set (do NOT add here — keep it subprocess) | `mesa.py` | `INPROCESS_HELPERS` | 153 |
| Viewer template to copy | `code/map_overview.py` | pywebview + Leaflet; `run(base_dir)`, `_render_html()`, `class Api` | 2474 / 2463 / 1068 |
| Base-dir resolution pattern | `code/map_overview.py` | `MESA_BASE_DIR` env + `base_dir()` | 47–86 |
| A–E colour-config read | `code/map_overview.py` | `get_color_mapping(cfg)` (config `[A]..[E]` `category_colour`) | 148–160 |
| Level enumeration | `code/segmentation.py` | `list_geocode_layers(gpq_dir)` (or glob `tbl_segmentation/*.parquet`) | 52 |
| Segmentation data | `code/segmentation.py` | `tbl_segmentation/<layer>.parquet` + `tbl_segmentation_profiles.parquet` | 256 / 301 |
| Signature colours to reuse | `code/segmentation.py` | `_RAMP`, `_signature_colour()` | 333 / 339 |
| Dissolve logic to promote | `devtools/dissolve_clusters.py` | `dissolve(by=<category>)` + style copy | whole file |
| Cluster palette to reuse | `devtools/test_segmentation.py` | `_QUAL_PALETTE`, `_colour_for_cluster()` | ~1147 / 1161 |
| Geometry source | `output/geoparquet/tbl_geocode_object.parquet` | filter `name_gis_geocodegroup == layer`, join on `code` | — |

> **Rule of thumb:** `map_overview.py` is the structural template end-to-end (window,
> Leaflet, Api bridge, basemap, base-dir). Copy it, then swap the data layer for
> dissolved segmentation GeoJSON.

---

## 3. Window layout (UX)

```
┌───────────────────────────────────────────────┬───────────────────────┐
│                                                │  Segmentations        │
│                                                │  ───────────────────  │
│                                                │  Level:  [basic_mosaic▼]│
│            Leaflet map                         │  Mode:   (•)Signatures │
│   (dissolved segmentation polygons,            │          ( )Clusters   │
│    coloured by signature or cluster,           │                       │
│    OSM basemap underneath)                     │  Legend               │
│                                                │   ■ A+B+C+D+E  32%     │
│   click a zone → popup:                        │   ■ B+D+E      23%     │
│     signature / cluster / sens_mean / n_assets │   …                   │
│                                                │                       │
│                                                │  Zones (table)        │
│                                                │   zone | n | sens | …  │
└───────────────────────────────────────────────┴───────────────────────┘
```

- **Level selector** — dropdown of levels found in `tbl_segmentation/`. Default
  `basic_mosaic` if present, else the first.
- **Mode toggle** — Signatures / Clusters. Clusters disabled when no `cluster_id`.
- **Legend** — signatures with share % (from `tbl_segmentation_profiles`); clusters by id
  with size. Colours: A–E ramp for signatures, qualitative palette for clusters.
- **Zones table** — `tbl_segmentation_profiles` rows for the active level+mode (zone,
  **total_area_km2**, n_polygons, sens_mean, mean_n_assets), **sorted by total area
  descending** by default (matching the report), and click-sortable on any column.
- **Popup** — on cell click, show that zone’s attributes.

---

## 4. Data & rendering strategy (the crux)

**Do not** convert raw per-cell polygons to GeoJSON: `basic_mosaic` ≈ 9M polygons and
`tbl_geocode_object.parquet` is ~1.2 GB. Leaflet cannot draw millions of features.

**Dissolve-per-category, then cache:**

1. For the requested (level, mode), read `tbl_segmentation/<level>.parquet` (slim, no
   geometry) and `tbl_geocode_object.parquet` filtered to that level (pyarrow filter),
   join on `code`.
2. Pick the category column: `signature` (Signatures) or `cluster_id` (Clusters).
3. `gdf.dissolve(by=<category>)` → one multipolygon per category (≤31 signatures, ≤k
   clusters). Optionally `simplify(tolerance)` for very large levels.
4. Attach a `fill` colour per feature (`_signature_colour` / cluster palette) and the
   profile stats, and write a small **cached GeoJSON** to
   `output/cache/segmentation_overview/<level>__<mode>.geojson`.
5. The viewer loads that tiny GeoJSON — instant render.

**Caching & invalidation:** rebuild the cache only when missing or older than
`tbl_segmentation/<level>.parquet`. Build on demand on first open behind a loading
state (an `Api` method the JS calls), so opening the window is fast and the heavy
dissolve happens once per level+mode.

**Memory discipline:** the dissolve reads geometry for **one level at a time** with a
pyarrow filter (mirror `segmentation._read_layer_stacked` / `_agglomerative_queen`),
inside the viewer's own subprocess — never all levels at once, never in the launcher.

**Colour conventions (reuse):**
- Signatures: `_RAMP` (A `#b2182b` … E `#2166ac`); multi-code = weighted average biased
  to A — already implemented in `segmentation._signature_colour`.
- Clusters: qualitative palette (`_QUAL_PALETTE` from the devtool); `-1` noise / `-999`
  unassigned rendered grey.

---

## 5. Phased implementation

### Phase 1 — Render-cache builder (in `code/segmentation.py`)
- Promote `devtools/dissolve_clusters.py` logic into a reusable
  `build_overview_geojson(gpq_dir, layer, mode, out_path, simplify_tolerance=0.0)`:
  join slim segmentation + filtered geometry → dissolve by category → per-feature colour
  + props → write GeoJSON. Return the path (and a small legend/profile summary).
- Reuse `_signature_colour`; add a `_cluster_colour(cid)` palette helper.

### Phase 2 — Viewer (`code/segmentation_overview.py`, mirror `map_overview.py`)
- `run(base_dir)`, `MESA_BASE_DIR` / `--original_working_directory` resolution, pywebview
  window, Leaflet map + OSM basemap (reuse map_overview's basemap, attribution, tile
  cache), right-hand panel.
- `class Api` methods the JS calls:
  - `list_levels()` → levels from `tbl_segmentation/`.
  - `modes_for(level)` → `["signatures"]` or `["signatures","clusters"]`.
  - `get_overview(level, mode)` → ensures the cached GeoJSON exists (builds via Phase 1
    if stale/missing), returns `{geojson_url_or_inline, legend, profiles}`.
  - `feature_info(...)` → popup payload.
- Default to `basic_mosaic` + Signatures.

### Phase 3 — Launcher wiring (`mesa.py`)
- Add handler near `open_maps_overview` (≈793):
  `def open_segmentations_overview(): _launch_helper_subprocess("segmentation_overview")`.
- Add the button to the Results group (3360–3369):
  `("Segmentations overview", open_segmentations_overview, "View the segmentation typology and zones per geocode level.")`.
- Keep it a **subprocess** helper (do not add to `INPROCESS_HELPERS`), like `map_overview`.

### Phase 4 — Polish
- Loading spinner while a level+mode dissolve builds; cache makes subsequent opens instant.
- Legend + zones table wired to `tbl_segmentation_profiles`.
- “Open output folder” / export current view (PNG) to match Results-map affordances.

### Phase 5 — Empty-state & validation
- If `tbl_segmentation/` is absent/empty: panel shows “No segmentation found — run the
  **Segment** stage (4b) in the processing runner first,” with no map error.
- Validate on a small level (H3_R6) first, then `basic_mosaic` via the dissolve cache.

---

## 6. Performance & memory guardrails (do not skip)

- **Never** load all levels at once; one level per render, pyarrow-filtered.
- **Never** emit raw per-cell GeoJSON for large levels — dissolve first (a handful of
  features per level/mode).
- Cache dissolved GeoJSON; rebuild only when the source partition is newer.
- The viewer is a **separate subprocess** (like Results map), so its memory never touches
  the launcher or the pipeline.

---

## 7. Acceptance criteria

- [ ] A **“Segmentations overview”** button in Workflows → Results opens a standalone
      window (separate process, like Results map).
- [ ] The window has a **level selector** (basic_mosaic, H3_R4…R9) and a **Signatures /
      Clusters** toggle.
- [ ] Each level renders **instantly** from a dissolved, cached GeoJSON — including
      basic_mosaic (≈9M source cells) — with the MESA A–E ramp (signatures) or a cluster
      palette.
- [ ] Legend + per-zone table (from `tbl_segmentation_profiles`) + click-cell popups work.
- [ ] Clean **empty-state** when no `tbl_segmentation` exists.
- [ ] No change to the processing pipeline; viewer memory stays out of the launcher.

---

## 8. Open questions / follow-ups

- **MBTiles vs vector:** dissolved vectors are recommended (categories are few). If a
  future need arises for smooth multi-zoom category rasters, generate per-level
  segmentation MBTiles in the Segment stage instead — bigger change, deferred.
- **Pre-build at stage time:** optionally have the Segment stage pre-write the dissolved
  overview cache so the first viewer open is instant even on basic_mosaic.
- **Determinants in the panel:** for Clusters, surface the top z-scored determinants
  (the devtool already computes these) so each zone gets a one-line “why”.

_Plan authored 2026-06-05. Anchors verified against the working tree (MESA 5.1, branch
`main`). If line numbers drift, match on the named symbols in §2._
