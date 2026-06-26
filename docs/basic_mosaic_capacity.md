# basic_mosaic — computation capacity baseline & tuning record

Purpose: a versioned record of what basic_mosaic cost *before* the 2026-06-26
tuning, what we changed, and how project size maps to machine size. Use this as
the historical starting point in release/version documentation.

## Reference run (historical starting point)

- **Date:** 2026-06-26
- **Machine:** Windows 11, 8 physical / 16 logical cores, 127 GB RAM, Python 3.14.6
- **Input:** `tbl_asset_object` = **3,526,097** asset polygons (635 MB).
  Mean ~16.8 vertices/polygon (p50 = 5, p90 = 65, max = 2,609); ~59 M vertices
  total; small features (~160 m bbox diagonal) over a ~300 × 540 km extent.
- **Output:** **9,013,622** atomic faces. Area sanity check: faces_area −
  coverage_area = **0 m²** (exact).
- **Total wall-clock:** 06:13:16 → 16:09:36 = **9 h 56 m**.

### Phase breakdown

| Phase | Time | Share | Nature |
|---|---|---|---|
| Load + project to metric CRS | 9.6 s | — | |
| Parallel boundary extraction (12 workers, 14,105 chunks of 250) | ~6 min | ~1 % | scales well |
| **Edge + coverage union tree-reduction** | **~8 h 39 m** | **~87 %** | the bottleneck |
| polygonize (GEOS Polygonizer, single thread) | ~1 h | ~10 % | **21.7 GB peak RSS** |
| Per-face coverage filter | seconds | — | cheap (prepared geometry) |
| Assemble + sanity + publish | ~10 min | ~2 % | |

### Root causes of the 87 %

- **Spawn-bound, not compute-bound.** The pairwise reduction ran ~2.2 merges/s on
  4 workers ≈ **~1.8 s/merge** — but a GEOS union of two small (chunk-level)
  geometries takes milliseconds. The cost was process **spawn + re-import**:
  `maxtasksperchild = 1` respawned a worker for every pair, and Windows/macOS
  (and this code on Linux) use the `spawn` start method, which re-imports
  geopandas/shapely each time. Round 1 alone had 7,053 merges.
- **Serial tail.** The last rounds are 1–2 huge unions that cannot parallelise.
- **Done twice.** Edges and coverage each reduce 14,105 partials independently.
- **polygonize is global + single-threaded** and holds the whole edge network +
  all 9 M faces in memory at once → the 21.7 GB peak.

## Adaptations applied (2026-06-26)

1. **Spawn fix (default on, output-identical).** `_tree_reduce_unions` now scales
   `maxtasksperchild` with the round's merge count: high in early rounds (many
   small geometries → amortise the spawn cost), `1` in late rounds (few large
   geometries → bounded RSS). Verified serial == parallel on synthetic geometry.
2. **Snap-rounding (opt-in, default off).** `mosaic_union_grid_size` (metres) snaps
   unions to a precision grid → fewer vertices, faster GEOS, fewer robustness
   retries, lower memory. `0` = exact, as before.
3. **Pre-flight memory gate (default on).** Estimates peak RAM from the asset count
   (`mosaic_preflight_gb_per_million_assets`, default 7.0) and, when it exceeds
   `available × mosaic_preflight_safety_frac` (0.8), **skips basic_mosaic with a
   clear message instead of OOM-crashing** mid-run. Scales with the host;
   override with `mosaic_preflight_allow_oversized`.
4. **Dead code removed (−158 lines).** The never-called tiled-mosaic remnants
   (`_mosaic_tile_worker`, `_plan_tiles_quadtree`, `_split_tile`) and their
   vestigial `mosaic_tile_*` / `mosaic_quadtree_*` / clip / dedup config keys.

## Expected effect

- The ~87 % reduction was spawn-dominated; removing the per-pair respawn targets
  exactly that. Expectation: the reduction drops from hours toward minutes on the
  same hardware. **Actual post-change wall-clock is pending a fresh full run — to
  be filled in here once measured.**
- Memory: a 16 GB machine no longer hard-crashes on oversized projects; it skips
  basic_mosaic gracefully (H3/QDGC grids are unaffected). High-RAM machines are
  never blocked.
- The polygonize 21.7 GB peak is **unchanged in cost** — only guarded by the gate.
  Truly bounding it needs a tiled overlay + membership-signature dissolve (Tier 2).

## Capacity guidance (project size vs machine)

The pre-flight gate uses ~7 GB of peak RAM per 1,000,000 assets and 80 % of
*available* memory as the budget. Approximate ceilings (available ≈ total − OS/app
overhead):

| Machine RAM | ~Available | basic_mosaic ceiling (assets) |
|---|---|---|
| 16 GB | ~12 GB | ~1.3–1.4 M (bigger → graceful skip) |
| 32 GB | ~26 GB | ~3.0 M |
| 64 GB | ~55 GB | ~6 M |
| 128 GB | ~100 GB | ~11 M (the 3.5 M reference run uses ~25 GB est., ~22 GB measured) |

Guidance for operators: 16 GB hosts can run small/medium projects but should not
attempt the largest AOIs; use H3/QDGC grids there, or run the mosaic on a larger
machine. The numbers are heuristic (asset-count based); a pathological
high-vertex dataset can exceed them, which is why the override exists.

See also: `learning.md` "Mosaic union reduction is spawn-bound, not compute-bound".
