# Sensitivity generalisation (multivariate segmentation)

> MESA v5+ feature. Additive and optional — it does **not** change the existing
> A–E sensitivity classification or any of its outputs. Helpers:
> `code/segmentation_setup.py` (config UI) and `code/segmentation_run.py` (compute).

## Why this exists — two complementary questions

MESA's sensitivity **classification** places each mosaic unit into one of five
classes **A–E** (Very low → Very high) by thresholding a single composite score
(importance × susceptibility, reduced by max/sum/OWA across the assets stacked under
that unit). It is a *univariate level of sensitivity*, scored one polygon at a time,
ignoring neighbours. It answers:

> **“How sensitive is this place?”**

**Sensitivity generalisation** is a different, complementary view. It works on the
full *stacked* per-asset profile of each polygon — not the flattened score — and
groups polygons into a configurable number of sensitivity **types** (composition /
character), optionally as spatially-contiguous regions. It answers:

> **“What kind of sensitivity pattern is this place part of?”**

Two areas can share class **C** yet be utterly different in *kind* — one dominated by
seagrass + sand, another by coral + mangrove. The classification collapses that; the
generalisation preserves it. Both views are valid; planners benefit from reading them
side by side. This follows the methods paper section *“Generalisation of sensitivity
patterns”* — a Frelat-style decomposition of the multivariate profile, with a
Blaschke-style OBIA analogy (objects-as-units rather than per-pixel scoring).

| | Classification (A–E) | Generalisation (types) |
|---|---|---|
| Operates on | flattened composite score | full stacked per-asset profile |
| Produces | levels (intensity) | types (composition / character) |
| Class count | fixed at 5 (A–E) | configurable (paper: 4–20+) |
| Neighbours | ignored | optional spatial contiguity |
| Question | how sensitive? | what kind of sensitivity? |

## How it works

1. **Read inputs** (per chosen geocode layer): `tbl_stacked` (read partition-by-
   partition with a pyarrow filter — never materialised whole), `tbl_geocode_object`
   (geometry + full polygon list), `tbl_asset_group` (asset-category labels).
2. **Build a feature vector per polygon** by aggregating its stacked rows:
   sum / mean / max / std of sensitivity, stack depth (row count), per-asset-group
   sensitivity sums (one column per asset group), and optionally a one-hot dominant
   asset group. Numeric features are standardised (`StandardScaler`). Polygons with
   no stacked rows become a zero vector, are flagged `no_data`, excluded from fitting,
   and assigned `cluster_id = NaN`.
3. **Cluster**:
   - **Attribute** (typology, geography-blind): KMeans (primary); HDBSCAN optional for
     an emergent-count comparison.
   - **Spatial** (contiguous regionalisation): SKATER (`spopt`) over a Queen-contiguity
     graph (`libpysal`). Above `segmv_skater_max_polys` polygons (or if `spopt` is
     unavailable) it falls back to KMeans + post-hoc contiguity enforcement (non-
     contiguous fragments are merged into their majority neighbour). The log states
     which path ran.
   - **Both** runs attribute and spatial.
   A class-count *list* (e.g. `4,8,16`) fits each value in one run.
4. **Profile + describe** each resulting type: mean/max/std sensitivity, top-3 asset
   groups by sensitivity contribution, polygon count, total area, and (optional) an
   AI-generated one-paragraph description.

## Parameters (`segmv_*` in config.ini / the setup UI)

| Key | UI control | Meaning | Default |
|---|---|---|---|
| `segmv_geocode_layer` | dropdown | which geocode layer to segment | basic_mosaic |
| `segmv_n_clusters` | text | class count — int or list `4,8,16` | 8 |
| `segmv_method` | radios | `attribute` / `spatial` / `both` | attribute |
| `segmv_pressure` | dropdown | single pressure, or aggregate across all | all (aggregate) |
| `segmv_features` | checkboxes | which aggregates form the feature vector | sum,mean,max,std,depth,group_sums |
| `segmv_min_area_m2` | spinbox | drop slivers below this area before clustering | 0 |
| `segmv_skater_max_polys` | (config) | above this count, spatial uses the fallback | 50000 |
| `segmv_ai_enabled` | checkbox | AI cluster descriptions (off by default) | 0 |
| `segmv_ollama_model` / `segmv_ollama_url` | (config) | local LLM for AI descriptions | mistral / localhost:11434 |

> **AI descriptions** call a local **Ollama** model first
> (`POST http://localhost:11434/api/generate`); if that is unavailable they fall back
> to a configured **OpenAI** key (`OPENAI_API_KEY` env or `secrets/openai.key`), reusing
> MESA's existing OpenAI integration. A network failure leaves the description blank and
> the run continues. Default **off**.

## Running it

- **From MESA**: Workflows tab → step 3 → **“Sensitivity generalisation”**. Configure,
  then **Save settings** (config only) or **Run now** (saves, then runs in a separate
  process).
- **From a terminal**:
  ```
  python code/segmentation_run.py --original_working_directory <project> \
         --layer basic_mosaic --method both --n-clusters 4,8,16
  ```
  `--run-id <id>` re-runs an existing run reproducibly (fixed seeds → identical output).

## Outputs

In `output/geoparquet/` (ZSTD-3, multiple runs co-exist by `run_id`):

- **`tbl_seg_mv.parquet`** — one row per (polygon, method, n_clusters, run_id): `code`,
  `name_gis_geocodegroup`, `cluster_id`, `cluster_label`, `no_data`, `sens_mean`,
  `method`, `n_clusters`, `run_id`. Join to `tbl_geocode_object` on `code` for geometry.
- **`tbl_seg_mv_profile.parquet`** — one row per (run_id, method, n_clusters,
  cluster_id): mean/max/std sensitivity, `top_asset_groups`, `n_polygons`,
  `total_area_km2`, `description_ai`.

In `output/segmentation_mv/<run_id>/`:

- Optional **`classification_results.gpkg`** (`segmv_export_gpkg`, off by default) —
  polygons + `cluster_id`, for direct use in QGIS. Nothing in MESA reads it back;
  the Maps window, the report and the shipped QGIS project all read the parquet
  tables above.
- **`summary.md`** — parameters, methods + path run, quality metrics (silhouette for
  attribute; a SKATER-style coherence objective for spatial), and the profile table.
- **`params.json`** — the exact parameter set, for reproducibility.
- Optional faceted **PNG** maps (`segmv_make_png`, off by default).

## Reading the output

- A **type** is a composition, not a rank. "type 3" is not "more sensitive" than
  "type 2" — read the profile (top asset groups + mean sensitivity) to see what it *is*.
- Use **attribute** types to find *like-with-like* regardless of location; use
  **spatial** types when you need contiguous planning units on the map.
- More classes = finer distinctions but smaller, less stable groups. Start coarse
  (4–8) for overview, go finer (16+) only when the planning purpose needs it.
- `no_data` polygons (no overlapping assets) are excluded from the types by design.

## Worked example — Uganda 36-asset dataset

On the Uganda dataset (36 asset groups), run on `basic_mosaic` with
`--method both --n-clusters 4,8,16`. The per-asset-group feature columns give the
clustering 36 composition dimensions on top of the scalar sensitivity aggregates, so
attribute types separate by *which* assets dominate (e.g. wetland-dominated vs
forest-dominated vs cultivation-dominated), while spatial types yield contiguous
regions suitable as planning zones. Compare the resulting `summary.md` profile table
against the A–E map for the same area: cells sharing a class often split across several
types, which is exactly the structure the generalisation is there to reveal.

## Report integration

Generate a report with `include_segmentation_mv=True` to add a **“Sensitivity
generalisation”** section that presents the types alongside the A–E classification for
the same area, explicitly framed as the *“what kind”* view vs the *“how sensitive”*
view, with a citation to the methods-paper section. The section is self-contained and
skips cleanly when no v2 run exists. The shipped overlap-signature segmentation section
is independent and unchanged.

## Relationship to the shipped `tbl_segmentation*` feature

MESA also ships a narrower, pipeline-integrated segmentation (`code/segmentation.py`,
the Segment stage, the Maps Segmentation tab, the report's "Segmentation" section)
using overlap **signatures** and KMeans/agglomerative zones, written to
`tbl_segmentation/` + `tbl_segmentation_profiles.parquet`. This generalisation is a
**separate, additive** capability in its own `tbl_seg_mv*` namespace; the two do not
interact and can both be used.
