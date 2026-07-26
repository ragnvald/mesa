# Classification (segmv) — specification for mesa-server

How MESA desktop computes **Classification**, precisely enough to reimplement, plus
how to run it against a PostGIS backend. Written for the mesa-server team.

Reference implementation: [`code/segmentation_run.py`](code/segmentation_run.py). Where this
document and that file disagree, the file wins — cite line numbers when reporting a mismatch.

Document language is English to match the rest of the repo and the column names; the content
is a spec, not user-facing copy.

---

## 1. What Classification is — and is not

Classification answers **"what kind of place is this?"** It groups geocode cells by the
*shape* of their sensitivity composition, not by how high their sensitivity is.

The reason it exists: MESA's `sensitivity` is a many-to-one product of
`importance × susceptibility`. A cell holding an (importance 5, susceptibility 1) asset and
one holding (importance 1, susceptibility 5) carry the **same** sensitivity code while
describing very different places. Classification clusters on the joint
(importance, susceptibility) histogram, so those two cells can land in different types.

Three things in MESA share the word "segment" and must not be conflated:

| Concept | Table | What it is |
| --- | --- | --- |
| **Classification** (segmv) | `tbl_seg_mv`, `tbl_seg_mv_profile` | GMM clustering on the (imp, sus) histogram. **This document.** |
| Segment / signatures | `tbl_segmentation/<layer>.parquet`, `tbl_segmentation_profiles` | Deterministic per-cell signature string. Used here only as a validation reference. |
| Line segments | `tbl_segments`, `tbl_segment_flat` | Geometry chopped along transect lines. Unrelated. |

---

## 2. Inputs

Per run, exactly **one** geocode layer is classified (`params.layer`, e.g. `basic_mosaic`,
`H3_R8`). Classify several layers by looping and reusing one `run_id` — see §6.3.

| Source | Columns used | Role |
| --- | --- | --- |
| `tbl_stacked` | `code`, `name_gis_geocodegroup`, `importance`, `susceptibility`, `ref_asset_group` | One row per (cell × overlapping asset). The only quantitative input. |
| `tbl_geocode_object` | `code`, `name_gis_geocodegroup`, `geometry` | Cell universe (so empty cells surface as `no_data`) and area. |
| `tbl_asset_group` | `id`, `title_fromuser`, `name_gis_assetgroup`, `importance`, `susceptibility` | Human labels for the top-3 listing; fallback valuations (§7.1). |
| `config.ini` `[VALID_VALUES] valid_input` | — | The valuation scale `S`. Falls back to `1,2,3,4,5`. |
| `config.ini` `segmv_*` | — | Run parameters (§3). |

A cell with **no** stacked rows is `no_data`: it is carried into the output with a null
`cluster_id` and label `no_data`, and is excluded from the fit.

---

## 3. Parameters

From `config.ini [DEFAULT]`, all `segmv_`-prefixed. Serialised verbatim into `params.json`
so a `run_id` round-trips.

| Key | Default | Meaning |
| --- | --- | --- |
| `segmv_geocode_layer` | — | Layer to classify. Blank → `basic_mosaic` if present, else first available. |
| `segmv_k_range` | `2-15` | Candidate cluster counts. Parsed from `min-max`, `min:max`, `min..max` or a single int. Swapped if reversed, floored at 2. |
| `segmv_transform` | `hellinger` | `hellinger` or `clr`. Anything else falls back to `hellinger`. |
| `segmv_coverage_weight` | `1.0` | Weight of the standardised stack-depth feature. `0` disables it (shape only). |
| `segmv_pressure` | *(empty)* | Optional pressure filter; empty/`all` = aggregate. The canonical MESA stack has no pressure column, so this is normally inert. |
| `segmv_min_area_m2` | `0` | Drop cells smaller than this before clustering. `0` keeps all. |
| `segmv_ai_enabled` | `0` | Optional plain-language per-type description via Ollama/OpenAI. Cosmetic; never affects clustering. |
| *(not exposed)* `seed` | `42` | GMM `random_state`. |

---

## 4. The algorithm

### 4.1 Valuation scale and bin layout

`S = sorted(set(valid_input))`, integers in `[0, 9999]`, default `[1,2,3,4,5]`. Both axes use
the same scale. The histogram has `|S|²` bins (25 by default), named and ordered
**importance-major**:

```
h_i1_s1, h_i1_s2, … h_i1_s5, h_i2_s1, … h_i5_s5
bin_index = imp_index * |S| + sus_index
```

The column order is part of the contract — the fingerprint vector in
`tbl_seg_mv_profile` is read positionally by the map's heatmap renderer.

### 4.2 Snap each stacked row to a bin

Each row's `importance` and `susceptibility` are snapped to the **nearest** level in `S`
independently. A row where either value is non-numeric/NULL is **dropped**.

> **Tie-breaking matters.** The reference uses `numpy.argmin`, which returns the *first*
> minimal index over an ascending scale — so a value exactly between two levels rounds
> **down**. With scale `1..5`, `importance = 1.5` → level 1, not 2. Reproduce this or your
> labels will drift on half-integer inputs.

Note the snap is unconditional: values outside the scale are clamped to its nearest end, not
rejected. With scale `1..5`, `importance = 0` snaps to level 1 (see §7.1 — this is exactly
how a project with unset valuations still produces output).

### 4.3 Per-cell histogram, depth, no_data

For each cell:

- `count[bin]` = number of surviving stacked rows in that bin
- `depth` = `sum(count)` — the **coverage/intensity index**, i.e. overlap stack depth
- `no_data` = `depth == 0`
- `p[bin]` = `count[bin] / depth` — proportions, summing to 1 (all-zero row for `no_data`)

Per-asset intersection areas are not persisted upstream, so within a cell every overlap is
weighted equally. Cross-cell comparability comes from area-weighting at the aggregation step
(§4.7), where `area_m2` *is* available.

### 4.4 Minimum-area filter

Applied to **cells**, before histogram construction, using area computed in the equal-area
CRS **EPSG:6933**. Cells below `segmv_min_area_m2` are removed from the run entirely — they
do not appear in the output at all (unlike `no_data` cells, which do).

### 4.5 Transform

Applied to the proportion matrix `P` (cells × bins), fitted cells only:

- **Hellinger** (default): `X = sqrt(P)`. Handles zeros without pseudocounts.
- **CLR**: `L = ln(P + 1e-6)`, then `X = L - rowmean(L)`. The compositional alternative.

### 4.6 Feature matrix and the fit

When `coverage_weight != 0`, one extra column is appended:

```
cov_z = zscore(depth) * coverage_weight        # over the FITTED cells of this layer only
```

The z-score is scikit-learn `StandardScaler`, i.e. **population** standard deviation
(`ddof = 0`). Using the sample standard deviation is a common and silent reimplementation
bug — it rescales the coverage feature relative to the histogram block and changes which
`k` wins.

`X = hstack([X_hist, cov_z])` → `n_cells × (|S|² + 1)` dims.

Then, for every `k` in `k_range` with `1 < k < n_cells`:

```
GaussianMixture(n_components=k,
                covariance_type="diag",
                random_state=seed,      # 42
                n_init=5,
                reg_covar=1e-6)
```

Pick the `k` with **minimum BIC**. If no candidate `k` satisfies `1 < k < n_cells`, fall back
to a single `k = min(2, n_cells - 1)`.

From the winning model:

- `proba` = posterior matrix (cells × k)
- `cluster_id` = `argmax(proba)` — a **hard** assignment; every cell lands in exactly one type
- `p_max` = `max(proba)` — per-cell certainty
- `entropy` = `-Σ p·ln(p)` in nats, with `p` clipped to `[1e-12, 1]`
- `cluster_label` = `"type " + (cluster_id + 1)` — so `cluster_id = 0` is displayed as *type 1*

### 4.7 Per-type profiles

One row per **non-empty** cluster (empty clusters are skipped, so the delivered type count can
be lower than the selected `k` — a legitimate reason a `k_range` of `7-9` yields 6 types).

Weights are cell area in km² (`w = area_km2`); if area is unavailable the weights fall back to
1 (equal weighting).

| Field | Definition |
| --- | --- |
| `n_polygons` | Cells in the cluster |
| `total_area_km2` | Sum of cell areas |
| `h_i*_s*` | Area-weighted mean of the cluster's `p` vectors — the **fingerprint** |
| `mean_importance` / `mean_susceptibility` | Area-weighted mean of each cell's mean importance / susceptibility (`nansum / Σw`) |
| `mean_coverage_index` | Area-weighted mean stack depth |
| `top_asset_groups` | The 3 asset groups with the most stacked rows inside the cluster's cells, as human labels, comma-joined |
| `area_basis` | `grid` for `H3_*`/`QDGC_*` layers (cell area generalises the true footprint), `polygon` otherwise |

> `top_asset_groups` is **descriptive, not a partition key**. The same asset group can top
> several types — a group that occurs across the whole study area will. It does not indicate
> duplicated data; each cell appears in exactly one type. Expect to answer this question from
> users, so surface the definition in the UI.

### 4.8 Validation against the deterministic signatures

If `tbl_segmentation/<layer>.parquet` exists, the run reports **ARI** (adjusted Rand index)
and **NMI** (normalised mutual information) between the GMM labels and the signature strings
for the same cells, over cells present in both with a non-empty signature. Requires ≥2 rows
and ≥2 distinct values on both sides, else omitted.

This is a report, not a gate. It is also the best available acceptance test for a
reimplementation — see §8.

---

## 5. Outputs

### 5.1 `tbl_seg_mv` — one row per cell

| Column | Type | Notes |
| --- | --- | --- |
| `code` | text | Cell id |
| `name_gis_geocodegroup` | text | Layer |
| `run_id` | text | `YYYY-MM-DD_HHMMSS` by default |
| `cluster_id` | int, nullable | NULL for `no_data` |
| `cluster_label` | text | `type N`, or `no_data` |
| `p_max` | float, nullable | Posterior certainty |
| `entropy` | float, nullable | Nats |
| `coverage_index` | int64 | Stack depth |
| `top_bins` | text | Compact `iIMPxSUS:prop` top-3 string, for tooltips |

### 5.2 `tbl_seg_mv_profile` — one row per type

`run_id`, `name_gis_geocodegroup`, `n_clusters`, `cluster_id`, `cluster_label`, `n_polygons`,
`total_area_km2`, `mean_importance`, `mean_susceptibility`, `mean_coverage_index`,
`top_asset_groups`, `description_ai`, `area_basis`, plus one `h_i{i}_s{s}` column per bin.

### 5.3 Idempotency

Both tables are written with **append-replacing** semantics: rows whose
`(run_id, name_gis_geocodegroup)` match the current run are dropped, then the new rows
appended. So:

- re-running the same `run_id` + layer replaces exactly that slice
- classifying several layers under one `run_id` **accumulates** instead of clobbering
- prior runs co-exist, which is what lets the map's **Run** dropdown offer history

If the column set no longer matches the existing file, the file is replaced wholesale rather
than merged into a half-NULL union.

---

## 6. Implementing this on PostGIS

### 6.1 Split the work: SQL extracts features, Python fits

The right seam is between §4.5 and §4.6.

Everything up to the transform is **set-based aggregation over the largest table** — do it in
SQL, next to the data, in one pass. The GMM fit is **iterative numerical optimisation over a
small dense matrix** — do it in a Python worker.

The matrix that crosses the boundary is tiny: `n_cells × (|S|² + 1)` doubles. For the
Mombasa reference project that is 3,318 × 26 ≈ 690 KB, extracted from 37,184 stacked rows.
Even a national run with 10⁷ stacked rows and 10⁶ cells yields ~200 MB — and the
`GaussianMixture` fit, not the SQL, becomes the bottleneck.

Do **not** put scikit-learn inside the database via PL/Python unless you have no worker tier.
It pins heavyweight deps to the postmaster, and a long `fit` holds a backend process.
MADlib is not a substitute: it offers k-means, not a diagonal-covariance GMM with BIC
selection, so results would not be comparable.

### 6.2 Feature extraction in SQL

Assumed server schema: `mesa.stacked(code, layer, ref_asset_group, importance,
susceptibility)`, `mesa.geocode_object(code, layer, geom)`, `mesa.asset_group(id, …)`.

**Scale and bin lookup** — materialise them; do not hard-code 25 columns.

```sql
CREATE TABLE mesa.valuation_scale (level int PRIMARY KEY);
INSERT INTO mesa.valuation_scale VALUES (1),(2),(3),(4),(5);

CREATE VIEW mesa.hist_bins AS          -- importance-major, matches hist_columns()
SELECT i.level AS imp_level,
       s.level AS sus_level,
       row_number() OVER (ORDER BY i.level, s.level) - 1 AS bin_ix,
       format('h_i%s_s%s', i.level, s.level) AS bin_name
FROM mesa.valuation_scale i CROSS JOIN mesa.valuation_scale s;
```

**Snap to nearest level** — the `ORDER BY abs(...), level` reproduces the round-down tie rule
from §4.2. Rows with a NULL on either axis drop out via the inner lateral.

```sql
CREATE VIEW mesa.stacked_binned AS
SELECT s.code, s.layer, s.ref_asset_group, i.level AS imp_level, u.level AS sus_level
FROM mesa.stacked s
JOIN LATERAL (SELECT level FROM mesa.valuation_scale
              WHERE s.importance IS NOT NULL
              ORDER BY abs(level - s.importance), level LIMIT 1) i ON true
JOIN LATERAL (SELECT level FROM mesa.valuation_scale
              WHERE s.susceptibility IS NOT NULL
              ORDER BY abs(level - s.susceptibility), level LIMIT 1) u ON true;
```

**Counts, depth, dense vector.** Keep counts long and densify only at the end — a sparse
group-by is far cheaper than 25 `FILTER` aggregates on a large table.

```sql
CREATE MATERIALIZED VIEW mesa.cell_hist AS
SELECT code, layer, imp_level, sus_level, count(*)::bigint AS n
FROM mesa.stacked_binned
GROUP BY code, layer, imp_level, sus_level;

CREATE UNIQUE INDEX ON mesa.cell_hist (layer, code, imp_level, sus_level);
```

```sql
-- Dense, ordered proportion vector + depth for one layer.
WITH cells AS (
    SELECT g.code
    FROM mesa.geocode_object g
    WHERE g.layer = $1
      AND ($2 = 0 OR ST_Area(ST_Transform(g.geom, 6933)) >= $2)   -- min_area_m2, §4.4
),
filled AS (
    SELECT c.code, b.bin_ix, COALESCE(h.n, 0) AS n
    FROM cells c
    CROSS JOIN mesa.hist_bins b
    LEFT JOIN mesa.cell_hist h
           ON h.layer = $1 AND h.code = c.code
          AND h.imp_level = b.imp_level AND h.sus_level = b.sus_level
),
depths AS (SELECT code, sum(n) AS depth FROM filled GROUP BY code)
SELECT f.code,
       d.depth,
       array_agg(f.n::float8 / d.depth ORDER BY f.bin_ix) AS p
FROM filled f
JOIN depths d USING (code)
WHERE d.depth > 0                       -- no_data cells excluded from the fit
GROUP BY f.code, d.depth;
```

`no_data` cells are the ones in `cells` but absent from this result — carry them straight to
the output with NULL cluster (§5.1) rather than dropping them.

**Transform in SQL** (or in the worker; either is fine — pick one place and keep it).
`WITH ORDINALITY` is required: array element order from a bare `unnest` is not guaranteed.

```sql
-- Hellinger
ARRAY(SELECT sqrt(v) FROM unnest(p) WITH ORDINALITY t(v, ord) ORDER BY ord)

-- CLR
ARRAY(SELECT ln(v + 1e-6) - (SELECT avg(ln(v2 + 1e-6)) FROM unnest(p) v2)
      FROM unnest(p) WITH ORDINALITY t(v, ord) ORDER BY ord)
```

**Coverage feature** — note `stddev_pop`, per §4.6. The window must span exactly the fitted
cells of this layer, nothing more.

```sql
SELECT code, depth,
       (depth - avg(depth) OVER ()) / NULLIF(stddev_pop(depth) OVER (), 0) * $3 AS cov_z
FROM fitted_cells;      -- $3 = coverage_weight
```

Guard the `NULLIF`: a layer where every cell has identical depth yields a zero denominator.
The reference produces zeros there (scikit-learn maps zero variance to 0); returning NULL
would poison the matrix.

### 6.3 The fit, and writing back

The worker receives `(code[], X)` , fits per §4.6, and returns
`(code, cluster_id, p_max, entropy)`. Keep `run_id` server-generated and pass it in, so
retries are idempotent.

```sql
CREATE TABLE mesa.seg_mv (
    run_id         text   NOT NULL,
    layer          text   NOT NULL,
    code           text   NOT NULL,
    cluster_id     int,
    cluster_label  text,
    p_max          float8,
    entropy        float8,
    coverage_index bigint NOT NULL DEFAULT 0,
    top_bins       text,
    PRIMARY KEY (run_id, layer, code)
);
```

The primary key gives you §5.3 for free — `INSERT … ON CONFLICT (run_id, layer, code) DO
UPDATE` replaces one run's slice of one layer and leaves every other run untouched. Do **not**
key on `(layer, code)` alone: run history is a product feature, not debris.

For the profile table, prefer `fingerprint float8[]` over 25 wide columns, with `mesa.hist_bins`
as the positional key. Widen to `h_i{i}_s{s}` only at the export boundary (§7.2).

Indexes and layout worth having up front:

- `mesa.stacked (layer, code)` btree — every read in §6.2 is layer-scoped
- BRIN on `mesa.stacked (layer)` if the table is append-ordered and large
- declarative partitioning of `mesa.stacked` by `layer` once it passes ~10⁸ rows; the whole
  pipeline is single-layer, so partition pruning is total
- refresh `mesa.cell_hist` per layer (or partition it) — a global `REFRESH MATERIALIZED VIEW`
  on a national dataset is wasted work

### 6.4 Determinism across engines

A fixed `seed` guarantees reproducibility only for the **same scikit-learn version on the same
platform**. Record `sklearn.__version__`, `numpy.__version__` and the platform alongside
`params` for every run, and treat a version bump as a reason to re-run rather than to compare.
Report agreement with ARI (§4.8), never by comparing `cluster_id` values directly — GMM
component numbering is arbitrary and a rerun may permute labels while partitioning
identically.

---

## 7. Two traps found in a real mesa-server import

Both from `admin-d42b8e65_processed.zip` (Mombasa reef, imported into MESA 5.6.0 desktop
2026-07-26). Neither is a desktop bug; both are export-side contract violations that produce
*plausible-looking but meaningless* output.

### 7.1 Unset valuations produce a degenerate classification, silently

In that export, all six asset groups had `importance = susceptibility = sensitivity = 0`, and
so did all 37,184 rows of `tbl_stacked`.

The consequence is not an error — it is a **wrong answer that looks right**:

1. Every stacked row snaps to the same bin (§4.2 clamps 0 to level 1).
2. Every cell's proportion vector is therefore identical: `1.0` in `h_i1_s1`, zero elsewhere.
3. The histogram block has **zero variance**. The only feature carrying information is
   `cov_z`.
4. The GMM cleanly partitions a 1-D feature: the delivered types were pure stack-depth bands,
   with `mean_coverage_index` of exactly 1.0, 2.0, 3.0, 4.0, 5.0 and 6.0.
5. `mean_importance` and `mean_susceptibility` read `0.0` for every type.
6. `top_asset_groups` repeated the same names across types — because the types were depth
   bands over one shared asset palette, not compositional types.

Every number was arithmetically correct. The classification simply answered "how many assets
overlap here", not "what kind of place is this" — and nothing in the output said so.

**Required of the server:** refuse, or loudly degrade, before spending the fit. A cheap
precondition:

```sql
SELECT count(DISTINCT (imp_level, sus_level)) AS occupied_bins
FROM mesa.stacked_binned WHERE layer = $1;
```

`occupied_bins <= 1` ⇒ the histogram cannot discriminate. Report it as a **precondition
failure naming the asset groups with unset valuations**, not as a completed run. Consider
extending the check to the composition itself: if the fitted histogram block has near-zero
variance across cells, the run is depth-only regardless of how many bins are occupied
globally. Record that verdict in the run metadata so the map can label the result honestly.

Worth doing on both sides — MESA desktop currently runs happily on such input too, and this
document is the record of why it should not.

### 7.2 Write NULL for "unknown", never 0

MESA desktop backfills `importance`/`susceptibility` from `tbl_asset_group` **only when the
whole column is NULL/non-numeric**. A present `0` is taken as a real measurement: it is not
backfilled, and it snaps to the bottom of the scale.

So an export that means "not assessed" must write **NULL**, not `0`. Writing `0` converts a
missing-data condition into a confident, wrong valuation — and it defeats the one repair
mechanism the desktop has.

Related contract notes for the export path:

- `ref_asset_group` must match `tbl_asset_group.id`, as text-comparable values — it is the
  join key for `top_asset_groups`, and the desktop casts both sides to `str`.
- Cells with no overlapping asset must still exist in `tbl_geocode_object`. That is how
  `no_data` becomes visible instead of silently absent.
- Emit the fingerprint columns literally as `h_i{i}_s{s}`, importance-major, matching the
  project's own `valid_input` scale. The map reads them positionally.
- `tbl_seg_mv` and `tbl_seg_mv_profile` must arrive together — a profile-less run renders a
  legend with no zones table.

---

## 8. Acceptance checklist

Run the server implementation and MESA desktop on the *same* project and compare:

1. **Same cells fitted.** Row count in `seg_mv` matches, and the `no_data` set is identical.
2. **Same partition.** ARI between server `cluster_id` and desktop `cluster_id` is `1.0`.
   Anything below 1.0 means a real divergence — chase it in this order: tie-breaking (§4.2),
   `stddev_pop` vs `stddev_samp` (§4.6), bin ordering (§4.1), min-area CRS (§4.4).
3. **Same model selected.** `n_clusters` agrees, and the BIC table agrees to ~1e-6 relative.
4. **Same certainties.** `p_max` and `entropy` agree to ~1e-9 after aligning label permutation.
5. **Same profiles.** `total_area_km2` and `mean_coverage_index` agree to 3 decimals;
   fingerprints agree to 1e-6.
6. **Idempotency.** Re-running one layer under an existing `run_id` leaves other layers' and
   other runs' rows byte-identical.
7. **Degenerate input is refused.** A project with unset valuations fails the §7.1
   precondition instead of producing depth bands.

Test 2 is the one that matters. Tests 3–5 tell you *where* 2 broke.
