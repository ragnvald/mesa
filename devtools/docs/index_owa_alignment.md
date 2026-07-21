# `index_owa`: desktop/server alignment and the pending method decision

Status 2026-07-21. Written while the method author is away (back 4 August). Until then
**desktop and server hold option A: both produce today's desktop numbers.** This file is the
readiness material for option B, so the change can be made once, quickly, when the answer comes.

## The exact algorithm desktop implements today

`processing_internal._compute_index_owa_from_counts`, called from the flatten stage.

Input: `owa_n25 … owa_n1` — counts of overlapping assets per rounded, clipped sensitivity value
1..25. Coerced numeric, NaN→0, cast to int, clipped at 0.

1. **Population**: the whole of `tbl_flat` — every geocode group pooled into one ranking.
2. If every row is all-zero, every row scores 0 and the rest is skipped.
3. Sort by `[owa_n25 … owa_n1]` descending, `kind="mergesort"` (stable — ties keep input order).
4. **Dense rank over distinct count-vectors**: `is_new[i] = row[i] != row[i-1]`, `dense = cumsum(is_new)`.
   One rank step per *distinct vector*, however many cells share it. `max_dense = dense[-1]`.
5. Invert so the best vector gets `max_dense`: `rank_high = max_dense - dense + 1`.
6. Scale: `round(rank_high / max(max_dense, 1) * 100)`, clipped to `[1, 100]`, int64.
7. All-zero rows forced back to 0.

Two details that matter for bit-exact reproduction:

- **The ranking runs before sliver cleanup.** Index is computed at `processing_internal.py:3535`;
  rows with `area_m2 < flatten_sliver_min_area_m2` (default 1.0) are dropped at `:3591`. The
  denominator therefore counts cells that never reach the published table — on the Uganda
  fixture, `max_dense` is 490 rather than 489, and 17 cells are counted then removed.
- **`np.round` is half-to-even.** `round(0.5) == 0`, `round(1.5) == 2`. A reimplementation using
  half-away-from-zero will differ by one wherever `rank_high / max_dense * 100` lands exactly on
  `.5`. It does not occur on the Uganda fixture (no `k` satisfies it for `max_dense = 490`), but
  it is not structurally impossible — check it per dataset rather than assuming.

Related and already aligned: the boundary-only join filter keeps rows whose cell∩asset overlap
is `>= 1.0 m²` (`_BOUNDARY_ONLY_OVERLAP_MIN_M2`). The server's independent implementation drops
the same rows.

## Where the two implementations diverge

| Aspect | Desktop 5.5.0 | Server |
|---|---|---|
| Ranking population | whole project, all groups pooled | per geocode layer |
| Rank unit | distinct count-vectors (dense rank) | cells, `rank(ties.method = "average")` |
| Sliver threshold | 1 m², applied **after** the index | none yet; planned **before** the index |

Measured impact: 19.9 % agreement overall on the population choice alone, 0.13 % on
`basic_mosaic`; on the rank unit, median 3 (desktop) against 45 (server R formulation).

## Option A — in force now

Server temporarily reproduces the desktop algorithm above, including the sliver threshold applied
*after* the index. Desktop does not change. Rationale: desktop has published artefacts (Zenodo
packages, demo datasets, wiki text); the server does not yet. Changing the side without published
output is cheaper, and the server has already demonstrated it can reproduce the desktop column
exactly.

## Option B — what changes if the method answer says per-group / per-cell

Code, all in `processing_internal.py`:

1. **Population.** Restore a per-group ranking in `_compute_index_owa_from_counts`. The old
   `labels` parameter was removed in `8e771c7` *because it never worked* — do not resurrect it
   from git as-is. The defect was that the call site keyed the label Series on `code` while the
   function reindexed on `tbl_flat`'s integer RangeIndex, so nothing matched. Any new version must
   assert that labels actually matched (`labels.reindex(index).notna().any()`), not silently fall
   back to one bucket.
2. **Rank unit**, if the author confirms the R formulation: rank *cells* with average ties rather
   than dense-ranking distinct vectors. This is the change with the largest numeric effect.
3. **Ordering.** Move the index computation to after the sliver cleanup, so the denominator only
   counts cells that reach the table.

4. **Which sensitivity values are counted at all.** The reference script counts ten exact values —
   4, 6, 8, 10, 12, 15, 16, 18, 20, 25 — and `18` is not a product of two classes on a 1–5 scale,
   while `9` (3×3) is missing from the list. On the Uganda fixture `tbl_stacked` holds **38,194
   rows at sensitivity 9 (32.85 %, the most common value) and 0 rows at 18**. An implementation
   that hardcodes the list as written scores 8,678 cells (17.8 %) as empty because all their
   overlaps sit at 9, and 20,653 cells have no overlap inside the list at all. Desktop is not
   affected — it counts every bin `owa_n1 … owa_n25` — but the answer determines whether the
   reference itself is fixed, and it is the same subject as the bins question below. Awaiting the
   author (this is the fourth question put to him).

Bins, worth settling in the same round because it is the same subject:

- `_compute_owa_counts_from_stacked` hardcodes bins 1..25 while the valuation scale is
  configurable (`[VALID_VALUES] valid_input`, default `1,2,3,4,5`). Eleven bins (7, 11, 13, 14,
  17, 18, 19, 21–24) can never be non-zero on a 1–5 scale, and conversely a 1–6 scale would
  produce 30 and 36, which fall outside the hardcoded ceiling and would vanish from the index
  silently. The real question is therefore "derive the bins from `valid_input`", not "delete the
  eleven dead columns". Note the columns are intermediate only — the write list at the end of
  the flatten stage keeps `index_owa` and drops every `owa_n*`, so nothing user-facing carries
  them and there is no schema gain in removing them. Deliberately deferred out of the parity
  window: a change inside the index computation has to be mirrored by the server for no benefit
  to either side.

Everything else:

- **Wiki**: `Indexes.md` currently describes pooled ranking (commit `139be4e`, pushed 2026-07-21).
  Option B makes the pre-`139be4e` text correct again — revert that commit rather than rewriting
  from scratch, and drop the provenance note with it.
- **learning.md**: the entry "Dead per-group ranking in index_owa" states that code and published
  docs disagree. Mark it superseded, per the repo convention — do not rewrite it.
- **Demo datasets and Zenodo packages**: every published `index_owa` value changes. The packages
  must be rebuilt and republished, not patched, and the release text needs a line saying index
  values from before and after are not comparable.
- **`qgis/mesa.qgz`**: carries `*_index_owa` mbtiles rasters; they are regenerated by the Tiles
  stage, so no project edit is needed, but the tiles must be rebuilt.

## Verification plan for B

The fixture is `D:\data\mesa_demodata\fixtures\mesa_5.5.0_uganda_full.zip` (48,615 cells across
eight geocode groups). Do not run the pipeline in `D:\code\mesa` — Prep deletes `tbl_stacked`
and `tbl_flat`.

1. Keep the current `tbl_flat` as the baseline before rerunning.
2. Rerun the flatten stage and compare `index_owa` per `code`: report the number of cells changed,
   the distribution of the change, and — the number that matters for reading a map — how many
   cells move between the deciles the legend uses.
3. Confirm the per-group ranking actually took effect: every geocode group must now contain at
   least one cell at 100. Under pooled ranking only `H3_R7` and `QDGC_L6` reach 100 while the
   other six layers cap at 97; that asymmetry disappearing is the signal the change is live.
4. Cross-check against the server's implementation on the same fixture. They have offered their
   specification and measurements for exactly this.
5. Re-run `pytest tests/ -q` (deps in `requirements_dev.txt`).
