# Scalable processing — bounded-working-set partitioning & adaptive sizing

**Status:** Proposal for **5.4+**, deferred out of the 5.3 stabilisation phase.
Nothing here is implemented. This is the roadmap surface; durable "why" lives in
`learning.md`, mosaic-specific detail in `docs/basic_mosaic_capacity.md`, and the
committed-but-not-done mosaic slice in `docs/further_development.md` **A2**.

This document generalises A2 (mosaic tiling) into a single architectural
principle that also covers the **intersect** stage and, crucially, upgrades the
**scaling mechanism** (`auto_tune.py`) that decides how hard to push each host.

---

## 1. The problem, in one sentence

MESA's heavy steps must scale from small laptops to large workstations. The
governing constraint is **peak memory**, and the guiding rule is the operator's:

> **Better that a run takes longer than that it stops.**

Today the pipeline honours that rule only partially, because two of the heaviest
operations have a working set that scales with **total problem size**, not with a
bounded chunk — so on a small machine with a large dataset there is no worker
setting that avoids the memory cliff, only a graceful **abort**.

## 2. Why not GPU (the direction this rules out)

Assessed 2026-07 (this proposal's origin). The dominant cost is **GEOS vector
geometry** — `gpd.sjoin`, `shapely.intersection`, `union_all`/`unary_union`,
`polygonize` (see `processing_internal.py` intersect stage and
`geocode_manage.py` mosaic). This class of work is a poor GPU fit and, more
decisively, has **no cross-platform GPU path**:

- GEOS has no GPU backend; polygon boolean ops / noding / polygonization are
  irregular, branch-heavy, data-dependent recursion — the antithesis of SIMD.
- NVIDIA cuSpatial (CUDA-only, Linux-first, no macOS) covers point-in-polygon and
  some joins, but **not** arbitrary polygon overlay/union/polygonize — the ops
  that dominate. So even on NVIDIA it cannot replace the hot path.
- Apple Silicon has no equivalent geometry library (Metal/MLX are for tensors).
- The GPU-amenable parts (per-cell index math, GMM classification) are **already
  cheap** (vectorised numpy / small feature matrices), so accelerating them yields
  little end-to-end (Amdahl).
- A GPU path would ship up to three backends (CUDA + Metal + CPU fallback), add
  gigabytes to the frozen app (conflicts with the footprint work), require vendor
  drivers, and risk non-deterministic geometry across vendors.

**Conclusion:** the portable, appropriate lever is **CPU parallelism via spatial
partitioning that bounds the working set** — the same algorithm on Windows / macOS
/ Linux, no vendor lock-in, no packaging bloat.

## 3. Evidence: config tuning alone is not enough

Benchmark (2026-07, isolated project, 10,777 overlapping benthic polygons →
~11,700 faces, this Windows 16-logical-core / 128 GB host):

| Config change | Speedup vs baseline |
|---|---|
| `mosaic_auto_worker_fraction` 0.75 → 1.0 | **0.91×** (noise — extraction is ~4 %) |
| `mosaic_reduce_workers` 4 → 8 | ~1.09× (within run-to-run variance) |
| `mosaic_union_grid_size` 1.0 / 10.0 | **0.61× / 0.79× (slower)**, and changes face count |

Reduction was ~90 % of wall-clock, matching the 3.5 M-asset production run
(~87 %, `basic_mosaic_capacity.md`). **Takeaway:** config-only tuning tops out at
~1.0–1.1× and the intuitive knobs (more extraction workers; grid snapping) do
nothing or hurt at this scale. Real gains — and the memory guarantee — require the
algorithmic change below, not more knobs.

---

## 4. Pillar I — bounded-working-set spatial partitioning

Partition the metric AOI into a grid of spatial tiles sized so **one tile fits a
memory budget**. Process tiles independently; combine **without** a global
all-at-once merge. Peak memory becomes `max(one-tile working set) × concurrency`
— **bounded and tunable independent of total size**.

This is the unifying principle behind both heavy steps:

### 4a. Mosaic (already scoped as A2 — the reference implementation of the idea)
Node + polygonize per tile in parallel; reconcile only tile-boundary faces by
**asset-membership-signature dissolve** (tag each face with its covering-asset
set; dissolve adjacent faces with identical signatures — this removes the moiré
seams that retired the old tiled path). Interior faces finalise immediately and
stream to output; the monolithic `polygonize` peak (~21.7 GB on the 3.5 M run)
collapses to per-tile peaks. See `docs/further_development.md` A2,
`docs/basic_mosaic_capacity.md`, `learning.md` "Mosaic union reduction is
spawn-bound".

### 4b. Intersect (new — the easier, higher-frequency win)
The intersect stage (`processing_internal.py`: `gpd.sjoin(geocode × asset,
predicate=intersects)` + `shapely.intersection` for overlap area) is **near-embarrassingly
partitionable**: a geocode cell belongs to exactly one tile, so each tile runs its
own sjoin + intersection over only the assets touching that tile, with **no
cross-tile merge at all** (rows are simply concatenated). This is strictly simpler
than the mosaic (no boundary faces to stitch) and runs far more often. Peak per
worker = one tile's cell+asset geometry, not the whole `tbl_stacked`.

### 4c. Why this delivers the small→large scaling the operator wants
- **Small machine:** small tiles + concurrency = 1 → peak = one small tile →
  **OOM impossible, just slow.** "Abort" becomes "takes longer" — the rule, honoured.
- **Large machine:** larger tiles + high concurrency → full core use; ample RAM →
  fewer/bigger tiles reduce boundary-stitching overhead.
- **Adaptive splitting:** if a tile's estimated/measured peak exceeds budget,
  split it into 4 and recurse. The working set is *guaranteed* to fit — "never
  stop" as an algorithmic invariant, not a heuristic.

## 5. Pillar II — improve the scaling mechanism (`auto_tune.py`)

`auto_tune.py` today is **open-loop, one-shot at pipeline start**, sizing each
stage as `min(RAM budget / static per-worker-GB guess, CPU)` (`_derive_max_workers`
et al.). Strengths to keep: uses *available* RAM × a platform-aware target
(`_mem_target`: 0.70 Apple / 0.85 else), physical-core cap for memory-heavy
flatten, cheap parquet-metadata data fingerprint, single audit log, respects
explicit overrides. Weaknesses this plan targets:

1. **Closed-loop calibration.** Run the first partition, measure its actual peak
   RSS (psutil on the child), then set concurrency for the rest from the *measured*
   cost — not the guess. Ramps up safely on big boxes (observe headroom before
   committing → no hubris) and clamps down on small ones. Removes both failure
   modes the operator named: under-utilisation *and* over-commitment.
2. **Partition size as the primary knob; worker count derived.** Invert today's
   design (worker count primary, `flatten_huge_partition_mb` a semi-static
   threshold): choose partition size so per-partition peak ≈ target, then
   concurrency = budget / per-partition-peak. Makes the memory guarantee explicit
   and the two knobs consistent.
3. **Data-derived per-worker estimate.** `auto_tune` already has the fingerprint
   (`_probe_data`: asset/geocode rows + sizes) but per-worker GB is a constant
   (`approx_gb_per_worker = 4.0`, flatten ×3, tiles 3.0, …). Replace with
   `bytes-per-row × rows-per-partition × overhead`, calibrated once from a real run
   and persisted (ties into the settings-store, `further_development.md` A1).
4. **Adaptive instead of abort.** Replace the mosaic pre-flight **abort**
   (`geocode_manage.py` `mosaic_preflight_*`) with pre-flight **choose finer
   tiles**. Abort only if a single minimal tile cannot fit (effectively never for a
   partitioned step). This is "better slow than stop" raised to a guarantee.
5. **Runtime memory watchdog + backpressure.** A monitor thread checks
   `virtual_memory().available`; below a floor it **pauses launching new
   partitions / shrinks the pool** rather than killing the run. Complements the
   existing lifetime panic watchdog (a graceful throttle, not a kill) and the
   already-listed "Mosaic memory watchdog" idea (`further_development.md` B).

## 6. Correctness & determinism (the real engineering cost)

- **Intersect:** trivially exact — tiles are disjoint by cell membership; results
  concatenate.
- **Mosaic:** the hard part is **exact noding at tile boundaries** (no slivers /
  gaps). Assets: the existing area sanity check (`coverage_area == faces_area`) is a
  correctness gate; `mosaic_union_grid_size` snapping stabilises boundary noding.
  The membership-signature dissolve (A2) is what makes boundary reconciliation
  produce the *same* faces as the global path.
- **Determinism:** union/intersection are commutative per region, so tile order
  does not change results — safe to parallelise and safe for reproducible reports.

## 7. Suggested landing order (incremental, low-risk first)

1. **Closed-loop calibration + watchdog (§5.1, §5.5).** Self-contained; touches
   only `auto_tune.py` + the pool runners, **not** the geometry algorithms. Safest
   first step; benefits every stage immediately.
2. **Intersect tiling (§4b).** Simpler than the mosaic (no boundary stitching);
   high frequency; bounds the intersect peak.
3. **Partition-size-primary + data-derived estimate (§5.2, §5.3).** Consolidates
   the sizing model once tiling exists to size against.
4. **Mosaic Tier 2 (§4a / A2).** The hardest (boundary dissolve), highest payoff
   on the largest AOIs; do last, gated by the sanity check.

Each step is independently shippable and measurable (re-run the mosaic/intersect
benchmark harness from this session, comparing wall-clock **and** peak RSS).

## 8. References

- `docs/further_development.md` A2 (mosaic Tier 2), B (mosaic watchdog, snap default)
- `docs/basic_mosaic_capacity.md` (phase breakdown, capacity table, pre-flight gate)
- `learning.md` "Mosaic union reduction is spawn-bound, not compute-bound"
- `code/auto_tune.py` (`_derive_*`, `_probe_hardware`, `_probe_data`, `_mem_target`)
- `code/geocode_manage.py` (`mosaic_faces_from_assets_parallel`, `_auto_worker_count`, `mosaic_preflight_*`)
- `code/processing_internal.py` (intersect: `gpd.sjoin`, `shapely.intersection`)
