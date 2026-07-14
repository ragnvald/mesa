# MESA processing server — hosted heavy compute (vision)

**Status:** Direction for **version 6** — a **separate server component** added to
the MESA portfolio. Nothing here is implemented; this is the roadmap surface, not a
committed design. It builds on the compute work in
`docs/SCALABLE_PROCESSING_PLAN.md` but does not depend on anything server-specific
landing in the desktop app.

**Scope guardrail — no desktop implications.** This is an **additive portfolio
extension**, not a change to the desktop application. The server *consumes*
artifacts the desktop already produces (backup ZIPs, headless entry points) as-is;
it introduces **no new desktop-side dependencies, no desktop build changes, and no
change to the local-first promise**. The desktop version ships and behaves exactly
as before whether or not the server exists.

**One line:** a hosted Linux service where a user uploads a project (as the
existing **backup ZIP**), the server runs the heavy pipeline on high-capacity
hardware under full environment control, and returns the processed results as a
downloadable backup ZIP.

---

## 1. Why this fits MESA unusually well

The pieces already exist; the server mostly *composes* them:

- **The backup ZIP is already the transport unit.** After the 5.3 export
  restructure (`create_backup_archive`, categories: input-always / databases /
  tiles / reports), an **upload = input-only backup** and a **download = output
  backup**. `restore_backup_archive` + the zip-slip guard (`_safe_zip_member_names`)
  already handle untrusted archives safely.
- **Processing is already headless.** `geocode_manage.py --nogui --mosaic`,
  `processing_pipeline_run.py`, and the tiles/segmentation helpers run without a
  GUI. The heavy work is cleanly separable from the desktop app.
- **AI-token stripping is already the default** in backups, so uploaded projects do
  not carry secrets by accident.

## 2. What full environment control unlocks (the real payoff)

The previous assessment (`SCALABLE_PROCESSING_PLAN.md` §2) ruled out GPU because of
the **cross-platform** constraint. A controlled Linux server removes that constraint
and unlocks wins that are impossible in the shipped desktop app:

- **`fork` instead of `spawn` — likely the single biggest free speedup.** The
  mosaic's ~87 % reduction was **spawn-bound**: Windows/macOS `spawn` re-imports
  geopandas/shapely for every worker (`learning.md` "Mosaic union reduction is
  spawn-bound"; `basic_mosaic_capacity.md`). On Linux with `fork`, workers inherit
  the already-imported interpreter → the per-merge respawn cost largely vanishes.
  Only available when we own the OS.
- **Huge RAM + the partitioning plan.** `SCALABLE_PROCESSING_PLAN.md`'s
  bounded-working-set tiling + adaptive sizing is exactly what lets a 256–512 GB /
  many-core server run projects that OOM everywhere else *and* saturate the cores.
  The server is where that plan's payoff is largest.
- **Pinned, optimised geometry stack.** One known GEOS/Shapely/pyogrio build, tuned
  compiler flags, no per-user driver roulette → reproducible outputs (valuable for a
  scientific tool whose results feed reports).
- **Optional GPU niche, only here.** Even with control, geometry (the hot path) has
  no GPU library — so GPU stays a narrow, optional accelerator for the pure numeric
  steps (per-cell index math, GMM classification), and only if it ever pays off.
  Do **not** make it the plan; `fork` + big-RAM + partitioning gets most of the win.

## 3. Shape of the service (sketch)

- **Job model:** async. `upload → validate → queue → process → notify → download`,
  with result expiry. A small REST API + a worker pool; the worker is essentially
  the existing headless pipeline invoked on a restored project directory.
- **Isolation:** one job per sandbox (container), resource-limited (cgroups memory
  ceiling) so a heavy job cannot OOM or starve neighbours. The auto_tune closed-loop
  sizing (`SCALABLE_PROCESSING_PLAN.md` §5) maps directly onto a per-container
  memory limit — the server sets the budget, auto_tune fills the cores.
- **Elastic capacity:** on-demand large instances per job vs a warm pool; size the
  instance to the uploaded project's fingerprint (`auto_tune._probe_data` already
  reads asset/geocode row counts from the upload without a full read).

## 4. Trust & privacy — the load-bearing product question

MESA's current selling point is **"all data stays local — nothing is uploaded."**
A hosted server **inverts** that promise, so it cannot be a silent default:

- **Strictly opt-in**, with plain-language data handling: what is uploaded, where it
  runs, retention window, deletion guarantees, encryption in transit and at rest.
- Keep the **local path as first-class** — the server is an *option* for users who
  lack the hardware for the largest AOIs, not a replacement.
- Consider a **self-hostable** server image (same container) so privacy-sensitive
  orgs can run it on their own infrastructure — this keeps the "your data, your
  machine" ethos while still delivering the big-box speedups.

## 5. Sequencing

1. **5.4 (shared engine, desktop-beneficial):** land
   `SCALABLE_PROCESSING_PLAN.md` (bounded-working-set partitioning + adaptive
   sizing) in the shared processing code. This improves the desktop app on its own;
   the server later inherits it for free. No server work required here.
2. **Version 6 (new server component):** wrap the headless pipeline as a job worker;
   add the REST/queue/API and container isolation; reuse backup upload/download.
   Built and shipped as **its own component — the desktop build is untouched**.
3. **Optionally:** self-hostable image; per-job elastic sizing; a narrow GPU
   experiment for the numeric steps only.

## 6. References

- `docs/SCALABLE_PROCESSING_PLAN.md` (the compute work this productises; GPU rationale)
- `docs/basic_mosaic_capacity.md`, `learning.md` "Mosaic union reduction is spawn-bound"
- `mesa.py` `create_backup_archive` / `restore_backup_archive` / `_safe_zip_member_names`
- `code/auto_tune.py` (`_probe_data`, `_probe_hardware`, per-stage sizing)
- Headless entry points: `code/geocode_manage.py --nogui`, `code/processing_pipeline_run.py`
