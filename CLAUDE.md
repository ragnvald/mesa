# CLAUDE.md

Guidance for Claude (and any AI assistant) working in this repo.

## Scope: work only inside mesa, never other projects

This session works on **mesa only**. Never write, edit, commit, or push in any other project — even a sibling repo on the same disk (e.g. `/Volumes/Beelink 1/code/rag-mepa`), and even when the user pastes an error whose traceback points outside mesa. Other projects share this machine and drive; a fix that looks obvious here is unowned context there.

When a problem clearly originates outside mesa, **stop and tell the user** — name the project and file, explain what's wrong, and let them open that project to fix it themselves. Diagnosing across the boundary (reading logs, pointing at the culprit line) is fine; *modifying* across it is not.

## Git: the user initiates commits and pushes

Do not run `git commit` or `git push` on your own. Finish the work, leave the changes in the working tree, and *ask* — "should I commit this?" — then wait. A "yes" authorises that one action; it is not standing permission for the next one, and permission to commit is never permission to push.

This holds even when the change is small, obviously correct, or already verified, and even when an earlier task in the same session ended in a commit. Attribute commits to the user only (`ragnvald@mindland.com`), no AI co-author trailer, and keep them granular: one purpose per commit, with a message that explains why.

## Document map — where every document lives

The root holds only what has to be there. Everything else is filed. A new document does **not** go in the root: roadmap or overview → `plans_oversight/`; design detail → `devtools/docs/`; letter to a sibling project → `docs/exchange/`.

- **Repo root** — entry points and the permanent developer notes only: `README.md` (developer entry point), `LICENSE`, `privacy-policy.md`, `CLAUDE.md`, `instructions.md`, [learning.md](learning.md), [history.md](history.md), [cooperation.md](cooperation.md). `instructions.md` must stay at the root — [.github/copilot-instructions.md](.github/copilot-instructions.md) reads it by that path.
- **[plans_oversight/](plans_oversight/)** — roadmap and overview material, moved off the root 2026-08-12: `plans.md` (index of decided-but-not-done work and ideas), `MESA_system_overview.md` (user-oriented walkthrough of the workflow buttons), `presentasjon.md` (methodology framing for planning processes, Norwegian).
- **[devtools/docs/](devtools/docs/)** — full design documents, capacity records, yEd `.graphml` diagram sources, the segmentation PoC. `build_all.py` removes `devtools/` from the distribution wholesale, so anything here is safe by construction.
- **[devtools/](devtools/)** (top level) — dated release and build notes: `PUBLISERE_*.md`, `RELEASE_NOTES_*.md`, `ZENODO_*.md`, `MAC_BUILD.md`, `COUNTING_LOC.md`. They record the state at that release; do not rewrite them to match a later one.
- **[docs/](docs/)** — **ships to end users.** The two manuals and `docs/templates/` (read at runtime by `report_generate`). `docs/exchange/` correspondence is filed here but `build_all.py` leaves it out of the distribution (since 2026-10-11). Never add developer material here; see learning.md "Developer docs must live outside docs/".
- **`mesa.wiki/`** — separate repo, canonical source for end-user prose.

Neither `plans_oversight/` nor `devtools/` is in the `build_all.py` copy allowlist, so neither can reach a build. That allowlist — not the filename-matched `DEVELOPER_ONLY_FILES` list — is the real guarantee.

## Where to put what

- **Code comments**: explain the "what" and the technical "why" (hidden constraints, non-obvious invariants, references to a learning.md section). Keep them tight - a sentence or two max. Never write multi-paragraph comment blocks.
- **Durable rules distilled from an incident**: go in [learning.md](learning.md), not in source files. Code comments rot; learning.md is the durable record. When a fix is informed by a past incident, the comment should *point at* the learning.md section (e.g. `# See learning.md "Parent-side memory in the pipeline"`), not retell the story.
- **What happened, and in what order**: goes in [history.md](history.md) — sessions, incidents, deliveries, decisions and what was reported to the sibling projects. The split is: history.md is the *narrative* ("on 2026-07-28 a 25-hour mosaic peaked at 85.65 GB"); learning.md is the *rule* extracted from it ("the pre-flight gate scales on asset count, but memory is driven by vertex count"). A one-off event with no reusable lesson belongs only in history.md; a rule with no incident behind it is usually not worth an entry at all.
- **Operator-facing tuning notes**: belong inline near the relevant `config.ini` key, but kept short (one sentence on what it does, plus a pointer to learning.md if there's a known failure mode behind it).
- **Correspondence with the sibling projects**: the full letter goes in [docs/exchange/](docs/exchange/) as `YYYY-MM-DD-<topic>-<type>.md`; the decision it settled goes in the "Agreed formats" section of [cooperation.md](cooperation.md), which is maintained in place. Never leave a letter loose in the repository root — four of them accumulated there before 2026-08-09, three never committed at all.

## Why this rule exists

Long historical comments inside source files turn every diff review into a history lesson, age badly as the code changes around them, and duplicate learning.md without the structure (date, rule, why, how to apply, non-regression). Keep code comments about *the code*; keep learning.md about *what we learned*.

## learning.md format

Each entry is a level-2 section dated by the day the lesson landed:

```
## <Topic> (YYYY-MM-DD)

- Rule: <the one-line takeaway, imperative>
- Why: <root cause + brief incident context, including measurements/numbers if relevant>
- How to apply: <concrete trigger - "when editing X, check Y">
- Non-regression guarantee: <only if the fix preserves prior behaviour by default>
```

Append, do not rewrite history. Mark a previous entry as superseded inline if a newer entry overrides it.

## history.md format

Each entry is a level-2 section headed by the date (or date range) and what happened:

```
## YYYY-MM-DD — <what happened, in a few words>

<Narrative. What was observed, what it turned out to be, what was decided, what shipped.
Cite commits and measured numbers. Say what was reported to mesa-server / mesa_demodata.>
```

Append at the bottom, newest last. Never rewrite an earlier entry — if a later one overturns
something, say so in the later entry. Both sibling projects (`mesa-server`, `mesa_demodata`) keep
the same file, so an incident that crosses the boundary should be logged on both sides.

## Pipeline-specific gotchas worth knowing about

- The processing pipeline (`code/processing_internal.py`) has been bitten more than once by parent-process memory blow-ups in the gaps *between* worker pools. Per-pool watchdogs, per-stage worker caps, and pre-flight checks all only run inside `Pool(...)` blocks - parent-side reads/merges between pools are unguarded. There is now a process-lifetime sentinel (`_start_lifetime_panic_watchdog`) as a backstop, but it costs the entire run's progress when it fires; do not rely on it to catch sloppy parent-side allocations. Before adding any read in the parent, check whether it materialises a known-large dataset (`tbl_stacked`, `tbl_flat`). See learning.md "Parent-side memory in the pipeline".
- `read_parquet_or_empty()` materialises the entire dataset including geometries. Read its docstring before calling it on anything other than small, known-bounded tables.

## Documentation in `docs/`

- `docs/MESA_User_Guide_en.docx` and `docs/MESA_User_Guide_pt.docx` are the end-user Word manuals (Home, What's new, Quickstart, User-interface, Data, Method, Indexes, QGIS, Definitions, Advanced, Troubleshooting). They target analysts/operators, not developers.
- Both are built by [devtools/build_user_guide.py](devtools/build_user_guide.py) from a hand-maintained EN/PT content tree inside that script, with the release stamped in its `VERSION` constant. Update the tree and re-run the script; do not hand-edit a .docx.
- Source-of-truth ordering: `mesa.wiki` is canonical prose; the .docx files are derived artefacts. When the wiki content changes substantially, mirror it into `build_user_guide.py` and regenerate so the two stay in sync.
