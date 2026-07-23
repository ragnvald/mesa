# CLAUDE.md

Guidance for Claude (and any AI assistant) working in this repo.

## Scope: work only inside mesa, never other projects

This session works on **mesa only**. Never write, edit, commit, or push in any other project — even a sibling repo on the same disk (e.g. `/Volumes/Beelink 1/code/rag-mepa`), and even when the user pastes an error whose traceback points outside mesa. Other projects share this machine and drive; a fix that looks obvious here is unowned context there.

When a problem clearly originates outside mesa, **stop and tell the user** — name the project and file, explain what's wrong, and let them open that project to fix it themselves. Diagnosing across the boundary (reading logs, pointing at the culprit line) is fine; *modifying* across it is not.

## Git: the user initiates commits and pushes

Do not run `git commit` or `git push` on your own. Finish the work, leave the changes in the working tree, and *ask* — "should I commit this?" — then wait. A "yes" authorises that one action; it is not standing permission for the next one, and permission to commit is never permission to push.

This holds even when the change is small, obviously correct, or already verified, and even when an earlier task in the same session ended in a commit. Attribute commits to the user only (`ragnvald@mindland.com`), no AI co-author trailer, and keep them granular: one purpose per commit, with a message that explains why.

## Where to put what

- **Code comments**: explain the "what" and the technical "why" (hidden constraints, non-obvious invariants, references to a learning.md section). Keep them tight - a sentence or two max. Never write multi-paragraph comment blocks.
- **Incident write-ups, post-mortems, historical narrative, dated decision logs**: go in [learning.md](learning.md), not in source files. Code comments rot; learning.md is the durable record. When a fix is informed by a past incident, the comment should *point at* the learning.md section (e.g. `# See learning.md "Parent-side memory in the pipeline"`), not retell the story.
- **Operator-facing tuning notes**: belong inline near the relevant `config.ini` key, but kept short (one sentence on what it does, plus a pointer to learning.md if there's a known failure mode behind it).

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

## Pipeline-specific gotchas worth knowing about

- The processing pipeline (`code/processing_internal.py`) has been bitten more than once by parent-process memory blow-ups in the gaps *between* worker pools. Per-pool watchdogs, per-stage worker caps, and pre-flight checks all only run inside `Pool(...)` blocks - parent-side reads/merges between pools are unguarded. There is now a process-lifetime sentinel (`_start_lifetime_panic_watchdog`) as a backstop, but it costs the entire run's progress when it fires; do not rely on it to catch sloppy parent-side allocations. Before adding any read in the parent, check whether it materialises a known-large dataset (`tbl_stacked`, `tbl_flat`). See learning.md "Parent-side memory in the pipeline".
- `read_parquet_or_empty()` materialises the entire dataset including geometries. Read its docstring before calling it on anything other than small, known-bounded tables.

## Documentation in `docs/`

- `docs/MESA_User_Guide.docx` is an end-user Word manual compiled from the `mesa.wiki` repository (Home, What's new, Quickstart, User-interface, Data, Method, Indexes, QGIS, Definitions, Advanced, Troubleshooting). It targets analysts/operators, not developers. When the wiki content changes substantially, regenerate the guide so the two stay in sync.
- The guide was generated using docx-js. The build script lives in the Cowork session output folder (`build_user_guide.js`) — not yet checked in. If a future session needs to rebuild it, recreate the script from the wiki content rather than searching for the old session artefact.
- Source-of-truth ordering: `mesa.wiki` is canonical; `docs/MESA_User_Guide.docx` is a derived artefact. Do not edit the .docx by hand and expect the wiki to follow.
- `docs/technical_docs.docx` is a separate, older developer-focused doc and should not be confused with the user guide.
