# Counting MESA's Python lines of code — guide

For an assistant with filesystem access to this repository (`d:\code\mesa`) that
needs to reproduce a "lines of Python code, excluding libraries" count.

## 1. Scope — what counts as MESA source

**Include** (MESA's own Python):

| Group | Path |
|---|---|
| Application | `mesa.py` (repo root) + everything under `code/`, **except** the vendored `code/qdgc_py/` |
| Dev tooling | `devtools/` |
| Tests | `tests/` |

**Exclude** — these are libraries or not MESA source, and counting them inflates or double-counts:

- **Virtual environments:** any directory whose name starts with `.venv` — `.venv/`, `.venv_compile/`, `.venv314/`. ⚠️ `.venv314` is a **symlink** to another environment; do not follow it.
- **Bytecode caches:** `__pycache__/`.
- **Git worktrees & agent data:** `.claude/` — the worktrees under `.claude/worktrees/` are **full copies of the repo**, so including them double-counts everything.
- **Vendored library:** `code/qdgc_py/` — a copied third-party package (Quarter Degree Grid Cells), not MESA code. Report it separately if the vendored total is wanted; otherwise leave it out.

## 2. Two metrics

- **Physical lines** — every line in each `.py` file (the usual "LOC" figure).
- **Code lines** — physical lines minus blank lines and lines whose first non-whitespace character is `#`. This is a simple, reproducible heuristic; it does **not** strip docstrings or trailing inline comments, so it is a lower bound on "real" code, not a tokenizer-accurate SLOC.

Report both; lead with physical lines.

## 3. Method A — PowerShell (this machine)

Run from the repo root. **Group by the path relative to the repo root**, never by an
absolute-path substring: the repo lives under `d:\code\mesa`, so a naive
`-match '\\code\\'` matches *every* file (via the `d:\code` prefix) and folds
`devtools/` + `tests/` into the application group. Count lines with
`[System.IO.File]::ReadAllLines(...).Length`, not `Measure-Object -Line` (the latter
undercounts).

```powershell
cd d:\code\mesa
$base = (Get-Location).Path + '\'
$files = Get-ChildItem -Recurse -Filter *.py -File | ForEach-Object {
  $rel = $_.FullName.Substring($base.Length)
  if ($rel -match '^\.venv' -or $rel -match '\\__pycache__\\' -or $rel -match '^\.claude\\') { return }
  [pscustomobject]@{ Full = $_.FullName; Rel = $rel }
}
function grp($rel){
  if     ($rel -like 'code\qdgc_py\*')            { 'vendored' }
  elseif ($rel -like 'devtools\*')                { 'devtools' }
  elseif ($rel -like 'tests\*')                   { 'tests' }
  elseif ($rel -like 'code\*' -or $rel -eq 'mesa.py') { 'app' }
  else   { 'other' }
}
$g = @{}
foreach ($f in $files) {
  $k  = grp $f.Rel
  $ls = [System.IO.File]::ReadAllLines($f.Full)
  $code = ($ls | Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') }).Count
  if (-not $g[$k]) { $g[$k] = [int[]]@(0,0,0) }
  $g[$k][0]++; $g[$k][1] += $ls.Length; $g[$k][2] += $code
}
$oF=0; $oL=0; $oC=0
foreach ($k in 'app','devtools','tests','other') {
  if ($g[$k]) { "{0,-10} files={1,4} lines={2,7} code={3,7}" -f $k,$g[$k][0],$g[$k][1],$g[$k][2]
                $oF+=$g[$k][0]; $oL+=$g[$k][1]; $oC+=$g[$k][2] } }
"{0,-10} files={1,4} lines={2,7} code={3,7}" -f 'MESA own',$oF,$oL,$oC
if ($g['vendored']) { "{0,-10} files={1,4} lines={2,7} code={3,7}  (excluded)" -f 'vendored',$g['vendored'][0],$g['vendored'][1],$g['vendored'][2] }
```

## 4. Method B — portable (Python), if PowerShell is unavailable

Prunes excluded directories during the walk (so symlinked venvs and worktrees are
never descended). Same grouping and metrics as Method A.

```python
import os
BASE = r"d:\code\mesa"
groups = {"app": [0,0,0], "devtools": [0,0,0], "tests": [0,0,0], "vendored": [0,0,0]}

def measure(path):
    phys = code = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            phys += 1
            s = line.strip()
            if s and not s.startswith("#"):
                code += 1
    return phys, code

for dp, dirs, files in os.walk(BASE):  # os.walk does not follow symlinks by default
    dirs[:] = [d for d in dirs
               if d != "__pycache__" and d != ".claude" and not d.startswith(".venv")]
    for f in files:
        if not f.endswith(".py"):
            continue
        rel = os.path.join(dp, f).replace("\\", "/")
        if "/qdgc_py/" in rel:      g = "vendored"
        elif "/devtools/" in rel:   g = "devtools"
        elif "/tests/" in rel:      g = "tests"
        else:                       g = "app"   # mesa.py + code/* (and any other root .py)
        p, c = measure(os.path.join(dp, f))
        groups[g][0] += 1; groups[g][1] += p; groups[g][2] += c

own = [sum(v[i] for k, v in groups.items() if k != "vendored") for i in range(3)]
for k in ("app", "devtools", "tests"):
    n, p, c = groups[k]; print(f"{k:10} files={n:4} lines={p:7} code={c:7}")
print(f"{'MESA own':10} files={own[0]:4} lines={own[1]:7} code={own[2]:7}")
n, p, c = groups["vendored"]
print(f"{'vendored':10} files={n:4} lines={p:7} code={c:7}   (excluded from MESA own)")
```

## 5. Baseline — verify against this

As of **2026-07-18**, commit **`07f894f`** (branch `main`). Cross-checked: Methods A
and B produce these figures identically.

| Group | Files | Lines | Code |
|---|---|---|---|
| Application (`mesa.py` + `code/`, excl. `qdgc_py`) | 24 | 43,709 | 36,404 |
| `devtools/` | 9 | 4,783 | 3,798 |
| `tests/` | 1 | 151 | 125 |
| **MESA own (excl. libraries)** | **34** | **48,643** | **40,327** |
| vendored `qdgc_py` (excluded) | 2 | 728 | 567 |

If your total differs materially, check the usual causes: a `.venv*` directory not
excluded, `.claude/worktrees/` (repo copies) counted, the `.venv314` symlink
followed, or — the trap that first bit this guide — grouping on an absolute-path
substring so the `d:\code` prefix pulls `devtools/`/`tests/` into the application
group. The count drifts as the code changes — treat these as a reference point,
not a fixed value; re-run against the current tree and note the commit.

## 6. Notes

- Run from the repo root; the grouping rules key off relative paths (`code/`, `devtools/`, `tests/`, `qdgc_py/`).
- Line-of-code counts are newline-based and unaffected by CRLF vs LF or file encoding.
- For a tokenizer-accurate SLOC (docstrings/inline comments handled properly), use a dedicated tool such as `cloc --exclude-dir=.venv,.venv_compile,.venv314,__pycache__,.claude,qdgc_py .` — expect its "code" figure to sit a little below the physical-line total.
