# Vendored: qdgc_py

This package is a vendored copy of the **qdgc_py** library (Quarter Degree Grid Cells).

- Upstream: https://github.com/ragnvald/qdgc (subdirectory `qdgc_py/`)
- Vendored version: `0.1.0`
- Files copied verbatim from `qdgc_py/src/qdgc_py/`: `__init__.py`, `core.py`

## Why vendored (not pip)

`qdgc_py` is not published on PyPI, and MESA ships as a frozen PyInstaller build that
must import it without network access. Vendoring the pure-stdlib source under `code/`
(already on `sys.path`) makes `from qdgc_py import ...` work in dev and frozen builds
alike, with zero transitive dependencies. This mirrors how `h3` is consumed, except h3
is a pinned pip dependency (`h3==4.2.2`) because it is on PyPI.

## Updating

Re-copy `__init__.py` and `core.py` from the upstream `src/qdgc_py/` directory and bump
the version note above. Do not hand-edit the vendored source; fix upstream and re-vendor.
