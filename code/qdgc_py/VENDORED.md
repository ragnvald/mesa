# Vendored: qdgc_py

This package is a vendored copy of the **qdgc_py** library (Quarter Degree Grid Cells).

- Upstream: https://github.com/ragnvald/qdgc (subdirectory `qdgc_py/`)
- Vendored version: `0.1.0`
- Files copied verbatim from `qdgc_py/src/qdgc_py/`: `__init__.py`, `core.py`

## Why vendored (not pip)

`qdgc-py` is now published on PyPI, but MESA 5 deliberately keeps the vendored copy
rather than switching to the pip package. MESA ships as a frozen PyInstaller build that
must import it without network access. Vendoring the pure-stdlib source under `code/`
(already on `sys.path`) makes `from qdgc_py import ...` work in dev and frozen builds
alike, with zero transitive dependencies.

A pip migration — mirroring how `h3` is pinned as a PyPI dependency (`h3==4.2.2`) — is on
the table for a future release, but not v5. Until then, updates flow by manual
re-vendoring (re-copy the files and bump the version note below).

## Updating

Re-copy `__init__.py` and `core.py` from the upstream `src/qdgc_py/` directory and bump
the version note above. Do not hand-edit the vendored source; fix upstream and re-vendor.
