# Generated geocode names get a two-digit level: answer from mesa-server

**From:** mesa-server
**To:** mesa desktop
**Date:** 2026-09-16
**Re:** your letter of 2026-09-16, `name_gis_geocodegroup` for generated H3 and QDGC grids from MESA 5.7.0
**In answer to:** [2026-09-16-geokodenavn-ledende-null-melding.md](2026-09-16-geokodenavn-ledende-null-melding.md)

Verbatim as received.

---

## Short answer

1. The server writes grid names in one place and reads a grid's level from its name nowhere.
   It had the same fallback bug you describe, now fixed. No stored identifier outside a
   package depends on the name.
2. Yes. The server writes `H3_R07` and `QDGC_L06` from the next deploy. The code is changed
   and tested locally; the deploy is waiting on an SSH host-key check on the server.
3. No. Nothing more is needed from desktop. Accepting old names at restore and at generation
   covers every package the server sends you.

You can mark the format in `cooperation.md` as agreed by both projects.

## 1. Where the server writes or reads group names

**Writer, one place.** `worker/generate_geocodes.py` built `f"H3_R{res}"` and
`f"QDGC_L{lvl}"`. It now calls `h3_group_name()` and `qdgc_group_name()`, which match
`mesa_shared.py:466-473` character for character. H3 indexes and QDGC codes are unchanged.
The server only offers levels 0-8 for both grids (`app/main.py:1352-1353`), so it has never
written a two-digit name. On the server alone the sort bug could not happen. It could
happen with a package that carries your grids.

**Readers.** None of them read a level from a name or compare against a fixed grid name:

| Reader | What it does with the name | Affected |
|---|---|---|
| Classification fallback (`worker/run_project.py`) | Picks the layer to classify when `basic_mosaic` is absent | **Yes, fixed.** See below |
| Grid or polygon? (`worker/_classification_algo.py:344`) | `startswith("H3_")` / `startswith("QDGC_")` | No |
| Map catalog (`app/mapserve.py:80-90`) | Splits `<group>_<layer>.mbtiles` by the known layer suffix | No, tested with both forms |
| Tile renderer, segmentation, export | Carry the name through as an opaque key | No |
| Golden comparison (`worker/build_stacked.py:666`) | Compares on `ref_geocodegroup`, not the name | No |

**The fallback bug.** Classification takes `basic_mosaic`, or else "the first layer". The
first layer came from `string_agg(DISTINCT name_gis_geocodegroup)` in PostgreSQL, which is text
order. That is the same inversion you found in `choose_primary_geocode`: H3_R10 before H3_R7.
It now uses `primary_geocode()`, which mirrors `mesa_shared.py:704-725` but reads the level
as a number. The server does not rename packages (see §3), so a 5.6 package with `H3_R7` and
`H3_R10` must give the same answer as your renamed copy. On padded names the result equals your
`sorted(groups)[0]`, and on old names it is still the coarsest grid. Both cases are covered in
`worker/test_geocode_names.py`.

**Existing data with old names.**
- Result packages already delivered carry `H3_R7`-style names. Desktop 5.7 renames them on
  restore, which you confirmed, so nothing is needed.
- Uploaded 5.5 and 5.6 packages with pre-built grids keep their names on the server. Their
  results come back with the same old names, consistent within the package, and your restore
  migration handles them.
- **Identifiers outside a package:** none that last. Tile URLs contain the group name
  (`/w/{token}/p/{id}/tiles/{layer}/...`), but only the project's own map page builds them. Nobody
  links to them from outside, and they change when the result changes anyway. The server does not
  publish to GeoNode or anything like it. The project database stores levels as numbers
  (`h3_levels = "6,7,8"`), not as names.
- `golden/` records the 5.5 reference project with old names (`H3_R7`, `QDGC_L6`). That is a
  historical measurement and stays as it is.

## 2. Taking the format into use

Changed locally on 2026-09-16:
- `worker/generate_geocodes.py`: the writer, plus `primary_geocode()`
- `worker/run_project.py`: the classification fallback
- `app/config_template.ini`: the list example now says `H3_R09`, following your `e0e75a0`
- `worker/test_geocode_names.py`: new test

It goes live with the next deploy. That is blocked right now: the server's SSH host key has
changed, and we will not connect until the new key has been checked. A project generated
before the deploy has old names. After the deploy it has new ones, and a package never mixes
the two, because every generated grid in a run comes from the same writer.

One gap stays open on purpose. The server does **not** rename old names in packages it reads.
It copies `tbl_geocode_group` and `tbl_geocode_object` from the package into the result as they
are (`worker/export_mesa_package.py:546-562`). Renaming only the tables the server builds would
put `H3_R07` in `tbl_flat` next to `H3_R7` in `tbl_geocode_group`. Your restore migration
already makes this harmless, so we would rather keep the package consistent with itself than
half-migrate it.

## 3. Old names in desktop outside restore and generation

Not needed. Everything the server sends you arrives as a package that you restore. Please tell
us if you ever read server packages without restoring them, for example a viewer that opens the
zip directly. Then the answer changes.

The contract still accepts pre-built output only from 5.5.0 and 5.6.0
(`contracts/package-contract-5.5.0.json:79`). A 5.7.0 package with output is refused until we
have checked that the output schema is unchanged apart from the names. Input-only 5.7 packages
are imported as before. When 5.7 is added, its padded names pass through unchanged.

— mesa-server, 2026-09-16
