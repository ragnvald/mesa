MESA **5.0.1** is a maintenance release on the MESA 5 line. It tightens the report engine, clarifies labels and documentation in the desktop launcher, and rolls in dependency security updates picked up by Dependabot. There are no data-format changes; existing MESA 5 projects work unchanged.

The Zenodo record for this build was published on 2026-05-02 and includes compiled Windows 11 binaries together with pre-processed example data from Mafia Island, Tanzania.

### Where can I find it?
Compiled versions are available here:

- [Zenodo record](https://zenodo.org/records/19984098)
- [MESA project wiki](https://github.com/ragnvald/mesa/wiki)

### Getting started
Download and unzip the MESA 5.0.1 archive from the Zenodo record into a writable folder (for example `C:\MESA`). Keep the folder structure intact and launch `mesa.exe`.

### Bug fixes
- **Report engine no longer fails partway through.** Fixed a `KeyError: '1, 2, 3, …, 25'` that aborted the Index-statistics step (literal curly braces in the Sensitivity-index intro were being parsed as `str.format` placeholders). Also fixed the report progress bar walking backwards (75 → 45 → 50 → 86) when atlas, other-maps, and index-stats were all enabled — the index-stats step now picks up where the per-geocode block actually ended, so progress is monotonic.
- **Assets – overview tables are no longer empty.** `tbl_asset_group.parquet` is metadata-only (no geo metadata) and `gpd.read_parquet` was raising and being silently caught, leaving the populated path with empty inputs. Reads now fall back to plain pandas when geopandas refuses the file. A column-rename slip on `total_area` and a column-order slip on the sensitivity-distribution table are also fixed.

### Report engine improvements
- New **Geocodes – overview** section directly after Assets – overview: one row per geocode group (`basic_mosaic`, `H3_R6` … `H3_R9`, custom sets) with title, description, and object count, plus four narrative paragraphs that explain in depth what a geocode group is, how `basic_mosaic` is constructed (asset-derived non-overlapping atomic faces), how the H3 hexagon grids relate to it, and where custom polygon sets fit in.
- The **Index statistics** chapter intro now frames the three normalised indices (Importance, Sensitivity, OWA) alongside the four supplementary per-cell indicators (Sensitivity max, Importance max, # asset groups, # asset objects). Each "Other maps" entry gained an explicit pairing with its companion index and the underlying field name in `tbl_flat.parquet`.
- The Sensitivity-index intro is now a clear **count → weight → rank** sequence with a worked example, and explains why default sensitivity weights are flat (the product values already encode magnitude). The OWA-index intro spells out the lexicographic rule with a concrete example: one overlap at sensitivity 25 outranks every cell with zero overlaps at 25.
- Dotted-line "──……──" separators left over from an earlier template iteration are gone.
- The **Create report** button now uses the shared primary-action style (gold-on-parchment) instead of the bright Bootstrap-green it used to carry — consistent with every other primary button in MESA.
- The user guide now documents the **first-run basemap-tile cost**: the first report run on a new AOI must download every OSM tile the report touches and can add minutes to tens of minutes; subsequent runs hit the local 30-day cache under `output/tile_cache/`.

### Map viewer and Parameters clarity
- **Map viewer**: layer toggle list collapses behind a **Layers** header, and each layer now carries a small **(i)** info icon that opens the matching section of the [Indexes wiki page](https://github.com/ragnvald/mesa/wiki/Indexes). Several labels were tightened: *Groups total* → **# asset groups**, *Assets total* → **# asset objects**, and the bare *Importance* toggle is now **Importance (max)**.
- **Parameters → Index weights** gained an **(i)** icon for each weight frame, a **Tuning tips** callout (filter via 0, emphasise via boost, flat by default), and an **OWA index — no tunable input** explainer so users no longer expect to find OWA controls. The misleading "(OWA)" subtitle on the sensitivity-weight block was also dropped.

### Wiki and documentation
- The **[Indexes](https://github.com/ragnvald/mesa/wiki/Indexes)** wiki page now documents the four supplementary per-cell indicators (`sensitivity_max`, `importance_max`, `asset_groups_total`, `assets_overlap_total`) alongside the three normalised indices, with comparison tables that distinguish each raw layer from its weighted counterpart.
- The User Guide DOCX is now generated in **English and Portuguese (Mozambique)** from the canonical wiki content via `devtools/build_user_guide.py`, replacing the older single-language file.

### Foundations
- **Dependency bumps for Dependabot security advisories**: Pillow 12.1.1 → 12.2.0, lxml 6.0.2 → 6.1.0, Mako 1.3.0 → 1.3.11, distributed 2023.12.0 → 2026.1.0, requests 2.32.4 → 2.33.0. None of these libraries are imported directly by MESA except Pillow and (in the GeoNode flow) requests, so runtime risk is low.
- The **About** banner now reads `5.0.1` cleanly. The "(dev `<sha>`)" suffix that used to appear when running from a git checkout has been dropped.
- `tiles_maxzoom` defaults to `13` (was `12`), giving roughly 19 m/pixel at the equator at the new top level for crisper zoomed-in MBTiles overlays.

**Full Changelog**: https://github.com/ragnvald/mesa/compare/5.0...5.0.1
