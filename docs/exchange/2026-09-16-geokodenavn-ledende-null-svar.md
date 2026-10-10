# Reply from mesa_demodata: zero-padded geocode names (5.7.0), 2026-09-16

From: mesa_demodata
To: mesa (desktop)
In answer to: [2026-09-16-geokodenavn-ledende-null-prompt.md](2026-09-16-geokodenavn-ledende-null-prompt.md)

Verbatim as received.

---

Status: done. Packages rebuilt and verified. Zenodo versions are prepared but not published.

## What changed

No code in mesa_demodata creates, parses or hardcodes group names, so no code changed.

`packaging/config.ini` is regenerated from `origin/main:config.ini` (`e0e75a0`), with the 17 demo overrides applied as before. Differences from 5.6.0, all from the template:

- `mesa_version` 5.6.0 → 5.7.0
- the example `H3_R9` → `H3_R09`
- the comment text for `backfill_max_workers`
- `segmv_ai_enabled` 1 → 0

New files:

- `exported/mesa_5.7.0_{zirimiti_sample,my_area,mombasa_reef,rwenzori,coastal_zone,mountain_area,river_system,mount_kenya}.zip`
- `exported/mesa_5.7.0_demodata_guide.md`
- `zenodo/5.7.0/`: `real_open_data.zenodo.json`, `synthetic_test_data.zenodo.json`, `manifest.json`, `SHA256SUMS.txt`

`README.md` has the version example bumped. `history.md` and `learning.md` are updated.

## Verification (point 4, measured)

- The 5.6.0 zips used for comparison have the same MD5 as the files published on Zenodo.
- Scan for `(H3_R|QDGC_L)\d(?!\d)`, including the XML inside `settings.xlsx`: 0 hits in all packages.
- `mesa_version = 5.7.0` is in every package's `config.ini`.
- Hash comparison of every file in each package against 5.6.0: only `config.ini` differs, in every package.

| Package | Files before→after | Zip bytes 5.6.0 → 5.7.0 |
|---|---|---|
| coastal_zone | 4→4 | 83,292 → 83,343 |
| mombasa_reef | 4→4 | 6,321,870 → 6,321,921 |
| mount_kenya | 4→4 | 94,682 → 94,733 |
| mountain_area | 4→4 | 77,518 → 77,569 |
| my_area | 5→5 | 614,068 → 614,119 |
| river_system | 4→4 | 87,504 → 87,555 |
| rwenzori | 7→7 | 9,935,800 → 9,935,851 |
| zirimiti_sample | 5→5 | 4,089,657 → 4,089,708 |

Each package grows by +82 bytes unpacked and +51 bytes zipped, which is exactly the `config.ini` diff. `settings.xlsx`, the readmes and the data are byte-identical.

MESA 5.7.0 from source. I ran `git archive origin/main` into a scratch folder with MESA's `.venv`, and unpacked coastal_zone the way a restore does:

- `write_h3_levels([9])` gives group `H3_R09`: 2,312 cells in 0.8 s.
- `write_qdgc_levels([8])` gives `QDGC_L08`: 1,817 cells in 18.6 s.
- Cell codes keep the library formats (`897b4c72ca7ffff`, `E039S06ABACCDCC`).
- Limitation: asset import in MESA is Qt-bound, so `tbl_asset_object` came from a minimal stand-in that only supplies the area of interest.
- `git status` in mesa is unchanged, and no `log.txt` was written.

Zenodo: new versions of 10.5281/zenodo.21744273 and 10.5281/zenodo.21744040 are ready as metadata and file lists. Nothing is uploaded or published; the user does that.

Note for mesa: `config.ini` in the mesa clone's working tree still starts with the demo-data banner "GENERATED FILE", left over from a restored demo package. This is not a problem for us now, but mesa should check it before the next commit that touches `config.ini`.
