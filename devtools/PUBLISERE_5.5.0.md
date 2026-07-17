# Publisere MESA 5.5.0 — arbeidsdokument

Arbeidsdokument for utgivelsen av 5.5.0 (kompilert Windows-utgave → Zenodo → GitHub).
Ligger i `devtools/` med vilje: `build_all.py` fjerner hele mappa fra distribusjonen,
så dette dokumentet følger aldri med ut til brukerne.

Status per 2026-07-17. Oppdater etter hvert som punktene lukkes.

---

## 1. Status

| Ting | Status |
|---|---|
| `mesa_version = 5.5.0` i config.ini | ✅ commit `0336f1b` |
| Programvaren rapporterer 5.5.0 | ✅ verifisert i alle tre kodestier (se §2) |
| Kanonisk config.ini gjenopprettet | ✅ demo-configen var skrevet over den |
| Grønn frozen build på Python 3.14 | ❌ **aldri kjørt** — se §3.1 |
| Zenodo-post | ⬜ |
| GitHub-release | ⬜ (krever Zenodo først) |
| Brukerveiledning oppdatert | ⬜ sier fortsatt 5.2 — se §3.2 |

Hopper 5.3/5.4 → 5.5.0. Begge ble bumpet i config, men aldri tagget eller utgitt.
Siste tag er `5.2` (2026-06-11). 64 commits i `5.2..HEAD`.

---

## 2. Versjon — hvordan MESA vet hva den heter

`config.ini` `[DEFAULT] mesa_version` er **eneste sannhet**. Ingen hardkodede kopier
i programvaren. Verifisert at alle tre lesestier gir `5.5.0`:

- `mesa.py` → banner, About-dialogen, `tbl_project_info`
- `mesa_shared.mesa_version_label()` → OSM tile-proxyens User-Agent (`analysis_setup`, `line_manage`, `special_focus`)
- `processing_internal._mesa_version_label()`

`build_info.json` genereres av bygget og skal **aldri** redigeres for hånd.
Pakket viser banneret `5.5.0 Build <dato> <tid>`.

**Unntak:** `devtools/build_user_guide.py:30` har sin egen `VERSION = "MESA 5.2.0"`.
Det er en dok-konstant, ikke programvaren. Se §3.2.

---

## 3. Blokkere og risiko — les før du bygger

### 3.1 Frozen build på Python 3.14 er aldri validert 🔴

Dette er den store. 3.14-migreringen er ferdig og validert **fra kildekode**, men
ingen har noensinne fått en grønn PyInstaller-build på 3.14.

- `requirements_compile_win.txt:12-15` lar `pyinstaller` og `pyinstaller-hooks-contrib`
  stå **upinnet**, med kommentaren: *"pin once a full build_all.py run on 3.14
  confirms a working version"*.
- `docs/further_development.md` A3: *"Not yet: … a green PyInstaller frozen build on 3.14."*
- `build_win_log.txt` er fra **Python 3.11.6** og `--collect-all fiona` — altså
  fra før pyogrio-migreringen. Den ~9m46s-tiden er en pekepinn, ikke en baseline.

**Konsekvens:** første `compile_win_11.bat` på 3.14 er et eksperiment, ikke en
rutinebygging. Sett av tid. Når den går grønn: **pinn pyinstaller-versjonen** i
`requirements_compile_win.txt` og noter versjonen her.

### 3.2 Brukerveiledningen sier fortsatt 5.2 🟠

`devtools/build_user_guide.py` har `VERSION = "MESA 5.2.0"` *og* prosa som nevner
"MESA 5.2" ~20 steder (inkl. et helt kapittel "2. What's new in MESA 5.2", på både
engelsk og portugisisk). Å bumpe kun konstanten gir en veiledning som er
selvmotsigende.

Valg: (a) skriv om kapittel 2 til 5.5.0-innhold og bump konstanten, (b) la
veiledningen stå på 5.2 og si det i release notes, (c) dropp veiledningen fra denne
utgivelsen. **Ikke avgjort.**

Merk også at `build_user_guide.py` ikke kalles av bygget og krever at `../mesa.wiki`
finnes som søsken-checkout.

### 3.3 `docs/` shipper rått — interne planer lekker ut 🟠

`build_all.py:864` kopierer hele `docs/`. `DEVELOPER_ONLY_FILES` (linje 931) fjerner
`CLAUDE.md`, `cooperation.md`, `instructions.md`, `learning.md` — men **ikke**
planleggingsdokumentene i `docs/`. Disse havner hos sluttbrukerne:

- `docs/further_development.md` (inneholder «ikke gjort»-lista vår)
- `docs/SEGMENTATION_INTEGRATION_PLAN.md`
- `docs/SCALABLE_PROCESSING_PLAN.md`
- `docs/CLOUD_PROCESSING_SERVER_PLAN.md`
- `docs/basic_mosaic_capacity.md`

**Avgjør før bygg.** Kapasitetsdokumentet kan være greit å dele; roadmap-ene er
neppe ment for publikum.

### 3.4 `output/` shipper rått — din egen cache blir med 🟠

`build_all.py:864` kopierer også `output/`. Akkurat nå: **111 filer / 20,6 MB**,
mesteparten `output/cache/osm_tiles/` — basiskart-fliser cachet fra dine egne
økter, pluss `processing_tuning_backup.json` og `area_stats.json`.

Backup-eksporten ble ryddet i `7318b8e` (basiskart-cache og runtime-lock aldri med),
men **bygget har ikke fått samme rydding**. Rydd `output/` før bygg, eller bestem
hva som skal være med.

### 3.5 `input/`-readme-ene forsvinner ved hver demo-restore 🟡

Kjent, dokumentert i learning.md. `restore_backup_archive` gjør ubetinget
`rmtree(input/)`, og `input/` inneholder tre sporede readme-filer som forklarer
drop-mappene. Hver demo-restore sletter dem. De må være på plass før bygg:

```
git status --porcelain input/     # skal være tom (utenom demodata)
```

### 3.6 Demo-config kan overskrive den kanoniske 🟡

Demo-pakken inneholder nå sin egen `config.ini` og **erstatter** repoets når du
restaurerer demodata inn i utviklingsrepoet. Det skjedde 2026-07-17 kl. 10:58.
Demo-configen slår bl.a. `segmv_ai_enabled` **på** (krever OpenAI-nøkkel) og mister
alle operatør-kommentarene.

**Sjekk alltid før bygg:**

```powershell
git diff --stat -- config.ini        # skal være tom
Get-Content config.ini -TotalCount 1 # skal være: "# In general - Do not make changes..."
```

Er første linje `# MESA demo data — config.ini`, har demo-restoren tatt den.
Gjenopprett: `git restore --source=HEAD -- config.ini`

Demo-pakkens egen header sier det rett ut: *"Restore demo data into a dedicated MESA
copy, or expect your own tuning to be overwritten."* Vurder å teste demodata i en
egen MESA-kopi framfor i repoet.

---

## 4. Slik bygges en kompilert utgave

Rekonstruert fra `README.md:155-179`, `instructions.md:130`, `learning.md:97,348-358`
og `devtools/`. Det finnes **ingen samlet release-skript** — dette er tre verktøy
pluss manuelle steg.

### Steg 0 — versjon
Allerede gjort (`0336f1b`). Kun `config.ini`.

### Steg 1 — miljøer
```
devtools\setup_venvs.bat
```
Lager `.venv` (utvikling, `requirements_py314_win.txt`) og `.venv_compile`
(bygg, `requirements_compile_win.txt`) med `py -3.14`.

### Steg 2 — bygg
```
devtools\compile_win_11.bat
```
Kanonisk inngang. Tvinger full ren build (`MESA_BUILD_MAIN=1`, `MESA_BUILD_HELPERS=1`,
`MESA_BUILD_CLEAN=1`), 4-veis parallell som standard. Python velges via
`MESA_COMPILE_PYTHON` → `.venv_compile` → `.venv` → PATH. Kaller `build_all.py`.

Valgfrie argumenter: `fast` (hopp over clean), `serial`, `parallel N`.

> Alltid full build — aldri kun helpers.

### Steg 3 — brukerveiledning (manuelt, ikke koblet til bygget)
```
python devtools\build_user_guide.py
```
Se §3.2 først.

### Steg 4 — zip + Zenodo (manuelt)
Se §6.

### Steg 5 — GitHub-release (etter Zenodo)
Se §7.

### Hva som produseres

Alt havner **utenfor repoet**, på D:-roten:

| Sti | Innhold |
|---|---|
| `D:\build` | PyInstaller work-dirs + genererte `.spec`-filer |
| `D:\dist\mesa` | `FINAL_DIST` — mappa som skal shippes |
| `D:\dist\mesa\tools` | helper-`.exe`-ene |
| `D:\dist\build_history.log` | append per bygg; ligger ett nivå over `FINAL_DIST` nettopp for å overleve `rmtree(FINAL_DIST)` |

Ingen `.spec`-filer er sjekket inn — alt er CLI-flagg-drevet og spec-ene genereres
inn i `D:\build\helper_specs`.

**Kjørbare filer:** `mesa.exe` + fire helpers i `tools\`:
`tiles_create_raster.exe`, `combined_map.exe`, `special_focus.exe`,
`segmentation_setup.exe`, `segmentation_run.exe`.

Sju tidligere helpers kjører nå **in-process i `mesa.exe`** via hidden imports:
`geocode_manage`, `asset_manage`, `atlas_manage`, `processing_setup`,
`processing_pipeline_run`, `report_generate`, `analysis_present`.

Helpers er `--onefile`; hovedprogrammet er `onedir` og flates opp i `FINAL_DIST`
slik at `mesa.exe` står ved siden av `config.ini`.

**Ingen kodesignering.** Usignerte PyInstaller-exe-er. Regn med
SmartScreen-advarsel hos nedlastere.

**Ingen CI.** `.github/` inneholder kun `copilot-instructions.md`.

---

## 5. Hva pakken skal inneholde

Bygget kopierer `qgis/ docs/ input/ output/ system_resources/` + `config.ini`,
skriver `build_info.json`, fjerner `devtools/`, og stripper utviklernotatene.
`secrets/` er bevisst utelatt.

### Demodata (jinja) — skal med

Ligger på disk nå og blir plukket opp automatisk siden bygget kopierer `input/`:

| Fil | Størrelse | Sporet i git? |
|---|---|---|
| `input/asset/jinja_sample.gpkg` | 7,88 MB | Nei — `.gitignore:18` (`*.gpkg`) |
| `input/evaluate_landuse/ESA_WorldCover_10m_2021_jinja_sample.tif` | 0,27 MB | Nei — usporet |
| `input/settings.xlsx` | 0,01 MB | Nei — usporet |

⚠️ **Ingen av dem er i git.** De finnes kun på denne maskinen. En ren klone bygger
uten demodata. Kilden er `D:\data\mesa_demodata\exported\jinja_sample.zip`.
Bekreft at de er på plass rett før bygg.

Datasettet: 8 MULTIPOLYGON-features polygonisert fra ESA WorldCover 10m,
~494k vertekser, EPSG:4326. Håndterbart — basic_mosaic bruker ~10 s på det.

---

## 6. Zenodo

Zenodo er **først** og er sannhetskilden: tittel, DOI, dato, beskrivelse og filnavn
hentes derfra av GitHub-skriptet.

Det finnes **ingen** `.zenodo.json` eller `CITATION.cff` i repoet, og ingen skript
som zipper `D:\dist\mesa`. Alt i dette steget er manuelt:

1. Rydd `D:\dist\mesa` etter §3.3/§3.4.
2. Zip `D:\dist\mesa` → navngi konsistent med tidligere utgivelser.
3. Last opp til Zenodo, community `mesatool` (https://zenodo.org/communities/mesatool/).
4. Tittel må inneholde `5.5.0` — GitHub-skriptet utleder taggen fra tittelen.
5. Publiser og noter record-id + DOI her: `_____________`

---

## 7. GitHub-release

**Binærfiler publiseres ikke på GitHub.** `github_release_from_zenodo.py:410-429`
kaller `gh release create` med tag/repo/target/title/notes — **uten** asset-argumenter.
Releasen er en tag + markdown-notat som lenker til Zenodo og wikien. Nedlastinger bor
på Zenodo.

```powershell
python devtools\github_release_from_zenodo.py <zenodo-record-id>            # forhåndsvis
python devtools\github_release_from_zenodo.py <zenodo-record-id> --publish  # opprett
```

- Tag utledes fra Zenodo-tittelen → `5.5.0`. Tag-stil er bar: `5.2`, `5.0.3`.
- Prerelease auto-detekteres fra alpha/beta/rc i taggen — ikke aktuelt her.
- Changelog bygges fra `git log --no-merges --format=%s` siden forrige tag,
  klassifisert i fire bøtter. **Merk:** siden 5.3/5.4 aldri ble tagget, spenner
  `5.2..HEAD` over 64 commits — mer enn én utgivelse. Les gjennom det auto-genererte
  notatet før `--publish`.

---

## 8. Utkast til release notes (engelsk)

Skrevet mot de fem begrunnelsene. **Hver påstand er sjekket mot målte tall i
learning.md/docs.** Se §9 for hva vi ikke kan påstå.

> ### MESA 5.5.0
>
> A performance, memory and platform release. Classification becomes a first-class
> part of the workflow, intersect is measurably ~3× faster, and the whole stack moves
> to Python 3.14.
>
> **Classification is now part of the pipeline**
> The classification engine shipped in 5.2 as a standalone tool you ran by hand after
> processing. In 5.5.0 it is integrated: it runs as a stage inside Process (between
> Data and Tiles), renders on the maps as Types and Certainty rasters with
> hover-identify, appears in reports with area charts and interpretive text, exports
> to QGIS as both raster and vector, and can span several geocode layers in one run.
> Setup moved to Configure. AI-generated descriptions are optional and off by default.
>
> **A configurable AI connection**
> Config gains an AI connection panel (OpenAI or Ollama). The token is stored in
> `secrets/ai_connection.parquet`, survives Clear output, and stays outside the
> default backup set. It powers classification descriptions, report prose and
> title-aware map styling — each with a deterministic fallback when no AI is
> configured.
>
> **Intersect: 3.18× faster, 21× less memory per worker**
> Intersect workers no longer each hold the full asset layer; the parent ships a
> per-chunk asset subset instead. Measured on a full pipeline run (3.5M assets,
> basic_mosaic, Python 3.14): per-worker RAM fell from 5.76 GB to ~0.27 GB, which
> lifted the RAM-derived worker cap from 3 to 10 and cut intersect wall-clock from
> **9.6 hours to 3.02 hours**. Output is byte-identical to the old run — 1,387 parts
> / 91,083,233 rows, zero errors across all 1,992 chunks.
>
> **basic_mosaic: graceful limits instead of out-of-memory crashes**
> A new pre-flight memory gate estimates peak RAM and skips basic_mosaic with a clear
> message rather than dying mid-run. It scales with the host, so high-RAM machines are
> never blocked. Indicative ceilings: ~1.3M assets on 16 GB, ~3M on 32 GB, ~6M on
> 64 GB, ~11M on 128 GB. Beyond that, use H3 or QDGC grids. We also diagnosed
> basic_mosaic's dominant cost as process-spawn overhead rather than geometry
> computation (~87% of a measured 9h56m reference run on 3.5M assets) and removed the
> per-pair worker respawn; post-change wall-clock measurement is pending.
>
> **Fixes that prevented data loss**
> - Importing geocodes silently deleted every other geocode group. One import removed
>   13.3M H3 objects and 83k QDGC objects. Imports now merge, and import is a separate
>   tab from manage so delete no longer sits beside it.
> - Restoring a backup deleted `config.ini` even when the archive had none to put back,
>   leaving MESA unable to start.
> - A lines- or analysis-only run wiped every map tile.
> - Two "Process all" windows could run against one project, racing outputs.
> - basic_mosaic could appear to hang indefinitely.
>
> Plus fixes to silently-ignored settings (inline `#` comments made pre-flight values
> unparseable), report sections that were always skipped, map ramps dominated by a
> single outlier cell, and certainty maps drawn with white holes.
>
> **Python 3.11 → 3.14**
> Windows and macOS both target CPython 3.14. numpy 2.5, pandas 3.0, shapely 2.1,
> pyogrio 0.12, PySide6 6.11, scikit-learn 1.9. fiona is gone — it has no cp314 wheel;
> pyogrio bundles GDAL and replaces it. If you maintain your own environment: pyogrio
> ≥ 0.8 and openpyxl ≥ 3.1.5 are hard requirements.

---

## 9. Hva vi IKKE kan påstå

Sjekket mot kildene. Ikke la disse snike seg inn i notatene.

| Ikke påstå | Hvorfor |
|---|---|
| «Ny AI-klassifisering» | Motoren shippet i 5.2. Det nye er *integrasjonen*. |
| «Mosaic fra timer til minutter» | `docs/basic_mosaic_capacity.md:64-65` sier ordrett at dette er en **forventning**, og at faktisk tid «is pending a fresh full run — to be filled in here once measured». Ingen måling etter endringen finnes. |
| Et tall på mosaic-forbedringen | Se over. Den ene benchmarken som finnes (`SCALABLE_PROCESSING_PLAN.md:53-66`) viser at config-knappene gir 0,61×–1,09× — altså at de *ikke* hjelper. |
| «Minne-watchdog i mosaic» | To-tiers-watchdogen er fra før 5.2 og er per `further_development.md:77-78` fortsatt **ikke** koblet til mosaic-poolen. |
| «Polygonize bruker mindre minne» | 21,7 GB-toppen er uendret — kun *beskyttet* av pre-flight-gaten. |
| «Ny settings-store» | `492969e` er Phase 1 og eksplisitt uten produksjonseffekt. Phase 2 er ikke gjort. |
| «Fullt validert på 3.14» | Lines og analysis ble aldri nådd i valideringskjøringen, og frozen build er uprøvd (§3.1). |
| De gamle 2,3× / ~4–4,5 t-tallene | Superseded av `learning.md:757`. Bruk 3,18× / 3,02 t. |

---

## 10. Åpne beslutninger

1. **§3.2** Brukerveiledning: skriv om til 5.5.0, la stå på 5.2, eller dropp?
2. **§3.3** Hvilke `docs/`-filer skal ut av distribusjonen?
3. **§3.4** Skal `output/` shippes i det hele tatt, og i så fall hva?
4. **§6** Filnavn/tittel på Zenodo-posten.
5. Skal demodataene sjekkes inn (i dag kun på denne maskinen)?

---

## 11. Rekkefølge

```
[ ] Rydd docs/ og output/ etter §3.3/§3.4
[ ] Bekreft config.ini er kanonisk + 5.5.0        (§3.6)
[ ] Bekreft input/-readmene er på plass           (§3.5)
[ ] Bekreft jinja-demodata ligger i input/        (§5)
[ ] devtools\setup_venvs.bat
[ ] devtools\compile_win_11.bat                   ← første 3.14-build, sett av tid (§3.1)
[ ] Pinn pyinstaller-versjonen når bygget er grønt
[ ] Røyktest D:\dist\mesa\mesa.exe — banner skal si 5.5.0 Build <dato>
[ ] Kjør demodataene gjennom i den kompilerte utgaven
[ ] python devtools\build_user_guide.py           (avhenger av §3.2)
[ ] Zip D:\dist\mesa
[ ] Last opp til Zenodo, noter record-id + DOI
[ ] python devtools\github_release_from_zenodo.py <id>
[ ] Les gjennom auto-changelogen (spenner 64 commits)
[ ] python devtools\github_release_from_zenodo.py <id> --publish
[ ] git tag 5.5.0 + push (du styrer pushing)
```
