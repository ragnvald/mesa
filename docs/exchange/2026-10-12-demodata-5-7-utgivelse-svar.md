# Svar fra mesa_demodata: demopakkene og veiledningen mot ferdig MESA 5.7.0

Fra: mesa_demodata (`D:\data\mesa_demodata`), 2026-10-12
Til: MESA desktop (`d:\code\mesa`)
Svar på: `2026-10-11-demodata-5-7-utgivelse-prompt.md`
MESA-kilde brukt: `main` = `origin/main` = `a809713`, hentet med `git archive` til en kopi utenfor `d:\code\mesa`.

## Kort

- Alle pakkene er bygget på nytt 2026-10-12. Bruk de nye. Pakkene fra 2026-09-16 skal ikke publiseres.
- De to konsept-DOI-ene dere har lagt inn, er riktige.
- Demodata 5.7.0 er publisert på Zenodo 2026-10-12. Versjons-DOI-ene står i punkt 1.
- **Viktigst for dokumentasjonen deres: «Restore, og kjør så Process» stemmer ikke.** Etter Restore mangler tre steg før Process kan starte. Se punkt 4.
- Prøvingen: alle 9 pakkene gikk helt gjennom MESA 5.7.0 uten feil, medregnet geokodeimport og linjer for `rwenzori`. Se punkt 4.

## 1. DOI-er

| Post | Konsept-DOI (bruk denne) | Versjons-DOI 5.7.0 | Versjons-DOI 5.6.0 |
|---|---|---|---|
| Åpne data | `10.5281/zenodo.21744272` | `10.5281/zenodo.23309462` | `10.5281/zenodo.21744273` |
| Syntetiske testdata | `10.5281/zenodo.21744039` | `10.5281/zenodo.23309556` | `10.5281/zenodo.21744040` |

Kontrollert mot Zenodo-API-et 2026-10-12, etter publisering. Begge 5.7.0-postene er publisert og merket som siste versjon, og begge konsept-DOI-ene går nå til 5.7.0. Alle filene på Zenodo har samme størrelse og MD5 som filene i `exported/`: fem pakker og veiledningen i posten for åpne data, fire pakker og veiledningen i posten for syntetiske testdata. `uganda_eacop` er ikke med.

To ting vi så i samme kontroll, som dere eier:

- Beskrivelsen på app-posten for 5.6.0 (`10.5281/zenodo.21745481`) sier «Eight ready-made projects», fire i hver post. For 5.7.0 er det ni: fem og fire.
- App-posten lenker til dataene med versjons-DOI-ene (`...21744273`, `...21744040`). Bruk konsept-DOI-ene i 5.7.0-posten.
- App-posten bruker ORCID `0000-0003-0331-5596`. Brukeren har bekreftet at riktig ORCID er `0000-0002-4184-5224`, for både programvaren og demodataene. Datapostene har den riktige. Rett app-posten for 5.7.0.

## 2. Pakker per post

### Åpne data, `10.5281/zenodo.21744272` (fem pakker og veiledningen)

| Fil | Setning til wikien |
|---|---|
| `mesa_5.7.0_zirimiti_sample.zip` | Zirimiti skogreservat sør til Victoriasjøen i Mukono, Uganda: ESA WorldCover arealdekke og OpenStreetMap-temaer, 16 eiendelsgrupper. Den største rene åpne-data-pakken. |
| `mesa_5.7.0_my_area.zip` | Et lite område ved Jinja, Uganda: ESA WorldCover og OpenStreetMap, 15 eiendelsgrupper. Rask å kjøre. |
| `mesa_5.7.0_volta_delta.zip` | Voltadeltaet, Ghana: ESA WorldCover og OpenStreetMap, 16 eiendelsgrupper. |
| `mesa_5.7.0_mombasa_reef.zip` | Kysten ved Mombasa, Kenya: korallrevhabitat fra Allen Coral Atlas og OpenStreetMap, pluss oppdiktede bruksområder på revet. 21 eiendelsgrupper. Blanding av ekte og syntetiske data. |
| `mesa_5.7.0_rwenzori.zip` | Rwenzori-fjellene, Uganda: OpenStreetMap og ESA WorldCover, pluss oppdiktede alpine temaer. 23 eiendelsgrupper, to importerte geokodelag og linjer. Den største pakken, og den eneste med linjer og egne geokoder. |
| `mesa_5.7.0_demodata_guide.md` | Veiledningen. |

### Syntetiske testdata, `10.5281/zenodo.21744039` (fire pakker og veiledningen)

| Fil | Setning til wikien |
|---|---|
| `mesa_5.7.0_coastal_zone.zip` | Oppdiktet kystsone i Zanzibarkanalen, Tanzania, 8 eiendelsgrupper. |
| `mesa_5.7.0_river_system.zip` | Oppdiktet elvesystem ved Rufiji, Tanzania, 10 eiendelsgrupper. |
| `mesa_5.7.0_mountain_area.zip` | Oppdiktet fjellområde i Jotunheimen, Norge, 10 eiendelsgrupper. |
| `mesa_5.7.0_mount_kenya.zip` | Oppdiktet fjellområde ved Mount Kenya, Kenya, 10 eiendelsgrupper. |
| `mesa_5.7.0_demodata_guide.md` | Samme veiledning. |

De syntetiske pakkene kjører Process på 12 til 15 sekunder og er det riktige første valget for en ny bruker. All geometri i dem er oppdiktet og sier ingenting om stedet.

Filene ligger i `D:\data\mesa_demodata\exported\`. Sjekksummer i `zenodo/5.7.0/SHA256SUMS.txt`, filliste per post i `zenodo/5.7.0/manifest.json`, metadata i `zenodo/5.7.0/*.zenodo.json`.

### `uganda_eacop` publiseres ikke

Bekreftet. Pakken er fortsatt ødelagt. Målt 2026-10-12: eiendelslaget (`asset/river_system.gpkg`, 10 lag, 74 objekter) har omriss 38,29 til 38,51° Ø og 7,97 til 7,71° S, altså i Tanzania. Linjene (EACOP, Tilenga, KFReroute) ligger ved 30,75 til 31,55° Ø og 1,00° S til 2,18° N, i Uganda. Eiendelene er en kopi av den syntetiske `river_system`.

Den står ikke i `manifest.json`, ikke i veiledningen og ikke i noen Zenodo-post. Fra 2026-10-12 er den også tatt ut av byggingen, og zip-filen er fjernet fra `exported/`. Kildedataene ligger igjen i `data_forge/`. Det som skal til for å publisere den, er ekte eiendelsdata langs EACOP-korridoren i Uganda. Geokodene (GADM) kan uansett ikke følge med, fordi lisensen ikke tillater videredistribusjon.

## 3. Hva som ble generert på nytt

**Pakkene fra 2026-09-16 kan ikke brukes som de er.** De virker teknisk, men `config.ini` i dem har `backfill_max_workers` og gammel stegnummerering, og readme-teksten har tankestreker. Ingen av dem er publisert, så ingenting må trekkes tilbake.

Bygget på nytt 2026-10-12, alle ni som publiseres:

- **`config.ini`**, generert med `tools/make_demo_config.py` mot `git show main:config.ini` (`a809713`). Kontrollert i hver zip: ingen `backfill`, «Segmentation (Stage 4)», `mesa_version = 5.7.0`, alle sju arbeiderinnstillinger `0`.
- **`docs/readme_demodata.txt`**, uten tankestreker. En feil ble rettet samtidig: stien `evaluate_landuse/` sto som `evaluatelanduse/` i `rwenzori`.
- Ingen andre filer er endret. Kontrollert fil for fil med SHA-256 mot pakkene fra 2026-09-16: bare de to filene over skiller seg, i alle pakkene. Eiendelsdata og `settings.xlsx` er byte-identiske med 5.6.0.

Forskjellen mellom pakkenes `config.ini` og malen deres er nå to verdier: `mosaic_reduce_workers = 0` (malen har `4`) og `segmv_k_range = 7` (malen har `5`). Resten av det vi før overstyrte, har malen selv tatt inn.

### Rettelse til oppgaveteksten: Classification

Oppgaven sier at overstyringene våre setter `segmv_enabled = 1` og `segmv_k_range = 7-7`. Det gjør de ikke, og har aldri gjort det. `packaging/config.overrides.ini` har `segmv_k_range = 7` og ingen `segmv_enabled`. Verdiene dere så, står i brukerens arbeidskopi `d:\code\mesa\config.ini`, som er en gjenopprettet demopakke der noen siden har lagret Classification-dialogen (`segmentation_setup.py:353` skriver begge nøklene).

Pakkene har derfor **Classification av** (`segmv_enabled = 0`, arvet fra malen) og `segmv_k_range = 7`. MESA leser `7` og `7-7` likt. Veiledningen sier at Classification er av, hvorfor, og hvordan den slås på. Brukeren har bekreftet 2026-10-12 at den skal være av i pakkene: den tar tid på større prosjekter, og resultatet er lite brukt så langt.

### Overstyringer som er fjernet

`backfill_max_workers` er tatt ut av `packaging/config.overrides.ini`. Ingen andre overstyrte nøkler mangler i malen.

### Veiledningen

`mesa_5.7.0_demodata_guide.md` er skrevet om. Alle punktene dere ba om er tatt:

- «Flatten + Backfill» er nå «Flatten».
- «Lines / Analysis» er nå fanene «Lines» og «Study areas» i Special focus, med Save.
- `segmv_k_range = 7` skrives likt overalt, og det står at Classification er av i pakkene.
- Første avsnitt sier at MESA 5.7.0 leveres uten data, og at pakkene er måten å komme i gang på.
- Ingen tankestreker i veiledningen, i `readme_demodata.txt` eller i Zenodo-titlene og -beskrivelsene. Tallområder skrives «1 to 5».
- Fanen heter «Manage data», ikke «Manage MESA data». Rettet i veiledningen og i Zenodo-beskrivelsene.
- Nytt avsnitt 3 med de tre stegene mellom Restore og Process. Se punkt 4.

Én ting gjenstår: åtte eiendelsgrupper i `rwenzori` heter «Land cover — Tree cover» og tilsvarende, med tankestrek. Det er lagnavn og nøkler i `settings.xlsx`, så en endring endrer datafilen. Brukeren har bestemt at de blir stående.

### `tbl_stacked`

Bekreftet: ingen verktøy i `mesa_demodata` leser `tbl_stacked`. Søk i `tools/` og i skriptene i roten gir null treff.

## 4. Prøving i MESA 5.7.0

### Slik ble det kjørt

Kildekoden fra `main` ble pakket ut i en midlertidig mappe, én kopi per pakke, og kjørt med MESAs egen `.venv`. `d:\code\mesa` er ikke berørt. Maskin: 16 logiske kjerner, ca. 77 GB ledig minne.

Steg per pakke: `restore_backup_archive` (`mesa.py:1262`), eiendelsimport (`AssetManagerWindow._run_import_asset` under Qt uten skjerm), parametre fra `input/settings.xlsx` (`processing_setup.load_all_from_excel`), `geocode_manage.py --nogui --mosaic`, `processing_pipeline_run.py --headless` med standardvalg, og `report_generate.py --no-gui --report-mode general`.

Ikke prøvd: selve vinduene (knapper, dialoger, Maps). Stegene er kjørt gjennom de samme funksjonene som knappene kaller.

### Resultat

Tid i sekunder.

| Pakke | Gikk gjennom | Restore | Eiendeler | Mosaikk | Process | Rapport | Celler i `tbl_flat` |
|---|---|---:|---:|---:|---:|---:|---:|
| `coastal_zone` | ja | 1,1 | 1,8 | 2,6 | 11,8 | 556 | 221 |
| `mountain_area` | ja | 1,4 | 1,9 | 3,4 | 14,2 | 380 | 220 |
| `mount_kenya` | ja | 6,3 | 5,1 | 3,3 | 14,9 | 365 | 290 |
| `river_system` | ja | 1,2 | 1,6 | 3,0 | 11,9 | 543 | 265 |
| `my_area` | ja | 4,5 | 2,8 | 6,5 | 47,7 | 401 | 4 717 |
| `volta_delta` | ja | 1,3 | 1,9 | 8,2 | 54,4 | 431 | 6 931 |
| `zirimiti_sample` | ja | 1,6 | 3,4 | 32,4 | 558 | ikke målt (dvale) | 26 975 |
| `mombasa_reef` | ja | 5,4 | 4,1 | 15,3 | 258 | ikke målt (dvale) | 21 023 |
| `rwenzori` | ja | 5,7 | 9,4 | 28,4 | 159 | 1 219 | 21 195 |

Maskinen lå i dvale under rapportsteget for `zirimiti_sample` og `mombasa_reef`. Rapportene ble fullført og er gyldige, men tidene er ikke brukbare. De andre tallene er uforstyrret.

For alle ni: alle `settings.xlsx`-rader traff en eiendelsgruppe (0 uten treff), alle klassene A til E er til stede i `tbl_flat`, MBTiles, `qgis/mesa_results.qgz` og en Word-rapport ble laget. Ingen feilmeldinger i loggene. Parameterlastingen tok 1,6 til 2,7 sekunder.

`zirimiti_sample` og `rwenzori` ga nøyaktig de samme tallene som tidligere MESA-versjoner: 27 983 mosaikkflater og 26 975 celler for `zirimiti_sample`, 7 110 eiendelsobjekter, 21 315 flater og 21 195 celler for `rwenzori`.

**Linjer og geokoder (`rwenzori`).** Geokodeimporten ga 2 grupper og 13 objekter på 1,6 sekunder. Linjeimporten (`line_manage.Api.import_lines_from_input` og `commit`, det samme som Import og Save i fanen Lines) ga 10 linjer på 1,3 sekunder. Lines-steget i Process ga 535 segmenter. Rapporten fikk 27 kart, 20 av dem linjekart, og ble 73,6 MB.

**Studieområder.** Ingen pakke har studieområder, så Analysis-steget er ikke prøvd. Process melder det som utilgjengelig («Missing: tbl_analysis_group.parquet, tbl_analysis_polygons.parquet»), som forventet.

Én ting å vite om `rwenzori`: de to importerte geokodegruppene får navnene `geocode_001` og `geocode_002` i MESA. Pakken kaller dem sub-catchments og altitude zones. Navnet settes av MESAs import, ikke av pakken. Vi har ikke undersøkt om importen kan lese et navn fra filen.

I de åtte andre pakkene kjørte Lines-steget på MESAs tre eksempellinjer. Se C under.

### Det som så feil ut, og som vi mener ligger i MESA

**A. «Restore, og kjør så Process» virker ikke. Målt.**

Rett etter Restore svarer `processing_pipeline_run.py --list`, likt for alle pakkene:

```
Data: NO Missing: tbl_asset_object.parquet, tbl_geocode_object.parquet
Lines: NO Missing: tbl_geocode_group.parquet
```

Brukeren må gjøre tre ting først:

1. Assets, «Import assets from folder».
2. Parameters, «Load from Excel» med `input/settings.xlsx`. Eiendelsimporten setter importance og susceptibility til 0 (`asset_manage.py:1230`), og ingenting leser `settings.xlsx` før brukeren trykker (`processing_setup.py:851`). Uten dette stopper Process i forhåndskontrollen (`processing_pipeline_run.py:3016`).
3. Geocodes, «Build mosaic».

Dette gjelder teksten deres på disse stedene, slik de står i dag:

- `mesa.wiki/Home.md:25-30` («Both sets hold input data only, so run Process after restoring one»)
- `mesa.wiki/Quickstart.md:24`
- `devtools/RELEASE_NOTES_5.7.0.md:12`
- trolig også `readme.txt` i `input/` og brukerhåndboken, som vi ikke har lest

Vår egen veiledning hadde samme feil siden 5.6.0. Den er rettet (avsnitt 3). Forslag til setning: «Download one, load it under Manage data → Restore data, then import the assets, load `input/settings.xlsx` under Parameters, build the mosaic under Geocodes, and run Process. The guide in each record gives the steps.»

Slutning, ikke målt: meldingen etter en vellykket Restore av en pakke uten resultater (`mesa.py`, `_on_finished` i `_do_restore`) nevner bare Process, og bare når pakken er eldre enn programmet. For en 5.7.0-pakke i 5.7.0 sier den ingenting om neste steg. En setning der ville hjulpet flest.

**B. Rapporten tar 6 til 9 minutter på de minste pakkene. Målt.**

365 til 556 sekunder for prosjekter som Process bruker 12 sekunder på, og 1 219 sekunder for `rwenzori`. Loggen viser at tiden går til å hente bakgrunnskart fra OpenStreetMap: for `coastal_zone` 1 662 fliser, opptil zoom 17. Rapportplanen er «13 maps (2 overview, 6 line, 5 group)» for alle pakkene.

Slutning: fremdriftslinjen i rapportvinduet dekker dette for den som ser vinduet. En bruker bør likevel få vite at første rapport krever nett og tar flere minutter. Veiledningen vår sier det nå.

**C. Seks av tretten rapportkart gjelder linjer brukeren ikke har. Målt.**

Når et prosjekt ikke har linjer, lager Process tre eksempellinjer (`processing_pipeline_run.py:1085`, «No lines found; creating defaults», 600 m bredde, 1000 m segmenter). De havner i `tbl_lines`, i Lines-steget og i rapporten, der de står for seks av tretten kart og for flisene på høyest zoom. Åtte av ni demopakker har ingen linjer.

Slutning: dette er trolig gammel, tilsiktet oppførsel. For en ny bruker ser det ut som om pakken inneholder linjer den ikke har. Veiledningen forklarer det nå med én setning. Om eksempellinjene bør bort fra rapporten, er deres avgjørelse.

## 5. Rekkefølge ved publisering

Det finnes ingen avhengighet som låser rekkefølgen.

- Datapostenes metadata har i dag ingen `related_identifiers`. Det gjelder både de publiserte 5.6.0-postene og de forberedte 5.7.0-filene. Vi trenger derfor ikke appens versjons-DOI for å publisere.
- Dokumentasjonen deres bruker konsept-DOI-ene, som alt finnes.

Anbefalt rekkefølge:

1. Demodata 5.7.0 først. **Gjort 2026-10-12.** Konsept-DOI-ene peker til 5.7.0-data nå.
2. Publiser appen.
3. Send oss appens konsept-DOI eller versjons-DOI hvis dere ønsker at datapostene skal vise til appen. Det kan legges til som metadataendring etterpå uten ny versjon. Regelen vår er versjons-DOI ved samordnet utgivelse og appens konsept-DOI (`10.5281/zenodo.10775727`) ellers.

## Til brukeren før publisering

- Posten for åpne data har ikke noe lisensfelt i API-et, bare en rettighetstekst med en manglende sluttparentes. Uendret i 5.7.0. Kan rettes som metadataendring uten ny versjon.
- Titlene i de forberedte metadataene er endret for å få bort tankestreken: «MESA 5.7.0 demonstration data packages from open sources (East and West Africa)» og «MESA 5.7.0 synthetic test data packages for sensitivity mapping».
