# Zenodo-post 5.7.0: ferdig tekst

Klipp og lim inn i Zenodo-skjemaet. Opprett posten som **New version** av app-posten (konsept-DOI `10.5281/zenodo.10775727`), ikke som en ny post.

Skrivestil som i 5.5.0-posten: flytende prosa, ingen overskrifter, punktlister eller kodeformatering i beskrivelsen, og ikke-teknisk. Målgruppen er analytikere og beslutningstakere som vurderer om de skal laste ned. Måletall og tabellnavn hører hjemme i `RELEASE_NOTES_5.7.0.md`. Teksten har ingen tankestreker.

## Rekkefølge

Avtalt med mesa_demodata 2026-10-12 (`docs/exchange/2026-10-12-demodata-5-7-utgivelse-svar.md`):

1. Demodata 5.7.0 først. **Gjort 2026-10-12.** Versjons-DOI-ene er `10.5281/zenodo.23309462` (åpne data) og `10.5281/zenodo.23309556` (syntetiske testdata), og konsept-DOI-ene peker nå til dem.
2. Publiser denne posten, med Windows-bygget alene. Besluttet 2026-10-12: macOS-bygget kan ikke lages nå og kommer som en ny versjon senere, se «macOS senere» nederst.
3. Send appens versjons-DOI til mesa_demodata hvis datapostene skal vise til appen. Det kan legges til som metadataendring uten ny versjon.

## Publisert

Windows-posten ble publisert 2026-10-12: `10.5281/zenodo.23314128` (record-id `23314128`), med filen `MESA_570_win11_2026_10_12.zip`, MD5 `a6e18c44a48a846bfdb6f8f009e640b0`, lik filen i `D:\dist`. Kontrollert mot Zenodo-API-et 2026-10-13. Fire felt avviker fra teksten i denne filen og kan rettes som metadataendring uten ny versjon:

- **ORCID** står som `0000-0003-0331-5596`. Riktig er `0000-0002-4184-5224`.
- **Tittelen** er «MESA 5.7.0: Mapping Environmentaly Sensitive assets (compiled Windows build)», med skrivefeil i «Environmentaly».
- **Keywords** og **Related identifiers** er tomme.
- **Affiliation** mangler.

## Dette må rettes fra 5.6.0-posten

Posten for 5.6.0 (`10.5281/zenodo.21745481`) har tre ting som ikke skal følge med videre:

- **ORCID.** 5.6.0-posten har `0000-0003-0331-5596`. Riktig er `0000-0002-4184-5224`, den samme som datapostene bruker.
- **Antall prosjekter.** 5.6.0 sier «Eight ready-made projects». For 5.7.0 er det ni: fem med åpne data og fire syntetiske.
- **Lenker til demodata.** 5.6.0 lenker med versjons-DOI-ene (`...21744273`, `...21744040`). Bruk konsept-DOI-ene, som alltid peker til nyeste versjon.

## Tittel

> MESA 5.7.0: Mapping Environmentally Sensitive Assets (compiled Windows build)

Tittelen **må** inneholde `5.7.0`. `github_release_from_zenodo.py` utleder GitHub-taggen fra den.

## Metadata

| Felt | Verdi |
|---|---|
| Resource type | Software |
| Version | 5.7.0 |
| Creator | Ragnvald Larsen, Norwegian Environment Agency, ORCID `0000-0002-4184-5224` |
| License | GNU General Public License v3.0 (GPL-3.0) |
| Community | `mesatool` |
| Language | English |
| Keywords | environmental sensitivity; oil spill preparedness; pollution response; environmental management; GIS; geospatial analysis; QGIS; sensitivity mapping; marine spatial planning |
| Related identifiers | *Is supplement to* → https://github.com/ragnvald/mesa (URL) · *Is documented by* → https://github.com/ragnvald/mesa/wiki (URL) · *References* → https://doi.org/10.5281/zenodo.21744272 (DOI, åpne demodata) · *References* → https://doi.org/10.5281/zenodo.21744039 (DOI, syntetiske demodata) |

## Filer

- `MESA_570_win11_2026_10_12.zip` (Windows 11, 64-bit)

Kontrollert 2026-10-12: zip-filen fra kl. 11:29 inneholder bygget fra kl. 09:02 (commit `752eb37`), 1,38 GB, 8 311 filer, MD5 `a6e18c44a48a846bfdb6f8f009e640b0`. Den har ikke rettelsen `0215efb` (konsollvindu som blinker under prosessering). Filen `MESA_570_win11_2026_10_12_done.zip` i samme mappe er det første bygget med gammel mappestruktur og skal ikke lastes opp.

Før zip-filen lages: kontroller at `D:\dist\mesa` ikke har noen `secrets`-mappe og ingen `log.txt`. Bygget lager ingen av dem, men de oppstår hvis `mesa.exe` er startet med gammel mappestruktur eller noe er kopiert inn for hånd.

## Beskrivelse (lim inn som den er)

This record provides a compiled build of MESA 5.7.0 for Windows 11. MESA is a desktop workflow for preparing, processing and publishing spatial sensitivity analyses using the MESA method. The MESA 5 line replaces all earlier MESA versions.

MESA 5.7.0 is a release about scale and safety. Projects take far less disk space and memory, because the analysis table no longer stores the outline of every cell on every row. The same project gives identical results in a table that is about 93 percent smaller, and study areas now work on large projects. Map tiles are drawn faster, and where a zoomed-out pixel covers many cells the highest value now wins, so two runs of the same data give the same picture.

A fault that could lose data in every earlier MESA 5 release is fixed. Adding an H3 or QDGC grid level to a project with a large mosaic could delete the mosaic without any message. If you have used an earlier MESA 5 version, open Geocodes after upgrading and check that the mosaic still has its objects. A second fault, which left the importance, sensitivity and susceptibility values empty in the results of some projects, is also fixed. Run Process again in the new version to be sure your figures are complete.

The program now asks before your work is lost. Every window with editable fields asks whether to save, discard or cancel before edits would disappear. Lines and study areas are edited in the same way, with Save and Discard in both. Destructive actions are marked as such and are no longer confirmed by pressing Enter, and a running job cannot be started twice.

The Word report has been reworked. Maps are wider and have a margin around the data, headings stay with their content, the maps have colour keys, and the classification of sensitivity types is shown in a table with a figure that explains each type. A status line shows what is being drawn and roughly how much time is left, and the background maps are downloaded several times faster than before.

Publishing to GeoNode now gives one map per geocode group with switchable layers styled as in MESA, optional dashboards, and a tab for managing what is on the server. The QGIS project is produced as its own processing step and includes interactive three-dimensional views.

The download is now only the program. The first time you start MESA it asks where to keep your data, and it proposes a folder outside OneDrive or similar services when your Documents folder is synchronised, because a project is often several gigabytes of files that change while MESA runs. When you upgrade, MESA asks whether to continue with your existing data or to start with an empty project. In the second case your old data is set aside under a new name, and nothing is deleted.

MESA is delivered without project data. Nine demonstration projects are published as two separate records. Four synthetic projects, at https://doi.org/10.5281/zenodo.21744039, are small and are the right first choice. Five projects built from open data for sites in Uganda, Kenya and Ghana are at https://doi.org/10.5281/zenodo.21744272. Each record includes a guide. A package holds input data only: after restoring it under Manage data you import the assets, load the parameters, build the mosaic and run Process. Everything runs locally on your own machine, without a server.

MESA can optionally connect to an AI service to draft descriptions and report text. This is switched off by default, and nothing leaves the machine unless it is deliberately configured and enabled.

MESA is intended for Windows 11, and benefits from at least 16 GB of memory and 8 processor cores. To get started, unzip the archive into a folder of its own and start mesa.exe. The file READ ME FIRST.txt in the same folder explains where your data is kept.

For method background, workflow details and the full list of changes, see the MESA project wiki: https://github.com/ragnvald/mesa/wiki

## macOS senere

Filene i en publisert Zenodo-post kan ikke byttes eller suppleres; bare metadata kan endres. macOS-bygget kommer derfor som en **New version** av 5.7.0-posten:

1. Last opp **begge** filene i den nye versjonen: Windows-zip-filen på nytt og `MESA_570_macos_<dato>.dmg`. Konsept-DOI-en peker til nyeste versjon, og den må ha en Windows-fil.
2. Sett Version til `5.7.0` også der.
3. Bytt tittelen til «MESA 5.7.0: Environmental sensitivity assessment tool (compiled builds for Windows and macOS)».
4. I beskrivelsen: første setning blir «This record provides compiled builds of MESA 5.7.0 for Windows 11 and for macOS on Apple Silicon.», og nest siste avsnitt begynner «MESA is intended for Windows 11 or macOS on Apple Silicon, ... To get started on Windows, unzip ...».
5. Ikke lag ny GitHub-tagg. Legg dmg-filen og den nye DOI-en til i den eksisterende 5.7.0-utgivelsen.

Dokumentasjonen lenker til konsept-DOI-en (`10.5281/zenodo.10775727`) for nedlasting, så den trenger ikke endres når macOS-utgaven kommer.

## Etter publisering

1. Før versjons-DOI og record-URL inn i `RELEASE_NOTES_5.7.0.md` (plassholderen under «Where can I find it?»).
2. Bytt lenkene til 5.6.0-posten i `mesa.wiki/Home.md` og `mesa.wiki/_Sidebar.md` til 5.7.0.
3. Kjør `python devtools\github_release_from_zenodo.py <record-id> --publish`.
4. Gi mesa_demodata appens versjons-DOI.
