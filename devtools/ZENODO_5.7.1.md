# Zenodo-post 5.7.1: ferdig tekst

Opprett posten som **New version** av 5.7.0-posten (`10.5281/zenodo.23314128`, konsept-DOI `10.5281/zenodo.10775727`), ikke som en ny post. Stegene står i `devtools/docs/release_procedure.md`.

Skrivestil som i 5.7.0-posten: flytende prosa, ingen overskrifter, punktlister eller kodeformatering, og ikke-teknisk. Teksten har ingen tankestreker.

## Tittel

> MESA 5.7.1: Mapping Environmentally Sensitive Assets (compiled Windows build)

Tittelen **må** inneholde `5.7.1`. `github_release_from_zenodo.py` utleder GitHub-taggen fra den.

## Metadata

| Felt | Verdi |
|---|---|
| Resource type | Software |
| Version | 5.7.1 |
| Creator | Larsen, Ragnvald. Norwegian Environment Agency. ORCID `0000-0002-4184-5224` |
| License | GNU General Public License v3.0 or later (`gpl-3.0-or-later`, som 5.7.0-posten) |
| Community | `mesatool` |
| Language | English (`eng`) |
| Keywords | environmental sensitivity; oil spill preparedness; pollution response; environmental management; GIS; geospatial analysis; QGIS; sensitivity mapping; marine spatial planning |
| Related identifiers | *Is supplement to* → https://github.com/ragnvald/mesa (URL) · *Is documented by* → https://github.com/ragnvald/mesa/wiki (URL) · *References* → https://doi.org/10.5281/zenodo.21744272 (DOI, åpne demodata) · *References* → https://doi.org/10.5281/zenodo.21744039 (DOI, syntetiske demodata) |

Kontrollert mot Zenodo-API-et 2026-10-13: 5.7.0-posten har nå riktig tittel, men oppføringen av forfatter er uten ORCID og uten tilknytning, og keywords og related identifiers er tomme. En ny versjon arver metadata fra forrige, så disse feltene må fylles inn her, ellers følger manglene med.

## Filer

- `MESA_571_win11_<dato>.zip` (Windows 11, 64-bit)

`MESA_571_win11_2026_10_13.zip`: 1 512 656 083 byte, 7 733 filer, MD5 `be718ea1309a8c246fb96ed3383a3856`. Bygget 2026-10-13 kl. 11:00 fra commit `e3179b5962847cc4a6ab86217231cb58381d9155`, med `mesa_version = 5.7.1`, uten `secrets` og uten `log.txt`. Et første bygg kl. 10:31 fra `54b634a` viste 5.7.0 i tittellinjen og ble forkastet.

Før zip-filen lages: kontroller at `D:\dist\mesa` ikke har noen `secrets`-mappe og ingen `log.txt`.

## Publisert

Publisert 2026-10-15: `10.5281/zenodo.23337078` (record-id `23337078`). Filen er lastet opp av brukeren fra en annen maskin og er en egen zip av samme bygg: `MESA_571_win11_2026_10_13.zip`, 1 502 844 542 byte, MD5 `2e26d6a906cf082fd1cdddd21c00ba53`. Innholdet er kontrollert mot den lokale zip-filen over: samme 7 733 filer med samme CRC og størrelse. GitHub-utgivelsen `5.7.1` peker på `e3179b5962847cc4a6ab86217231cb58381d9155`.

## Beskrivelse (lim inn som den er)

This record provides a compiled build of MESA 5.7.1 for Windows 11. MESA is a desktop workflow for preparing, processing and publishing spatial sensitivity analyses using the MESA method. The MESA 5 line replaces all earlier MESA versions.

MESA 5.7.1 is a correction of 5.7.0 and adds nothing new. The name of the program is now spelled out correctly on the Welcome and About tabs: MESA is Mapping Environmentally Sensitive Assets. The Lines tab in Special focus, which could open empty with an error message, now opens with its lines. Windows that were too small for their content no longer cut off text and tick boxes. A command-line window no longer flashes by during processing. Projects and results are not affected, and nothing needs to be run again after upgrading from 5.7.0.

The 5.7 line is about scale and safety. Projects take far less disk space and memory, because the analysis table no longer stores the outline of every cell on every row. The same project gives identical results in a table that is about 93 percent smaller, and study areas now work on large projects. Map tiles are drawn faster, and where a zoomed-out pixel covers many cells the highest value now wins, so two runs of the same data give the same picture.

A fault that could lose data in every MESA 5 release before 5.7 is fixed. Adding an H3 or QDGC grid level to a project with a large mosaic could delete the mosaic without any message. If you have used a MESA 5 version older than 5.7, open Geocodes after upgrading and check that the mosaic still has its objects. A second fault, which left the importance, sensitivity and susceptibility values empty in the results of some projects, is also fixed. Run Process again in the new version to be sure your figures are complete.

The program asks before your work is lost. Every window with editable fields asks whether to save, discard or cancel before edits would disappear. Lines and study areas are edited in the same way, with Save and Discard in both. Destructive actions are marked as such and are not confirmed by pressing Enter, and a running job cannot be started twice.

The Word report has wider maps with a margin around the data, headings that stay with their content, colour keys, and a table and a figure that explain each sensitivity type. A status line shows what is being drawn and roughly how much time is left.

Publishing to GeoNode gives one map per geocode group with switchable layers styled as in MESA, optional dashboards, and a tab for managing what is on the server. The QGIS project is produced as its own processing step and includes interactive three-dimensional views.

The download is only the program. The first time you start MESA it asks where to keep your data, and it proposes a folder outside OneDrive or similar services when your Documents folder is synchronised. When you upgrade, MESA asks whether to continue with your existing data or to start with an empty project. In the second case your old data is set aside under a new name, and nothing is deleted.

MESA is delivered without project data. Nine demonstration projects are published as two separate records. Four synthetic projects, at https://doi.org/10.5281/zenodo.21744039, are small and are the right first choice. Five projects built from open data for sites in Uganda, Kenya and Ghana are at https://doi.org/10.5281/zenodo.21744272. Each record includes a guide. A package holds input data only: after restoring it under Manage data you import the assets, load the parameters, build the mosaic and run Process. Everything runs locally on your own machine, without a server.

MESA can optionally connect to an AI service to draft descriptions and report text. This is switched off by default, and nothing leaves the machine unless it is deliberately configured and enabled.

MESA is intended for Windows 11, and benefits from at least 16 GB of memory and 8 processor cores. To get started, unzip the archive into a folder of its own and start mesa.exe. The file READ ME FIRST.txt in the same folder explains where your data is kept.

For method background, workflow details and the full list of changes, see the MESA project wiki: https://github.com/ragnvald/mesa/wiki

## macOS senere

Som for 5.7.0: en publisert posts filer kan ikke suppleres, så macOS-bygget kommer som en ny versjon med begge filene, fortsatt med Version `5.7.1`. Det bygges da fra samme commit som Windows-bygget.
