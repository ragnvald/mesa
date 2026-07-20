# Zenodo-post 5.5.0 — ferdig tekst

Klipp-og-lim inn i Zenodo-skjemaet. Se `PUBLISERE_5.5.0.md` §6 for rekkefølgen
(Zenodo først, deretter GitHub-release som utleder taggen fra tittelen her).

Skrivestil: følg 5.2-posten — flytende prosa, ingen overskrifter, punktlister eller
kodeformatering i beskrivelsen, og hold det ikke-teknisk. Målgruppen er analytikere og
beslutningstakere som vurderer om de skal laste ned, ikke utviklere. Måletall,
bibliotekversjoner og interne tabellnavn hører hjemme i release-notatet, ikke her.

**Gjenbruk ved neste utgivelse.** Denne teksten er ment som startpunkt. Behold åpnings-
og avslutningsavsnittene, bytt versjonsnummer, og skriv om midtdelen til det som faktisk
er nytt. Sjekk på nytt hver gang:

- **Attribusjon** — avsnittet nederst gjelder demodataene som ligger i buildet. Kildene
  står i `docs/readme_demodata.txt` i pakken (seksjonen SOURCES AND CREDITS); endres
  demodataene, må avsnittet oppdateres. For 5.5.0: ESA WorldCover + OpenStreetMap.
- **Demodata** — 5.2 brukte Mafia Island, 5.5.0 bruker Zirimiti. Stedsnavnet er bevisst
  utelatt her fordi demodata skal ha sin egen Zenodo-post (`PUBLISERE` §6b).
- **Systemkrav** — hentet fra wikiens Home-side.
- Bakgrunnskart som Bing/Google/Waze er tjenester brukeren kobler seg til, ikke data som
  distribueres. Ikke omtal dem som inkludert data.

---

## Tittel

> MESA 5.5.0 — Environmental sensitivity assessment tool (compiled Windows build)

⚠️ Tittelen **må** inneholde `5.5.0` — `github_release_from_zenodo.py` utleder
GitHub-taggen fra den. Demodata-posten må tilsvarende **ikke** inneholde `5.5.0`.

## Metadata

| Felt | Verdi |
|---|---|
| Resource type | Software |
| Version | 5.5.0 |
| License | GNU General Public License v3.0 (GPL-3.0) |
| Community | `mesatool` |
| Language | English |
| Keywords | environmental sensitivity; oil spill preparedness; pollution response; environmental management; GIS; geospatial analysis; QGIS; sensitivity mapping; marine spatial planning |
| Related identifiers | *Is supplement to* → https://github.com/ragnvald/mesa (URL) · *Is documented by* → https://github.com/ragnvald/mesa/wiki (URL) |

---

## Beskrivelse (lim inn som den er)

This package provides compiled Windows 11 binaries for MESA 5.5.0, a desktop workflow for preparing, processing, and publishing spatial sensitivity analysis deliverables using the MESA method. The MESA 5.x line fully replaces all earlier MESA versions and provides a clear end-to-end workflow from inputs to published outputs.

The distribution includes a complete, pre-processed demonstration project, allowing immediate exploration of maps, classification and reporting without additional setup. Everything runs locally on your own machine, without a server or a network connection.

Where MESA 5.2 introduced classification as a separate tool that had to be run by hand after processing, with its results confined to a single window, MESA 5.5.0 makes it part of the workflow itself. Classification now runs automatically as a step within processing, appears as a selectable section in the Word report with its own charts and written interpretation, is available as map layers showing both the assigned types and how confident the classification is at each location, and is exported to QGIS alongside the other results. A single run can cover several sets of analytical units at once. This turns classification from a supplementary experiment into a normal part of producing deliverables.

Processing is also substantially faster. The heaviest step in the pipeline, where assets are combined with analytical units, has been reworked so that each worker receives only the data it needs. On a large reference dataset this cut the time for that step to roughly a third, while producing identical results. Datasets that previously risked exhausting memory are now assessed before the run begins: where the finest-grained analytical units would not fit in available memory, MESA explains this clearly and suggests a coarser grid, rather than failing partway through a long run.

MESA can optionally connect to an AI service to draft classification descriptions, report prose and map styling. This is switched off by default, nothing leaves the machine unless it is deliberately configured and enabled, and every AI-assisted feature falls back to a deterministic result when no connection is present.

Loading data is no longer limited to restoring your own backup. Manage data now opens any MESA project package, whether that is a backup of your own, a project shared by a colleague, or a demonstration dataset. Demonstration packages carry a short description of themselves, covering their data sources and the credits those sources require, which is offered for reading once the package has been loaded.

This release also resolves a number of faults that could cost work, including imports of analytical units that removed previously imported sets, restores that could leave a project unable to start, and partial re-runs that discarded existing map tiles.

The reporting system remains Word-first (.docx), improving reproducibility and easing downstream editing, while advanced cartographic layout continues to be supported through the bundled QGIS project. Lightweight status monitoring on the Status tab continues to support iterative and team-based work. The underlying platform has been modernised throughout, which is invisible in the packaged build but keeps the tool current and maintainable.

MESA is intended for Windows 11, and benefits from at least 16 GB of memory and 8 processor cores. To get started, unzip the archive into a writable folder, keep the folder structure intact, and launch mesa.exe.

The bundled demonstration project contains open data that requires attribution: ESA WorldCover 10 m 2021 v200 (© ESA WorldCover project; contains modified Copernicus Sentinel data; CC-BY 4.0) and OpenStreetMap (© OpenStreetMap contributors; ODbL 1.0; openstreetmap.org/copyright). Map backgrounds rendered by MESA use OpenStreetMap unless another provider is selected. Verify licence terms before redistributing this package or publishing results derived from it.

For method background, workflow details, and ongoing updates, refer to the MESA project wiki:

https://github.com/ragnvald/mesa/wiki

---

## Etter publisering

1. Noter record-id + DOI i `PUBLISERE_5.5.0.md` §6a.
2. Fyll DOI/record-URL inn i `RELEASE_NOTES_5.5.0.md` (to plassholdere nederst).
3. Kjør `python devtools\github_release_from_zenodo.py <record-id> --publish`.
4. Krysslenk mot demodata-posten når den finnes.
