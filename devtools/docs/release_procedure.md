# Utgivelse av MESA: stegene, og hva som kan automatiseres

Skrevet 2026-10-13 under klargjøringen av 5.7.1, ut fra hvordan 5.5.0, 5.6.0 og 5.7.0 faktisk ble gitt ut. Formålet er at hvert steg skal stå med kommando, inndata, utdata og kontroll, slik at det kan settes sammen til ett skript.

Tegnforklaring i kolonnen **Auto**: `ja` = kan kjøres uten menneske i dag, `skript` = kan automatiseres, men skriptet finnes ikke ennå, `menneske` = krever en vurdering eller en handling som ikke bør automatiseres.

`<V>` er versjonen (`5.7.1`), `<VV>` er den uten punktum (`571`), `<D>` er datoen zip-filen lages (`2026_10_13`).

## Oversikt

| # | Steg | Auto |
|---|---|---|
| 1 | Avgjør hva utgivelsen inneholder | menneske |
| 2 | Stemple versjonen | skript |
| 3 | Skriv utgivelsestekstene | menneske, med utkast fra commit-loggen |
| 4 | Test fra kildekoden | ja |
| 5 | Commit stempelet | ja, etter et ja |
| 6 | Bygg | ja |
| 7 | Kontroller bygget | delvis |
| 8 | Røyktest den kompilerte utgaven | menneske |
| 9 | Lag zip-filen | skript |
| 10 | Zenodo: ny versjon, fil, metadata | skript (API) |
| 11 | Zenodo: publiser | menneske trykker, eller API med eget flagg |
| 12 | Før DOI inn i dokumentene | skript |
| 13 | Push | ja, etter et ja |
| 14 | GitHub-utgivelse | ja |
| 15 | Wiki | delvis |
| 16 | Kunngjøring på GitHub Discussions | skript |
| 17 | history.md, og beskjed til søsterprosjektene | menneske |

## Stegene

### 1. Avgjør hva utgivelsen inneholder

`git log <forrige tagg>..HEAD --oneline` viser hva som er kommet til. Forrige bygg er ikke nødvendigvis bygget fra taggen: 5.7.0 ble bygget fra `752eb37`, og `0215efb` kom etter. Byggets commit står i `ZENODO_<forrige>.md`.

Avgjør også om demodata må gis ut på nytt. Det må de bare når formatet på en MESA-pakke eller `config.ini`-malen er endret. Er de det, publiseres demodata først (avtalt med mesa_demodata 2026-10-12).

### 2. Stemple versjonen

To steder, og ingen andre:

- `config.ini`, `[DEFAULT] mesa_version`. Programmet leser versjonen herfra alle steder.
- `devtools/build_user_guide.py`, konstanten `VERSION`, og kapittel 2 hvis utgivelsen har noe en bruker må vite.

Deretter `.\.venv\Scripts\python.exe devtools\build_user_guide.py`, som skriver de to `.docx`-filene i `docs/` og henter skjermbilder fra `..\mesa.wiki\images`.

Fallgruve: arbeidskopien av `config.ini` er vanligvis en demokonfigurasjon som ikke skal committes. Bare linjen med `mesa_version` skal inn. Et skript bør derfor endre linjen i `HEAD`-utgaven og committe den alene, ikke `git add config.ini`.

### 3. Skriv utgivelsestekstene

- `devtools/RELEASE_NOTES_<V>.md`: for GitHub. Lenken til Zenodo har plassholderne `ZENODO_RECORD_ID` og `ZIP_DATE` til steg 12.
- `devtools/ZENODO_<V>.md`: tittel, metadata og beskrivelse.

Tittelen på Zenodo må inneholde versjonsnummeret, for `github_release_from_zenodo.py` utleder taggen fra den. Ingen tankestreker i ny tekst.

### 4. Test fra kildekoden

`.\.venv\Scripts\python.exe -m pytest tests -q`. Skal være grønn før bygget.

### 5. Commit stempelet

Bygget henter `config.ini`, `input/` og `qgis/` fra `HEAD`, ikke fra arbeidsmappen. Stempelet må derfor være committet før bygget, ellers får bygget forrige versjonsnummer. `docs/` kopieres fra arbeidsmappen, så `.docx`-filene må være bygget på nytt før bygget, og committes sammen med stempelet for at commit og bygg skal stemme overens. Push er ikke nødvendig for å bygge.

### 6. Bygg

Kontroller først at ingenting bruker `D:\dist\mesa` (`mesa.exe` må ikke kjøre): en byggestart tømmer mappen.

```
set MESA_BUILD_MAIN=1
set MESA_BUILD_HELPERS=1
set MESA_BUILD_CLEAN=1
set MESA_BUILD_PARALLEL=4
.\.venv_compile\Scripts\python.exe devtools\build_all.py
```

`devtools\compile_win_11.bat` gjør det samme. Det må være `.venv_compile`: `.venv` har ikke PyInstaller, og en start med feil tolk tømmer likevel `D:\dist\mesa`. Tar om lag ti minutter. Resultatet er `D:\dist\mesa`, og en linje i `D:\dist\build_history.log`.

### 7. Kontroller bygget

Bygget avslutter selv med `verify_distribution()`, som stopper hvis noe annet enn det git sporer har kommet med. I tillegg:

- `D:\dist\mesa\_internal\config.ini` har `mesa_version = <V>`.
- Ingen `secrets`-mappe og ingen `log.txt` i `D:\dist\mesa`.
- Noter byggets commit (`git rev-parse HEAD`). Den skal inn i `ZENODO_<V>.md` og er målet for GitHub-taggen.

De tre punktene kan legges i et skript. De finnes ikke i `verify_distribution()` i dag.

Disse kontrollene viser hva som er pakket, ikke hva programmet viser. Det første 5.7.1-bygget besto alle tre og kalte seg likevel 5.7.0 (se learning.md "The program's version is the program's, not the data folder's").

### 8. Røyktest den kompilerte utgaven

Start `D:\dist\mesa\mesa.exe` mot en datamappe som er brukt med forrige versjon, og svar at du fortsetter med dataene. Se at versjonen i tittellinjen og i banneret er `<V>`, at fanene vises hele, og at det utgivelsen retter faktisk er rettet. Dette er steget som fant feil i 5.7.0-rundene, og det bør forbli et menneske.

Etter røyktesten: kontroller på nytt at `D:\dist\mesa` ikke har fått `secrets` eller `log.txt`. Med mappestrukturen fra 5.7 havner brukerdata i datamappen, ikke her, men kontrollen er billig.

### 9. Lag zip-filen

Navn: `D:\dist\MESA_<VV>_win11_<D>.zip`, med mappen `mesa` som øverste nivå.

```
Compress-Archive -Path D:\dist\mesa -DestinationPath D:\dist\MESA_<VV>_win11_<D>.zip
Get-FileHash D:\dist\MESA_<VV>_win11_<D>.zip -Algorithm MD5
```

Før størrelse, antall filer og MD5 inn i `ZENODO_<V>.md`. MD5 brukes til å kontrollere opplastingen i steg 10.

### 10 og 11. Zenodo

Se neste kapittel. For hånd: åpne forrige post, **New version**, slett den gamle filen, last opp den nye, lim inn tittel, beskrivelse og metadata fra `ZENODO_<V>.md`, **Publish**.

### 12. Før DOI inn i dokumentene

Record-id og DOI fra Zenodo skal inn i:

- `devtools/RELEASE_NOTES_<V>.md` (plassholderne under "Where can I find it?")
- `devtools/ZENODO_<V>.md` (avsnitt "Publisert")
- `..\mesa.wiki\Home.md` (linjen "Current release") og `..\mesa.wiki\_Sidebar.md`

Konsept-DOI-en `10.5281/zenodo.10775727` endres aldri og er den dokumentasjonen ellers lenker til.

### 13. Push

`git push origin main` i mesa. Taggen i steg 14 må peke på en commit som finnes på GitHub.

### 14. GitHub-utgivelse

```
.\.venv\Scripts\python.exe devtools\github_release_from_zenodo.py <record-id> --target <full commit-id> --publish
```

Skriptet henter tittel og beskrivelse fra Zenodo, utleder taggen fra tittelen og kaller `gh release create`. `--target` må være en gren eller den fulle commit-id-en på 40 tegn: en forkortet id gir "Release.target_commitish is invalid" (skjedde 2026-10-12). Uten `--publish` skrives bare teksten ut, som er en trygg prøvekjøring.

### 15. Wiki

- `Mesa-changelog-overview.md`: et avsnitt for utgivelsen.
- `Home.md` og `_Sidebar.md`: lenkene fra steg 12.
- `Troubleshooting.md`: fjern eller juster omtaler av feil som nå er rettet.
- Skjermbilder ved behov: `.\.venv\Scripts\python.exe devtools\capture_ui_active_batch.py`. Det åpner vinduer på skjermen og kan ikke kjøres mens noen arbeider på maskinen.

Wikien er et eget repo (`..\mesa.wiki`, gren `master`).

### 16. Kunngjøring

GitHub Discussions, kategorien Announcements, i samme form som for 5.6.0 og 5.7.0 (nr. 38). Gjøres med `gh api graphql` (`createDiscussion`), siden `gh` ikke har en egen kommando for diskusjoner.

### 17. Etterarbeid

- `history.md`: hva som ble gitt ut, fra hvilken commit, DOI, og hva som ble meldt til mesa-server og mesa_demodata.
- mesa_demodata får appens versjons-DOI hvis datapostene skal vise til den.
- macOS: bygges på Mac-en fra samme commit (`devtools/MAC_BUILD.md`) og gis ut som enda en ny versjon på Zenodo med begge filene.

## Zenodo via API: vurdering

Kort: hele steg 10 og 11 kan gjøres med API-et, og det er verdt å gjøre. Det fjerner de feilene som faktisk har skjedd for hånd (feil ORCID, skrivefeil i tittelen, tomme keywords og related identifiers på 5.7.0-posten). Publiseringen bør likevel stå bak et eget flagg, for den kan ikke angres.

Det som følger er lest i Zenodos API-dokumentasjon (developers.zenodo.org) og i InvenioRDM-dokumentasjonen 2026-10-13. Ingenting av det er prøvd mot Zenodo ennå. Punktene merket **uavklart** må prøves i sandkassen først.

### Tilgang

- Et personlig token lages på zenodo.org under Applications. Det trenger to rettigheter: `deposit:write` (opprette, laste opp, endre metadata) og `deposit:actions` (ny versjon, publisere, redigere publisert post).
- Sendes som `Authorization: Bearer <token>`. JSON-kall må ha `Content-Type: application/json`.
- Tokenet skal ligge i `secrets/` (som ikke spores og ikke bygges) eller i en miljøvariabel, aldri i et skript, en logg eller et dokument.
- Grenser: 100 kall i minuttet og 5000 i timen. En utgivelse bruker under 20.

### Kallene, i rekkefølge

Med det eldre deponerings-API-et, som er det Zenodo dokumenterer:

1. **Ny versjon.** `POST /api/deposit/depositions/<id>/actions/newversion`, der `<id>` er record-id til den nyeste publiserte versjonen (ikke konsept-id). Svaret er den gamle posten; utkastet ligger i `links.latest_draft`. Hent den adressen og les `id` og `links.bucket`. Bare ett upublisert utkast kan finnes om gangen, så et nytt kall gjør ingenting før utkastet er publisert eller slettet.
2. **Fjern den gamle filen.** `GET .../depositions/<utkast>/files`, og `DELETE .../files/<file_id>` for hver. **Uavklart:** Zenodos dokumentasjon sier at utkastet arver filene fra forrige versjon, InvenioRDM-dokumentasjonen sier at det er tomt. Skriptet må liste filene og slette det som finnes, så virker det i begge tilfeller.
3. **Last opp.** `PUT <links.bucket>/MESA_<VV>_win11_<D>.zip` med filen som strøm (ikke lest inn i minnet). Grensen er 50 GB per post og 100 filer; zip-filen er om lag 1,5 GB. Svaret har `checksum` som `md5:...`: sammenlign med MD5 fra steg 9, og stopp hvis de er ulike.
4. **Metadata.** `PUT /api/deposit/depositions/<utkast>` med `{"metadata": {...}}`. Send alle feltene hver gang, ikke bare de som endres; dokumentasjonen sier ikke om feltene flettes eller erstattes.
5. **Kontroll.** `GET` utkastet og skriv ut tittel, versjon, forfatter med ORCID, filnavn og sjekksum, og adressen til forhåndsvisningen. Her stopper skriptet uten flagg.
6. **Publiser.** `POST .../depositions/<utkast>/actions/publish`. Svaret har DOI og record-id, som går rett inn i steg 12.

Metadata for MESA, i API-ets form:

```json
{
  "metadata": {
    "title": "MESA 5.7.1: Mapping Environmentally Sensitive Assets (compiled Windows build)",
    "upload_type": "software",
    "version": "5.7.1",
    "publication_date": "2026-10-13",
    "description": "<p>...</p>",
    "creators": [{"name": "Larsen, Ragnvald",
                  "affiliation": "Norwegian Environment Agency",
                  "orcid": "0000-0002-4184-5224"}],
    "access_right": "open",
    "license": "gpl-3.0-or-later",
    "language": "eng",
    "keywords": ["environmental sensitivity", "oil spill preparedness", "..."],
    "related_identifiers": [
      {"identifier": "https://github.com/ragnvald/mesa", "relation": "isSupplementTo"},
      {"identifier": "https://github.com/ragnvald/mesa/wiki", "relation": "isDocumentedBy"},
      {"identifier": "10.5281/zenodo.21744272", "relation": "references"},
      {"identifier": "10.5281/zenodo.21744039", "relation": "references"}
    ],
    "communities": [{"identifier": "mesatool"}]
  }
}
```

- `description` er HTML. Tillatt er blant annet `p`, `a`, `ul`, `li`, `strong`, `em`, `code`; overskrifter og bilder er ikke tillatt. Beskrivelsen i `ZENODO_<V>.md` er rene avsnitt, så omformingen er å pakke hvert avsnitt i `<p>` og gjøre adressene til lenker.
- `version` og `publication_date` må settes uttrykkelig: de kan være tømt i et nytt utkast.
- Lisens-id-en `gpl-3.0-or-later` er den 5.7.0-posten har i dag.
- **Uavklart:** 5.7.0-posten har også feltene kodelager, programmeringsspråk og utviklingsstatus (`custom`, `code:codeRepository` og så videre) og en referanseliste. Det eldre API-et dokumenterer ikke disse. De arves trolig til utkastet, men det må ses i sandkassen om en `PUT` av metadata lar dem stå.
- **Uavklart:** fellesskapet `mesatool`. I det eldre API-et er det et metadatafelt, og eieren av fellesskapet får en forespørsel. Siden du eier `mesatool` og forrige versjon er med der, følger medlemskapet trolig med, men det er ikke dokumentert.

### Rette en publisert post

Metadata i en publisert post kan endres uten ny versjon: `POST .../actions/edit`, `PUT` med metadata, `POST .../actions/publish`. Det samme skriptet kan altså rette 5.7.0-posten (ORCID, tilknytning, keywords, related identifiers). Filene kan ikke endres via API-et etter publisering.

### Det nyere API-et

Zenodo kjører på InvenioRDM, som har et nyere API (`POST /api/records/<id>/versions`, `PUT /api/records/<id>/draft`, opplasting i tre kall, `POST .../draft/actions/publish`). Det har en annen metadataform og dekker også de egendefinerte feltene. Zenodos egen dokumentasjon beskriver fortsatt bare det eldre API-et og sier ikke at det er på vei ut, så det eldre er det trygge valget nå. Skriptet bør holde Zenodo-kallene i én modul, slik at et bytte senere berører ett sted.

### Sandkassen

`https://sandbox.zenodo.org` er en egen installasjon med egen konto og eget token, og DOI-er som ikke er ekte (`10.5072/...`). Poster derfra finnes ikke i produksjon, så en prøve begynner med å opprette en post der, publisere den, og så kjøre ny-versjon-løpet mot den. Sandkassen kan tømmes når som helst.

### Skriptet

`devtools/zenodo_release.py`, skrevet 2026-10-13. Det leser tittel og beskrivelse fra `ZENODO_<V>.md`; de faste metadataene (forfatter, lisens, keywords, related identifiers, fellesskap) står i skriptet. Tokenet leses fra `ZENODO_TOKEN` eller fra `secrets/zenodo.ini` (`[zenodo]`, `token = ...`, og `sandbox_token = ...` for sandkassen):

```
zenodo_release.py 5.7.1 --zip D:\dist\MESA_571_win11_2026_10_13.zip               utkast, stopper før publisering
zenodo_release.py 5.7.1 --zip ... --publish                                        samme utkast, og publiserer
zenodo_release.py 5.7.1 --discard                                                  sletter utkastet
zenodo_release.py 5.7.0 --fix-metadata --record 23314128 --publish                 retter en publisert post
... --sandbox --record <id>                                                        mot sandbox.zenodo.org
```

Slik oppfører det seg:

- Uten `--publish` lages bare utkast. Et utkast kan slettes; en publisert post kan ikke.
- Det skal kunne kjøres på nytt: finnes det et utkast, brukes det. Ligger riktig fil der med riktig MD5, lastes den ikke opp igjen.
- Det stopper før opplasting hvis zip-filen inneholder `secrets/` eller `log.txt`, og hvis `mesa_version` i zip-filens `config.ini` ikke er versjonen som gis ut.
- Tokenet skrives aldri ut.
- Etter publisering skriver det record-id og DOI til en liten fil som steg 12 og 14 leser.

Med dette skriptet, et for steg 2, 7 og 9, og et for steg 12, gjenstår tre menneskelige punkter i en utgivelse: tekstene, røyktesten og ordet "publiser".
