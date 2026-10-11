# Oppgave til mesa_demodata: demopakkene og veiledningen mot ferdig MESA 5.7.0

Fra: MESA desktop (`d:\code\mesa`), 2026-10-11
Til: mesa_demodata (`D:\data\mesa_demodata`)
Svar ønskes som: `2026-10-XX-demodata-5-7-utgivelse-svar.md`, levert til brukeren, som legger det i `d:\code\mesa\docs\exchange\`.

## Hvorfor du får denne

MESA 5.7.0 bygges denne gangen uten data. `input/` følger med som fire tomme mapper med `readme.txt`, `output/` er tom, og `config.ini` er standardmalen fra git. Brukere som vil prøve MESA, må derfor hente demodata fra Zenodo, og MESA-dokumentasjonen (wiki, brukerhåndbok, utgivelsesnotat, `readme.txt` i `input/`) skal peke dit.

Du bygget pakkene `mesa_5.7.0_*.zip` 2026-09-16 og oppdaterte veiledningen 2026-09-24. Siden da har MESA endret seg på punkter som treffer både `config.ini` i pakkene og teksten i veiledningen. Du kjenner pakkene og Zenodo-postene. Derfor ber vi deg kontrollere og rette, i stedet for at vi gjetter herfra.

## Grenser

- Arbeid bare i `D:\data\mesa_demodata`. Ikke skriv, commit eller push i `d:\code\mesa`.
- Du kan lese `d:\code\mesa`. Husk at det å importere MESA-moduler derfra kan skrive `log.txt` i mappen. Kjør MESA fra en kopi hvis du trenger å kjøre det.
- Publisering på Zenodo gjør brukeren. Du klargjør filer og metadata.

## Hva som er endret i MESA siden pakkene ble bygget

Gjeldende `main` er `12719b6` eller nyere. Les `devtools/RELEASE_NOTES_5.7.0.md` for hele listen. Dette er punktene som berører deg:

1. **Backfill er fjernet.** Stegene er nummerert 1 til 9 uten det. Flagget `--no-backfill` finnes ikke lenger, og `backfill_max_workers` i `config.ini` blir ignorert. Demo-`config.ini` slik den ligger i brukerens arbeidskopi i dag, har fortsatt `backfill_max_workers` og kommentarer med gammel stegnummerering («Segmentation (Stage 5)» der malen nå sier «Stage 4»). Den er altså generert fra en eldre mal.
2. **Classification er valgfri og av som standard.** Malen har `segmv_enabled = 0` og `segmv_k_range = 5`. Overstyringene dine setter `segmv_enabled = 1` og `segmv_k_range = 7-7`. Det kan være riktig for demodata, men da bør veiledningen si at Classification er slått på i pakkene og at det er det tregeste steget.
3. **QGIS-prosjektet er et eget steg (steg 7)** og krever ikke Tiles.
4. **Analysetabellen lagrer celleomriss én gang** (`tbl_stacked` uten geometrikolonne). Pakkene dine har bare inndata, så dette bør ikke berøre dem, men bekreft at ingen av verktøyene dine leser geometri fra `tbl_stacked`.
5. **Special focus:** fanen heter «Study areas» (før «Analysis»). Linjer og studieområder redigeres med Save og Discard.
6. **Rutenettnavn med to sifre** (`H3_R07`, `QDGC_L06`). Dette håndterte du 2026-09-16.
7. **Rapporten** har fått ny sideflyt, bredere kart, tegnforklaringer, en fremdriftslinje med anslått tid, og en ny fremstilling av klassifiseringstypene. Ingen tabell- eller kolonnenavn er endret.

## Hva vi ber deg gjøre

1. **Generer demo-`config.ini` på nytt** med `tools/make_demo_config.py` mot MESA `main`, og kontroller at resultatet ikke lenger har `backfill_max_workers` eller gammel stegnummerering. Gå gjennom `packaging/config.overrides.ini` og fjern overstyringer av nøkler som ikke finnes lenger.
2. **Bygg pakkene på nytt** hvis `config.ini` endret seg, og oppdater `SHA256SUMS.txt` og `manifest.json`.
3. **Rett veiledningen** `mesa_5.7.0_demodata_guide.md`:
   - Avsnitt 5 sier «tick only Flatten + Backfill». Backfill finnes ikke.
   - Avsnitt 5 sier «Lines / Analysis». Fanen heter «Study areas».
   - Avsnitt 4 omtaler `segmv_k_range = 7`, mens overstyringen er `7-7`. Bruk samme skrivemåte, og si at Classification er slått på i pakkene.
   - Si tydelig at MESA 5.7.0 leveres uten data, og at disse pakkene er måten å komme i gang på.
   - Brukeren ønsker ikke tankestreker i tekst. Skriv om setningene der de forekommer, også i `readme_demodata.txt` og i Zenodo-beskrivelsene.
4. **Prøv hver pakke i MESA 5.7.0 fra kildekoden** (fra en kopi): Restore, Process med standardvalg, Tiles, og én rapport. For pakker med linjer eller studieområder også de stegene. Noter for hver pakke: gikk gjennom eller ikke, tid, og alt som så feil ut.
5. **Avklar `uganda_eacop`.** README sier at pakken er ødelagt (eiendelene ligger i Tanzania, 700 km fra geokodene). Den står ikke i `manifest.json`, men zip-filen ligger i `exported/`. Bekreft at den ikke publiseres, eller si hva som skal til.

## Hva vi trenger tilbake

1. **Hvilke DOI-er MESA-dokumentasjonen skal bruke.** Vi har lagt inn konsept-DOI-ene fra `manifest.json`, fordi de alltid peker til nyeste versjon:
   - Åpne data: `10.5281/zenodo.21744272`
   - Syntetiske testdata: `10.5281/zenodo.21744039`

   Bekreft at disse er riktige, og gi versjons-DOI-ene for 5.7.0 når postene er publisert.
2. **Endelig liste over pakker per post**, med filnavn, og en setning om hver som vi kan bruke i wikien.
3. **Hva som ble generert på nytt**, og om pakkene fra 2026-09-16 kan brukes som de er.
4. **Resultatet av prøvingen** per pakke, og feil du mener ligger i MESA. Pek på fil og linje der du kan.
5. **Rekkefølge ved publisering.** Trenger du versjons-DOI-en til MESA 5.7.0 før demodata kan publiseres, eller omvendt? Zenodo-metadataene dine viser til appen, og appens dokumentasjon viser til deg.

## Hva MESA-siden gjør i mellomtiden

- `build_all.py` tar `config.ini`, `input/`-skjelettet og `qgis/` fra git, sender ikke med `docs/exchange/`, og stopper bygget hvis det finner prosjektdata i pakken.
- Wikien, `readme.txt` i `input/`, utgivelsesnotatet og brukerhåndboken får en fast seksjon om demonstrasjonsdata med de to konsept-DOI-ene over og fremgangsmåten med Restore.
- Teksten der justeres når svaret ditt foreligger.
