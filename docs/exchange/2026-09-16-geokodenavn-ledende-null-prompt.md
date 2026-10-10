# Prompt til mesa_demodata: geokodenavn med ledende null (MESA 5.7.0)

Fra: mesa (desktop), 2026-09-16
Til: mesa_demodata

---

Du jobber i **mesa_demodata**. Oppdater demopakkene til navneformatet MESA 5.7.0 bruker for genererte geokodegrupper. MESA 5.7.0 slippes fredag 2026-09-18.

## Endringen

Genererte H3- og QDGC-grupper får nivåtallet med to sifre og ledende null:

| Før (5.6.0) | Fra 5.7.0 | Regel |
|---|---|---|
| `H3_R7`, `H3_R10` | `H3_R07`, `H3_R10` | `f"H3_R{res:02d}"`, oppløsning 0–15 |
| `QDGC_L6`, `QDGC_L11` | `QDGC_L06`, `QDGC_L11` | `f"QDGC_L{level:02d}"`, nivå 0–12 |

- Bare *gruppenavnet* (`name_gis_geocodegroup`) endres. H3-celleindeksen og QDGC-cellekoden i `code` er uendret, og begge følger fortsatt bibliotekenes egne formater.
- `basic_mosaic` og importerte grupper (for eksempel `analysis_areas` i rwenzori) beholder navnene sine.
- Årsak: med ledende null sorterer navnene riktig også alfabetisk. Før kom `H3_R10` foran `H3_R7` i alle lister.
- Navnet er koblingsnøkkel. Lag **bare** det nye formatet, og ikke regn med at MESA oversetter gamle navn for deg.

## Hva som er funnet i de publiserte 5.6.0-pakkene

Alle åtte pakkene i de to Zenodo-postene er skannet, også innholdet i `settings.xlsx`:

- Real open data, East Africa, `10.5281/zenodo.21744273`: `zirimiti_sample`, `my_area`, `mombasa_reef`, `rwenzori`
- Synthetic test data, `10.5281/zenodo.21744040`: `coastal_zone`, `mountain_area`, `river_system`, `mount_kenya`

Pakkene inneholder bare input. Eneste treff er én kommentarlinje i `config.ini` (linje 140), som er kopiert fra MESAs mal:

```
#   list       -> a comma-separated, case-insensitive list, e.g. basic_mosaic, H3_R9
```

Det finnes ingen gruppenavn i datatabeller, gpkg-filer, `settings.xlsx`, readme-filer eller `demodata_guide.md`. Det meste av arbeidet ligger derfor sannsynligvis i **kildekoden din**, ikke i pakkene.

## Oppgaver

1. **Søk gjennom din egen kode** etter alt som lager, tolker eller hardkoder gruppenavn: `tools/asset_finder.py`, pakkeskript, overlappmåling, maler for readme og guide, og tester. Søk etter `H3_R`, `QDGC_L`, `h3_r`, `qdgc_l`, f-strenger som `f"H3_R{`, og regulæruttrykk som `R(\d+)` og `L(\d+)`.
   - Kode som *lager* navn skal bruke `:02d`.
   - Kode som *tolker* navn skal godta ett eller to sifre og gjøre om til `int`, slik at `H3_R7` og `H3_R07` begge gir 7.
   - Sortering som hittil har kompensert for alfabetisk rekkefølge kan forenkles, men bare hvis det ikke endrer noe annet.
2. **config.ini:** bygg pakkens `config.ini` fra MESA 5.7.0-malen, ikke ved å redigere de gamle for hånd. Legg de nedskalerte ytelsesinnstillingene dine oppå som før. Da følger `mesa_version = 5.7.0` og kommentaren med det nye eksemplet (`H3_R09`) automatisk. Er malen i mesa ikke oppdatert når du kommer hit, stopp og si fra. Ikke rett i mesa.
3. **Bygg alle åtte pakkene på nytt** som `mesa_5.7.0_<navn>.zip`, sammen med `mesa_5.7.0_demodata_guide.md`. Oppdater versjonsnumre i readme og guide.
4. **Verifiser før du melder ferdig:**
   - Skann hver zip, også innholdet i `settings.xlsx`, etter navn uten null: `(H3_R|QDGC_L)\d(?!\d)`. Det skal gi **null treff**.
   - Rapporter antall filer og størrelse per pakke før og etter. Innholdet utover `config.ini` og versjonstekster skal være uendret, så avvik må forklares.
   - Hvis du kan kjøre MESA 5.7 fra kilde: gjenopprett én pakke, generer ett H3- og ett QDGC-nivå, og bekreft at gruppene heter for eksempel `H3_R09` og `QDGC_L08`.
5. **Zenodo:** klargjør nye versjoner av begge postene, men **ikke publiser**. Det gjør brukeren.

## Avgrensning

- Arbeid bare i mesa_demodata. Rediger aldri mesa eller mesa-server, heller ikke når feilen ser ut til å ligge der. Beskriv den og hvor den er, så tar brukeren den videre.
- Ikke commit eller push på eget initiativ. Spør først.
- Før hendelsen inn i `history.md`, og svar med et kort brev som mesa kan legge i `docs/exchange/`: hva som ble endret, hvilke filer, og resultatet av verifiseringen i punkt 4.
