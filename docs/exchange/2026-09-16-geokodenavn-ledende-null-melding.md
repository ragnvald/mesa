# Genererte geokodenavn får tosifret nivå: H3_R07, QDGC_L06

**Fra:** mesa desktop
**Til:** mesa-server
**Dato:** 2026-09-16
**Gjelder:** `name_gis_geocodegroup` for genererte H3- og QDGC-nett, fra MESA 5.7.0
**Svar ønskes på:** de tre spørsmålene i §5

---

## 1. Endringen

Fra 5.7.0 skriver desktop nivåtallet med to sifre og ledende null i navnet på genererte nett:

| Før (5.6.0 og eldre) | Fra 5.7.0 | Regel |
|---|---|---|
| `H3_R7`, `H3_R10` | `H3_R07`, `H3_R10` | `H3_R{res:02d}`, oppløsning 0–15 |
| `QDGC_L6`, `QDGC_L11` | `QDGC_L06`, `QDGC_L11` | `QDGC_L{level:02d}`, nivå 0–12 |

- Bare *etiketten* i `name_gis_geocodegroup` endres. H3-celleindeksen og QDGC-cellekoden i `code`
  er uendret og følger fortsatt bibliotekenes egne formater.
- `basic_mosaic` og importerte grupper beholder navnene sine.
- Kode: `mesa_shared.h3_group_name()` og `qdgc_group_name()`, commit `e0e2412`. Malen i
  `config.ini` fulgte i `e0e75a0`.
- Formatet står under «Agreed formats» i `cooperation.md`, foreløpig avtalt mellom desktop og
  mesa_demodata. Dere er ikke med på den avtalen før dere har svart.

## 2. Hvorfor

Navnene ble sortert som tekst, så `H3_R10` kom foran `H3_R7` i rapportens kartvalg og i kartene.
Den samme sorteringen snudde `choose_primary_geocode`. Når et prosjekt mangler `basic_mosaic`, skal
den falle tilbake på det *groveste* H3-nettet. Med R6 og R10 i prosjektet valgte den R10, det
fineste nettet og det dyreste å regne på. Med to sifre er alfabetisk rekkefølge lik
nivårekkefølge, og feilen forsvinner uten at noen sorteringskode må endres.

## 3. Hva desktop gjør med gamle navn

`migrate_geocode_group_names()` døper om gamle navn i et prosjekt. Den kjøres når MESA starter,
etter hver restore og før H3- eller QDGC-generering. Den siste er nødvendig fordi generering
erstatter grupper etter navn: uten migrering ville `H3_R07` blitt skrevet ved siden av `H3_R7`.

Migreringen omfatter:
- alle strengkolonner under `output/geoparquet` der **hele verdien** er et gammelt navn, blant
  annet `tbl_geocode_group`, `tbl_geocode_object`, `tbl_stacked`, `tbl_flat` og
  `tbl_seg_mv*`
- filer navngitt etter gruppa i `output/mbtiles/` og `tbl_segmentation/`, pluss `name` og
  `description` i MBTiles-metadataene
- verdier i `config.ini` som `segment_geocode_layer` og `segmv_geocode_layer`, og fritekst i
  `tbl_settings`

Migreringen tar **ikke** navn som står inni en lengre verdi, for eksempel `wards_H3_R7` eller en
sammensatt ID. Det er bevisst, for slike verdier kan ikke tolkes sikkert.

Målt på en kopi av en ekte 5.6-arbeidsmappe (11 grupper, 69 MBTiles): 4 tabeller og 40 filer ble
omdøpt på 0,50 s. Alle 24 tabellene var identiske før og etter, bortsett fra navnene. Det gjelder
radantall, alle andre kolonner og `geo`-headeren.

**For dere betyr det:** en pakke fra mesa-server med gamle navn åpnes riktig i desktop 5.7, fordi
den blir omdøpt ved restore. Det motsatte gjelder ikke. En pakke fra desktop 5.7 har nye navn, og
leserne deres får dem uten noen oversettelse.

## 4. Hva dette kan berøre hos dere

Vi kjenner ikke kodebasen deres, så dette er punkter å sjekke, ikke påstander:

- **Skrivere.** Der dere lager H3- eller QDGC-grupper, vil navnene bli ulike for samme nett i de
  to systemene.
- **Lesere.** Kode som tolker nivået fra navnet bør godta ett eller to sifre og gjøre det om til
  `int`. Kode som sammenligner hele navn mot en konstant (`== "H3_R8"`) slutter å treffe pakker fra
  desktop.
- **Varige identifikatorer.** Er navnet del av noe som publiseres eller lagres utenfor en pakke,
  som primærnøkler i PostGIS, URL-er eller MBTiles-filnavn som noen lenker til, endrer en
  omdøping de identifikatorene. Desktop har ett slikt tilfelle: GeoNode-eksporten publiserer
  datasett som heter `mesa_<gruppenavn>`. Lag som alt er publisert beholder det gamle navnet, og
  en ny eksport etter omdøpingen lager et *nytt* datasett i stedet for å erstatte det gamle, fordi
  eksporten bare spør om erstatning når navnet er likt. Den migreringen når ikke en ekstern server.
  Har dere noe tilsvarende, gjelder det samme.
- **Klassifiseringsspesifikasjonen** (`2026-07-26-klassifisering-spesifikasjon.md`, §2) bruker
  `H3_R8` som eksempel på `params.layer`. Brevet er fryst og blir ikke rettet. Les eksempelet som
  `H3_R08`.

## 5. Spørsmål

1. Hvor lager eller tolker mesa-server gruppenavn for genererte nett, og har dere eksisterende data
   eller identifikatorer med gamle navn?
2. Tar dere i bruk formatet med ledende null i skriverne, og i så fall når?
3. Trenger dere at desktop godtar gamle navn andre steder enn ved restore og generering, for
   eksempel for pakker som leses uten restore?

Svar gjerne med et brev. Da legges det i `docs/exchange/`, og formatet oppdateres i
`cooperation.md` til å gjelde begge prosjektene.

— mesa desktop, 2026-09-16
