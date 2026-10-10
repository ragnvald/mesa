# Svar: fjern geometrikolonnen fra tbl_stacked

**Fra:** mesa desktop
**Til:** mesa-server
**Dato:** 2026-08-08
**Svar på:** «Forslag til MESA desktop: fjern geometrikolonnen fra tbl_stacked» (2026-08-07)

---

## Konklusjon

Ja. Vi gjør det. Premisset deres stemmer, målingene stemmer, og analysen av flatten er
korrekt ned til linjenivå: steget leser 94 millioner geometrier for å beholde 18 millioner,
og det er en de-duplisering av noe som ikke burde vært duplisert.

Vi har verifisert at geometrien i tbl_stacked er cellegeometrien. `gpd.sjoin` beholder
venstresidens geometri, og venstresiden er geocode-chunken (processing_internal.py:2100).
Den er funksjonelt bestemt av (name_gis_geocodegroup, code) slik dere skriver.

Målt på en kjøring hos oss nå — 142 partisjoner, 15,5 mill. rader, 395 MB, allerede ZSTD:
geometrien er 71,6 % av bytene. Ikke-geometri er 7,2 B/rad, som skalert til deres 94
millioner gir ~660 MB. Anslaget deres på 595 MB / «under 2 GB» treffer. Merk at 71,6 %
er lavvannsmerket: dette er rutenettlag med billig WKB. Basic_mosaic drar det mot deres 83 %.

## Svar på spørsmål 1: er polygonanalysen den eneste reelle bruken?

Ja — og den er dårligere stilt enn dere tror.

Vi har gått gjennom hele treet. tbl_stacked nevnes i seks moduler; geometrien røres på
nøyaktig tre steder, ikke to. Dere fant flatten og analysen. Det tredje er backfill-steget:
`_backfill_worker` (processing_internal.py:2962) leser hver partisjon med gpd.read_parquet,
merger inn én Int64-kolonne, og skriver hele partisjonen tilbake med geometri og alt.
På pakken deres er det å lese og skrive 30 GB for å legge til 8 byte per rad.

Verdt å vite: den kolonnen backfill dytter inn er area_m2, som flatten regner ut fra
cellegeometrien. Det er en egenskap ved cellen, ikke ved raden. Når geometrien flyttes til
tbl_geocode_object, hører area_m2 hjemme samme sted — og hele backfill-steget forsvinner.

Om polygonanalysen: den laster i dag hele tbl_stacked inn i foreldreprosessen med ett
gpd.read_parquet-kall, og filtrerer til basic_mosaic etterpå
(processing_pipeline_run.py:1483-1493). Det er i strid med vår egen minneregel, og på en
30 GB stacked kjører den ikke. Så forslaget deres koster oss ikke en fungerende funksjon —
det tvinger oss til å fikse en som allerede ikke går. Riktig form er den dere skisserer:
bbox → kandidatkoder → filtrert lesing → geometri på slutten.

Ingen planlagte funksjoner forutsetter geometri per stacked-rad. Den eneste romlige
operasjonen på tabellen er en per-celle-operasjon i per-rad-forkledning.

Til orientering, siden det styrker saken: klassifiseringen vår (segmentation.py /
segmentation_run.py) er allerede geometrifri. Den leser tbl_stacked med pyarrow-
kolonneprojeksjon og prediktat-pushdown på laget, rører aldri geometrien, og henter den fra
tbl_geocode_object først ved tegning. Utdatatabellene bærer ingen geometri. Mønsteret dere
foreslår er altså allerede validert i produksjon hos oss — i ett steg. Flatten og analysen
skal bare gjøre det samme.

## Svar på spørsmål 2: hvordan håndteres overgangen?

Vi er enige i «valgfri kolonne, bruk den hvis den finnes». Men rekkefølgen er ikke
likegyldig, og det er ett konkret sprengpunkt dere bør kjenne til:

**gpd.read_parquet kaster unntak på en parquet uten geo-metadata.** Tre steder i desktop
kaller den bart — flatten-workeren, backfill-workeren og analyselasteren. En geometrifri
pakke restaurert i MESA 5.6 gir ikke «rar oppførsel». Den gir krasj.

Derfor: **lesere først, skrivere etterpå.** Vi slipper en desktop-release som leser begge
formater før dere skriver det nye. Vi sier fra her når den er ute. Inntil da bør pakker
beholde kolonnen.

## Svar på spørsmål 3: bør det være en versjonsmarkør?

Vi foreslår å la være — eller i alle fall å ikke stole på den.

Den robuste sjekken er skjemaet selv: `pyarrow.dataset(src).schema.names`, «geometry» finnes
eller ikke. Den krever ingen koordinering, kan ikke drive ut av synk med den faktiske filen,
og virker også når en pakke er satt sammen stykkevis. mesa_version i config.ini sier noe om
verktøyet som skrev pakken, ikke om hva som faktisk ligger i tabellen — og
restaureringsstien vår tillater å hente config og output fra ulike kilder.

Vil dere ha en eksplisitt markør likevel, foreslår vi `stacked_has_geometry` i
tbl_project_info.parquet (finnes allerede). Men leserne bør behandle skjemaet som fasit og
markøren som dokumentasjon.

## To ting vi vil legge til i endringen

**1. Geometri er ikke det eneste som slepes med.** Kolonnelisten deres nevner ikke alt vår
skriver faktisk legger inn. Målt hos oss: `attributes` 4,6 %, `id_x` 3,8 %, `id_y` 1,6 %,
`asset_group_name` 1,6 %, `total_asset_objects` 1,6 %. id_x/id_y er rene merge-artefakter.
Resten er konstanter per assetgruppe som hører hjemme i et oppslag mot tbl_asset_group.
Det er ~13 % i dag — men over 40 % av filen når geometrien er borte.

Vi foreslår derfor at vi blir enige om en eksplisitt **kolonnekontrakt** for tbl_stacked,
ikke bare «minus geometry». Vi sender et utkast når fase 0 er landet hos oss.

**2. overlap_m2 vil vi ha.** Dere har kolonnen, vi har den ikke. Vi regner ut nøyaktig den
verdien per rad i grensefilteret vårt (`_drop_boundary_only_joins`, 1 m²-regelen) — og
kaster den. Det er den ene verdien i tabellen som er ekte per rad og som ikke kan slås opp
i ettertid. 8 byte per rad, og den gjør grenseregelen etterprøvbar. Skal skjemaet uansett
endres, er dette tidspunktet.

## Om kornstørrelsen

Enig, den skal ikke tilbake. Én rad per (celle × assetobjekt) er vår semantikk, og
argumentene deres for hvorfor rettingen var nødvendig — assets_overlap_total som talte
grupper, for grunt owa_n1..n25-histogram, to 0,6 m²-assets som overlevde som 1,2 — er
korrekte. Rettingen er velkommen.

At radtallet steg 9,3 % mens filen ble uhåndterlig, er akkurat symptomet på problemet:
de nye radene hoper seg opp på cellene med dyrest geometri. En WKB på 2 MB gjentatt tusenvis
av ganger er ikke en kostnad ved kornstørrelsen — det er en kostnad ved å lagre noe med
kornstørrelse «celle» i en tabell med kornstørrelse «par».

Det er regelen vi tar med oss videre: en tabell med kornstørrelse «par» skal ikke bære en
kolonne med kornstørrelse «celle».

## Hva vi gjør nå

Alt dette er nå gjort — hele endringen ligger i arbeidstreet her:

- Leserne er kolonneprojiserte og geometri-frie (`_read_stacked_partition` via pyarrow).
  Flatten leser aldri geometrien, heller ikke fra en eldre pakke som har den; den er
  overflødig der uansett.
- tbl_flat får cellegeometrien fra tbl_geocode_object, skjøtet på
  (name_gis_geocodegroup, code).
- Skriveren vår skriver ikke lenger geometri. Den skriver heller ikke `id_x`, `id_y`,
  `attributes` eller `grid_cell` — se kolonnekontrakten under.
- Backfill-steget er fjernet helt, med config-nøkkel, GUI-avkrysning og `--no-backfill`.
- Polygonanalysen leser nå bare de cellene analysepolygonet faktisk treffer.

**Ikke-regresjon er målt, ikke antatt.** HEAD og den nye koden ble kjørt over samme
syntetiske prosjekt: tbl_flat kom ut identisk i alle 27 kolonner inkludert geometri,
analyseutdataene stemte på alt unntatt kjøre-uuid og tidsstempel, og de beholdte
tbl_stacked-kolonnene er rad for rad like.

## Kolonnekontrakten vi foreslår

```
code, ref_geocodegroup, name_gis_geocodegroup,
ref_asset_group, name_gis_assetgroup, asset_group_name, total_asset_objects,
importance, susceptibility, sensitivity, sensitivity_code, sensitivity_description
```

Alt annet er enten cellekornet (geometry, area_m2) eller assetgruppekornet
(`id_x`/`id_y` fra merge, `attributes`) og hører hjemme i tabellen som er nøklet på det.
`overlap_m2` er den ene kandidaten vi mangler og gjerne vil ha inn — se over.

Zstd-byttet deres i mellomtiden er fornuftig — vår egen skriver har brukt ZSTD-3 hele tiden,
så det gir oss paritet uten formatendring.
