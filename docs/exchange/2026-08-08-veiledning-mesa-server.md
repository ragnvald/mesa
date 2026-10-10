# Speiling til mesa-server: hva desktop gjorde, og hvor minnet ryker

**Fra:** mesa desktop
**Til:** mesa-server
**Dato:** 2026-08-08
**Gjelder:** `tbl_stacked`-formatet, GeoParquet-metadata, og flisrendereren
**Forutsetning:** skrevet for en maskin med **32 GB**. Alle minnetall er merket
som målt eller ekstrapolert.

---

## 0. Kort om hva som er gjort

Tre ting landet på desktop i denne økten:

1. **`tbl_stacked` mistet geometrikolonnen** — deres forslag, gjennomført i sin helhet, inkludert
   at Backfill-steget er fjernet.
2. **To feil i GeoParquet-metadata rettet** — en skriver som mistet headeren, og lesere som
   krasjet på filer uten den.
3. **Flisrendereren fikk to runder ytelsesarbeid** (kalt Landing A og B her), som er de delene av
   deres eget renderingsforslag som ikke endrer hva et piksel betyr.

Svaret på geometriforslaget står i `2026-08-08-tbl-stacked-geometri-svar.md`, i samme mappe.
Dette dokumentet er oppskriften, ikke diskusjonen.

---

## 1. Minnet først: hvor det ryker på 32 GB

Rangert etter hvor farlig de er på en 32 GB-maskin. Dette er delen å lese hvis dere bare leser én.

### 1.1 Flis-rendering: WKB-nyttelasten kopieres inn i hver arbeider

**Dette er de 26,7 GB deres.**

Deres egen `pack_wkb`-docstring (arvet fra oss) sier det rett ut:

> *the pool runs under 'spawn', so everything in initargs is pickled once per worker and each
> worker keeps a private copy … the blob form is ~3.2 GB*

3,2 GB × 8 arbeidere = 25,6 GB. Dere målte 26,7 GB anslått og ble OOM-drept på 30,9. Det er ikke
et tilfeldig sammenfall — per-arbeider-kopien **er** taket.

**Fiks:** skriv blobben til fil én gang, send stien, la hver arbeider `mmap`-e den read-only.
Operativsystemet holder én fysisk kopi i sidecachen uansett hvor mange arbeidere det er.

```python
def spill_wkb(geoms, out_path, chunk=500_000):
    arr = np.asarray(geoms, dtype=object)
    offsets = np.zeros(len(arr) + 1, dtype=np.int64)
    with open(out_path, "wb") as fh:
        for start in range(0, len(arr), chunk):
            parts = shapely.to_wkb(arr[start:start + chunk])
            stop = start + len(parts)
            lengths = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
            np.cumsum(lengths, out=offsets[start + 1:stop + 1])
            offsets[start + 1:stop + 1] += offsets[start]
            fh.write(b"".join(parts))
    return offsets                      # blobben ligger aldri i minnet, heller ikke i forelderen

def _worker_init(wkb_path, wkb_offsets, ...):
    global _G_WKB_FH, _G_WKB
    _G_WKB_FH = open(wkb_path, "rb")
    _G_WKB = mmap.mmap(_G_WKB_FH.fileno(), 0, access=mmap.ACCESS_READ)
```

`mmap` støtter `blob[a:b]`-slicing akkurat som `bytes`, så resten av renderkoden er uendret.
Husk å slette spill-fila når poolen er lukket.

**Merk om Linux:** med `fork` gir copy-on-write en viss deling gratis, så gevinsten er mindre
enn på Windows-spawn — men ikke null, fordi Pythons referansetelling skriver til objekthoder og
utløser kopiering. Og i Python 3.14 er standard startmetode på Linux endret til `forkserver`,
som *ikke* arver forelderens minne. Sjekk hva dere faktisk kjører med
(`multiprocessing.get_start_method()`) før dere antar CoW redder dere.

### 1.2 Analyse mot polygon: hele `tbl_stacked` inn i orkestratoren

Vår `AssetAnalyzer._load_stacked_dataset` gjorde `gpd.read_parquet(katalog)` på **hele**
stacked-datasettet i foreldreprosessen, og filtrerte til basic_mosaic etterpå. På deres 30 GB
pakke kjører det ikke i det hele tatt.

**Fiks:** per polygon, hent kandidatkodene først, og les bare de radene:

```python
flt = pc.field("code").isin(codes) & (pc.field("name_gis_geocodegroup") == layer)
df = ds.dataset(src, format="parquet").to_table(columns=wanted, filter=flt).to_pandas()
```

Klippet er uansett en **per-celle**-operasjon. Den gamle koden gjentok samme snitt én gang per
asset som traff cella.

### 1.3 Å dekode `tbl_geocode_object` i geopandas

Målt hos oss, to ganger, på den ekte 19,4-millioners tabellen:

| hva | målt | ekstrapolert til 19,4 mill. celler |
|---|---|---|
| full tabell gjennom `gpd.read_parquet` | 1 269 B/rad | **~23 GB** |
| tre kolonner (`code`, gruppe, `geometry`) | 1,47 GB per 3 mill. celler | **~9,5 GB** |

Begge tallene er for store for 32 GB når noe annet også kjører.

**Dette treffer nå oss også**, og dere bør ikke kopiere vår løsning uendret: da geometrien
forsvant fra `tbl_stacked`, må `tbl_flat` hente cellegeometrien et sted, og vi gjør det med én
join mot `tbl_geocode_object` i forelderen på slutten av flatten. På vår maskin (127 GB) er det
25 sekunder og uproblematisk. **På 32 GB vil vi anbefale å gjøre den joinen per geokodegruppe**,
slik at toppen begrenses av den største gruppa og ikke av hele tabellen. Vi har ikke gjort det
selv ennå, og sier fra når vi gjør det.

Der geometrien bare skal kopieres og ikke tolkes: strøm radgrupper gjennom pyarrow og la WKB-en
være de bytene den allerede er. Vi flyttet sammenslåingen av `tbl_geocode_object` til den formen
tidligere, og den samme operasjonen gikk fra en anslått 23 GB til **målt 0,50 GB topp, 15 s** for
å legge 250 000 objekter til en 2,3 GB-tabell.

### 1.4 Flatten: hele partisjoner gjennom `gpd.read_parquet`

Hver arbeider leste en hel `tbl_stacked`-partisjon og dekodet all WKB til shapely — for så å
kaste geometrien etter en `drop_duplicates`.

**Fiks:** kolonneprojisert pyarrow-lesing, og les aldri geometri:

```python
dataset = ds.dataset(str(path), format="parquet")
wanted = [c for c in columns if c in set(dataset.schema.names) and c != "geometry"]
df = dataset.to_table(columns=wanted).to_pandas()
```

Målt på ekte partisjoner: **0,87 µs/rad → 0,06 µs/rad, 13,5× raskere**, og minnet faller med
hele geometrien. Vi leser ikke geometrien selv når en eldre pakke fortsatt har den — den er
overflødig der uansett, og det gjør at det ikke trengs to kodeveier.

### 1.5 Backfill-steget: leste og skrev hele tabellen tilbake

Steget leste hver partisjon med `gpd.read_parquet`, merget inn én `Int64`-kolonne, og skrev hele
partisjonen tilbake — geometri og alt. På 30 GB er det 30 GB lest og 30 GB skrevet for å legge
til 8 byte per rad.

Kolonnen det gjaldt, `area_m2`, er en egenskap ved **cella**, ikke ved (celle × asset)-paret.
**Vi har fjernet steget helt.** Konsumentene henter arealet fra `tbl_flat`, som har én rad per
celle.

### 1.6 Flis-planlegging: STRtree i forelderen

`plan_tile_tasks` bygde en `sindex` over alle cellene bare for å spørre én gang per flis.
~1,8 GB på basic_mosaic, enkelttrådet, og den vokser med flisantallet.

**Fiks:** flisindekser i Web Mercator er en lukket funksjon av lon/lat, så hver celles
flisområde er ren aritmetikk på bounding-boksene, og bøttingen er én `argsort`. Ingen indeks,
ingen 1,8 GB. Vi verifiserte at utvalget er identisk — se §4.

---

## 2. Kolonnekontrakten for `tbl_stacked`

Dette er formatet vi nå skriver. Vi foreslår at dere skriver det samme.

```
code, ref_geocodegroup, name_gis_geocodegroup,
ref_asset_group, name_gis_assetgroup, asset_group_name, total_asset_objects,
importance, susceptibility, sensitivity, sensitivity_code, sensitivity_description
```

**Fjernet, og hvorfor:**

| kolonne | kornstørrelse | hvor den hører hjemme |
|---|---|---|
| `geometry` | celle | `tbl_geocode_object` |
| `area_m2` | celle | `tbl_flat` (én rad per celle) |
| `id_x`, `id_y` | merge-artefakter | ingen steder |
| `attributes` | assetobjekt | `tbl_asset_object` |
| `grid_cell` | intern chunking | ingen steder |

Målt hos oss: geometrien var **71,6 %** av bytene i et 15,5-millioners partisjonssett (og det var
rutenettlag med billig WKB — deres 83 % på basic_mosaic er det riktige tallet for polygonlag).
Resten av de fjernede kolonnene utgjør ytterligere ~10 % i dag, men **over 40 % av fila** når
geometrien er borte.

Ikke-geometri måler 7,2 B/rad hos oss, som skalert til deres 94 millioner rader gir ~660 MB — det
treffer deres eget 595 MB-anslag.

Skriv tabellen som **ren parquet**, ikke GeoParquet. Det er en funksjon, ikke en detalj: da kan
ingen leser komme til å bruke `gpd.read_parquet` på den ved et uhell.

### `overlap_m2`

Vi har den fortsatt ikke, og vi vil gjerne ha den. Vi regner ut nøyaktig den verdien per rad i
grensefilteret vårt (1 m²-regelen) og kaster den. Det er den ene verdien i tabellen som er ekte
per rad og ikke kan slås opp i ettertid. 8 byte/rad, og den gjør grenseregelen etterprøvbar.

---

## 3. GeoParquet-metadata: en felle dere sannsynligvis har

**Dette stoppet en kjøring hos oss i dag, og feilmeldingen pekte ingen steder i nærheten av
årsaken.**

GeoParquet registrerer geometrikolonnen to steder: en `geo`-nøkkel på **skjemanivå**, som
`gpd.read_parquet` krever, og **per-felt** geoarrow-utvidelsesmetadata. `GeoDataFrame.to_arrow()`
gir dere den andre gratis og mister den første.

Enhver skriver som setter sammen sitt eget Arrow-skjema — altså enhver strømmende skriver — vil
produsere en fil som er **komplett** (geometri og CRS overlever begge) men som geopandas nekter
å åpne med «Missing geo metadata».

Hos oss dukket det opp som:

```
Error during processing: The CRS attribute of a GeoDataFrame without an active
geometry column is not defined.
```

…fordi lesefallbacken vår returnerte en GeoDataFrame med `geometry=None`, og neste linjes
`.crs` kastet. Ingen filnavn, ingen omtale av metadata.

**To ting å gjøre:**

*Skriveren* — legg inn headeren når tabellen bygges fra bunnen. Hent den fra geopandas selv, så
JSON-en ikke kan drive fra spec-en:

```python
buf = io.BytesIO()
gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_parquet(buf, index=False)
buf.seek(0)
meta = dict(pq.ParquetFile(buf).schema_arrow.metadata or {})
# sett bbox fra de faktiske grensene, og bruk meta[b"geo"] i pa.schema(fields, metadata=...)
```

*Leserne* — tål at headeren mangler, i stedet for å feile. Filen kan reddes fullt ut gjennom
Arrow:

```python
try:
    return gpd.read_parquet(path, columns=cols)
except Exception as exc:
    if "Missing geo metadata" not in str(exc):
        raise
table = pq.read_table(path, columns=cols, filters=filters)
return gpd.GeoDataFrame.from_arrow(table)     # geoarrow-feltmetadata bærer geometri OG CRS
```

Verifisert på vår skadde fil: 29 682 celler, alle geometrier intakte, EPSG:4326 hele veien.
**Ingen regenerering nødvendig.**

**Rekkefølge, hvis dere skifter format:** lesere først, skrivere etterpå. En geometrifri pakke
levert til en desktop som ikke er oppdatert gir ikke «rar oppførsel» — den gir krasj.

---

## 4. Flisrendereren: hva som er verdt å speile

Vår renderer har aldri hatt hybriden, senterpunktspredningen, hullfyllingen eller det adaptive
taket deres. Den tegner alltid polygon. Så flekkene deres kan ikke oppstå hos oss, og dette er
ikke en liste over fikser for det problemet — det er listen over ting som gjorde rendringen
raskere uten å endre hva et piksel betyr.

**Målt, ekte prosjekt (28 674 celler, z6–z11, 4 grupper, 20 mbtiles, 195 fliser):**

```
før    63,3 s    360 KB
etter  37,3 s    347 KB
```

**Renderkjernen alene** — én z6-flis med alle 26 975 basic_mosaic-celler som kandidater:

```
før    1488 ms    55,2 µs per celle
etter   485 ms    18,0 µs per celle       3,1x
```

### 4.1 Projeksjonen er den varme løkka — og den må batches

`ring_lonlat_to_pixels` kalte `lonlat_to_tile_px` per verteks, med `math.radians/tan/cos/log` i
Python. Målt: **0,78 µs per verteks**, over 1 027 039 vertekser for 26 975 celler, gjentatt for
hvert zoomnivå og hvert lag.

**Fellen vi gikk i:** første vektorisering gjorde det *per ring* og ga **1,1×**. Gjennomsnittscella
er 38 vertekser, og numpys per-kall-overhead spiste hele gevinsten. Først batchet over alle
ringene i flisen ble det 3,1×.

Mønsteret er: samle alle ringenes koordinater, ett `np.concatenate`, **ett** projeksjonskall,
og deretter `np.minimum.reduceat` / `np.maximum.reduceat` over ring-offsetene for å avgjøre
vektorisert hvilke ringer som faller utenfor flisen og hvilke som kollapser til ett punkt. Da
kjører den sekvensielle desimeringsløkka bare for de få ringene som faktisk trenger den.

### 4.2 Tegnerekkefølgen avgjør pikselen — gi den mening

Én grov piksel dekker hundrevis av celler, og PIL-fyll er «siste vinner». Utfallet avhang altså
av hvilken rekkefølge den romlige indeksen tilfeldigvis returnerte kandidatene i.

Sorter kandidatene stigende etter verdi, så vinner den høyeste verdien. **Det er
maks-aggregering**, det dere vurderte og forkastet som for dyrt — og det koster tre linjer:

```python
order = _G_ORDER_BY_MODE.get(mode)
if order is not None and len(cand_idx) > 1:
    v = order[cand_idx]
    cand_idx = cand_idx[np.argsort(np.where(np.isnan(v), -np.inf, v), kind="stable")]
```

**Én felle:** rekkefølgenøkkelen er ikke alltid fargeverdien. Sensitivitetslaget farges fra en
kategorikode hvis alfabetiske rekkefølge (A..E) går **motsatt** av alvorlighet, så å sortere på
den ville gitt pikselen til den *minst* sensitive cella. Vi holder nøkkelen i et eget oppslag.

### 4.3 PNG: `optimize=True` er ikke verdt det

**1,99 ms/flis mot 0,67 ms** for å spare 6,5 % av bytene. Bruk `compress_level=6`.

Vi prøvde også palettisering, som så ut som 5× gevinst på en syntetisk tett flis — og målte
**tregere og større** på ekte fliser, fordi ekte fliser stort sett er gjennomsiktige og RGBA
komprimerer lange transparente løp bedre enn en indeksflate pluss palett. Vi skrev
palettmaskineriet, målte det, og slettet det igjen.

### 4.4 Om punktsampling — kort, siden vi svarte utfyllende andre steder

Vi anbefaler den ikke, og grunnen er MESAs egne standardverdier: linje- og punktassets bufres til
30 m, som er under én piksel på **hvert** zoomnivå (z6 = 2446 m/px, z12 = 38,2 m/px). Målt over
seks helninger, andel kolonner truffet for et sammenhengende bånd:

```
båndbredde   PIL polygonfyll     pikselsentersampling
    0,05px   verst 189/248       verst   0/248   snitt 38,7
    0,50px   verst 196/248       verst   0/248   snitt 94,5
    1,50px   verst 197/248       verst 196/248   snitt 239,3
```

En bufret elv blir stiplet, og forsvinner helt i verste fall. De to metodene konvergerer først
ved ~1,5 piksel.

**Alternativet vi vil peke på:** operatøren vår foreslo å rendre bare det fineste zoomnivået og
aggregere oppover. Målt hos oss koster hvert zoomnivå omtrent én full gjennomgang av datasettet
(1,00–1,02× av N, fordi flisantallet firedobles mens kandidatene per flis firedeles), så en
pyramide er verdt ~6,9× — og kombinert med å dele én geometri-gjennomgang på alle lagene, ~49×.
Maks-aggregering oppover pyramiden er gratis når man først har verdirasteren, og **den beholder
de tynne trekkene**. Vi har ikke bygget det ennå, og vil gjerne avklare det med dere først.

---

## 5. Hvordan vi verifiserte, og hvorfor det er verdt å kopiere

Ingen av disse endringene ble akseptert på at «totalen ser riktig ut». Hver del ble målt for seg,
og det er den eneste grunnen til at vi oppdaget hva som faktisk skjedde.

- **Flisplanleggeren:** identiske kandidatsett på alle 41 fliser, ingenting tapt eller lagt til,
  men **ulik rekkefølge**. Det forklarte 1,32 % endrede piksler som ellers ville sett ut som en
  regresjon — og beviste samtidig at den gamle utdataen aldri var determinert.
- **Projeksjonen:** bane-identisk med den gamle over 26 984 ringer på tre zoomnivåer i to
  grupper, 0 avvik, inkludert hvilke ringer som droppes helt.
- **`tbl_flat`:** A/B mot forrige versjon på samme syntetiske prosjekt ga identiske verdier i alle
  27 kolonner, geometri inkludert.

To ting vi bygde og så slettet på bevisene: palett-PNG-en, og den per-ring-vektoriserte
projeksjonen. I begge tilfellene var det **målingen**, ikke koden, som hadde vært feil.

Den generelle regelen vi tar med oss: *sjekk hvor stor arrayen den vektoriserte koden faktisk får
i produksjon, ikke i benchmarken.* En benchmark som batcher 153 651 vertekser når produksjonen
batcher 38, måler noe annet enn programmet.

---

## 6. Sjekkliste for 32 GB

Rekkefølgen er den vi ville tatt den i.

1. **`mmap` WKB-nyttelasten i flisrendereren.** Fjerner taket. Liten endring, størst effekt.
2. **Kolonneprojiser hver lesing av `tbl_stacked`.** Aldri `gpd.read_parquet` på den.
3. **Fjern Backfill-steget.** `area_m2` er per celle.
4. **Kodeavgrens polygonanalysen.** Aldri hele tabellen i orkestratoren.
5. **Gjør geometri-joinen mot `tbl_geocode_object` per geokodegruppe**, ikke over hele tabellen —
   se §1.3. Dette er det ene punktet der dere ikke bør kopiere oss uendret.
6. **Bytt `plan_tile_tasks` til bbox-aritmetikk.** Fjerner STRtree-en.
7. **Batch projeksjonen** over hele flisen, ikke per ring.
8. **`compress_level=6`**, ikke `optimize=True`.

Og en advarsel om parallellitet som gjelder uansett: `work_mem` er per operasjon, ikke per
server, og det samme gjelder per-arbeider-nyttelaster i en prosesspool. Elleve samtidige backender
med 3,2 GB hver er 35 GB på en 32 GB-maskin. Dimensjoner arbeiderne etter et **målt** tall per
arbeider, ikke etter kjerneantallet.

---

## 7. Det vi skylder dere

- Å gjøre geometri-joinen per gruppe (§1.3), som gagner dere mer enn oss.
- Et svar på hva `overlap_m2` skal hete og hvilken presisjon den skal ha, hvis dere vil ta den inn.
- En avklaring på pyramiden i §4.4 før noen av oss endrer hvordan grove fliser ser ut.

Si fra hvis noe her er uklart, eller hvis tallene deres avviker fra våre — de er målt på vår
maskin og våre datasett, og forskjeller er informasjon.
