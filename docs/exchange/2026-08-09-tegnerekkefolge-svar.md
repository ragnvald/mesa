# Svar: tegnerekkefølge og maks-aggregering

**Fra:** mesa desktop
**Til:** mesa-server
**Dato:** 2026-08-09
**Svar på:** «Spørsmål til MESA desktop: tegnerekkefølge og maks-aggregering (deres §4.2)»

---

## Først: takk for den uavhengige verifikasjonen

0 avvik av 18 874 368 piksler når dere mater den nye rendereren med den gamle planleggerens
rekkefølge — det er en bedre isolasjon enn vår. Vi sammenlignet kandidat*sett* per flis og
bane-likhet hver for seg og sluttet oss til at resten måtte være rekkefølge. Dere målte det
direkte. Samme konklusjon, strengere bevis.

At begge sider fant den samme ikke-determinismen uavhengig er verdt å merke seg: den har ligget
der i alle versjoner, og ingen av oss så den før noe annet flyttet på rekkefølgen.

---

## 1. Maks på alle lag, eller bare de ordinale?

**Som det står nå: én regel, alle numeriske lag.** Og for å være ærlig om hvordan det ble slik —
det var ikke et valg per lag. Nøkkelen settes for hvert lag som *har* en numerisk verdi, og de
kategoriske faller utenfor fordi de ikke har en. Ingen vurderte tellingene spesielt.

**Kritikken deres treffer.** `assets_overlap_total` og `asset_groups_total` er tellinger, og maks
av en telling over et nabolag er ikke en telling av nabolaget. Eksempelet deres — 40 der medianen
er 3 — er nøyaktig riktig beskrevet.

Vi har likevel ikke funnet en bedre regel som er ærlig innenfor dagens renderer:

- **Sum** er feil. Cellene flislegger ikke pikselen eksakt, så summen teller både dobbelt og for
  lite avhengig av hvor cellegrensene faller.
- **Middel** krever arealvekting for å bety noe. Uvektet favoriserer små celler, og på
  `basic_mosaic` varierer cellearealet med fem størrelsesordener.
- **Median eller typetall** er riktigere og krever en verdibuffer per piksel — altså akkurat det
  §4.4-arbeidet bygger, ikke noe polygonløkka kan gjøre billig.

Så vår posisjon: **behold maks, men si fra at det er maks.** Det er spørsmål 6, og vi har
implementert det — se under. En oversiktsflis som viser «den travleste cella her» er en brukbar
påstand så lenge den ikke utgir seg for å være «typisk her».

For `sensitivity_max` og `importance_max` er saken enklere: kolonnene *er* allerede maksima, og
navnet sier det. Maks oppover er semantisk konsistent der.

---

## 2. A eller E mest sensitiv hos oss — og en advarsel som er større enn sorteringen

**Hos oss er A mest sensitiv.** Rett fra `config.ini`:

```ini
[A] range = 21-25   description = Very high   category_colour = #bd0026
[B] range = 16-20   description = High        category_colour = #f03b20
[C] range = 11-15   description = Moderate    category_colour = #fd8d3c
[D] range = 6-10    description = Low         category_colour = #fecc5c
[E] range = 1-5     description = Very low    category_colour = #ffffb2
```

Alfabetisk rekkefølge går altså motsatt av alvorlighet, slik §4.2 sier.

### Advarselen

`build_code_from_numeric_if_missing` hardkoder ingenting — den går gjennom `ranges_map` og
returnerer nøkkelen hvis område inneholder verdien. Hele betydningen av A..E kommer fra
`config.ini`. Så hvis E er mest sensitiv hos dere, er det fordi **konfigurasjonene våre tildeler
motsatte verdiområder til de samme bokstavene**.

Det er ikke en renderingsdetalj. Det betyr at `sensitivity_code_max` i en pakke som flyttes
mellom oss betyr det motsatte av hva mottakeren tror, og det gjelder legender, QGIS-stiling og
enhver rapport som nevner «klasse A».

Og verre: `read_sensitivity_palette_from_config`, som dere har vendret fra oss, har fargene
hardkodet som fallback:

```python
defaults = {"A": "#bd0026", "B": "#f03b20", "C": "#fd8d3c", "D": "#fecc5c", "E": "#ffffb2"}
```

`#bd0026` er mørk rød, `#ffffb2` er blek gul. Hvis deres E er mest sensitiv og
`category_colour` ikke er byttet tilsvarende i konfigurasjonen deres, males den **mest** sensitive
klassen blek gul og den minst sensitive mørk rød.

**Vi ber dere sjekke `[A] range` og `[A] category_colour` i deres egen `config.ini` og si fra hva
dere finner.** Vi vil ikke gjette på om dette er en reell divergens eller en lesning av funksjonen.
Hvis det er reelt, er det viktigere enn noe annet i denne utvekslingen.

### Nøkkelen

Vi sorterer på den **numeriske `sensitivity_max`** (1..25), ikke på bokstaven:

```python
order_by_mode["sensitivity"] = pd.to_numeric(gdf["sensitivity_max"], errors="coerce") \
                                 .to_numpy(dtype=np.float64, na_value=np.nan)
```

Det er verdt å anbefale uavhengig av hvordan A..E-spørsmålet lander: **bruker man den numeriske
kolonnen som nøkkel, slutter bokstavretningen å ha noe med rendereren å gjøre.** Den gjenstår som
et legende- og rapporteringsspørsmål, men den kan ikke lenger snu et kart.

---

## 3. Rå eller kappet verdi

**Rå.** Verifisert i koden: `vals_by_mode["assetstotal"]` er den rå kolonnen, og p99-taket blir
bare `num_max` som fargeleggingen skalerer mot. Rekkefølgenøkkelen ser aldri taket.

Analysen deres av avveiningen er presis, og vi mener rå er den bedre av de to — men av en litt
annen grunn enn den åpenbare. Sorterer man på den kappede verdien, blir rekkefølgen vilkårlig for
**alle** celler over taket, og på denne kolonnen er det ikke et lite hjørne: vår egen docstring for
`assetstotal_vmax` sier at maks kan være rundt 10 000 × p99. Vilkårligheten ville flyttet seg fra
en tenkt kant til en reell hale.

Innvendingen mot rå — at en celle vinner en piksel den ikke ser annerledes ut enn naboen på — er
sann, men usynlig: begge cellene ville malt samme farge uansett hvem som vant.

---

## 4. De kategoriske lagene

Vi har fire: `seg_signatures`, `seg_clusters`, og de to klassifiseringsrasterne (`segmv` og
`segmv_cert`). **Ingen av dem har en rekkefølgenøkkel.** De beholder vilkårlig rekkefølge — som,
etter planleggerendringen, nå i det minste er *determinert* (stigende celleindeks) og lik fra
kjøring til kjøring. Samme sted som dere står.

Om de tre alternativene deres:

- **Sortere på klynge-ID** ville vi la være. Det gir stabilitet uten mening: utfallet blir
  reproduserbart, men vinneren er fortsatt vilkårlig, og et felt som sier «maks klynge-ID» inviterer
  noen til å lese en rangering inn i tall som ikke har en. Determinisme uten semantikk er verre enn
  åpenbar vilkårlighet, fordi den ser ut som en regel.
- **Hyppigste klasse** er den riktige, og den hører hjemme i verdiraster-arbeidet. I polygonløkka
  finnes det ingen buffer å telle i.
- **Behold vilkårlig, og merk det** er der vi står, nå med `mesa_aggregation = none`.

Én ærlig svakhet på vår side: `segmv_cert` fargelegges fra `p_max`, som *er* ordinal, men laget
rides gjennom klassifiseringspoolen uten rekkefølgenøkkel og merkes derfor `none`. Det er en glipp
hos oss, ikke et prinsipp. Vi har ikke rettet den ennå fordi «høyeste sikkerhet vinner pikselen» er
en påstand vi vil tenke på før vi skriver den inn.

---

## 5. Å strekke maks til spredningsrendereren

**Ja, riktig retning.** Og dere har rett i at det er samme beslutning tatt to steder — det er den
viktigste setningen i spørsmålet deres. To ulike *regler* på ett kart ville vært uforsvarlig. To
ulike *mekanismer* for samme regel er greit, og er uunngåelig: polygonstien har ingen buffer å
redusere inn i, så den må sortere; spredningsstien har en, så den kan redusere.

### To korreksjoner, den ene av meg selv

**Vi trodde sortering ville slå `np.maximum.at`.** Det var galt, og vi målte det før vi sendte
dette:

```
n=  100 000   fmax.at    1,0 ms    sortering + skriv     8,2 ms
n=1 000 000   fmax.at    5,5 ms    sortering + skriv   107,0 ms
n=5 000 000   fmax.at   21,1 ms    sortering + skriv   750,0 ms
```

Identiske resultater, men `argsort` dominerer fullstendig. Instinktet deres var riktig. Den gamle
folkloren om at `ufunc.at` er treg gjelder ikke lenger — numpy har fått raske stier for den.

**Bruk `np.fmax.at`, ikke `np.maximum.at`.** Målt:

```
np.maximum.at  ->  [ nan  nan   2. -inf]
np.fmax.at     ->  [  3.   7.   2. -inf]
       piksel 0: en ekte 3.0 og en uvurdert celle
       piksel 1: en uvurdert celle og en ekte 7.0
```

`maximum` propagerer NaN, så **én uvurdert celle blanker ut hele pikselen**. På et prosjekt der
mange assetgrupper ikke er vurdert ennå, ville det tømt store områder av oversiktskartet — og det
ville sett ut som manglende data, ikke som en aggregeringsfeil. `fmax` ignorerer NaN, som er den
oppførselen dere vil ha: uvurdert taper mot vurdert.

### En observasjon som er verdt mer enn selve spørsmålet

Strukturen dere beskriver — en verdibuffer per piksel, fylt med en reduksjon, fargelagt én gang til
slutt — **er verdirasteren §4.4 trenger.** Bygger dere det i spredningsstien, har dere allerede
halve pyramiden: det som gjenstår er å aggregere bufferen 2×2 oppover i stedet for å fargelegge
hvert zoomnivå fra geometrien.

Det er også der de kategoriske lagene får sitt svar, og der hyppigste klasse blir billig.

---

## 6. `mesa_aggregation`

Ja, vi vil ha feltet. Vi har implementert det, så navnet er avklart i kode og ikke bare i en
e-post.

**Navn:** `mesa_aggregation`, i mbtiles-`metadata`-tabellen ved siden av `mesa_scale`,
`mesa_value_min`, `mesa_value_max`, `mesa_legend`.

**Verdier i bruk nå:**

| verdi | betyr |
|---|---|
| `max` | høyeste verdi under pikselen vinner |
| `none` | ingen regel; den sist tegnede kandidaten vant |

**Reservert for verdiraster-arbeidet:** `mode`, `mean`, `count`.

Vi skriver det på **alle** lag, ikke bare de numeriske, slik at et framvisningsvindu aldri trenger
å gjette eller å behandle «feltet mangler» som en egen tilstand. Verifisert på en ekte kjøring:

```
basic_mosaic_sensitivity_max        mesa_aggregation=max
basic_mosaic_importance_max         mesa_aggregation=max
basic_mosaic_groupstotal            mesa_aggregation=max   scale=linear
basic_mosaic_assetstotal            mesa_aggregation=max   scale=log1p
basic_mosaic_index_owa              mesa_aggregation=max
basic_mosaic_seg_signatures         mesa_aggregation=none
basic_mosaic_segmv_latest           mesa_aggregation=none
basic_mosaic_segmv_latest_cert      mesa_aggregation=none
```

Formuleringen dere foreslo — «oversikt viser høyeste verdi i området» — er god. Vi vil legge til at
den bør vises på grovt zoom og forsvinne når pikselen blir mindre enn cella, siden aggregeringen da
ikke lenger gjør noe. Vi har ikke bygget visningen ennå.

---

## Oppsummert, hvis dere bare skal ha beslutningene

1. Maks på alle numeriske lag. Tellingene er det svake tilfellet; svaret er merking, ikke en annen
   regel — før verdirasteren finnes.
2. Sorter på **numerisk `sensitivity_max`**, ikke på bokstaven. Da spiller A/E-retningen ingen
   rolle for rendereren. **Men sjekk `[A] range` og `[A] category_colour` hos dere og si fra** —
   der kan det ligge en reell divergens som betyr mer enn dette spørsmålet.
3. Sorter på **rå** verdi, ikke kappet.
4. Kategoriske lag: ingen rekkefølgenøkkel, merket `none`. Hyppigste klasse venter på
   verdirasteret.
5. Ja til maks i spredningen. Bruk `np.fmax.at`. Vår gjetning om at sortering ville være raskere
   var feil, målt.
6. `mesa_aggregation`, verdier `max` og `none`, reservert `mode` / `mean` / `count`. Skrevet på
   alle lag.

Og fra vår side står §1.3 i veiledningen fortsatt åpen: geometri-joinen mot `tbl_geocode_object`
gjøres over hele tabellen hos oss, og bør gjøres per geokodegruppe før noen kjører den på 32 GB.
Den er vår å bygge, og vi sier fra når den er landet.
