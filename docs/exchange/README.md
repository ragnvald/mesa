# Exchange — brev til og fra søsterprosjektene

Fullstendige, ordrette dokumenter utvekslet med `mesa-server` (og eventuelt `mesa_demodata`).
Filnavn er `YYYY-MM-DD-<tema>-<type>.md`, datert den dagen dokumentet ble sendt.

**Dette er revisjonssporet, ikke oppslagsverket.** Skal du vite hva som *gjelder* — hvilke
kolonner `tbl_stacked` har, hvilke verdier `mesa_aggregation` kan ha, hva som fortsatt er
uavklart — så les «Agreed formats» i [`cooperation.md`](../../cooperation.md). Den vedlikeholdes
og er alltid gjeldende. Brevene her er frosne i tid og kan være overkjørt av senere brev.

Fortellingen om hva som skjedde og i hvilken rekkefølge står i
[`history.md`](../../history.md); reglene destillert ut av den i
[`learning.md`](../../learning.md).

## Innhold

| dato | dokument | retning |
|---|---|---|
| 2026-07-26 | `2026-07-26-klassifisering-spesifikasjon.md` — Classification (segmv), presis nok til å reimplementeres mot PostGIS | desktop → server |
| 2026-08-08 | `2026-08-08-tbl-stacked-geometri-svar.md` — ja til å fjerne geometrikolonnen; nei til versjonsmarkør; lesere før skrivere | desktop → server |
| 2026-08-08 | `2026-08-08-veiledning-mesa-server.md` — speilingsveiledning med minneseksjon skrevet mot 32 GB | desktop → server |
| 2026-08-09 | `2026-08-09-tegnerekkefolge-svar.md` — svar på seks spørsmål om tegnerekkefølge og maks-aggregering | desktop → server |
| 2026-09-16 | `2026-09-16-geokodenavn-ledende-null-prompt.md` — oppgave: bygg demopakkene om til navn med ledende null (5.7.0) | desktop → demodata |
| 2026-09-16 | `2026-09-16-geokodenavn-ledende-null-svar.md` — åtte pakker bygget om, bare `config.ini` endret, null gamle navn | demodata → desktop |
| 2026-09-16 | `2026-09-16-geokodenavn-ledende-null-melding.md` — navneformatet, migreringen, og tre spørsmål | desktop → server |
| 2026-09-16 | `2026-09-16-geokodenavn-ledende-null-server-svar.md` — ja til formatet; samme reservefeil rettet hos server; 5.7.0-pakker med output avvises inntil kontrakten er utvidet | server → desktop |

## Arkivet er ensidig

Alle fire dokumentene er sendt **fra** desktop. Mesa-servers to forslag — geometriforslaget
(2026-08-07) og renderingsforslaget (2026-08-08) — samt spørsmålsbrevet (2026-08-09) kom som
innlimt tekst i en arbeidsøkt og ble aldri lagret som filer. De bør hentes inn fra mesa-server og
legges her, ellers leser mappa som en monolog og svarene mangler det de svarer på.
