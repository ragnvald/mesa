# MESA sample package — mount kenya

Region: **Mount Kenya, Kenya (equatorial afro-alpine massif)**. Synthetic, deliberately overlapping asset data for testing MESA end-to-end (processing, sensitivity, segmentation, Classification).

## Use

1. Copy `mount_kenya.gpkg` into `input/asset/` (replace or alongside existing assets).
2. Copy `settings.xlsx` into `input/` (it carries the importance/susceptibility per group).
3. Run Import assets → set up geocodes → process. Importance/susceptibility are matched
   to each gpkg layer by `name_original`.

## Asset groups (layer = group name)

| Group | Importance | Susceptibility | Sensitivity | Code |
|---|---:|---:|---:|:--:|
| Endemic species habitat | 3 | 4 | 12 | C |
| Montane cloud forest | 5 | 5 | 25 | A |
| Glacier ice cap | 1 | 1 | 1 | E |
| Afro-alpine moorland | 5 | 4 | 20 | B |
| Headwater tarn | 5 | 2 | 10 | D |
| Bamboo zone | 3 | 2 | 6 | D |
| Grazing pasture | 2 | 3 | 6 | D |
| Smallholder farmland | 4 | 3 | 12 | C |
| Trekking route | 2 | 3 | 6 | D |
| Logging concession | 4 | 3 | 12 | C |

Groups overlap by design, so stacked cells carry a mix of (importance, susceptibility) pairs — giving the Classification tool real histogram depth to cluster on.
