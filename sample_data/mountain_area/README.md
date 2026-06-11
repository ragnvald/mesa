# MESA sample package — mountain area

Region: **Jotunheimen, Norway (alpine range)**. Synthetic, deliberately overlapping asset data for testing MESA end-to-end (processing, sensitivity, segmentation, Classification).

## Use

1. Copy `mountain_area.gpkg` into `input/asset/` (replace or alongside existing assets).
2. Copy `settings.xlsx` into `input/` (it carries the importance/susceptibility per group).
3. Run Import assets → set up geocodes → process. Importance/susceptibility are matched
   to each gpkg layer by `name_original`.

## Asset groups (layer = group name)

| Group | Importance | Susceptibility | Sensitivity | Code |
|---|---:|---:|---:|:--:|
| Endemic species habitat | 5 | 5 | 25 | A |
| Glacier snowfield | 4 | 5 | 20 | B |
| Old-growth forest | 5 | 4 | 20 | B |
| Headwater stream | 4 | 4 | 16 | B |
| Alpine meadow | 4 | 3 | 12 | C |
| Grazing pasture | 2 | 3 | 6 | D |
| Ski resort | 2 | 2 | 4 | E |
| Hiking trail corridor | 2 | 1 | 2 | E |
| Mining concession | 1 | 2 | 2 | E |
| Hydropower intake | 2 | 1 | 2 | E |

Each group is scattered across the area as a few organic patches plus outer satellites, overlapping other groups at shared hotspots — so stacked cells carry a mix of (importance, susceptibility) pairs, giving the Classification tool real histogram depth to cluster on.
