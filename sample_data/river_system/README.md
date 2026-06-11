# MESA sample package — river system

Region: **Rufiji River, Tanzania (large lowland river)**. Synthetic, deliberately overlapping asset data for testing MESA end-to-end (processing, sensitivity, segmentation, Classification).

## Use

1. Copy `river_system.gpkg` into `input/asset/` (replace or alongside existing assets).
2. Copy `settings.xlsx` into `input/` (it carries the importance/susceptibility per group).
3. Run Import assets → set up geocodes → process. Importance/susceptibility are matched
   to each gpkg layer by `name_original`.

## Asset groups (layer = group name)

| Group | Importance | Susceptibility | Sensitivity | Code |
|---|---:|---:|---:|:--:|
| Fish spawning reach | 5 | 5 | 25 | A |
| Wetland floodplain | 5 | 4 | 20 | B |
| Riparian forest | 4 | 4 | 16 | B |
| Drinking water intake | 5 | 3 | 15 | C |
| River channel | 4 | 3 | 12 | C |
| Irrigation abstraction | 3 | 3 | 9 | D |
| Hydropower reservoir | 3 | 2 | 6 | D |
| Agricultural land | 2 | 2 | 4 | E |
| Settlement urban | 2 | 1 | 2 | E |
| Sand mining | 1 | 1 | 1 | E |

Groups overlap by design, so stacked cells carry a mix of (importance, susceptibility) pairs — giving the Classification tool real histogram depth to cluster on.
