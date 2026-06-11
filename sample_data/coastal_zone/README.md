# MESA sample package — coastal zone

Region: **Zanzibar Channel, Tanzania (tropical coast)**. Synthetic, deliberately overlapping asset data for testing MESA end-to-end (processing, sensitivity, segmentation, Classification).

## Use

1. Copy `coastal_zone.gpkg` into `input/asset/` (replace or alongside existing assets).
2. Copy `settings.xlsx` into `input/` (it carries the importance/susceptibility per group).
3. Run Import assets → set up geocodes → process. Importance/susceptibility are matched
   to each gpkg layer by `name_original`.

## Asset groups (layer = group name)

| Group | Importance | Susceptibility | Sensitivity | Code |
|---|---:|---:|---:|:--:|
| Coral reef | 5 | 5 | 25 | A |
| Seagrass meadow | 4 | 4 | 16 | B |
| Mangrove | 5 | 4 | 20 | B |
| Saltmarsh | 3 | 4 | 12 | C |
| Fish breeding ground | 4 | 3 | 12 | C |
| Tourism beach | 3 | 2 | 6 | D |
| Aquaculture | 2 | 2 | 4 | E |
| Shipping lane | 2 | 1 | 2 | E |

Each group is scattered across the area as a few organic patches plus outer satellites, overlapping other groups at shared hotspots — so stacked cells carry a mix of (importance, susceptibility) pairs, giving the Classification tool real histogram depth to cluster on.
