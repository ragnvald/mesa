# MESA System Overview

## 1. What MESA Delivers
MESA (Methods for Environmental Sensitivity Assessment) is a guided desktop workspace for teams that need to understand which assets are most vulnerable. Instead of stitching together numerous GIS tools, MESA offers one launcher with clearly labelled buttons. Each button opens a focused helper window that walks you through a single task—importing assets, setting up scoring rules, processing results, or exporting maps and reports. The helpers share the same project data, so once a step is completed every other screen can immediately build on it. The result is a predictable workflow that produces comparable outputs every time.

## 2. A Typical Assessment Journey
1. **Bring assets into the project.** Start with the *Import* button to load wildlife sites, infrastructure, or other layers. The importer checks projection, duplicates, and attribute names so later steps run smoothly.
2. **Prepare supporting geography.** Use *Geocodes* if you need analytical tiles/polygons and *Atlas* to create optional atlas tiles.
3. **Agree on scoring rules.** Open *Processing setup* to review sensitivity categories, weights, naming conventions, and default colours so the entire team uses the same assumptions.
4. **Run the heavy lifting.** *Process areas* calculates all indices for polygon layers, while *Process lines* handles pipelines, rivers, or roads. These steps can take time on large regions, but they only need to be run when inputs or scoring rules change.
5. **Explore and fine-tune.** *Asset maps* lets you toggle groups, drag layers into folders, and even request AI-generated colour themes for quick presentations. *Analysis maps* gives a broader overview for QA.
6. **Tell the story.** Use *Analysis design* and *Compare study areas* to define study zones and compare them. Finish by clicking *Report engine* to export a Word report (`.docx`) for sharing and further work.

## 3. Workflows Tab – Button Guide
| Button | What happens when you click it | When to use it |
| --- | --- | --- |
| **Area assets** | Imports asset datasets from `input/asset/` into the shared GeoParquet store. | First step of every project or whenever you receive updated asset data. |
| **Geocodes** | Builds analysis geocodes (grids/polygons) or refreshes existing ones. `basic_mosaic` is the standard, always-available base grid used to attach results to stable cells across runs. It is also the default geocode basis used by the report engine to keep reporting consistent and simpler. | When you need consistent zones for statistics or heat maps. |
| **Line assets** | Opens the line admin/editor tool for importing lines and tuning segmentation parameters. | When working with pipelines, rivers, transmission lines, etc. |
| **Atlas** | Creates or refreshes atlas tiles used for QGIS atlas layouts and optional per-tile outputs. | When you maintain atlas tiles or want per-tile outputs. |
| **Area processing parameters** | Adjust sensitivity categories, weights, descriptions, and defaults used during processing. | Any time policies change or before sharing the project with a new audience. |
| **Analysis design** | Define study polygons and groups used for comparisons. | When preparing scenario comparisons or defining management areas. |
| **Process area** | Runs the core engine that intersects assets with geocodes and calculates the indices. | After imports or configuration edits; required before reporting. |
| **Process line** | Processes line assets into analysis-ready segments. | When you want segment-level sensitivity along lines. |
| **Process area analysis** | Builds analysis outputs for the configured study areas. | After defining study areas and after area processing. |
| **Raster tiles** | Optional: generate raster MBTiles for fast map viewing. | When you want raster tiles in addition to vector outputs. |
| **Asset map** | Opens the interactive asset layer viewer. | For quick workshops, storyboards, or to verify that processed data looks right. |
| **Analysis map** | Opens the maps overview viewer. | For rapid QA or when demonstrating coverage to partners. |
| **Compare study areas** | Presents dashboards, charts, and comparisons for defined study areas. | To brief decision makers on differences between zones. |
| **Report engine** | Generates a Word report (`.docx`) from the most recent processed data. | Final step before sharing results externally. |

## 4. Settings Tab – Polishing Tools
These buttons keep the project information tidy and easy to understand:
- **Edit config** – Update project name, contact details, theme colours, and other global settings without opening a text editor.
- **Backup / restore** – Create a ZIP backup of `input/`, `output/`, and `config.ini`, or restore from a previous backup.
- **Edit geocodes** – Rename grid cells or administrative units so atlases and charts speak the same language as local partners.

Asset and atlas metadata editing is available under **Workflows → Configure processing**.

## 5. Other Tabs in the Launcher
- **Status** – Live dashboard that shows key counters, recent activity, and quick links.
- **About** – Background on the method, partner organisations, and links to training material.

The older Register tab has been removed in MESA 5.

## 6. Practical Tips for Users
- Work from top to bottom in the Activities list when starting a fresh region; the buttons are intentionally ordered to match the recommended workflow.
- If you change scoring rules or import new data, rerun the relevant processing step before exploring maps or exporting reports.
- Launch helpers from inside the MESA hub whenever possible. The hub keeps track of your progress, ensures the right project is open, and centralises log messages if something goes wrong.
- Save time during workshops by customising colours or folders inside Asset maps; those adjustments are remembered the next time you open the viewer.

By keeping the focus on clear tasks and reusable helpers, MESA lets environmental teams move from raw data to polished decision material without worrying about the underlying technical plumbing.
