# MESA System Overview

## 1. What MESA Delivers
MESA (Methods for Environmental Sensitivity Assessment) is a guided desktop workspace for teams that need to understand which assets are most vulnerable. Instead of stitching together numerous GIS tools, MESA offers one launcher with clearly labelled buttons. Each button opens a focused helper window that walks you through a single task—importing assets, setting up scoring rules, processing results, or exporting maps and reports. The helpers share the same project data, so once a step is completed every other screen can immediately build on it. The result is a predictable workflow that produces comparable outputs every time.

## 2. A Typical Assessment Journey
1. **Bring assets into the project.** Start with the *Import* button to load wildlife sites, infrastructure, or other layers. The importer checks projection, duplicates, and attribute names so later steps run smoothly.
2. **Prepare supporting geography.** Use *Grids* if you need analytical tiles and *Define map tiles* to create printable atlas areas.
3. **Agree on scoring rules.** Open *Processing setup* to review sensitivity categories, weights, naming conventions, and default colours so the entire team uses the same assumptions.
4. **Run the heavy lifting.** *Process areas* calculates all indices for polygon layers, while *Process lines* handles pipelines, rivers, or roads. These steps can take time on large regions, but they only need to be run when inputs or scoring rules change.
5. **Explore and fine-tune.** *Asset maps* lets you toggle groups, drag layers into folders, and even request AI-generated colour themes for quick presentations. *Analysis maps* gives a broader overview for QA.
6. **Tell the story.** Use *Analysis setup* and *Analysis results* to define study zones and compare them. Finish by clicking *Export reports* to package PDFs, atlases, and summary tables for stakeholders.

## 3. Activities Tab – Button Guide
| Button | What happens when you click it | When to use it |
| --- | --- | --- |
| **Import** | Opens a step-by-step wizard that ingests raw shapefiles, GeoPackages, or spreadsheets and standardises their fields. | First step of every project or whenever you receive updated asset data. |
| **Grids** | Builds analysis grids (hexagons, squares, admin areas) or refreshes existing ones. | When you need consistent zones for statistics or heat maps. |
| **Define map tiles** | Creates or refreshes atlas boundaries with suggested map scales. | Prior to producing printed or PDF atlas pages. |
| **Processing setup** | An interactive checklist for sensitivity categories, weights, descriptions, and colour suggestions. | Any time policies change or before sharing the project with a new audience. |
| **Process areas** | Runs the core engine that intersects assets with grids and calculates the sensitivity scores. | After imports or configuration edits; required before analysis or reporting. |
| **Process lines** | Similar to Process areas but optimised for linear networks. | When working with pipelines, rivers, transmission lines, etc. |
| **Asset maps** | Launches an interactive viewer with layer toggles, drag-and-drop folders, AI styling, and PNG export. | For quick workshops, storyboards, or to verify that processed data looks right. |
| **Analysis maps** | Shows a consolidated map of current data with ready-made base layers. | For rapid QA or when demonstrating coverage to partners. |
| **Analysis setup** | Lets you digitise or import study polygons and tag them with metadata. | When preparing scenario comparisons or defining management areas. |
| **Analysis results** | Presents dashboards, charts, and comparison tables for the study areas created earlier. | To brief decision makers on differences between zones. |
| **Export reports** | Generates PDF reports, atlases, and tables using the most recent processed data. | Final step before sharing results externally. |

## 4. Settings Tab – Polishing Tools
These buttons keep the project information tidy and easy to understand:
- **Edit config** – Update project name, contact details, theme colours, and other global settings without opening a text editor.
- **Edit assets** – Give each asset group a friendly title, description, and optional styling notes; these appear in reports and the Asset maps viewer.
- **Edit geocodes** – Rename grid cells or administrative units so atlases and charts speak the same language as local partners.
- **Edit lines** – Adjust default buffer widths, segment lengths, and descriptions for line datasets before processing.
- **Edit map tiles** – Refine the atlas layout created earlier: merge tiles, tweak names, and confirm the sequence used in reports.

## 5. Other Tabs in the Launcher
- **Statistics** – Live dashboard that shows when each major step was last completed and highlights outstanding actions.
- **About** – Background on the method, partner organisations, and links to training material.
- **Register** – Optional form where you can provide a name and email so the project can acknowledge contributors or share updates. Consent toggles control whether anonymised usage statistics are uploaded when you are online.

## 6. Practical Tips for Users
- Work from top to bottom in the Activities list when starting a fresh region; the buttons are intentionally ordered to match the recommended workflow.
- If you change scoring rules or import new data, rerun the relevant processing step before exploring maps or exporting reports.
- Launch helpers from inside the MESA hub whenever possible. The hub keeps track of your progress, ensures the right project is open, and centralises log messages if something goes wrong.
- Save time during workshops by customising colours or folders inside Asset maps; those adjustments are remembered the next time you open the viewer.

By keeping the focus on clear tasks and reusable helpers, MESA lets environmental teams move from raw data to polished decision material without worrying about the underlying technical plumbing.
