A correction release. MESA 5.7.1 fixes five things in 5.7.0 and adds nothing new. Projects and results are not affected, and nothing needs to be run again.

### Where can I find it?

- [Zenodo record](https://doi.org/10.5281/zenodo.23481060): `MESA_571_win11_2026_10_13.zip` (Windows 11, 64-bit) and `MESA_571_macos_2026_10_17.dmg` (macOS, Apple Silicon; signed and notarised).
- [All versions, and the reference to cite](https://doi.org/10.5281/zenodo.10775727)
- [MESA project wiki](https://github.com/ragnvald/mesa/wiki), with the full [changelog](https://github.com/ragnvald/mesa/wiki/Mesa-changelog-overview)

### Demonstration data

Unchanged from 5.7.0. The download contains no project data. Nine demonstration packages are published as two records of their own, each with a guide:

- [Synthetic test data](https://doi.org/10.5281/zenodo.21744039): four small projects. The right first choice.
- [Open data from Uganda, Kenya and Ghana](https://doi.org/10.5281/zenodo.21744272): five projects.

## Fixes

### The name of the program

The Welcome and About tabs spelled MESA out as "Method for an Easy Sensitivity Assessment". MESA is **Mapping Environmentally Sensitive Assets**. The Welcome text had been wrong since 5.1 and the About text since 5.0. The wiki, the user guide and the reports were right.

### The Lines tab could open empty

In **Special focus**, the Lines tab could open with no lines and the message "API error: Error: no bridge method get_state". It depended on timing and came with 5.7.0. The lines themselves were never affected: closing Special focus and opening it again showed them. The tab now waits until it can reach its data.

### Windows that cut off text and tick boxes

Config, Manage data, the report window, the GeoNode window and Process in Advanced mode could open too small for what they hold, and text and tick boxes were cut at the edge. These windows now open large enough for their content where the screen allows, and show a scroll bar where it does not. Two tick boxes in the GeoNode window have shorter labels, with the detail in the tooltip.

### A command-line window during processing

A black command-line window could appear and close during processing, typically in the Classification stage. It was harmless and is gone.

### The version shown after an upgrade

A build that was started with a data folder kept from an earlier version showed the earlier version number in the title bar and the banner. It now shows its own.

### Smaller

- Three hints in **Compare study areas** named a script instead of saying what to do. They now say to run Process with Analysis ticked.

## Upgrading

On Windows, unzip 5.7.1 into a folder of its own and start `mesa.exe`. On a Mac, open the disk image and drag MESA to Applications. Your data folder is used as it is. MESA asks whether to continue with the existing data, since it was last used with another version: answer that you continue. Nothing needs to be imported or processed again.
