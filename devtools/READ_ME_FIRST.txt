MESA
====

Start MESA by double-clicking mesa.exe.

This folder is the program. It holds no project data, and nothing you put here is read.

Where your data is kept
-----------------------
The first time you start MESA it asks where it should keep your data. That data folder holds
your input files, the results, the reports and the settings (config.ini). MESA proposes a
folder called MESA in your Documents. If your Documents are synchronised by OneDrive or a
similar service, it proposes a folder outside the synchronised area instead, because a MESA
project is often several gigabytes of files that change while MESA runs.

The location of the data folder is shown at the bottom of the MESA window, with a button that
opens it. You can move the folder under the Config tab.

Put your own files in the data folder, not here:

    input\asset     the assets (GeoPackage, Shapefile or GeoParquet)
    input\geocode   grids or other areas that results are summed up for
    input\lines     lines, for sensitivity along a line
    input\images    pictures for the atlas

No data of your own yet?
------------------------
MESA is delivered without project data. Demonstration packages are published on Zenodo.

    Synthetic test data (small, start here):   https://doi.org/10.5281/zenodo.21744039
    Open data from Uganda, Kenya and Ghana:    https://doi.org/10.5281/zenodo.21744272

A package holds input data only. To get from a package to results:

    1. Manage data:  Restore data, and pick the zip file
    2. Assets:       Import assets from folder
    3. Parameters:   Load from Excel (reads input\settings.xlsx)
    4. Geocodes:     Build mosaic
    5. Process

MESA lists these steps when the restore has finished. Each record on Zenodo includes a guide.

Upgrading
---------
Unzip a new version into a folder of its own and start it. Your data folder is not touched.
If the data folder was last used with another version, MESA asks whether to continue with
that data or to start with an empty project. In the second case the old folder is renamed
and kept; nothing is deleted.

More help
---------
Wiki:        https://github.com/ragnvald/mesa/wiki
User guide:  the docs folder inside your data folder
