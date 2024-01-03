# README for MESA Tool (Python Implementation)

## Overview
The MESA (Methods for Environmental Sensitivity Assessment) tool is a Python-based application designed to assess and map environmental sensitivity, particularly in relation to potential pollution incidents in the petroleum industry. It provides a systematic framework for evaluating the vulnerability and sensitivity of environmental assets such as coastal areas, ecosystems, and wildlife to pollution impacts.

The python implementation of the MESA tool is rather complex to get up and running. Installing and running Python on a computer could be demanding. On top of that comes the need to install several python libraries. 

The code can be compiled to a series of executable files (.exe) which run nicely on a Windows computer. The compiled files are rather big and in total they are at the last compilation 1.2 GB. This means distributing the files using Github is not possible.

* Environmental Sensitivity Assessment: Systematic evaluation of various environmental assets' vulnerability to pollution.
* Mapping Capability: Visual representation of sensitivity assessments for easier interpretation and decision-making.
* User-Friendly GUI: Developed using Tkinter for easy navigation and usability.
* Compiled Executable: Distributable .exe version for convenient deployment and usage.
* Customizable Framework: Allows adaptation to different environmental contexts and sensitivity parameters.


## Script Descriptions

**01_import.py**
Importing data or modules (specific details not provided).

**03_data_structure.py**
Shared/general functions, possibly for data structure definitions.

**04_edit_asset_group.py**
Shared/general functions for asset group management.

**04_edit_geocode_group.py**
Reading configuration files and managing geocoded data groups.

**04_edit_input.py**
Shared/general functions for input management.

**05_main_statistics.py**
Likely handles main statistical computations (specific details not provided).

**06_process.py**
Main processing logic of the MESA tool (specific details not provided).

**07_edit_atlas.py**
Functions for editing or managing atlas data.

**user_interface.py**
Functions related to checking and creating folders, and user interface aspects.


## System Requirements

### Libraries
To run this code uncompiled in a python environment the following libraries should be installed:
+ **os** - Provides a portable way of using operating system dependent functionality.
+ **datetime** - Supplies classes for manipulating dates and times.
+ **tkinter** - Standard Python interface to the Tk GUI toolkit.
+ **fiona** - Used for reading and writing geographic data files.
+ **subprocess** - Allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
+ **configparser** - Used for handling configuration files.
+ **PIL** (Python Imaging Library) - Adds support for opening, manipulating, and saving many different image file formats.
+ **webbrowser** - Provides a high-level interface to allow displaying Web-based documents to users.
+ **osgeo** - Typically associated with GDAL/OGR library, which is used for reading and writing vector and raster geospatial data formats.
+ **pandas** - Offers data structures and operations for manipulating numerical tables and time series.
+ **threading** - Used to run code concurrently.
+ **sqlalchemy** - A SQL toolkit and Object-Relational Mapping (ORM) library for Python.
+ **geopandas** - Extends the datatypes used by pandas to allow spatial operations on geometric types.
+ **glob** - Used for Unix style pathname pattern expansion.
+ **shapely** - Used for manipulation and analysis of planar geometric objects.

The plan is to replace the spatial libraries and the user interface with QGIS native libraries/methods.

### Pocessing capabilities required
This depends very a lot on the input data to be handled. Generally having a complex geocode with many objects will increase the calculation resource requirements.


## Method background
Read more about the method and tool here: https://www.mesamethod.org/wiki/Main_Page