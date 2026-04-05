# -*- coding: utf-8 -*-
"""Shared constants for MESA helper scripts.

Centralises file names, directory paths, and other values that were
previously duplicated as magic strings across multiple helpers.  Import from
here rather than hard-coding literals so that a rename or path change only
needs to be made in one place.

Intentionally stdlib-only and side-effect-free.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

PARQUET_SUBDIR = "output/geoparquet"
"""Default sub-path (relative to project root) for GeoParquet outputs."""

# ---------------------------------------------------------------------------
# GeoParquet table file names
# ---------------------------------------------------------------------------

# Asset tables
TABLE_ASSET_GROUP       = "tbl_asset_group.parquet"
TABLE_ASSET_OBJECT      = "tbl_asset_object.parquet"
TABLE_ASSET_HIERARCHY   = "tbl_asset_hierarchy.parquet"

# Geocode tables
TABLE_GEOCODE_GROUP     = "tbl_geocode_group.parquet"
TABLE_GEOCODE_OBJECT    = "tbl_geocode_object.parquet"

# Line tables
TABLE_LINES             = "tbl_lines.parquet"
TABLE_LINES_ORIGINAL    = "tbl_lines_original.parquet"

# Atlas
TABLE_ATLAS             = "tbl_atlas.parquet"

# Processing / flat result tables
TABLE_FLAT              = "tbl_flat.parquet"
TABLE_STACKED           = "tbl_stacked.parquet"
TABLE_SEGMENT_FLAT      = "tbl_segment_flat.parquet"
TABLE_SEGMENTS          = "tbl_segments.parquet"

# Analysis tables
TABLE_ANALYSIS_POLYGONS = "tbl_analysis_polygons.parquet"
TABLE_ANALYSIS_GROUP    = "tbl_analysis_group.parquet"
TABLE_ANALYSIS_FLAT     = "tbl_analysis_flat.parquet"
TABLE_ANALYSIS_STACKED  = "tbl_analysis_stacked.parquet"

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------

DEFAULT_CRS = "EPSG:4326"
"""WGS-84 geographic CRS used throughout the project."""
