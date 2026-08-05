"""Generate qgis/mesa_results.qgz — styled vector result layers per geocode group.

The shipped qgis/mesa.qgz is a hand-maintained template that only carries the
always-present `basic_mosaic` results. The geocode layers a project actually has
(H3/QDGC levels, uploaded admin layers) vary run to run, so their result layers
cannot be frozen into the template. This module clones the template's own styled
vector layers once per geocode group present in tbl_flat, retargeting the subset
filter, and writes a second project alongside the template — the seed is never
modified.

For each geocode group it emits three vector layers from tbl_flat:
  - Sensitivity  (rule renderer on sensitivity_code_max, cloned from the seed)
  - Importance   (rule renderer on importance_code_max, cloned from the seed)
  - Overlap index (OWA) — a 5-class rule renderer on index_owa, built here
    because the seed carries OWA only as a raster.

Every geocode group also gets interactive saved 3D map views — two each, one
geocode layer per view:
  - asset overlap — extruded by assets_overlap_total relative to that level's
    maximum, on a recoloured green-to-red A–E ramp
  - sensitivity — extruded by the A–E code itself, A tallest and E short but
    present, on the seed's own sensitivity palette so it matches the 2D map
Both cap the tallest column at a multiple of the grid's cell width.

Best-effort by contract: any failure logs and returns without raising, so it can
never break the processing stage that calls it. See learning.md
"Generated QGIS results project".
"""
from __future__ import annotations

import copy
import json
import math
import re
import uuid
import warnings
import zipfile
from array import array
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

DOCTYPE = "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"

# The sensitivity layer is the single styling reference; Importance reuses its
# A–E renderer with the filter column swapped to importance_code_max. Deriving
# both from one reference keeps them in visual lockstep and independent of the
# seed's separate Importance layer.
_SENS_REF = "Sensitivity - screen"
_SENS_ATTR = "sensitivity_code_max"
_IMP_ATTR = "importance_code_max"

_ASSET_COUNT_ATTR = "assets_overlap_total"
_AREA_ATTR = "area_m2"
_GRID_LEVEL_PATTERNS = {
    "H3": re.compile(r"^H3(?:[_ -]?R)?(\d+)$", re.IGNORECASE),
    "QDGC": re.compile(r"^QDGC(?:[_ -]?L)?(\d+)$", re.IGNORECASE),
}
# basic_mosaic and uploaded polygon layers are geocodes like any other and get
# their own 3D views, but they carry no grid level, so they stay out of
# _GRID_LEVEL_PATTERNS and have their cell width counted instead.
_MOSAIC_GROUP = "basic_mosaic"
_MOSAIC_FAMILY = "MOSAIC"
_OTHER_FAMILY = "OTHER"
_MAX_3D_FEATURES_PER_CHUNK = 25_000

# A clear low-to-high vulnerability ramp for the 3D landscapes. The seed's
# yellow-to-dark-red ramp reads as brown/rust once shaded on extruded geometry.
_SENSITIVITY_3D_COLORS = {
    "E": (26, 152, 80),    # very low — green
    "D": (145, 207, 96),  # low — light green
    "C": (255, 255, 191), # moderate — pale yellow
    "B": (252, 141, 89),  # high — orange
    "A": (215, 48, 39),   # very high — red
}

# Keep the columns legible without letting them dominate the underlying map.
# These are deliberately half the original 0.12 / 50 m / 5000 m scale.
_LANDSCAPE_HEIGHT_RATIO = 0.06
_LANDSCAPE_MIN_HEIGHT_M = 25.0
_LANDSCAPE_MAX_HEIGHT_M = 2500.0
# The scene span alone gives every grid the same column height, so a fine grid
# ends up with columns tens of times taller than the cell is wide — they read as
# pipes, and their tops never face the camera. Cap the tallest column at this
# multiple of the cell width. See learning.md "3D landscape proportions".
_LANDSCAPE_CELL_ASPECT = 4.0

# Nominal cell width per grid level, needed for that cap and derived from the
# level alone so no geometry is read. H3's average edge shrinks by sqrt(7) per
# resolution from 1107.7 km at R0, and a hexagon is two edges across; a QDGC
# level-n cell is 2^-n degrees square.
_H3_R0_EDGE_M = 1_107_712.591
_METRES_PER_DEGREE = 111_320.0

# Glassy columns: enough transparency to see the ones behind, a tight white
# highlight, and a darker ambient so the lit and shaded faces separate. Dial
# opacity back up towards 1 if the columns start reading as hollow — QGIS draws
# transparent geometry without depth writes. See learning.md "3D landscape
# proportions".
_LANDSCAPE_OPACITY = "0.93"
# basic_mosaic cells follow the assets they were cut from, so a handful are
# enormous: in the Kampala demo the largest is 186 km2, 153,000x the median, and
# at full height it roofs over the whole scene. Cells past this multiple of the
# group's median area keep their height but turn to glass, so the fine structure
# under them stays readable. A regular grid never trips it — every cell is the
# same size. See learning.md "Outsized mosaic cells in 3D".
_OUTSIZED_CELL_AREA_MULTIPLE = 100
_OUTSIZED_OPACITY = "0.3"
# The second landscape category: height straight from the A–E sensitivity code,
# A tallest. E is deliberately short rather than flat, so a fully assessed low
# area still reads as covered instead of missing.
_SENSITIVITY_HEIGHT_RATIOS = {"A": 1.0, "B": 0.75, "C": 0.52, "D": 0.32, "E": 0.15}
_LANDSCAPE_SHININESS = "48"
_LANDSCAPE_SPECULAR_K = "0.45"
_LANDSCAPE_AMBIENT_FACTOR = 0.55

# OWA 0..100 → five classes, pale-yellow to dark-red (sequential, high = more).
_OWA_BINS = [
    ('"index_owa" > 0 AND "index_owa" <= 20', (255, 255, 178), "1–20"),
    ('"index_owa" > 20 AND "index_owa" <= 40', (254, 204, 92), "21–40"),
    ('"index_owa" > 40 AND "index_owa" <= 60', (253, 141, 60), "41–60"),
    ('"index_owa" > 60 AND "index_owa" <= 80', (240, 59, 32), "61–80"),
    ('"index_owa" > 80', (189, 0, 38), "81–100"),
]


def _tbl_flat_source(group: str, *, data_only: bool = False) -> str:
    escaped_group = group.replace("'", "''")
    subset = f'\"name_gis_geocodegroup\" = \'{escaped_group}\''
    if data_only:
        subset += f' AND \"{_ASSET_COUNT_ATTR}\" > 0'
    return (f"../output/geoparquet/tbl_flat.parquet|layername=tbl_flat|"
            f"subset={subset}")


def _qcolor(rgb, a: int = 255) -> str:
    r, g, b = rgb
    return f"{r},{g},{b},{a},rgb:{r/255:.6f},{g/255:.6f},{b/255:.6f},{a/255:.6f}"


def _find_maplayer(root: ET.Element, layername: str) -> ET.Element | None:
    for ml in root.iter("maplayer"):
        if ml.findtext("layername") == layername:
            return ml
    return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _set_text(el: ET.Element, tag: str, text: str) -> None:
    child = el.find(tag)
    if child is not None:
        child.text = text


def _clone_cloned_renderer_layer(ref: ET.Element, group: str, layer_id: str,
                                  display: str) -> ET.Element:
    """Deep-copy a reference maplayer, retargeting id, datasource and name.

    The renderer is reused verbatim: it keys on sensitivity_code_max /
    importance_code_max, which every geocode group carries in tbl_flat.
    """
    ml = copy.deepcopy(ref)
    _set_text(ml, "id", layer_id)
    _set_text(ml, "datasource", _tbl_flat_source(group))
    _set_text(ml, "layername", display)
    # customproperties may cache the old display name; keep it consistent.
    for opt in ml.iter("Option"):
        if opt.get("name") == "cached_name":
            opt.set("value", display)
    return ml


def _clone_importance_layer(sens_ref: ET.Element, group: str, layer_id: str,
                            display: str) -> ET.Element:
    """Clone the sensitivity layer and re-key its rules to importance_code_max,
    producing a correct A–E importance map with the same palette."""
    ml = _clone_cloned_renderer_layer(sens_ref, group, layer_id, display)
    rv = ml.find("renderer-v2")
    if rv is not None:
        for rule in rv.findall(".//rules/rule"):
            filt = rule.get("filter")
            if filt:
                rule.set("filter", filt.replace(_SENS_ATTR, _IMP_ATTR))
    return ml


def _build_owa_layer(sens_ref: ET.Element, group: str, layer_id: str,
                     display: str) -> ET.Element:
    """Clone the sensitivity maplayer but replace its renderer with a 5-class
    rule renderer on index_owa, reusing the seed's fill-symbol structure."""
    ml = _clone_cloned_renderer_layer(sens_ref, group, layer_id, display)
    rv = ml.find("renderer-v2")
    if rv is None:
        return ml

    base_symbol = rv.find("symbols/symbol")
    if base_symbol is None:
        return ml  # can't restyle without a template symbol; leave as sensitivity
    base_template = copy.deepcopy(base_symbol)

    # Rebuild <symbols> with one recoloured fill per bin.
    symbols = rv.find("symbols")
    for child in list(symbols):
        symbols.remove(child)
    for i, (_filt, rgb, _label) in enumerate(_OWA_BINS):
        sym = copy.deepcopy(base_template)
        sym.set("name", str(i))
        for opt in sym.iter("Option"):
            if opt.get("name") == "color":
                opt.set("value", _qcolor(rgb))
        symbols.append(sym)

    # Rebuild <rules>.
    rules = rv.find("rules")
    if rules is not None:
        for child in list(rules):
            rules.remove(child)
        for i, (filt, _rgb, label) in enumerate(_OWA_BINS):
            ET.SubElement(rules, "rule", {
                "key": "{" + str(uuid.uuid4()) + "}",
                "symbol": str(i),
                "label": label,
                "filter": filt,
            })
    return ml


def _tree_layer_node(layer_id: str, source: str, display: str,
                     checked: bool) -> ET.Element:
    node = ET.Element("layer-tree-layer", {
        "id": layer_id,
        "legend_exp": "",
        "legend_split_behavior": "0",
        "source": source,
        "patch_size": "-1,-1",
        "providerKey": "ogr",
        "checked": "Qt::Checked" if checked else "Qt::Unchecked",
        "name": display,
        "expanded": "1",
    })
    cp = ET.SubElement(node, "customproperties")
    opt_map = ET.SubElement(cp, "Option", {"type": "Map"})
    ET.SubElement(opt_map, "Option",
                  {"type": "QString", "value": display, "name": "cached_name"})
    return node


def _group_node(name: str, checked: bool) -> ET.Element:
    return ET.Element("layer-tree-group", {
        "name": name,
        "checked": "Qt::Checked" if checked else "Qt::Unchecked",
        "expanded": "0",
        "groupLayer": "",
    })


def _scan_flat(flat: Path) -> tuple[list[str], dict[str, int], dict | None, dict[str, dict]]:
    """Stream the few small attributes needed here; never materialise tbl_flat.

    Per group it returns the cell count, which proportions the columns where
    there is no grid level to measure, and the median cell area, which decides
    which cells are outsized. Areas are kept in an array.array, not a list, so a
    multi-million-row tbl_flat costs 8 bytes a row rather than a Python float.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(flat)
    available = set(parquet.schema_arrow.names)
    if "name_gis_geocodegroup" not in available:
        return [], {}, None, {}

    columns = ["name_gis_geocodegroup"]
    has_asset_count = _ASSET_COUNT_ATTR in available
    if has_asset_count:
        columns.append(_ASSET_COUNT_ATTR)
    has_area = _AREA_ATTR in available
    if has_area:
        columns.append(_AREA_ATTR)

    groups: set[str] = set()
    maxima: dict[str, int] = {}
    cells: dict[str, int] = {}
    areas: dict[str, array] = {}
    for batch in parquet.iter_batches(columns=columns, batch_size=131_072):
        names = batch.column(0).to_pylist()
        counts = batch.column(1).to_pylist() if has_asset_count else [None] * len(names)
        sizes = batch.column(columns.index(_AREA_ATTR)).to_pylist() if has_area \
            else [None] * len(names)
        for raw_name, raw_count, raw_area in zip(names, counts, sizes):
            if raw_name is None:
                continue
            name = str(raw_name)
            groups.add(name)
            cells[name] = cells.get(name, 0) + 1
            if raw_count is not None:
                try:
                    maxima[name] = max(maxima.get(name, 0), int(raw_count))
                except (TypeError, ValueError, OverflowError):
                    pass
            if raw_area is not None:
                try:
                    areas.setdefault(name, array("d")).append(float(raw_area))
                except (TypeError, ValueError, OverflowError):
                    pass

    stats = {
        name: {"cells": count, "median_area": _median(areas.get(name))}
        for name, count in cells.items()
    }
    return list(groups), maxima, _parquet_geo_metadata(flat, parquet=parquet), stats


def _median(values: array | None) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _parquet_geo_metadata(path: Path, *, parquet=None) -> dict | None:
    """Read only GeoParquet metadata, without materialising any geometries."""
    try:
        if parquet is None:
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
        metadata = parquet.metadata.metadata or {}
        if b"geo" not in metadata:
            return None
        return json.loads(metadata[b"geo"].decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _grid_family(group: str) -> str:
    """Which cell-width rule applies to this group."""
    if group == _MOSAIC_GROUP:
        return _MOSAIC_FAMILY
    for family, pattern in _GRID_LEVEL_PATTERNS.items():
        if pattern.fullmatch(group.strip()):
            return family
    # Uploaded polygon layers carry no grid level; their cell width is counted.
    return _OTHER_FAMILY


def _selected_3d_groups(groups: list[str]) -> list[tuple[str, str]]:
    """Every geocode group gets both 3D categories, in the layer tree's order."""
    return [(_grid_family(group), group) for group in groups]


def _geo_metadata_geometry(geo_metadata: dict | None) -> tuple[list[float], dict] | None:
    if not geo_metadata:
        return None
    primary = geo_metadata.get("primary_column")
    column = (geo_metadata.get("columns") or {}).get(primary) if primary else None
    if not isinstance(column, dict):
        return None
    bbox = column.get("bbox")
    crs = column.get("crs")
    if (not isinstance(bbox, list) or len(bbox) < 4
            or not isinstance(crs, dict)):
        return None
    try:
        clean_bbox = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in clean_bbox):
        return None
    return clean_bbox, crs


def _transformed_bbox(geo_metadata: dict | None, target_crs) -> tuple[float, float, float, float] | None:
    """Transform a GeoParquet bbox, sampling edges as well as corners."""
    geometry_metadata = _geo_metadata_geometry(geo_metadata)
    if geometry_metadata is None:
        return None
    from pyproj import CRS, Transformer

    bbox, crs_json = geometry_metadata
    source_crs = CRS.from_json_dict(crs_json)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    xs = [bbox[0], (bbox[0] + bbox[2]) / 2, bbox[2]]
    ys = [bbox[1], (bbox[1] + bbox[3]) / 2, bbox[3]]
    transformed = [transformer.transform(x, y) for x in xs for y in ys]
    valid = [point for point in transformed if all(math.isfinite(value) for value in point)]
    if not valid:
        return None
    tx, ty = zip(*valid)
    return min(tx), min(ty), max(tx), max(ty)


def _centre_lonlat(geo_metadata: dict | None) -> tuple[float, float] | None:
    """Dataset centre in WGS84 degrees; the QDGC cell width depends on latitude."""
    geometry_metadata = _geo_metadata_geometry(geo_metadata)
    if geometry_metadata is None:
        return None

    from pyproj import CRS, Transformer

    bbox, crs_json = geometry_metadata
    to_wgs84 = Transformer.from_crs(
        CRS.from_json_dict(crs_json), CRS.from_epsg(4326), always_xy=True
    )
    lon, lat = to_wgs84.transform((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return (lon, lat) if math.isfinite(lon) and math.isfinite(lat) else None


def _scene_spatial_context(geo_metadata: dict | None) -> tuple[object, tuple[float, float, float, float]] | None:
    """Choose a local metric CRS and transform the GeoParquet dataset bounds."""
    centre = _centre_lonlat(geo_metadata)
    if centre is None:
        return None

    from pyproj import CRS

    lon, lat = centre
    if -80 <= lat <= 84 and -180 <= lon <= 180:
        zone = max(1, min(60, int((lon + 180) // 6) + 1))
        target_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    else:
        target_crs = CRS.from_epsg(3857)

    transformed_bbox = _transformed_bbox(geo_metadata, target_crs)
    return (target_crs, transformed_bbox) if transformed_bbox is not None else None


def _set_main_canvas_extent(root: ET.Element, geo_metadata: dict | None) -> bool:
    """Set the startup 2D map canvas to the supplied coverage bounding box."""
    canvas = root.find("mapcanvas")
    extent = canvas.find("extent") if canvas is not None else None
    if canvas is None or extent is None:
        return False

    from pyproj import CRS
    authid = (canvas.findtext("./destinationsrs/spatialrefsys/authid")
              or root.findtext("./projectCrs/spatialrefsys/authid"))
    wkt = (canvas.findtext("./destinationsrs/spatialrefsys/wkt")
           or root.findtext("./projectCrs/spatialrefsys/wkt"))
    try:
        target_crs = CRS.from_user_input(authid or wkt)
    except (TypeError, ValueError):
        return False

    bounds = _transformed_bbox(geo_metadata, target_crs)
    if bounds is None:
        return False
    for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), bounds):
        child = extent.find(tag)
        if child is None:
            child = ET.SubElement(extent, tag)
        child.text = str(value)
    rotation = canvas.find("rotation")
    if rotation is not None:
        rotation.text = "0"
    return True


def _crs_element(crs) -> ET.Element:
    wrapper = ET.Element("crs")
    spatial = ET.SubElement(wrapper, "spatialrefsys", {"nativeFormat": "Wkt"})
    ET.SubElement(spatial, "wkt").text = crs.to_wkt()
    epsg = crs.to_epsg()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You will likely lose important projection information")
        ET.SubElement(spatial, "proj4").text = crs.to_proj4()
    ET.SubElement(spatial, "srsid").text = "0"
    ET.SubElement(spatial, "srid").text = str(epsg or 0)
    ET.SubElement(spatial, "authid").text = f"EPSG:{epsg}" if epsg else ""
    ET.SubElement(spatial, "description").text = crs.name
    ET.SubElement(spatial, "projectionacronym").text = "utm" if epsg and 32601 <= epsg <= 32760 else "merc"
    ET.SubElement(spatial, "ellipsoidacronym").text = "EPSG:7030"
    ET.SubElement(spatial, "geographicflag").text = "false"
    return wrapper


def _empty_property_collection(parent: ET.Element) -> None:
    option = ET.SubElement(parent, "Option", {"type": "Map"})
    ET.SubElement(option, "Option", {"type": "QString", "value": "", "name": "name"})
    ET.SubElement(option, "Option", {"type": "Map", "name": "properties"})
    ET.SubElement(option, "Option", {"type": "QString", "value": "collection", "name": "type"})


def _extrusion_property_collection(parent: ET.Element, expression: str) -> None:
    option = ET.SubElement(parent, "Option", {"type": "Map"})
    ET.SubElement(option, "Option", {"type": "QString", "value": "", "name": "name"})
    properties = ET.SubElement(option, "Option", {"type": "Map", "name": "properties"})
    extrusion = ET.SubElement(properties, "Option", {"type": "Map", "name": "extrusionHeight"})
    ET.SubElement(extrusion, "Option", {"type": "bool", "value": "true", "name": "active"})
    ET.SubElement(extrusion, "Option", {"type": "QString", "value": expression, "name": "expression"})
    ET.SubElement(extrusion, "Option", {"type": "int", "value": "3", "name": "type"})
    ET.SubElement(option, "Option", {"type": "QString", "value": "collection", "name": "type"})


def _parse_rgb(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    try:
        rgb = tuple(int(part) for part in value.split(",")[:3])
    except (TypeError, ValueError):
        return None
    return rgb if len(rgb) == 3 else None


def _sensitivity_3d_rules(sens_ref: ET.Element, *, seed_colors: bool = False
                          ) -> list[tuple[str, str, tuple[int, int, int], str | None]]:
    """The seed's A–E rules, as (filter, label, colour, code).

    The overlap landscape recolours the ramp green-to-red; the sensitivity
    landscape keeps the seed's own palette so it matches the 2D sensitivity map.
    """
    renderer = sens_ref.find("renderer-v2")
    if renderer is None:
        return []
    colors: dict[str, tuple[int, int, int]] = {}
    for symbol in renderer.findall("./symbols/symbol"):
        color = next((opt.get("value") for opt in symbol.iter("Option")
                      if opt.get("name") == "color"), None)
        parsed = _parse_rgb(color)
        if parsed is not None and symbol.get("name") is not None:
            colors[symbol.get("name")] = parsed
    result = []
    for rule in renderer.findall(".//rules/rule"):
        filter_expression = rule.get("filter")
        if not filter_expression:
            continue
        code_match = re.search(r"'([A-E])'", filter_expression, re.IGNORECASE)
        code = code_match.group(1).upper() if code_match else None
        seed_color = colors.get(rule.get("symbol", ""))
        color = seed_color if seed_colors else (_SENSITIVITY_3D_COLORS.get(code) or seed_color)
        if color is None:
            color = seed_color
        if color is not None:
            result.append((filter_expression, rule.get("label", ""), color, code))
    return result


def _polygon_3d_symbol(rgb: tuple[int, int, int], expression: str,
                       opacity: str = _LANDSCAPE_OPACITY) -> ET.Element:
    symbol = ET.Element("symbol", {"type": "polygon", "material_type": "phong"})
    ET.SubElement(symbol, "data", {
        "alt-clamping": "absolute",
        "alt-binding": "centroid",
        "offset": "0",
        "extrusion-height": "0",
        "culling-mode": "no-culling",
        "invert-normals": "0",
        "add-back-faces": "0",
        "rendered-facade": "3",
    })
    ambient = tuple(max(0, round(value * _LANDSCAPE_AMBIENT_FACTOR)) for value in rgb)
    material = ET.SubElement(symbol, "material", {
        "ambient": _qcolor(ambient),
        "diffuse": _qcolor(rgb),
        "specular": _qcolor((255, 255, 255)),
        "shininess": _LANDSCAPE_SHININESS,
        "opacity": opacity,
        "ka": "1",
        "kd": "1",
        "ks": _LANDSCAPE_SPECULAR_K,
    })
    material_ddp = ET.SubElement(material, "data-defined-properties")
    _empty_property_collection(material_ddp)
    symbol_ddp = ET.SubElement(symbol, "data-defined-properties")
    _extrusion_property_collection(symbol_ddp, expression)
    ET.SubElement(symbol, "edges", {
        "enabled": "0", "width": "0.25", "color": _qcolor((70, 70, 70), 100)
    })
    return symbol


def _outsized_variants(filter_expression: str, label: str, area_threshold: float | None
                       ) -> list[tuple[str, str, str]]:
    """Split an A–E rule into a solid and a glass variant on cell area."""
    if area_threshold is None:
        return [(filter_expression, label, _LANDSCAPE_OPACITY)]
    limit = f"{area_threshold:.1f}"
    return [
        # A null area renders solid: unknown is not outsized.
        (f'({filter_expression}) AND coalesce(\"{_AREA_ATTR}\", 0) <= {limit}',
         label, _LANDSCAPE_OPACITY),
        (f'({filter_expression}) AND \"{_AREA_ATTR}\" > {limit}',
         f"{label} — outsized cell", _OUTSIZED_OPACITY),
    ]


def _add_3d_renderer(layer: ET.Element, layer_id: str,
                     rules: list[tuple[str, str, tuple[int, int, int], str | None]],
                     expression: str | Callable[[str | None], str],
                     area_threshold: float | None = None) -> None:
    """Rule-based 3D renderer. `expression` may vary per A–E code."""
    old = layer.find("renderer-3d")
    if old is not None:
        layer.remove(old)
    renderer = ET.SubElement(layer, "renderer-3d", {"type": "rulebased", "layer": layer_id})
    ET.SubElement(renderer, "vector-layer-3d-tiling", {
        # QGIS 4 uses max-chunk-features; QGIS 3 uses zoom-levels-count.
        # Writing both keeps the generated project useful in either version.
        "zoom-levels-count": "3",
        "max-chunk-features": str(_MAX_3D_FEATURES_PER_CHUNK),
        "show-bounding-boxes": "0",
    })
    root_rule = ET.SubElement(renderer, "rules", {"key": "{" + str(uuid.uuid4()) + "}"})
    for filter_expression, label, rgb, code in rules:
        height = expression(code) if callable(expression) else expression
        for variant, description, opacity in _outsized_variants(
            filter_expression, label, area_threshold
        ):
            rule = ET.SubElement(root_rule, "rule", {
                "filter": variant,
                "description": description,
                "key": "{" + str(uuid.uuid4()) + "}",
            })
            rule.append(_polygon_3d_symbol(rgb, height, opacity))


def _grid_cell_width_m(family: str, group: str, latitude: float) -> float | None:
    """Nominal cell width in metres, from the grid level alone."""
    pattern = _GRID_LEVEL_PATTERNS.get(family)
    match = pattern.fullmatch(group.strip()) if pattern is not None else None
    if match is None:
        return None
    level = int(match.group(1))
    if family == "H3":
        return 2 * _H3_R0_EDGE_M / (math.sqrt(7) ** level)
    # QDGC cells are square in degrees, so they narrow with the latitude.
    narrowing = max(math.cos(math.radians(latitude)), 0.01)
    return (2.0 ** -level) * _METRES_PER_DEGREE * narrowing


def _counted_cell_width_m(extent: tuple[float, float, float, float], cells: int | None) -> float | None:
    """Mean cell width for a group with no grid level, from the cell count.

    The extent is the whole dataset while the cells cover only the assessed part
    of it, so this reads high — it caps the tallest columns without flattening
    the mosaic, which is what it is for.
    """
    if not cells or cells < 1:
        return None
    area = (extent[2] - extent[0]) * (extent[3] - extent[1])
    return math.sqrt(area / cells) if area > 0 else None


def _landscape_max_height(span: float, cell_width: float | None = None) -> float:
    height = span * _LANDSCAPE_HEIGHT_RATIO
    if cell_width:
        height = min(height, cell_width * _LANDSCAPE_CELL_ASPECT)
    return max(_LANDSCAPE_MIN_HEIGHT_M, min(_LANDSCAPE_MAX_HEIGHT_M, height))


def _add_flyby_animation(view: ET.Element, center_x: float, center_y: float,
                         span: float, max_height: float) -> None:
    """Add an approach followed by one complete, comfortably distant orbit."""
    orbit_distance = max(span * 1.25, max_height * 3)
    orbit_z = max_height * 0.25
    keyframes = [
        # A short oblique approach across the landscape.
        (0, center_x - span * 0.20, center_y + span * 0.12, max_height * 0.18,
         max(span * 1.90, max_height * 5), 36, 285),
        (4, center_x - span * 0.08, center_y + span * 0.05, max_height * 0.22,
         max(span * 1.50, max_height * 4), 43, 305),
        # Ease into the orbit while maintaining the approach's angular speed.
        (8, center_x - span * 0.02, center_y + span * 0.012, max_height * 0.24,
         max(span * 1.30, max_height * 3.2), 48, 325),
        (12, center_x, center_y, orbit_z, orbit_distance, 50, 345),
    ]
    # Start the turn at 5°/s, accelerate through 10°/s, then settle at
    # 15°/s. The seven legs still add up to one exact 360-degree orbit.
    orbit_yaws = (365, 405, 465, 525, 585, 645, 705)
    keyframes.extend(
        (16 + index * 4, center_x, center_y, orbit_z, orbit_distance, 50, yaw)
        for index, yaw in enumerate(orbit_yaws)
    )

    animation = ET.SubElement(
        view, "animation3d", {"interpolation": "0", "widget-visible": "1"}
    )
    keyframes_element = ET.SubElement(animation, "keyframes")
    for time, x, y, z, distance, pitch, yaw in keyframes:
        ET.SubElement(keyframes_element, "keyframe", {
            "time": str(time),
            "x": str(x),
            "y": str(y),
            "z": str(z),
            "dist": str(distance),
            "pitch": str(pitch),
            "yaw": str(yaw),
        })


def _build_3d_view(name: str, layer_id: str, base_layer_ids: list[str], crs,
                   extent: tuple[float, float, float, float], max_height: float) -> ET.Element:
    xmin, ymin, xmax, ymax = extent
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    padding = max(width, height) * 0.04
    xmin, ymin, xmax, ymax = xmin - padding, ymin - padding, xmax + padding, ymax + padding
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin)

    view = ET.Element("view", {"name": name, "isOpen": "0"})
    qgis3d = ET.SubElement(view, "qgis3d")
    ET.SubElement(qgis3d, "origin", {"x": str(center_x), "y": str(center_y), "z": "0"})
    ET.SubElement(qgis3d, "extent", {
        "xMin": str(xmin), "yMin": str(ymin), "xMax": str(xmax), "yMax": str(ymax),
        "showIn2dView": "0",
    })
    ET.SubElement(qgis3d, "camera", {
        "field-of-view": "45", "projection-type": "1",
        "camera-navigation-mode": "terrain-based-navigation",
        "camera-movement-speed": str(max(5.0, span / 120)),
    })
    ET.SubElement(qgis3d, "color", {
        "background": _qcolor((214, 228, 238)), "selection": _qcolor((255, 255, 0))
    })
    qgis3d.append(_crs_element(crs))
    lights = ET.SubElement(qgis3d, "lights")
    # A high key light for the highlight on the column tops, and a dim fill from
    # the opposite side so the shaded walls keep their colour instead of going
    # flat. Warm key, cool fill — the difference is what makes the columns read
    # as solid.
    ET.SubElement(lights, "directional-light", {
        "x": "-0.35", "y": "-0.45", "z": "-0.82",
        "color": _qcolor((255, 250, 240)), "intensity": "1.25",
    })
    ET.SubElement(lights, "directional-light", {
        "x": "0.55", "y": "0.40", "z": "-0.30",
        "color": _qcolor((205, 225, 245)), "intensity": "0.35",
    })
    terrain = ET.SubElement(qgis3d, "terrain", {
        "terrain-rendering-enabled": "1", "shading-enabled": "0",
        "map-theme": "", "show-labels": "0", "exaggeration": "1",
        "texture-size": "512", "max-terrain-error": "3",
        "max-ground-error": "1", "elevation-offset": "0",
    })
    terrain_layers = ET.SubElement(terrain, "layers")
    for scene_layer_id in [*base_layer_ids, layer_id]:
        ET.SubElement(terrain_layers, "layer", {"id": scene_layer_id})
    ET.SubElement(terrain, "generator", {"type": "flat"})

    ET.SubElement(view, "camera", {
        "xMap": str(center_x), "yMap": str(center_y), "zMap": str(max_height * 0.25),
        "dist": str(max(span * 1.25, max_height * 3)), "pitch": "50", "yaw": "325",
    })
    _add_flyby_animation(view, center_x, center_y, span, max_height)
    return view


def build_results_project(base_dir: str | Path,
                          log: Callable[[str], None] | None = None) -> Path | None:
    """Write qgis/mesa_results.qgz from the seed template + tbl_flat's groups.

    Returns the output path, or None if it could not be produced (seed or data
    missing, or any error). Never raises.
    """
    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    try:
        base = Path(base_dir)
        seed = base / "qgis" / "mesa.qgz"
        flat = base / "output" / "geoparquet" / "tbl_flat.parquet"
        if not seed.is_file():
            _log(f"[QGIS] Results project skipped: seed {seed} not found.")
            return None
        if not flat.is_file():
            _log(f"[QGIS] Results project skipped: {flat} not found (run processing first).")
            return None

        groups, asset_maxima, geo_metadata, group_stats = _scan_flat(flat)
        # basic_mosaic first, then the rest alphabetically, for a stable tree.
        groups.sort(key=lambda g: (g != "basic_mosaic", g))
        if not groups:
            _log("[QGIS] Results project skipped: tbl_flat has no geocode groups.")
            return None

        # Read the seed (qgs member + the styles .db carried over unchanged).
        with zipfile.ZipFile(seed) as zin:
            names = zin.namelist()
            qgs_name = next(n for n in names if n.endswith(".qgs"))
            qgs_bytes = zin.read(qgs_name)
            extras = {n: zin.read(n) for n in names if n != qgs_name}

        root = ET.fromstring(qgs_bytes)
        coverage_path = base / "output" / "geoparquet" / "tbl_data_extent.parquet"
        coverage_geo_metadata = _parquet_geo_metadata(coverage_path) or geo_metadata
        if not _set_main_canvas_extent(root, coverage_geo_metadata):
            _log("[QGIS] Main map extent unchanged: no usable coverage bounds/CRS metadata.")
        sens_ref = _find_maplayer(root, _SENS_REF)
        if sens_ref is None:
            _log(f"[QGIS] Results project skipped: reference layer {_SENS_REF!r} not found in seed.")
            return None

        projectlayers = root.find("projectlayers")
        # Nest the new group inside "Results (step 4)" when present, else the root.
        tree_root = root.find("layer-tree-group")
        results_group = None
        for g in tree_root.findall("layer-tree-group"):
            if (g.get("name") or "").startswith("Results"):
                results_group = g
                break
        parent_group = results_group if results_group is not None else tree_root

        custom_order = root.find(".//custom-order")

        container = _group_node("Geocode results (vector)", checked=True)
        parent_group.append(container)

        n_layers = 0
        for group in groups:
            sub = _group_node(group, checked=(group == "basic_mosaic"))
            container.append(sub)
            specs = [
                ("Sensitivity", "sens", group == "basic_mosaic"),
                ("Importance", "imp", False),
                ("Overlap index (OWA)", "owa", False),
            ]
            for label, kind, checked in specs:
                lid = _new_id("tbl_flat")
                display = f"{group} — {label}"
                if kind == "owa":
                    ml = _build_owa_layer(sens_ref, group, lid, display)
                elif kind == "imp":
                    ml = _clone_importance_layer(sens_ref, group, lid, display)
                else:
                    ml = _clone_cloned_renderer_layer(sens_ref, group, lid, display)
                projectlayers.append(ml)
                sub.append(_tree_layer_node(lid, _tbl_flat_source(group), display, checked))
                if custom_order is not None:
                    ET.SubElement(custom_order, "item").text = lid
                n_layers += 1

        # Add interactive 3D landscapes for every geocode group that has been
        # assessed. These layers stay off in the main 2D canvas; each saved 3D
        # view has its own independent layer set.
        selected_3d = [(family, group) for family, group in _selected_3d_groups(groups)
                       if asset_maxima.get(group, 0) > 0]
        spatial_context = _scene_spatial_context(geo_metadata) if selected_3d else None
        n_views = 0
        if selected_3d and spatial_context is None:
            _log("[QGIS] 3D views skipped: tbl_flat has no usable GeoParquet CRS/bounds metadata.")
        elif selected_3d:
            crs, scene_extent = spatial_context
            span = max(scene_extent[2] - scene_extent[0], scene_extent[3] - scene_extent[1])
            centre = _centre_lonlat(geo_metadata)
            latitude = centre[1] if centre is not None else 0.0
            overlap_rules = _sensitivity_3d_rules(sens_ref)
            sensitivity_rules = _sensitivity_3d_rules(sens_ref, seed_colors=True)
            if not overlap_rules:
                _log("[QGIS] 3D views skipped: sensitivity colours could not be read from the seed.")
            else:
                overlap_node = _group_node("3D asset overlap landscapes", checked=False)
                sensitivity_node = _group_node("3D sensitivity landscapes", checked=False)
                container.append(overlap_node)
                container.append(sensitivity_node)
                views = root.find("mapViewDocks3D")
                if views is None:
                    views = ET.Element("mapViewDocks3D")
                    root.append(views)
                osm = _find_maplayer(root, "OSM Standard")
                base_layer_ids = [osm.findtext("id")] if osm is not None and osm.findtext("id") else []
                for family, group in selected_3d:
                    maximum = asset_maxima[group]
                    stats = group_stats.get(group) or {}
                    cell_width = _grid_cell_width_m(family, group, latitude)
                    if cell_width is None:
                        cell_width = _counted_cell_width_m(scene_extent, stats.get("cells"))
                    max_height = _landscape_max_height(span, cell_width)
                    median_area = stats.get("median_area")
                    area_threshold = (median_area * _OUTSIZED_CELL_AREA_MULTIPLE
                                      if median_area else None)
                    overlap_expression = (
                        f'coalesce(\"{_ASSET_COUNT_ATTR}\", 0) / {maximum} * {max_height:.6f}'
                    )

                    def sensitivity_expression(code: str | None, ceiling: float = max_height) -> str:
                        ratio = _SENSITIVITY_HEIGHT_RATIOS.get(
                            code or "", min(_SENSITIVITY_HEIGHT_RATIOS.values())
                        )
                        return f"{ceiling * ratio:.6f}"

                    categories = (
                        (overlap_node, overlap_rules, overlap_expression, "3D asset overlap"),
                        (sensitivity_node, sensitivity_rules, sensitivity_expression, "3D sensitivity"),
                    )
                    for node, rules, expression, label in categories:
                        layer_id = _new_id("tbl_flat_3d")
                        display = f"{group} — {label} landscape"
                        layer = _clone_cloned_renderer_layer(sens_ref, group, layer_id, display)
                        _set_text(layer, "datasource", _tbl_flat_source(group, data_only=True))
                        _add_3d_renderer(layer, layer_id, rules, expression, area_threshold)
                        projectlayers.append(layer)
                        node.append(_tree_layer_node(
                            layer_id, _tbl_flat_source(group, data_only=True), display, False
                        ))
                        if custom_order is not None:
                            ET.SubElement(custom_order, "item").text = layer_id
                        views.append(_build_3d_view(
                            f"{label} — {group}", layer_id, base_layer_ids,
                            crs, scene_extent, max_height
                        ))
                        n_layers += 1
                        n_views += 1

        out = base / "qgis" / "mesa_results.qgz"
        xml_body = ET.tostring(root, encoding="unicode")
        qgs_out = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + DOCTYPE + xml_body).encode("utf-8")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr("mesa_results.qgs", qgs_out)
            for n, data in extras.items():
                zout.writestr(n, data)

        _log(f"[QGIS] Wrote {out.name}: {n_layers} vector layer(s) across "
             f"{len(groups)} geocode group(s), {n_views} interactive 3D view(s) "
             f"[{', '.join(groups)}].")
        return out
    except Exception as e:
        _log(f"[QGIS] Results project generation failed ({type(e).__name__}: {e}).")
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate qgis/mesa_results.qgz")
    ap.add_argument("--original_working_directory", default=".")
    args = ap.parse_args()
    build_results_project(args.original_working_directory, log=print)
