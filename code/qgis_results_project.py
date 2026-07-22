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

Best-effort by contract: any failure logs and returns without raising, so it can
never break the processing stage that calls it. See learning.md
"Generated QGIS results project".
"""
from __future__ import annotations

import copy
import uuid
import zipfile
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

# OWA 0..100 → five classes, pale-yellow to dark-red (sequential, high = more).
_OWA_BINS = [
    ('"index_owa" > 0 AND "index_owa" <= 20', (255, 255, 178), "1–20"),
    ('"index_owa" > 20 AND "index_owa" <= 40', (254, 204, 92), "21–40"),
    ('"index_owa" > 40 AND "index_owa" <= 60', (253, 141, 60), "41–60"),
    ('"index_owa" > 60 AND "index_owa" <= 80', (240, 59, 32), "61–80"),
    ('"index_owa" > 80', (189, 0, 38), "81–100"),
]


def _tbl_flat_source(group: str) -> str:
    return (f"../output/geoparquet/tbl_flat.parquet|layername=tbl_flat|"
            f"subset=\"name_gis_geocodegroup\" = '{group}'")


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

        import pandas as pd
        groups = (pd.read_parquet(flat, columns=["name_gis_geocodegroup"])
                    ["name_gis_geocodegroup"].dropna().astype(str).unique().tolist())
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

        out = base / "qgis" / "mesa_results.qgz"
        xml_body = ET.tostring(root, encoding="unicode")
        qgs_out = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + DOCTYPE + xml_body).encode("utf-8")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr("mesa_results.qgs", qgs_out)
            for n, data in extras.items():
                zout.writestr(n, data)

        _log(f"[QGIS] Wrote {out.name}: {n_layers} vector layer(s) across "
             f"{len(groups)} geocode group(s) [{', '.join(groups)}].")
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
