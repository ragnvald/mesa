"""config.ini in the data folder gains the settings a newer MESA expects, and nothing else changes."""
from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "code"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mesa_shared as ms  # noqa: E402

TEMPLATE = """\
# In general - Do not make changes to the configuration file.

[DEFAULT]
input_folder_asset              = input/asset
mosaic_coverage_union           = true

# Repair imported geometry at asset import.
import_validate_geometries      = true

mesa_version                    = 5.7.1

# Classification (sensitivity-pattern clustering).
# segmv_enabled: 1 = Classification joins Process.
segmv_enabled = 0
segmv_geocode_layer = basic_mosaic
segmv_make_png                 = 0

# Write classification_results.gpkg per run.
segmv_export_gpkg              = 0
# AI cluster descriptions are opt-in.
segmv_ai_enabled               = 0

[A]
category_colour = #bd0026
description     = Very high

[VALID_VALUES]
valid_input = 1,2,3,4,5
"""

OLDER = """\
# In general - Do not make changes to the configuration file.

[DEFAULT]
input_folder_asset              = input/my_assets
mosaic_coverage_union           = true

mesa_version                    = 5.5.0

# Classification (sensitivity-pattern clustering).
segmv_geocode_layer = H3_R07
segmv_make_png                 = 1
# AI cluster descriptions are opt-in.
segmv_ai_enabled               = 1

[A]
category_colour = #ff0000
"""


def _values(path: Path) -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False, interpolation=None)
    cfg.read(path, encoding="utf-8-sig")
    out = {("DEFAULT", k): v for k, v in cfg.defaults().items()}
    for section in cfg.sections():
        for key, value in cfg._sections[section].items():
            out[(section, key)] = value
    return out


def _write(path: Path, text: str, newline: str = "\n", bom: bool = False) -> Path:
    data = text.replace("\n", newline).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)
    return path


def test_missing_keys_and_sections_are_added_and_values_kept(tmp_path: Path) -> None:
    template = _write(tmp_path / "template.ini", TEMPLATE)
    target = _write(tmp_path / "config.ini", OLDER)
    before = _values(target)

    added = ms.upgrade_config_file(target, template)

    assert added == ["DEFAULT.import_validate_geometries", "DEFAULT.segmv_enabled",
                     "DEFAULT.segmv_export_gpkg", "A.description", "VALID_VALUES.valid_input"]
    after = _values(target)
    assert set(after) == set(_values(template))
    # what the user had wins, even where the template's default differs
    assert all(after[k] == v for k, v in before.items())
    assert after[("DEFAULT", "mesa_version")] == "5.5.0"
    assert after[("DEFAULT", "segmv_enabled")] == "0"


def test_a_key_comes_with_its_comment_and_a_shared_comment_is_not_repeated(tmp_path: Path) -> None:
    template = _write(tmp_path / "template.ini", TEMPLATE)
    target = _write(tmp_path / "config.ini", OLDER)
    ms.upgrade_config_file(target, template)
    text = target.read_text(encoding="utf-8")

    assert "# Repair imported geometry at asset import.\nimport_validate_geometries" in text
    assert text.count("# Classification (sensitivity-pattern clustering).") == 1
    assert ("# Classification (sensitivity-pattern clustering).\n"
            "# segmv_enabled: 1 = Classification joins Process.\n"
            "segmv_enabled = 0\n"
            "segmv_geocode_layer = H3_R07\n") in text
    # a commented key is set off by blank lines from its neighbours
    assert ("segmv_make_png                 = 1\n\n"
            "# Write classification_results.gpkg per run.\n"
            "segmv_export_gpkg              = 0\n\n"
            "# AI cluster descriptions are opt-in.\n") in text
    assert "[VALID_VALUES]\nvalid_input = 1,2,3,4,5\n" in text


def test_a_second_run_changes_nothing(tmp_path: Path) -> None:
    template = _write(tmp_path / "template.ini", TEMPLATE)
    target = _write(tmp_path / "config.ini", OLDER)
    ms.upgrade_config_file(target, template)
    once = target.read_bytes()
    assert ms.upgrade_config_file(target, template) == []
    assert target.read_bytes() == once


def test_line_endings_and_byte_order_mark_are_kept(tmp_path: Path) -> None:
    template = _write(tmp_path / "template.ini", TEMPLATE)
    target = _write(tmp_path / "config.ini", OLDER, newline="\r\n", bom=True)
    assert ms.upgrade_config_file(target, template)
    data = target.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in data.replace(b"\r\n", b"")
    assert ("DEFAULT", "segmv_export_gpkg") in _values(target)


def test_nothing_is_written_when_nothing_is_missing(tmp_path: Path) -> None:
    template = _write(tmp_path / "template.ini", TEMPLATE)
    target = _write(tmp_path / "config.ini", TEMPLATE.replace("5.7.1", "5.7.0"))
    os.utime(target, (1_000_000, 1_000_000))
    assert ms.upgrade_config_file(target, template) == []
    assert target.stat().st_mtime == 1_000_000


def test_the_same_file_is_left_alone(tmp_path: Path) -> None:
    # Running from source, the data folder's config.ini is the template itself.
    target = _write(tmp_path / "config.ini", OLDER)
    assert ms.upgrade_config_file(target, target) == []
    assert target.read_text(encoding="utf-8") == OLDER


def test_the_shipped_config_restores_what_an_older_copy_lacks(tmp_path: Path) -> None:
    template = ROOT / "config.ini"
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    keys = [i for i, line in enumerate(lines) if line[:1].isalpha() and "=" in line]
    dropped = {keys[3], keys[len(keys) // 2], keys[-1]}
    older = tmp_path / "config.ini"
    older.write_text("".join(line for i, line in enumerate(lines) if i not in dropped), encoding="utf-8")

    added = ms.upgrade_config_file(older, template)

    assert len(added) == 3
    assert _values(older) == _values(template)


def test_mesa_upgrades_the_data_folder_from_the_template_it_ships(tmp_path: Path, monkeypatch) -> None:
    import mesa

    shipped = tmp_path / "_internal"
    shipped.mkdir()
    _write(shipped / "config.ini", TEMPLATE)
    data = tmp_path / "MESA"
    data.mkdir()
    target = _write(data / "config.ini", OLDER)
    logged: list[str] = []
    monkeypatch.setattr(mesa, "RESOURCE_BASE", str(shipped))
    monkeypatch.setattr(mesa, "config_file", str(target))
    monkeypatch.setattr(mesa, "log_to_logfile", logged.append)

    added = mesa._upgrade_config_from_template()

    assert "DEFAULT.segmv_enabled" in added
    assert ("DEFAULT", "segmv_enabled") in _values(target)
    assert logged and "segmv_enabled" in logged[0]
