"""MESA is Mapping Environmentally Sensitive Assets, and nothing else."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Mapping Environmentally Sensitive Assets"
# The Portuguese user guide translates the name; that is the one other form allowed.
ALLOWED = (NAME, "Mapeamento de Activos Ambientalmente Sensíveis")

# Everything a user or a new developer reads: the application, what ships with it,
# and the documents that describe it. The dated records (history, learning, letters,
# old release notes) quote the past and are left alone.
SOURCES = (
    [ROOT / "mesa.py", ROOT / "README.md", ROOT / "instructions.md"]
    + sorted((ROOT / "code").glob("*.py")) + sorted((ROOT / "code").glob("*.md"))
    + sorted((ROOT / "docs" / "templates").glob("*.md"))
    + sorted((ROOT / "plans_oversight").glob("*.md"))
    + [ROOT / "devtools" / "build_user_guide.py", ROOT / "devtools" / "READ_ME_FIRST.txt",
       ROOT / "devtools" / "RELEASE_NOTES_5.7.0.md"]
    + sorted((ROOT / "input").glob("*/readme.txt"))
)

# "MESA (Something Capitalised ...)" is how the name gets spelled out.
SPELLED_OUT = re.compile(r"\bMESA \(([A-Z][A-Za-zÀ-ÿ]+(?: [A-Za-zÀ-ÿ]+){2,})\)")
# The words of the name in any order of mistakes we have seen.
LOOKALIKE = re.compile(r"\b(?:Methods? for (?:an? )?(?:Easy|Environmental) Sensitivity Assessment"
                       r"|Mapping Environmentaly Sensitive Assets"
                       r"|Mapping Environmental(?:ly)? Sensitiv\w* Area\w*)\b", re.IGNORECASE)


def test_the_name_is_spelled_out_in_one_way_only() -> None:
    wrong: list[str] = []
    for path in SOURCES:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for match in SPELLED_OUT.finditer(line):
                if match.group(1) not in ALLOWED:
                    wrong.append(f"{path.relative_to(ROOT)}:{number}: MESA ({match.group(1)})")
            for match in LOOKALIKE.finditer(line):
                wrong.append(f"{path.relative_to(ROOT)}:{number}: {match.group(0)}")
    assert not wrong, "MESA is 'Mapping Environmentally Sensitive Assets':\n  " + "\n  ".join(wrong)


def test_the_application_spells_the_name_out_where_it_introduces_itself() -> None:
    text = (ROOT / "mesa.py").read_text(encoding="utf-8")
    assert text.count(f"MESA ({NAME})") >= 2          # the Welcome tab and the About tab
