from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "br"}:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "ul", "ol"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.parts))
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> str:
    proc = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    return proc.stdout.strip()


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "mesa-release-helper/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"failed to fetch {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc.reason}") from exc


def _extract_zenodo_record_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    match = re.search(r"zenodo\.org/(?:records|api/records)/(\d+)", value, re.IGNORECASE)
    if match:
        return match.group(1)
    raise ValueError(f"could not determine Zenodo record id from: {value}")


def _strip_html(value: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(value or "")
    return parser.text()


def _detect_repo(cwd: Path) -> str:
    remote = _run(["git", "remote", "get-url", "origin"], cwd=cwd)
    remote = remote.strip()
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote, re.IGNORECASE)
    if not match:
        raise RuntimeError(f"could not derive GitHub repo from origin remote: {remote}")
    return match.group(1)


def _default_target(cwd: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=cwd)


def _version_label_from_title(title: str) -> str:
    title = (title or "").strip()
    match = re.search(r"version\s+(.+)$", title, re.IGNORECASE)
    if match:
        suffix = match.group(1).strip()
    else:
        suffix = re.sub(r"(?i)^mesa(?: tool)?\s*", "", title).strip()
    suffix = re.sub(r"\s+", " ", suffix)
    if not suffix.lower().startswith("mesa "):
        return f"MESA {suffix}".strip()
    return suffix


def _derive_tag(version_label: str) -> str:
    tag = re.sub(r"(?i)^mesa\s+", "", version_label).strip().lower()
    tag = tag.replace("/", "-")
    tag = re.sub(r"\s+", "-", tag)
    tag = re.sub(r"[^0-9a-z._-]+", "-", tag)
    tag = re.sub(r"-{2,}", "-", tag).strip("-")
    return tag


def _is_prerelease(*values: str) -> bool:
    joined = " ".join(values).lower()
    return any(token in joined for token in ("alpha", "beta", "rc", "pre-release", "prerelease"))


def _latest_previous_release_tag(repo: str, *, exclude_tag: str | None = None) -> str | None:
    payload = _run(["gh", "api", f"repos/{repo}/releases?per_page=30"])
    releases = json.loads(payload)
    for release in releases:
        if release.get("draft"):
            continue
        tag = release.get("tag_name")
        if not tag or tag == exclude_tag:
            continue
        return tag
    return None


def _release_exists(repo: str, tag: str) -> bool:
    proc = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode == 0


def _collect_commit_subjects(cwd: Path, previous_tag: str | None, target: str) -> list[str]:
    if not previous_tag:
        return []
    log_range = f"{previous_tag}..{target}"
    output = _run(["git", "log", "--no-merges", "--format=%s", log_range], cwd=cwd)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _normalize_subject(subject: str) -> str:
    subject = subject.strip()
    subject = re.sub(r"^(feat|fix|docs|chore|refactor|build)\(([^)]+)\):\s*", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"^(feat|fix|docs|chore|refactor|build):\s*", "", subject, flags=re.IGNORECASE)
    subject = subject.rstrip(".")
    if subject:
        subject = subject[0].upper() + subject[1:]
    return subject


def _classify_subject(subject: str) -> str:
    text = subject.lower()
    if any(word in text for word in ("geonode", "publish", "oauth", "report", "presentation")):
        return "Publishing and reporting"
    if any(word in text for word in ("build", "bundle", "startup", "runtime", "venv", "icon")):
        return "Packaging and runtime"
    if any(word in text for word in ("pyside6", "desktop", "ui", "launcher", "label", "tab", "workflow")):
        return "User interface and workflow"
    if any(word in text for word in ("process", "processing", "mosaic", "tile", "analysis", "grid", "flatten")):
        return "Processing and outputs"
    return "Other changes"


def _skip_subject(subject: str) -> bool:
    text = subject.lower()
    return any(
        marker in text
        for marker in (
            "screenshot",
            "capture",
            "docs",
            "readme",
            "ignore",
            "rename",
            "layout",
            "labels",
            "local probe",
            "workspace artifacts",
            "learning log",
            "temporary",
        )
    )


def _select_changes(subjects: list[str]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {
        "User interface and workflow": [],
        "Processing and outputs": [],
        "Publishing and reporting": [],
        "Packaging and runtime": [],
    }
    seen: set[str] = set()
    for raw_subject in subjects[:60]:
        subject = _normalize_subject(raw_subject)
        key = subject.lower()
        if not subject or key in seen or _skip_subject(subject):
            continue
        section = _classify_subject(subject)
        if section not in selected or len(selected[section]) >= 4:
            continue
        selected[section].append(subject)
        seen.add(key)
        if all(len(items) >= 3 for items in selected.values()):
            break
    return {section: items for section, items in selected.items() if items}


def _description_highlights(description_text: str) -> list[str]:
    text = description_text.lower()
    highlights: list[str] = []
    if "fully replaces all earlier mesa versions" in text:
        highlights.append("Version 5.0 fully replaces the earlier MESA releases.")
    if "end-to-end workflow" in text or "from inputs to published outputs" in text:
        highlights.append("New end-to-end workflow from project inputs to published outputs.")
    if "word-first" in text or ".docx" in text:
        highlights.append("Reporting is now Word-first (`.docx`) for easier downstream editing.")
    if "backup/restore" in text:
        highlights.append("Built-in backup and restore support for iterative project work.")
    if "status monitoring" in text:
        highlights.append("Lightweight status monitoring is included in the desktop workflow.")
    if "mafia island" in text:
        highlights.append("Includes pre-processed example data from Mafia Island, Tanzania.")
    return highlights[:6]


def _first_file_key(record: dict) -> str | None:
    files = record.get("files") or []
    if not files:
        return None
    return files[0].get("key")


def _build_notes(
    *,
    version_label: str,
    record_url: str,
    doi_url: str,
    publication_date: str,
    description_text: str,
    file_key: str | None,
    previous_tag: str | None,
    new_tag: str,
    change_sections: dict[str, list[str]],
) -> str:
    intro = (
        f"This marks the release of {version_label}. "
        f"This release packages the current MESA desktop workflow for preparing, "
        f"processing, and publishing spatial sensitivity analysis deliverables "
        f"using the [MESA method](https://www.mesamethod.org/)."
    )
    summary = (
        f"The Zenodo record for this build was published on {publication_date}. "
        "It includes compiled Windows binaries and project documentation links."
    )
    lines: list[str] = [intro, "", summary]

    if _is_prerelease(version_label, description_text):
        lines.extend(
            [
                "",
                "### Beta notice",
                "This is still a beta release intended for testing and feedback. "
                "Documentation is still being finalized, parts of the interface are still evolving, "
                "and calculation or stability issues may occur. Results should not be used for "
                "decision-making or production analysis.",
            ]
        )

    highlights = _description_highlights(description_text)
    if highlights:
        lines.extend(["", "### Highlights"])
        for item in highlights:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "### Where can I find it?",
            "Compiled versions are available here:",
            "",
            f"- [Zenodo record]({record_url})",
            f"- [DOI landing page]({doi_url})",
            "- [MESA project wiki](https://github.com/ragnvald/mesa/wiki)",
        ]
    )

    if file_key:
        lines.extend(
            [
                "",
                "### Getting started",
                f"Download and unzip `{file_key}` from the Zenodo record into a writable folder, "
                "keep the folder structure intact, and launch `mesa.exe`.",
            ]
        )

    if change_sections:
        lines.extend(["", "## Changelog"])
        for section, items in change_sections.items():
            lines.append(f"### {section}")
            for item in items:
                lines.append(f"- {item}")

    if previous_tag:
        lines.extend(
            [
                "",
                f"**Full Changelog**: https://github.com/ragnvald/mesa/compare/{previous_tag}...{new_tag}",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or publish a GitHub release body from a Zenodo record."
    )
    parser.add_argument("zenodo", help="Zenodo record id or URL")
    parser.add_argument("--repo", help="GitHub repo in owner/name form; defaults to origin remote")
    parser.add_argument("--target", help="Git commit/tag/branch to release; defaults to current HEAD")
    parser.add_argument("--tag", help="Override generated git tag")
    parser.add_argument("--title", help="Override generated release title")
    parser.add_argument("--notes-out", help="Write generated markdown to this path")
    parser.add_argument("--publish", action="store_true", help="Create the GitHub release with gh")
    parser.add_argument(
        "--prerelease",
        action="store_true",
        help="Force the release to be created as a prerelease",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Mark the created release as latest (not recommended for beta/alpha releases)",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    repo = args.repo or _detect_repo(cwd)
    target = args.target or _default_target(cwd)

    record_id = _extract_zenodo_record_id(args.zenodo)
    record = _fetch_json(f"https://zenodo.org/api/records/{record_id}")

    title = record.get("metadata", {}).get("title") or record.get("title") or f"Zenodo record {record_id}"
    description_html = record.get("metadata", {}).get("description") or ""
    description_text = _strip_html(description_html)
    version_label = args.title or _version_label_from_title(title)
    tag = args.tag or _derive_tag(version_label)
    prerelease = args.prerelease or _is_prerelease(version_label, description_text)
    previous_tag = _latest_previous_release_tag(repo, exclude_tag=tag)
    subjects = _collect_commit_subjects(cwd, previous_tag, target)
    change_sections = _select_changes(subjects)

    notes = _build_notes(
        version_label=version_label,
        record_url=record.get("links", {}).get("self_html") or f"https://zenodo.org/records/{record_id}",
        doi_url=record.get("links", {}).get("doi") or record.get("doi_url") or "",
        publication_date=record.get("metadata", {}).get("publication_date") or "",
        description_text=description_text,
        file_key=_first_file_key(record),
        previous_tag=previous_tag,
        new_tag=tag,
        change_sections=change_sections,
    )

    if args.notes_out:
        notes_path = Path(args.notes_out)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(notes, encoding="utf-8")
    else:
        notes_path = Path(tempfile.gettempdir()) / f"mesa_release_{tag}.md"
        notes_path.write_text(notes, encoding="utf-8")

    print(f"repo: {repo}")
    print(f"target: {target}")
    print(f"title: {version_label}")
    print(f"tag: {tag}")
    print(f"prerelease: {'yes' if prerelease else 'no'}")
    print(f"notes: {notes_path}")
    print()
    print(notes)

    if not args.publish:
        return 0

    if _release_exists(repo, tag):
        raise RuntimeError(f"release/tag {tag!r} already exists in {repo}")

    cmd = [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        repo,
        "--target",
        target,
        "--title",
        version_label,
        "--notes-file",
        str(notes_path),
    ]
    if prerelease:
        cmd.append("--prerelease")
    if args.latest:
        cmd.append("--latest")

    _run(cmd, cwd=cwd)
    print(f"created: https://github.com/{repo}/releases/tag/{tag}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
