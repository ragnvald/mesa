"""Put a MESA build on Zenodo as a new version of the application record.

Without --publish everything stops at a draft, which can be inspected on zenodo.org
and thrown away with --discard. Publishing cannot be undone.

    zenodo_release.py 5.7.1 --zip D:\\dist\\MESA_571_win11_2026_10_13.zip     draft
    zenodo_release.py 5.7.1 --zip ... --publish                              draft, then publish
    zenodo_release.py 5.7.1 --discard                                        delete the draft
    zenodo_release.py 5.7.0 --fix-metadata --record 23314128 [--publish]     correct a published record
    ... --sandbox --record <id>                                              sandbox.zenodo.org

Title and description come from devtools/ZENODO_<version>.md. The token is read from the
environment (ZENODO_TOKEN, ZENODO_SANDBOX_TOKEN) or from secrets/zenodo.ini, section
[zenodo], keys token and sandbox_token. It needs the scopes deposit:write and
deposit:actions. See devtools/docs/release_procedure.md.
"""
from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import hashlib
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONCEPT_RECID = "10775727"
TITLE_FORM = "MESA {version}: Mapping Environmentally Sensitive Assets"

# What every version of the application record carries. A new version inherits the
# previous one's metadata, gaps included, so these are written every time.
FIXED_METADATA = {
    "upload_type": "software",
    "access_right": "open",
    "license": "gpl-3.0-or-later",
    "language": "eng",
    "creators": [{
        "name": "Larsen, Ragnvald",
        "affiliation": "Norwegian Environment Agency",
        "orcid": "0000-0002-4184-5224",
    }],
    "keywords": [
        "environmental sensitivity", "oil spill preparedness", "pollution response",
        "environmental management", "GIS", "geospatial analysis", "QGIS",
        "sensitivity mapping", "marine spatial planning",
    ],
    "related_identifiers": [
        {"identifier": "https://github.com/ragnvald/mesa", "relation": "isSupplementTo"},
        {"identifier": "https://github.com/ragnvald/mesa/wiki", "relation": "isDocumentedBy"},
        {"identifier": "10.5281/zenodo.21744272", "relation": "references"},
        {"identifier": "10.5281/zenodo.21744039", "relation": "references"},
    ],
    "communities": [{"identifier": "mesatool"}],
}


class Stop(Exception):
    """A condition that ends the run with a message and no traceback."""


# ----------------------------------------------------------------------------- text
def read_release_text(version: str) -> tuple[str, str]:
    """Title and HTML description from devtools/ZENODO_<version>.md."""
    path = REPO / "devtools" / f"ZENODO_{version}.md"
    if not path.is_file():
        raise Stop(f"{path} does not exist.")
    sections: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    title_lines = [l[1:].strip() for l in sections.get("tittel", []) if l.startswith(">")]
    body_key = next((k for k in sections if k.startswith("beskrivelse")), None)
    if not title_lines or body_key is None:
        raise Stop(f"{path.name} needs a '## Tittel' section with a '> ' line and a '## Beskrivelse' section.")
    title = title_lines[0]
    if version not in title:
        raise Stop(f"The title does not contain {version}: {title!r}")
    if not title.startswith(TITLE_FORM.format(version=version)):
        raise Stop(f"The title must begin with {TITLE_FORM.format(version=version)!r}: {title!r}")
    paragraphs = [p.strip() for p in "\n".join(sections[body_key]).split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        raise Stop(f"The description in {path.name} has only {len(paragraphs)} paragraphs.")
    return title, "".join(f"<p>{_paragraph_html(p)}</p>" for p in paragraphs)


def _paragraph_html(text: str) -> str:
    escaped = html.escape(" ".join(text.split()), quote=False)
    return re.sub(r"(https?://[^\s<]+?)(?=[.,;:)]?(?:\s|$))", r'<a href="\1">\1</a>', escaped)


# ------------------------------------------------------------------------------ zip
def check_zip(path: Path, version: str) -> tuple[int, str]:
    """Refuse a zip that is not this version or that carries what must not ship."""
    if not path.is_file():
        raise Stop(f"{path} does not exist.")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        lowered = [n.lower() for n in names]
        for n in lowered:
            parts = n.split("/")
            if "secrets" in parts or parts[-1] == "log.txt":
                raise Stop(f"The zip contains {n}, which must not be published.")
        cfg = [n for n in names if n.lower().endswith("_internal/config.ini")]
        if len(cfg) != 1:
            raise Stop("The zip has no single _internal/config.ini; is this a MESA build?")
        m = re.search(r"(?mi)^\s*mesa_version\s*=\s*(.+?)\s*$", zf.read(cfg[0]).decode("utf-8", "ignore"))
        if not m or m.group(1) != version:
            raise Stop(f"The zip is version {m.group(1) if m else 'unknown'}, not {version}.")
        if not any(n.lower().endswith("/mesa.exe") for n in names):
            raise Stop("The zip has no mesa.exe.")
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return path.stat().st_size, digest.hexdigest()


# ------------------------------------------------------------------------------ api
class Zenodo:
    def __init__(self, sandbox: bool):
        self.host = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
        self._token = _read_token(sandbox)

    def call(self, method: str, url: str, body=None, *, data=None, length: int | None = None,
             ok=(200, 201, 202, 204)):
        if not url.startswith("http"):
            url = self.host + url
        if not url.startswith(self.host + "/"):
            raise Stop(f"Refusing to send the token to {urllib.parse.urlsplit(url).netloc}.")
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif data is not None:
            headers["Content-Type"] = "application/octet-stream"
            headers["Content-Length"] = str(length)
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1500]
            raise Stop(f"{method} {urllib.parse.urlsplit(url).path} gave {exc.code}:\n{detail}") from None
        if status not in ok:
            raise Stop(f"{method} {url} gave {status}.")
        return json.loads(raw) if raw else None


def _read_token(sandbox: bool) -> str:
    env, key = ("ZENODO_SANDBOX_TOKEN", "sandbox_token") if sandbox else ("ZENODO_TOKEN", "token")
    token = os.environ.get(env, "").strip()
    if not token:
        ini = REPO / "secrets" / "zenodo.ini"
        if ini.is_file():
            cp = configparser.ConfigParser(interpolation=None)
            cp.read(ini, encoding="utf-8")
            token = cp.get("zenodo", key, fallback="").strip()
    if not token:
        raise Stop(f"No token. Set {env}, or put '{key} = ...' under [zenodo] in secrets/zenodo.ini.")
    return token


def latest_record_id() -> str:
    """Record id of the newest published version, from the public search."""
    url = ("https://zenodo.org/api/records?q=conceptrecid:" + CONCEPT_RECID
           + "&sort=mostrecent&size=1")
    with urllib.request.urlopen(url, timeout=60) as resp:
        hits = json.loads(resp.read())["hits"]["hits"]
    if not hits:
        raise Stop("Could not find the newest version of the record.")
    return str(hits[0]["id"])


# ---------------------------------------------------------------------------- steps
def open_draft(z: Zenodo, previous: str) -> dict:
    """The one unpublished new version of `previous`, created if it is not there."""
    record = z.call("POST", f"/api/deposit/depositions/{previous}/actions/newversion")
    draft_url = (record.get("links") or {}).get("latest_draft")
    if not draft_url:
        raise Stop("Zenodo did not return a draft for the new version.")
    draft = z.call("GET", draft_url)
    if draft.get("submitted"):
        raise Stop(f"Deposition {draft.get('id')} is already published; nothing was changed.")
    return draft


def put_file(z: Zenodo, draft: dict, zip_path: Path, size: int, md5: str) -> None:
    files_url = f"/api/deposit/depositions/{draft['id']}/files"
    present = False
    for f in z.call("GET", files_url) or []:
        if f.get("filename") == zip_path.name and f.get("checksum") == md5:
            present = True
            print(f"  already uploaded: {zip_path.name}")
        else:
            print(f"  removing from the draft: {f.get('filename')}")
            z.call("DELETE", f"{files_url}/{f['id']}")
    if present:
        return
    bucket = (draft.get("links") or {}).get("bucket")
    if not bucket:
        raise Stop("The draft has no upload address (links.bucket).")
    print(f"  uploading {zip_path.name}, {size / 1e9:.2f} GB ...")
    with open(zip_path, "rb") as fh:
        answer = z.call("PUT", f"{bucket}/{urllib.parse.quote(zip_path.name)}", data=fh, length=size)
    got = str((answer or {}).get("checksum", "")).removeprefix("md5:")
    if got != md5:
        raise Stop(f"Checksum after upload is {got or 'missing'}, expected {md5}. The draft is left as it is.")
    print("  uploaded, checksum matches")


def put_metadata(z: Zenodo, deposition: dict, version: str, title: str, description: str,
                 *, new_version: bool) -> dict:
    # Start from what is there, so that fields this script does not know are sent back unchanged.
    metadata = dict(deposition.get("metadata") or {})
    metadata.update(FIXED_METADATA)
    metadata.update({"title": title, "description": description, "version": version})
    if new_version or not metadata.get("publication_date"):
        metadata["publication_date"] = _dt.date.today().isoformat()
    return z.call("PUT", f"/api/deposit/depositions/{deposition['id']}", {"metadata": metadata})


def show(deposition: dict, z: Zenodo) -> None:
    m = deposition.get("metadata") or {}
    creator = (m.get("creators") or [{}])[0]
    print(f"  id            {deposition.get('id')}")
    print(f"  title         {m.get('title')}")
    print(f"  version       {m.get('version')}   date {m.get('publication_date')}")
    print(f"  creator       {creator.get('name')} | {creator.get('affiliation')} | {creator.get('orcid')}")
    print(f"  license       {m.get('license')}   communities {[c.get('identifier') for c in m.get('communities') or []]}")
    print(f"  keywords      {len(m.get('keywords') or [])}   related identifiers {len(m.get('related_identifiers') or [])}")
    for f in z.call("GET", f"/api/deposit/depositions/{deposition['id']}/files") or []:
        print(f"  file          {f.get('filename')}  {f.get('filesize')}  md5 {f.get('checksum')}")
    print(f"  preview       {z.host}/uploads/{deposition.get('id')}")


def publish(z: Zenodo, deposition_id, note_beside: Path | None) -> dict:
    done = z.call("POST", f"/api/deposit/depositions/{deposition_id}/actions/publish")
    result = {"record_id": str(done.get("record_id") or done.get("id")), "doi": done.get("doi"),
              "url": (done.get("links") or {}).get("record_html") or (done.get("links") or {}).get("html")}
    print(f"\nPublished: record {result['record_id']}, DOI {result['doi']}")
    if note_beside is not None:
        out = note_beside.with_suffix(".zenodo.json")
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Written to {out}")
    return result


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="for example 5.7.1")
    ap.add_argument("--zip", type=Path, help="the build to upload")
    ap.add_argument("--record", help="record id to make a new version of, or to correct (default: the newest)")
    ap.add_argument("--publish", action="store_true", help="publish; cannot be undone")
    ap.add_argument("--discard", action="store_true", help="delete the unpublished draft")
    ap.add_argument("--fix-metadata", action="store_true", help="correct the metadata of a published record")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    args = ap.parse_args()

    try:
        if args.sandbox and not args.record:
            raise Stop("--sandbox needs --record: the sandbox has its own records.")
        title, description = read_release_text(args.version)
        z = Zenodo(args.sandbox)

        if args.fix_metadata:
            if not args.record:
                raise Stop("--fix-metadata needs --record.")
            current = z.call("GET", f"/api/deposit/depositions/{args.record}")
            if (current.get("metadata") or {}).get("version") != args.version:
                raise Stop(f"Record {args.record} is version {(current.get('metadata') or {}).get('version')}, not {args.version}.")
            print(f"Correcting the metadata of record {args.record} on {z.host}")
            if not args.publish:
                print("  (nothing is changed without --publish; this is what would be written)")
                print(f"  title         {title}\n  creator       {FIXED_METADATA['creators'][0]}")
                return 0
            z.call("POST", f"/api/deposit/depositions/{args.record}/actions/edit")
            put_metadata(z, z.call("GET", f"/api/deposit/depositions/{args.record}"), args.version, title, description,
                         new_version=False)
            publish(z, args.record, None)
            return 0

        previous = args.record or latest_record_id()
        if args.discard:
            record = z.call("GET", f"/api/deposit/depositions/{previous}")
            draft_url = (record.get("links") or {}).get("latest_draft")
            if not draft_url or draft_url.rstrip("/").endswith(f"/{previous}"):
                raise Stop("There is no unpublished draft to delete.")
            z.call("DELETE", draft_url)
            print("The draft is deleted.")
            return 0

        if not args.zip:
            raise Stop("--zip is required.")
        print(f"Checking {args.zip.name}")
        size, md5 = check_zip(args.zip, args.version)
        print(f"  {size} bytes, md5 {md5}")
        print(f"New version of record {previous} on {z.host}")
        draft = open_draft(z, previous)
        put_file(z, draft, args.zip, size, md5)
        draft = put_metadata(z, draft, args.version, title, description, new_version=True)
        print("Draft:")
        show(draft, z)
        if not args.publish:
            print("\nNot published. Look at the preview, then run again with --publish, or --discard.")
            return 0
        publish(z, draft["id"], args.zip)
        return 0
    except Stop as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
