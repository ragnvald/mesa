# -*- coding: utf-8 -*-
"""Shared OpenStreetMap tile proxy helpers for embedded MESA viewers.

This module provides a small local HTTP proxy that fronts
``https://tile.openstreetmap.org`` with:

- an explicit application ``User-Agent``
- local on-disk tile caching
- cache expiry derived from upstream response headers
- stale-on-error fallback when cached content is available

It is intended for desktop/webview viewers where browser-side requests may not
send a suitable ``Referer`` and can therefore be blocked by OSM's tile policy.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib import error as urlerror
from urllib import request as urlrequest


LogFn = Optional[Callable[[str], None]]


@dataclass
class OsmTileProxy:
    server: ThreadingHTTPServer
    thread: threading.Thread
    base_url: str
    cache_dir: Path


def build_osm_user_agent(
    product_name: str,
    version: str,
    contact_url: str = "https://github.com/ragnvald/mesa",
) -> str:
    product = (product_name or "MESA").strip().replace(" ", "-") or "MESA"
    ver = (version or "dev").strip().replace(" ", "_") or "dev"
    return f"{product}/{ver} (+{contact_url})"


def choose_cache_dir(candidates: Iterable[Path], fallback_name: str = "mesa_osm_tiles") -> Path:
    for candidate in candidates:
        try:
            path = Path(candidate)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    fallback = Path(tempfile.gettempdir()) / fallback_name
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def start_osm_tile_proxy(
    cache_dir: Path,
    user_agent: str,
    log: LogFn = None,
    thread_name: str = "mesa-osm-tile-proxy",
) -> OsmTileProxy:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        if log is None:
            return
        try:
            log(message)
        except Exception:
            pass

    def _cache_paths(z: str, x: str, y: str) -> tuple[Path, Path]:
        tile_path = cache_dir / z / x / f"{y}.png"
        meta_path = cache_dir / z / x / f"{y}.json"
        return tile_path, meta_path

    def _load_cached_tile(tile_path: Path, meta_path: Path) -> tuple[Optional[bytes], float]:
        data: Optional[bytes] = None
        expires_at = 0.0
        try:
            if tile_path.exists():
                data = tile_path.read_bytes()
        except Exception:
            data = None
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                expires_at = float(meta.get("expires_at") or 0.0)
        except Exception:
            expires_at = 0.0
        return data, expires_at

    def _save_cached_tile(tile_path: Path, meta_path: Path, data: bytes, expires_at: float) -> None:
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile_path.write_bytes(data)
        meta = {
            "expires_at": expires_at,
            "saved_at": time.time(),
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def _cache_expiry_from_headers(headers) -> float:
        now = time.time()
        cache_control = headers.get("Cache-Control", "")
        for token in cache_control.split(","):
            part = token.strip().lower()
            if not part.startswith("max-age="):
                continue
            try:
                return now + max(0, int(part.split("=", 1)[1]))
            except Exception:
                break
        expires = headers.get("Expires")
        if expires:
            try:
                parsed = parsedate_to_datetime(expires)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except Exception:
                pass
        return now + (7 * 24 * 60 * 60)

    def _build_handler():
        class _OsmTileProxyHandler(BaseHTTPRequestHandler):
            server_version = "MesaOsmProxy/1.0"
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                raw_path = self.path.split("?", 1)[0]
                if raw_path == "/health":
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                parts = raw_path.strip("/").split("/")
                if (
                    len(parts) != 4
                    or parts[0] != "osm"
                    or not parts[1].isdigit()
                    or not parts[2].isdigit()
                    or not parts[3].endswith(".png")
                ):
                    self.send_error(404)
                    return

                z = parts[1]
                x = parts[2]
                y = parts[3][:-4]
                if not y.isdigit():
                    self.send_error(404)
                    return

                tile_path, meta_path = _cache_paths(z, x, y)
                cached_data, expires_at = _load_cached_tile(tile_path, meta_path)
                now = time.time()
                if cached_data is not None and expires_at > now:
                    self._send_png(cached_data, expires_at - now)
                    return

                upstream = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                request = urlrequest.Request(
                    upstream,
                    headers={
                        "User-Agent": user_agent,
                        "Accept": "image/png,image/*;q=0.9,*/*;q=0.8",
                    },
                )
                try:
                    with urlrequest.urlopen(request, timeout=20) as response:
                        payload = response.read()
                        expiry = _cache_expiry_from_headers(response.headers)
                    _save_cached_tile(tile_path, meta_path, payload, expiry)
                    self._send_png(payload, max(60.0, expiry - time.time()))
                except urlerror.HTTPError as exc:
                    if cached_data is not None:
                        _log(f"OSM tile proxy using stale cache for {raw_path} after HTTP {exc.code}")
                        self._send_png(cached_data, 60.0)
                        return
                    self.send_error(exc.code)
                except Exception as exc:
                    if cached_data is not None:
                        _log(f"OSM tile proxy using stale cache for {raw_path} after error: {exc}")
                        self._send_png(cached_data, 60.0)
                        return
                    _log(f"OSM tile proxy failed for {raw_path}: {exc}")
                    self.send_error(502, f"Tile fetch failed: {exc}")

            def _send_png(self, payload: bytes, ttl_seconds: float) -> None:
                ttl = max(60, int(ttl_seconds))
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", f"public, max-age={ttl}")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)

        return _OsmTileProxyHandler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, name=thread_name, daemon=True)
    thread.start()
    proxy = OsmTileProxy(server=server, thread=thread, base_url=base_url.rstrip("/"), cache_dir=cache_dir)
    _log(f"OSM tile proxy started at {proxy.base_url} (cache={cache_dir})")
    return proxy


def stop_osm_tile_proxy(proxy: Optional[OsmTileProxy], log: LogFn = None) -> None:
    if proxy is None:
        return

    def _log(message: str) -> None:
        if log is None:
            return
        try:
            log(message)
        except Exception:
            pass

    try:
        proxy.server.shutdown()
    except Exception:
        pass
    try:
        proxy.server.server_close()
    except Exception:
        pass
    try:
        proxy.thread.join(timeout=2.0)
    except Exception:
        pass
    _log("OSM tile proxy stopped")
