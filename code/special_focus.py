#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""special_focus.py — one window hosting the Lines and Analysis setup tools as
in-place button-tabs (the same UX as the Maps window).

WHY A NEW WINDOW
    Lines (line_manage) and Analysis (analysis_setup) are each a full pywebview
    + Leaflet editing app with its own HTML page and Python JS-bridge. They even
    share a bridge method name (update_geometry), so they cannot share one page
    or one flat bridge. To present them as in-place tabs (rather than separate
    pop-up windows) we embed each app, unchanged, in an <iframe> served from one
    loopback origin, and reach the Python bridge cross-frame.

HOW THE BRIDGE WORKS ACROSS FRAMES
    pywebview injects `window.pywebview` only into the TOP frame; a same-origin
    iframe reaches it via `window.parent.pywebview` / `window.top.pywebview`
    (verified on pywebview 6.x). Each app's bridge methods are exposed once,
    namespaced as `lines__<m>` / `analysis__<m>` (window.expose), and a tiny shim
    injected into each iframe defines a local `window.pywebview` whose `.api`
    proxies bare calls to the namespaced parent bridge. The shim also synthesises
    the `pywebviewready` event (which only fires on the top frame) so each app
    bootstraps normally.

CALLED BY
    mesa.py -> open_special_focus -> _launch_helper_subprocess("special_focus")
"""
from __future__ import annotations

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import webview  # pip install pywebview
except Exception:
    webview = None

import mesa_shared
from mesa_shared import leaflet_bundle, mesa_version_label, read_config

# The two embedded apps. Importing runs their module-level setup (OSM proxies
# etc.); we drive their Api/WebApi + HTML directly rather than their run()/main().
import line_manage as _lines
import analysis_setup as _analysis


# ---------------------------------------------------------------------------
# Per-iframe bridge shim. Runs before the app's own scripts: makes
# window.pywebview.api proxy to the namespaced parent bridge, and fires the
# pywebviewready the app is waiting for.
# ---------------------------------------------------------------------------
_SHIM_JS = r"""(function(){
  var NS='__NS__';
  function real(){
    try{ return (window.parent&&window.parent.pywebview&&window.parent.pywebview.api)
              || (window.top&&window.top.pywebview&&window.top.pywebview.api) || null; }
    catch(e){ return null; }
  }
  var handler={ get:function(_t,m){
    // Must NOT look like a Promise/thenable: apps probe `typeof api.then` to
    // decide whether the bridge is a promise. Returning a function for every key
    // (incl. 'then') makes them await the bridge instead of calling it, so the
    // call (e.g. bootstrap) never fires. Hand back undefined for those probes.
    if(typeof m!=='string' || m==='then' || m==='catch' || m==='finally') return undefined;
    return function(){
      var args=Array.prototype.slice.call(arguments);
      var r=real();
      if(!r) return Promise.reject(new Error('bridge not ready'));
      var fn=r[NS+'__'+m];
      if(typeof fn!=='function') return Promise.reject(new Error('no bridge method '+m));
      return fn.apply(r, args);
    };
  } };
  // Expose window.pywebview ONLY once the parent bridge is reachable. Apps probe
  // `window.pywebview` synchronously and call immediately if present, so defining
  // it early makes them fire before pywebview has injected the top-frame bridge
  // (the 'bridge not ready' error). Defining it here, together with dispatching
  // pywebviewready, guarantees every call path sees a live bridge.
  function ready(){
    if(!window.pywebview){ window.pywebview={ api:new Proxy({}, handler) }; }
    try{ window.dispatchEvent(new Event('pywebviewready')); }catch(e){}
  }
  if(real()){ ready(); }
  else { var n=0, iv=setInterval(function(){
    if(real()){ clearInterval(iv); ready(); } else if(++n>200){ clearInterval(iv); }
  }, 50); }
})();"""


def _inject_shim(html: str, ns: str) -> str:
    shim = "<script>" + _SHIM_JS.replace("__NS__", ns) + "</script>"
    low = html.lower()
    i = low.find("<head>")
    if i != -1:
        pos = i + len("<head>")
        return html[:pos] + shim + html[pos:]
    return shim + html


def _build_payload(html_const: str, osm_proxy, base_dir, cfg, ns: str) -> str:
    """Reproduce an app's html_payload (the 3 template replaces its run()/main()
    does) and inject the cross-frame bridge shim."""
    bundle = leaflet_bundle(base_dir, include_draw=True)
    html = (
        html_const
        .replace("__MESA_LEAFLET_HEAD__", bundle.head_block)
        .replace("__MESA_LEAFLET_BODY_OPEN__", bundle.body_open)
        .replace("__MESA_OSM_TILE_URL__", osm_proxy.tile_layer_url(base_dir, mesa_version_label(cfg)))
    )
    return _inject_shim(html, ns)


# ---------------------------------------------------------------------------
# Combined bridge: a tiny host api (tabs/exit) + namespaced wrappers for each
# app's public bridge methods, registered via window.expose.
# ---------------------------------------------------------------------------
def _host_exit(*_args, **_kwargs):
    """Close the whole Special focus window."""
    try:
        webview.destroy_window()
    except Exception:
        import os
        os._exit(0)
    return {"ok": True}


class _HostApi:
    def host_exit(self):
        return _host_exit()


def _namespaced_wrappers(instance, ns: str) -> list:
    """One forwarding function per public, callable method of `instance`, named
    '<ns>__<method>' so window.expose registers it under that name. Each app's own
    exit_app is rerouted to close the combined window (the app's module-level
    `webview` global isn't initialised here — only the host owns the window)."""
    wrappers = []
    for name in dir(instance):
        if name.startswith("_"):
            continue
        attr = getattr(instance, name, None)
        if not callable(attr):
            continue
        target = _host_exit if name == "exit_app" else attr
        wrappers.append(_forward(target, f"{ns}__{name}"))
    return wrappers


def _forward(fn, exposed_name: str):
    def _w(*args, **kwargs):
        return fn(*args, **kwargs)
    _w.__name__ = exposed_name
    return _w


# ---------------------------------------------------------------------------
# Loopback server: host page at /, each app at /lines and /analysis.
# ---------------------------------------------------------------------------
_PAGES: dict[str, str] = {}
_CTX: dict = {}


def _page(ns: str) -> str:
    """Lazily build (and cache) an embedded app's iframe payload. Deferring this
    out of run() keeps window creation snappy — the host chrome paints first and
    each tab's heavy Leaflet page is built only when its iframe is first fetched
    (the same lazy-tab idea as the Maps window)."""
    if ns in _PAGES:
        return _PAGES[ns]
    ctx = _CTX
    if not ctx:
        return "<h3>Not ready</h3>"
    if ns == "lines":
        html = _build_payload(_lines.HTML, _lines.OSM_PROXY, ctx["base_dir"], ctx["cfg"], "lines")
    elif ns == "analysis":
        html = _build_payload(_analysis.HTML_TEMPLATE, _analysis.OSM_PROXY, ctx["base_dir"], ctx["cfg"], "analysis")
    else:
        return "<h3>Unknown page</h3>"
    _PAGES[ns] = html
    return html


HOST_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>MESA — Special focus</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body{height:100%;margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;font-size:13px;color:#3f3528}
  #app{display:flex;flex-direction:column;height:100vh}
  #bar{display:flex;align-items:center;gap:8px;padding:8px 12px;background:#f3ecdf;border-bottom:2px solid #cbb791}
  .tab{background:#e6dac2;color:#5c4a2f;border:1px solid #c6b089;padding:6px 14px;cursor:pointer;border-radius:6px;font-size:13px}
  .tab:hover{background:#eadbbd}
  .tab:active{transform:translateY(1px)}
  .tab.active{background:#d9bd7d;color:#3f3018;border-color:#9b7c3d;font-weight:600}
  #spacer{flex:1}
  #exit{background:#e6dac2;color:#5c4a2f;border:1px solid #c6b089;padding:6px 12px;border-radius:6px;cursor:pointer}
  #exit:hover{background:#eadbbd}
  #views{position:relative;flex:1}
  .view{position:absolute;inset:0;width:100%;height:100%;border:0;display:none;background:#fff}
  .view.active{display:block}
</style></head>
<body>
<div id="app">
  <div id="bar">
    <button class="tab active" data-tab="lines">Lines</button>
    <button class="tab" data-tab="analysis">Analysis</button>
    <span id="spacer"></span>
    <button id="exit" title="Close Special focus">Exit</button>
  </div>
  <div id="views">
    <iframe class="view active" id="view-lines"></iframe>
    <iframe class="view" id="view-analysis"></iframe>
  </div>
</div>
<script>
  function show(tab){
    document.querySelectorAll('.tab').forEach(function(b){ b.classList.toggle('active', b.dataset.tab===tab); });
    document.querySelectorAll('.view').forEach(function(v){ v.classList.remove('active'); });
    var f=document.getElementById('view-'+tab);
    if(!f.getAttribute('src')) f.setAttribute('src','/'+tab);  // lazy: load each app on first open
    f.classList.add('active');
  }
  // Paint the host chrome first, then load the default (Lines) tab — keeps the
  // window from appearing blank while a heavy embedded app initialises.
  window.addEventListener('load', function(){ show('lines'); });
  document.querySelectorAll('.tab').forEach(function(b){ b.addEventListener('click', function(){ show(b.dataset.tab); }); });
  document.getElementById('exit').addEventListener('click', function(){
    try{ window.pywebview.api.host_exit(); }catch(e){}
  });
</script>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def _send_html(self, body: str):
        data = body.encode("utf-8", errors="replace")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            self._send_html(HOST_HTML)
        elif p == "/lines":
            self._send_html(_page("lines"))
        elif p == "/analysis":
            self._send_html(_page("analysis"))
        elif p == "/favicon.ico":
            self.send_response(204); self.end_headers()
        else:
            self.send_response(404); self.end_headers()


def _start_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


def run(base: str | None = None) -> None:
    if webview is None:
        raise RuntimeError("'pywebview' is not installed. Install it with: pip install pywebview")
    base_dir = Path(mesa_shared.find_base_dir(base))
    cfg = read_config(base_dir)

    # Each app's bridge is built up front (cheap) so its namespaced methods exist
    # before any iframe calls them; the heavy HTML payloads are built lazily by the
    # server on first fetch (see _page) so the window appears promptly.
    lines_api = _lines.Api(str(base_dir), cfg)
    analysis_api = _analysis.WebApi(base_dir, cfg)
    _CTX.update({"base_dir": base_dir, "cfg": cfg})

    url = _start_server()
    window = webview.create_window(
        title="MESA — Special focus (Lines / Analysis)",
        url=url,
        js_api=_HostApi(),
        width=1320, height=880, resizable=True,
    )
    # Register the two apps' bridges under namespaced names.
    window.expose(*_namespaced_wrappers(lines_api, "lines"))
    window.expose(*_namespaced_wrappers(analysis_api, "analysis"))
    webview.start(gui="edgechromium", debug=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="MESA — Special focus (Lines / Analysis tabs)")
    ap.add_argument("--original_working_directory", required=False, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    run(args.original_working_directory)
    return 0


if __name__ == "__main__":
    if webview is None:
        sys.stderr.write("ERROR: 'pywebview' is not installed in this environment.\n")
        raise SystemExit(1)
    raise SystemExit(main())
