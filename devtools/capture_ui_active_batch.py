from __future__ import annotations

import argparse
import ctypes
import hashlib
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import psutil
from PIL import ImageGrab

DWMWA_EXTENDED_FRAME_BOUNDS = 9
SW_RESTORE = 9
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_TAB = 0x09
VK_SHIFT = 0x10

user32 = ctypes.windll.user32

TARGET_SCRIPT_NAMES = {
    "mesa.py",
    "geocode_manage.py",
    "asset_manage.py",
    "processing_setup.py",
    "processing_pipeline_run.py",
    "atlas_manage.py",
    "map_overview.py",
    "asset_map_view.py",
    "report_generate.py",
    "analysis_setup.py",
    "analysis_present.py",
    "line_manage.py",
}


@dataclass(frozen=True)
class HelperCapture:
    key: str
    args: list[str]
    title_hint: str
    wait_seconds: float


def set_dpi_awareness() -> None:
    shcore = getattr(ctypes.windll, "shcore", None)
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    if shcore is not None:
        try:
            shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


def process_tree_pids(root_pid: int) -> set[int]:
    out = {root_pid}
    try:
        root = psutil.Process(root_pid)
        for child in root.children(recursive=True):
            out.add(child.pid)
    except Exception:
        pass
    return out


def cleanup_existing_gui_processes() -> None:
    current_pid = psutil.Process().pid
    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid <= 0 or pid == current_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            cmd_text = " ".join(str(x).lower() for x in cmdline)
            if not cmd_text:
                continue
            if "python" not in cmd_text:
                continue
            if not any(name in cmd_text for name in TARGET_SCRIPT_NAMES):
                continue
            psutil.Process(pid).kill()
            killed += 1
        except Exception:
            continue
    if killed:
        print(f"Cleanup: terminated {killed} existing MESA/helper python process(es)")
        time.sleep(1.0)


def enum_windows() -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.strip()
        if not title:
            return True
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found.append((hwnd, int(pid.value), title))
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return found


def window_bounds(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    dwmapi = getattr(ctypes.windll, "dwmapi", None)
    if dwmapi is not None:
        try:
            result = dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd),
                ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS),
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            if result == 0:
                return rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            pass

    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetWindowRect failed")
    return rect.left, rect.top, rect.right, rect.bottom


def ensure_on_screen(hwnd: int) -> None:
    left, top, right, bottom = window_bounds(hwnd)
    width = right - left
    height = bottom - top

    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)

    x = left
    y = top
    if left < 0:
        x = 8
    if top < 0:
        y = 8
    if right > sw:
        x = max(8, sw - width - 8)
    if bottom > sh:
        y = max(8, sh - height - 8)

    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetWindowPos(hwnd, 0, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.35)


def find_app_window(root_pid: int, title_hint: str, timeout: float = 300.0):
    deadline = time.time() + timeout
    hint = title_hint.lower().strip()
    while time.time() < deadline:
        pids = process_tree_pids(root_pid)
        cands = []
        for hwnd, pid, title in enum_windows():
            if pid not in pids and hint not in title.lower():
                continue
            try:
                left, top, right, bottom = window_bounds(hwnd)
            except Exception:
                continue
            w = right - left
            h = bottom - top
            if w < 420 or h < 260:
                continue
            cands.append((w * h, hwnd, title))
        if cands:
            cands.sort(reverse=True)
            _area, hwnd, title = cands[0]
            return hwnd, title
        time.sleep(0.4)
    return None, None


def capture_hwnd(hwnd: int, out_path: Path) -> str:
    left, top, right, bottom = window_bounds(hwnd)
    if right <= left or bottom <= top:
        raise RuntimeError("Invalid bounds")
    img = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
    img.save(out_path)
    return hashlib.md5(out_path.read_bytes()).hexdigest()


def click_client(hwnd: int, x: int, y: int) -> None:
    pt = wintypes.POINT(x, y)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    user32.SetCursorPos(pt.x, pt.y)
    time.sleep(0.05)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _press_vk(vk_code: int, key_up: bool = False) -> None:
    flags = KEYEVENTF_KEYUP if key_up else 0
    user32.keybd_event(vk_code, 0, flags, 0)


def send_ctrl_tab(reverse: bool = False) -> None:
    _press_vk(VK_CONTROL, key_up=False)
    if reverse:
        _press_vk(VK_SHIFT, key_up=False)
    time.sleep(0.02)
    _press_vk(VK_TAB, key_up=False)
    time.sleep(0.02)
    _press_vk(VK_TAB, key_up=True)
    if reverse:
        _press_vk(VK_SHIFT, key_up=True)
    _press_vk(VK_CONTROL, key_up=True)


def capture_mesa_tabs(repo: Path, py: Path, wiki_images: Path) -> None:
    proc = subprocess.Popen([str(py), "mesa.py"], cwd=repo)
    try:
        hwnd, title = find_app_window(proc.pid, "5.0 beta", timeout=300.0)
        if hwnd is None:
            print("FAIL mesa_desktop: window not found")
            return
        ensure_on_screen(hwnd)
        print("Waiting 45s for mesa desktop render...")
        time.sleep(45)

        base_path = wiki_images / "ui_mesa_desktop.png"
        first_hash = capture_hwnd(hwnd, base_path)
        print(f"OK   mesa_desktop: {title} -> {base_path.name}")

        # Focus the notebook area once, then cycle tabs with Ctrl+Tab.
        click_client(hwnd, 120, 135)
        time.sleep(0.2)

        tab_files = [
            "ui_mesa_desktop_tab2.png",  # Status
            "ui_mesa_desktop_tab3.png",  # Config
            "ui_mesa_desktop_tab4.png",  # Manage MESA data
            "ui_mesa_desktop_tab5.png",  # About
        ]

        for filename in tab_files:
            ensure_on_screen(hwnd)
            send_ctrl_tab(reverse=False)
            time.sleep(1.2)
            out_path = wiki_images / filename
            capture_hwnd(hwnd, out_path)
            print(f"OK   mesa_desktop tab -> {filename}")

        # Return to first tab for predictable end-state before cleanup.
        for _ in tab_files:
            send_ctrl_tab(reverse=True)
            time.sleep(0.08)
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)


def capture_helper(helper: HelperCapture, repo: Path, py: Path, wiki_images: Path) -> None:
    proc = subprocess.Popen([str(py), *helper.args], cwd=repo)
    try:
        hwnd, title = find_app_window(proc.pid, helper.title_hint, timeout=300.0)
        if hwnd is None:
            print(f"FAIL {helper.key}: window not found")
            return
        ensure_on_screen(hwnd)
        print(f"Waiting {int(helper.wait_seconds)}s for {helper.key} render...")
        time.sleep(helper.wait_seconds)
        ensure_on_screen(hwnd)
        out_path = wiki_images / f"ui_{helper.key}.png"
        capture_hwnd(hwnd, out_path)
        print(f"OK   {helper.key}: {title} -> {out_path.name}")
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)


def _default_repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_wiki_images(repo: Path) -> Path:
    candidates = [
        repo.parent / "mesa.wiki" / "images",
        repo.parent.parent / "mesa.wiki" / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-capture MESA helper UIs into wiki image files.")
    parser.add_argument("--repo", default="", help="Path to mesa repo (default: inferred from script location)")
    parser.add_argument("--wiki-images", default="", help="Path to mesa.wiki/images (default: auto-detected)")
    args = parser.parse_args()

    set_dpi_awareness()
    cleanup_existing_gui_processes()

    repo = Path(args.repo).resolve() if args.repo else _default_repo_dir()
    wiki_images = Path(args.wiki_images).resolve() if args.wiki_images else _default_wiki_images(repo)
    wiki_images.mkdir(parents=True, exist_ok=True)
    py = repo / ".venv" / "Scripts" / "python.exe"

    if not py.exists():
        raise SystemExit(f"Python executable not found: {py}")

    print(f"Repo: {repo}")
    print(f"Output dir: {wiki_images}")

    capture_mesa_tabs(repo, py, wiki_images)

    helpers = [
        HelperCapture("geocode_create", ["code/geocode_manage.py", "--start-tab", "h3", "--original_working_directory", str(repo)], "geocode manage", 35.0),
        HelperCapture("asset_manage", ["code/asset_manage.py", "--original_working_directory", str(repo)], "asset", 35.0),
        HelperCapture("processing_setup", ["code/processing_setup.py", "--original_working_directory", str(repo)], "setup", 40.0),
        HelperCapture("processing_pipeline_run", ["code/processing_pipeline_run.py", "--original_working_directory", str(repo)], "process all", 40.0),
        HelperCapture("atlas_manage", ["code/atlas_manage.py", "--original_working_directory", str(repo)], "atlas", 35.0),
        HelperCapture("map_overview", ["code/map_overview.py"], "maps overview", 50.0),
        HelperCapture("asset_map_view", ["code/asset_map_view.py"], "asset layers", 50.0),
        HelperCapture("report_generate", ["code/report_generate.py", "--original_working_directory", str(repo)], "report generator", 35.0),
        HelperCapture("analysis_setup", ["code/analysis_setup.py", "--original_working_directory", str(repo)], "area analysis", 40.0),
        HelperCapture("analysis_present", ["code/analysis_present.py", "--original_working_directory", str(repo)], "comparison", 40.0),
        HelperCapture("geocode_group_edit", ["code/geocode_manage.py", "--start-tab", "edit", "--original_working_directory", str(repo)], "geocode manage", 35.0),
        HelperCapture("line_manage", ["code/line_manage.py", "--original_working_directory", str(repo)], "edit line", 40.0),
    ]

    for helper in helpers:
        capture_helper(helper, repo, py, wiki_images)


if __name__ == "__main__":
    main()
