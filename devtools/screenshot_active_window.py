from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab

DWMWA_EXTENDED_FRAME_BOUNDS = 9


def _set_dpi_awareness() -> None:
    user32 = ctypes.windll.user32
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


def _get_window_bounds(hwnd: int, client_only: bool) -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32

    if client_only:
        client = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            raise RuntimeError("GetClientRect failed")
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise RuntimeError("ClientToScreen failed")
        left = origin.x
        top = origin.y
        right = left + (client.right - client.left)
        bottom = top + (client.bottom - client.top)
        return left, top, right, bottom

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


def main() -> None:
    default_out = Path(__file__).resolve().parent.parent / "output" / "screenshots"
    parser = argparse.ArgumentParser(description="Capture the currently active window (Alt+PrintScreen style).")
    parser.add_argument("--out", default=str(default_out), help="Output directory")
    parser.add_argument("--name", default="active_window", help="Base filename")
    parser.add_argument("--client-only", action="store_true", help="Capture only client area (inside window chrome)")
    args = parser.parse_args()

    _set_dpi_awareness()
    user32 = ctypes.windll.user32

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        raise SystemExit("No active foreground window found.")

    title_len = user32.GetWindowTextLengthW(hwnd)
    title_buf = ctypes.create_unicode_buffer(title_len + 1)
    user32.GetWindowTextW(hwnd, title_buf, title_len + 1)
    title = title_buf.value.strip() or "(untitled)"

    left, top, right, bottom = _get_window_bounds(hwnd, client_only=args.client_only)
    if right <= left or bottom <= top:
        raise SystemExit("Invalid window bounds.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{args.name}_{stamp}.png"

    image = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
    image.save(out_file)

    print(f"Captured: {out_file}")
    print(f"Title: {title}")
    print(f"Bounds: ({left}, {top}, {right}, {bottom})")


if __name__ == "__main__":
    main()
