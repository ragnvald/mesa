"""Shared helpers for Windows locale resilience.

ttkbootstrap triggers locale.setlocale calls at import time (e.g. dialogs.DatePickerDialog).
On some Windows environments this raises locale.Error('unsupported locale setting'), which
would crash frozen helper executables.

Keep this module stdlib-only so it is always safe to import early.
"""

from __future__ import annotations

import locale
import os


def harden_locale_for_ttkbootstrap() -> None:
    """Harden locale initialization so ttkbootstrap imports can't crash.

    Safe to call multiple times.
    """
    try:
        if os.name != "nt":
            return

        for key in ("LC_ALL", "LC_CTYPE", "LANG"):
            value = os.environ.get(key)
            if value and ("utf-8" in value.lower()) and ("_" in value) and ("." in value):
                os.environ.pop(key, None)

        orig = locale.setlocale

        def safe_setlocale(category, value=None):  # type: ignore[no-untyped-def]
            try:
                if value is None:
                    return orig(category)
                return orig(category, value)
            except locale.Error:
                for fallback in ("", "C"):
                    try:
                        return orig(category, fallback)
                    except Exception:
                        pass
                try:
                    return orig(category)
                except Exception:
                    return "C"

        locale.setlocale = safe_setlocale  # type: ignore[assignment]

        try:
            locale.setlocale(locale.LC_ALL, "")
        except Exception:
            try:
                locale.setlocale(locale.LC_ALL, "C")
            except Exception:
                pass
    except Exception:
        # Never fail during bootstrap.
        pass
