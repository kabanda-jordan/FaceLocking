"""Small Linux/Qt compatibility helpers for the live OpenCV window.

The OpenCV 5 Qt wheel currently ships the xcb platform plugin but omits the
``cv2/qt/fonts`` directory that older Qt bundles used to provide.  On a
Wayland desktop Qt then prints a font warning every time a window is opened.
This module prepares the small compatibility pieces before the GUI is created.

The helper is deliberately best-effort: image-only scripts and headless
machines must continue to work even when Qt or a desktop font directory is not
available.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _font_source() -> Optional[Path]:
    """Return the first installed system font suitable for Qt."""
    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def prepare_qt() -> bool:
    """Prepare Qt for OpenCV's bundled GUI plugin.

    Returns ``True`` when the OpenCV package was found, but the function is
    intentionally non-fatal: a missing GUI/font must not prevent model setup,
    image processing, or headless tests from running.
    """
    if not sys.platform.startswith("linux"):
        return False

    # The OpenCV wheel contains the xcb plugin, but not a Wayland plugin.  If
    # this process has an X display, prefer xcb and keep Qt from warning about
    # the desktop's Wayland session type.  Respect an explicitly selected
    # platform so advanced users can still override this behavior.
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    if (
        os.environ.get("QT_QPA_PLATFORM") == "xcb"
        and os.environ.get("XDG_SESSION_TYPE") == "wayland"
    ):
        os.environ["XDG_SESSION_TYPE"] = "xcb"

    try:
        import cv2  # noqa: PLC0415 - deliberately delayed until app import
    except Exception:
        return False

    try:
        cv2_root = Path(cv2.__file__).resolve().parent
        fonts_dir = cv2_root / "qt" / "fonts"
        fonts_dir.mkdir(parents=True, exist_ok=True)
        # Some installations already provide a real directory or a font.  Do
        # not replace it; only fill an empty directory with one known font.
        if not any(fonts_dir.iterdir()):
            source = _font_source()
            if source is not None:
                shutil.copy2(source, fonts_dir / source.name)
    except (OSError, AttributeError, TypeError):
        # The warning is cosmetic; inability to write into a system-managed
        # OpenCV installation must not stop the application.
        return False

    return True


__all__ = ["prepare_qt"]
