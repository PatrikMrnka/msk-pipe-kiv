# SPDX-License-Identifier: Apache-2.0
"""Linux: the shared libraries Qt takes from the system.

The PySide6 wheels bring Qt itself but expect the client libraries of a desktop system:

* X11: Qt's ``xcb`` platform plugin needs the xcb-util family and libxkbcommon-x11.
  Ubuntu installs only a part of them (24.04 has no ``libxcb-cursor0``, needed since
  Qt 6.5; the Wayland-only desktop of 26.04 lacks seven), a server or WSL none. The pixi
  environment therefore has its own copies (conda-forge, ``[target.linux-64]`` in
  pixi.toml). Qt's libraries look only into their own folder and the system (``RUNPATH``),
  so :func:`preload` loads every copy the system does not have before Qt starts; the
  loader then finds it already there.
* OpenGL, fonts, D-Bus and GLib (:data:`SYSTEM_LIBRARIES`) belong to the graphics stack of
  the system and are part of every desktop installation. Without them Qt cannot be
  imported; :func:`missing_hint` names the packages to install (WSL, minimal systems).

Standard library only and no Qt: imported by ``mskpipe.gui`` before anything else.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# libraries of the environment Qt may need, dependencies first
BUNDLED_LIBRARIES = (
    "libXau.so.6",
    "libXdmcp.so.6",
    "libxcb.so.1",
    "libX11.so.6",
    "libX11-xcb.so.1",
    "libXrender.so.1",
    "libxcb-render.so.0",
    "libxcb-shm.so.0",
    "libxcb-randr.so.0",
    "libxcb-shape.so.0",
    "libxcb-sync.so.1",
    "libxcb-xfixes.so.0",
    "libxcb-xkb.so.1",
    "libxcb-glx.so.0",
    "libxcb-util.so.1",
    "libxcb-image.so.0",
    "libxcb-render-util.so.0",
    "libxcb-icccm.so.4",
    "libxcb-keysyms.so.1",
    "libxcb-cursor.so.0",
    "libxkbcommon.so.0",
    "libxkbcommon-x11.so.0",
)
# library -> Ubuntu/Debian package; not bundled: they must match the system (drivers, fonts)
SYSTEM_LIBRARIES = {
    "libGL.so.1": "libgl1",
    "libEGL.so.1": "libegl1",
    "libfontconfig.so.1": "libfontconfig1",
    "libfreetype.so.6": "libfreetype6",
    "libdbus-1.so.3": "libdbus-1-3",
    "libglib-2.0.so.0": "libglib2.0-0t64",
}
SYSTEM_DIRS = (
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib64",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
)


def system_has(name: str) -> bool:
    """Whether a system library folder has ``name`` (the usual folders of x86_64 Linux and
    ``LD_LIBRARY_PATH``)."""
    extra = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    return any((Path(folder) / name).exists() for folder in (*SYSTEM_DIRS, *extra))


def preload(prefix: Path | None = None) -> list[str]:
    """Load the environment's copy of every library in :data:`BUNDLED_LIBRARIES` that the
    system lacks; returns their names. Does nothing outside Linux, never raises."""
    if not sys.platform.startswith("linux"):
        return []
    folder = (prefix or Path(sys.prefix)) / "lib"
    loaded = []
    for name in BUNDLED_LIBRARIES:
        path = folder / name
        if system_has(name) or not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        loaded.append(name)
    return loaded


def missing_system_libraries() -> list[str]:
    """Libraries of :data:`SYSTEM_LIBRARIES` the system does not have (Linux only)."""
    if not sys.platform.startswith("linux"):
        return []
    return [name for name in SYSTEM_LIBRARIES if not system_has(name)]


def missing_hint() -> str | None:
    """What to install so that the GUI can start, or None if nothing is missing."""
    missing = missing_system_libraries()
    if not missing:
        return None
    packages = " ".join(SYSTEM_LIBRARIES[name] for name in missing)
    return (
        "The graphical interface needs the graphics libraries of a desktop system; missing: "
        f"{', '.join(missing)}. On Ubuntu/Debian install them with:\n"
        f"  sudo apt install {packages}\n"
        "The command line (mskpipe run ...) works without them."
    )
