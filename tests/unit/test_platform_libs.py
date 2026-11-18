# SPDX-License-Identifier: Apache-2.0
"""Linux: X11 libraries of the environment for Qt (mskpipe.gui.platform_libs)."""

import ctypes
import importlib.metadata
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

from mskpipe.gui import platform_libs as libs

ROOT = Path(__file__).resolve().parents[2]
LINUX = sys.platform.startswith("linux")


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """A Linux whose only system library folder is ``tmp_path/system``."""
    system = tmp_path / "system"
    system.mkdir()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(libs, "SYSTEM_DIRS", (str(system),))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    return system


def test_preload_loads_only_what_the_system_lacks(linux, tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    (prefix / "lib").mkdir(parents=True)
    for name in ("libxcb.so.1", "libxcb-cursor.so.0", "libxkbcommon.so.0", "libxcb-icccm.so.4"):
        (prefix / "lib" / name).write_bytes(b"")
    (linux / "libxcb.so.1").write_bytes(b"")  # the system has its own
    loaded = []

    def cdll(path, mode=0):
        if path.endswith("libxcb-icccm.so.4"):
            raise OSError("cannot open shared object file")
        loaded.append((Path(path), mode))

    monkeypatch.setattr(libs.ctypes, "CDLL", cdll)
    assert libs.preload(prefix) == ["libxcb-cursor.so.0", "libxkbcommon.so.0"]
    assert loaded == [
        (prefix / "lib" / "libxcb-cursor.so.0", ctypes.RTLD_GLOBAL),
        (prefix / "lib" / "libxkbcommon.so.0", ctypes.RTLD_GLOBAL),
    ]
    monkeypatch.setenv("LD_LIBRARY_PATH", str(prefix / "lib"))  # the user brings them
    assert libs.preload(prefix) == []


def test_nothing_outside_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(libs.ctypes, "CDLL", lambda *a, **k: pytest.fail("loaded"))
    assert libs.preload() == []
    assert libs.missing_system_libraries() == [] and libs.missing_hint() is None


def test_hint_names_the_packages(linux):
    assert libs.missing_system_libraries() == list(libs.SYSTEM_LIBRARIES)
    for name in libs.SYSTEM_LIBRARIES:
        if name != "libGL.so.1":
            (linux / name).write_bytes(b"")
    hint = libs.missing_hint()
    assert "libGL.so.1" in hint and "sudo apt install libgl1\n" in hint
    (linux / "libGL.so.1").write_bytes(b"")
    assert libs.missing_hint() is None


def test_environment_brings_the_libraries():
    """pixi.toml: the conda packages with BUNDLED_LIBRARIES, on Linux only."""
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    packages = set(pixi["target"]["linux-64"]["dependencies"])
    assert packages >= {
        "xorg-libx11",
        "xorg-libxrender",
        "libxcb",
        "xcb-util",
        "xcb-util-cursor",
        "xcb-util-image",
        "xcb-util-keysyms",
        "xcb-util-renderutil",
        "xcb-util-wm",
        "libxkbcommon",
    }
    assert not packages & set(pixi["dependencies"])
    assert len(set(libs.BUNDLED_LIBRARIES)) == len(libs.BUNDLED_LIBRARIES)
    assert not set(libs.BUNDLED_LIBRARIES) & set(libs.SYSTEM_LIBRARIES)


@pytest.mark.skipif(not LINUX, reason="shared libraries of Linux")
def test_libraries_are_in_this_environment():
    folder = Path(sys.prefix) / "lib"
    if not (folder / "libxcb-cursor.so.0").is_file():
        pytest.skip("not a pixi environment of this project")
    missing = [name for name in libs.BUNDLED_LIBRARIES if not (folder / name).is_file()]
    assert missing == []
    assert libs.preload() == [n for n in libs.BUNDLED_LIBRARIES if not libs.system_has(n)]


@pytest.mark.skipif(not LINUX, reason="shared libraries of Linux")
def test_qt_imports():
    """The GUI tests start with pytest.importorskip: a missing library must fail here."""
    try:
        importlib.metadata.version("PySide6_Essentials")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("PySide6 not installed")
    try:
        importlib.import_module("PySide6.QtWidgets")
    except ImportError as exc:
        pytest.fail(f"{exc}\n{libs.missing_hint()}")


def test_gui_package_preloads_before_qt():
    text = (ROOT / "src" / "mskpipe" / "gui" / "__init__.py").read_text(encoding="utf-8")
    assert "preload()" in text and "PySide6" not in text.split('"""')[2]
    main = (ROOT / "src" / "mskpipe" / "gui" / "__main__.py").read_text(encoding="utf-8")
    assert main.index("missing_hint") < main.index("mskpipe.gui.app")
    source = Path(libs.__file__).read_text(encoding="utf-8")
    assert "PySide6" not in source.split('"""')[2]  # standard library only
    assert importlib.util.find_spec("mskpipe.gui.platform_libs") is not None
