# SPDX-License-Identifier: Apache-2.0
"""GPU support on demand for the Windows release (``mskpipe setup gpu``).

The pixi environments ``cpu`` and ``gpu`` differ only in two wheels: ``torch`` and
``torchvision`` (``+cpu`` vs ``+cu128``; checked by a test against ``pixi.lock``). The
release ships the CPU environment; GPU support downloads the two CUDA wheels pinned in
``pixi.lock`` (URL and SHA-256, copied to ``gpu_wheels.json``) and unpacks them into an
*overlay* folder shared by all msk-pipe versions::

    <MSKPIPE_HOME>/gpu/<tag>/site-packages/   torch + torchvision (CUDA build)
    <MSKPIPE_HOME>/gpu/<tag>/overlay.json     what was installed (written last)

Only the child processes of a run whose device resolved to ``cuda`` (and the torch probe)
get the overlay prepended to ``PYTHONPATH``; CPU runs keep the CPU build. The packages
seen by a GPU run are the same as in the pixi ``gpu`` environment.

``MSKPIPE_TORCH_OVERLAY`` points to another overlay ``site-packages`` folder (tests,
custom builds).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from collections.abc import Callable, Mapping
from importlib import metadata
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

from mskpipe.resources.download import Progress, download, file_sha256
from mskpipe.resources.paths import home
from mskpipe.resources.tool import Log

OVERLAY_ENV = "MSKPIPE_TORCH_OVERLAY"
SPEC_FILE = "gpu_wheels.json"
MARKER = "overlay.json"
SITE = "site-packages"
OVERLAY_PACKAGES = ("torch", "torchvision")  # the only packages that differ cpu <-> gpu
MIN_FREE_BYTES = 10 * 2**30  # downloads (~3.5 GB) + unpacked overlay (~6 GB)


class GpuSetupError(RuntimeError):
    """GPU support cannot be installed here or the installation failed."""


class Wheel(BaseModel):
    name: str
    version: str
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return url_filename(self.url)


class GpuSpec(BaseModel):
    format: str = "mskpipe.gpu_wheels"
    version: int = 1
    tag: str  # e.g. torch-2.7.1+cu128
    python: str  # e.g. cp311
    platform: str  # e.g. win_amd64
    wheels: list[Wheel]


class GpuStatus(BaseModel):
    tag: str
    installed: bool
    path: str | None = None
    source: str  # "overlay", "env" (MSKPIPE_TORCH_OVERLAY) or "none"
    packages: dict[str, str] = {}
    supported: bool  # the release platform (Windows, CPython 3.11)


# ---------------------------------------------------------------- spec


def url_filename(url: str) -> str:
    from urllib.parse import unquote, urlparse

    return unquote(PurePosixPath(urlparse(url).path).name)


def load_spec() -> GpuSpec:
    text = files("mskpipe.resources").joinpath(SPEC_FILE).read_text(encoding="utf-8")
    return GpuSpec.model_validate_json(text)


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_packages(lock: Mapping[str, Any], env: str, platform: str) -> dict[str, str]:
    """Package name -> version of one locked environment (conda and PyPI)."""
    pypi = {p["pypi"]: p for p in lock["packages"] if "pypi" in p}
    out: dict[str, str] = {}
    for entry in lock["environments"][env]["packages"][platform]:
        if "pypi" in entry:
            info = pypi.get(entry["pypi"])
            if info is not None:  # git and path packages: the URL stands for the version
                out[_canonical(info["name"])] = str(info.get("version") or entry["pypi"])
        elif "conda" in entry:
            stem = url_filename(entry["conda"]).removesuffix(".conda").removesuffix(".tar.bz2")
            name, version, _build = stem.rsplit("-", 2)
            out[_canonical(name)] = version
    return out


def env_differences(
    lock: Mapping[str, Any], a: str = "cpu", b: str = "gpu", platform: str = "win-64"
) -> dict[str, tuple[str | None, str | None]]:
    """Packages whose version differs between two locked environments. The local version
    label is kept (``2.7.1+cpu`` vs ``2.7.1+cu128``); conda vs PyPI origin is ignored."""
    pa, pb = _lock_packages(lock, a, platform), _lock_packages(lock, b, platform)
    return {
        name: (pa.get(name), pb.get(name))
        for name in sorted(set(pa) | set(pb))
        if pa.get(name) != pb.get(name)
    }


def spec_from_lock(lock: Mapping[str, Any], env: str = "gpu", platform: str = "win-64") -> GpuSpec:
    """GPU wheels (torch, torchvision) of a locked environment with URL and SHA-256."""
    pypi = {p["pypi"]: p for p in lock["packages"] if "pypi" in p}
    wheels: dict[str, Wheel] = {}
    for entry in lock["environments"][env]["packages"][platform]:
        info = pypi.get(entry.get("pypi", ""))
        if info is None or _canonical(info["name"]) not in OVERLAY_PACKAGES:
            continue
        wheels[_canonical(info["name"])] = Wheel(
            name=_canonical(info["name"]),
            version=str(info["version"]),
            url=entry["pypi"],
            sha256=info["sha256"],
        )
    missing = [p for p in OVERLAY_PACKAGES if p not in wheels]
    if missing:
        raise GpuSetupError(f"pixi.lock environment '{env}' has no wheel for {missing}")
    torch = wheels["torch"]
    parts = torch.filename.removesuffix(".whl").split("-")  # name-ver-py-abi-platform
    return GpuSpec(
        tag=f"torch-{torch.version}",
        python=parts[2],
        platform=parts[4],
        wheels=[wheels[p] for p in OVERLAY_PACKAGES],
    )


# ---------------------------------------------------------------- location / status


def overlay_root() -> Path:
    return home() / "gpu"


def overlay_dir(spec: GpuSpec | None = None) -> Path:
    return overlay_root() / (spec or load_spec()).tag


def platform_supported(spec: GpuSpec | None = None) -> bool:
    """The overlay wheels fit this interpreter (Windows x64, the pinned CPython)."""
    spec = spec or load_spec()
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return sys.platform == "win32" and spec.platform == "win_amd64" and spec.python == py


def installed_overlay() -> Path | None:
    """``site-packages`` of the GPU overlay to use, or None."""
    custom = os.environ.get(OVERLAY_ENV)
    if custom:
        return Path(custom) if Path(custom).is_dir() else None
    try:
        spec = load_spec()
    except (OSError, ValueError):
        return None
    folder = overlay_dir(spec)
    try:
        marker = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if marker.get("tag") != spec.tag:
        return None
    site = folder / SITE
    return site if site.is_dir() else None


def overlay_packages(site: Path) -> dict[str, str]:
    """Distribution name -> version found in an overlay folder."""
    return {
        _canonical(d.metadata["Name"]): d.version
        for d in metadata.distributions(path=[str(site)])
        if d.metadata["Name"]
    }


def gpu_status() -> GpuStatus:
    spec = load_spec()
    site = installed_overlay()
    source = "none" if site is None else "env" if os.environ.get(OVERLAY_ENV) else "overlay"
    return GpuStatus(
        tag=spec.tag,
        installed=site is not None,
        path=str(site) if site else None,
        source=source,
        packages=overlay_packages(site) if site else {},
        supported=platform_supported(spec),
    )


def with_overlay(env: Mapping[str, str], site: Path | None) -> dict[str, str]:
    """``PYTHONPATH`` for a child process: the overlay first, then what was there."""
    out = dict(env)
    if site is None:
        return out
    current = out.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    out["PYTHONPATH"] = os.pathsep.join(p for p in (str(site), current) if p)
    return out


# ---------------------------------------------------------------- install


def extract_wheel(wheel: Path, site: Path) -> None:
    """Unpack a wheel like ``pip install --target --no-deps``: package files and the
    ``.dist-info`` into ``site``; ``<dist>.data/{purelib,platlib}`` merged into ``site``,
    other ``.data`` parts (scripts, headers) are not needed and skipped."""
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise GpuSetupError(f"Unsafe path in {wheel.name}: {info.filename}")
            parts = name.parts
            if parts and parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                    continue
                parts = parts[2:]
            if info.is_dir() or not parts:
                continue
            target = site.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)


def install_gpu(
    log: Log,
    *,
    progress: Callable[[Wheel], Progress] | None = None,
    force: bool = False,
    keep_downloads: bool = False,
    spec: GpuSpec | None = None,
    wheel_dir: Path | None = None,
) -> Path:
    """Download (or take from ``wheel_dir``) the pinned CUDA wheels, check their SHA-256
    and unpack them into the overlay. Returns the overlay ``site-packages``."""
    spec = spec or load_spec()
    if not platform_supported(spec):
        raise GpuSetupError(
            f"GPU support for the release needs Windows x64 with {spec.python}; from the "
            "sources use the pixi GPU environment instead: pixi run -e gpu mskpipe ..."
        )
    final = overlay_dir(spec)
    if (final / MARKER).is_file() and not force:
        log(f"GPU support ({spec.tag}) is already installed: {final}")
        return final / SITE
    root = overlay_root()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise GpuSetupError(
            f"Not enough free disk space in {root}: {free / 2**30:.1f} GiB, "
            f"needs {MIN_FREE_BYTES / 2**30:.0f} GiB"
        )
    downloads = root / "downloads"
    paths = []
    for wheel in spec.wheels:
        path = _fetch_wheel(wheel, downloads, wheel_dir, log, progress)
        paths.append(path)

    staging = root / f"{spec.tag}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    for wheel, path in zip(spec.wheels, paths, strict=True):
        log(f"Unpacking {wheel.filename}")
        extract_wheel(path, staging / SITE)
    found = overlay_packages(staging / SITE)
    for wheel in spec.wheels:
        if found.get(wheel.name) != wheel.version:
            raise GpuSetupError(f"{wheel.filename}: unpacked {found.get(wheel.name)}")
    marker = {"tag": spec.tag, "wheels": [w.model_dump() for w in spec.wheels]}
    (staging / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if final.exists():
        _move_aside(final, root / f"{spec.tag}.old")
    staging.replace(final)
    if not keep_downloads:
        for path in paths:
            if path.parent == downloads:
                path.unlink(missing_ok=True)
    log(f"GPU support installed: {final}")
    return final / SITE


def _fetch_wheel(
    wheel: Wheel,
    downloads: Path,
    wheel_dir: Path | None,
    log: Log,
    progress: Callable[[Wheel], Progress] | None,
) -> Path:
    candidates = [p / wheel.filename for p in (wheel_dir, downloads) if p is not None]
    for path in candidates:
        if path.is_file():
            if file_sha256(path) == wheel.sha256:
                log(f"Using {path}")
                return path
            log(f"Ignoring {path} (checksum differs)")
    target = downloads / wheel.filename
    log(f"Downloading {wheel.filename}")
    download(
        wheel.url,
        target,
        sha256=wheel.sha256,
        progress=progress(wheel) if progress else None,
    )
    return target


IN_USE = (
    "GPU support is in use, probably by a run on the GPU (Windows does not allow changing "
    "loaded DLLs). Wait until it finishes, close the other msk-pipe windows and try again."
)


def _move_aside(folder: Path, old: Path) -> None:
    """Rename ``folder`` out of the way, then delete it. The rename fails as a whole on
    Windows while one of its DLLs is loaded, so a GPU run never sees a half-deleted
    overlay; deleting the renamed copy may leave files behind, removed next time."""
    shutil.rmtree(old, ignore_errors=True)
    try:
        folder.replace(old)
    except OSError as exc:
        raise GpuSetupError(f"{IN_USE} ({exc})") from exc
    shutil.rmtree(old, ignore_errors=True)


def remove_gpu(log: Log, spec: GpuSpec | None = None) -> bool:
    """Delete the overlay and leftover downloads; False if nothing was installed.
    Raises :class:`GpuSetupError` if the overlay is in use."""
    root = overlay_root()
    folder = overlay_dir(spec)
    existed = folder.exists()
    if existed:
        _move_aside(folder, root / f"{folder.name}.old")
    for path in (root / f"{folder.name}.partial", root / "downloads"):
        shutil.rmtree(path, ignore_errors=True)
    log(f"GPU support removed: {folder}" if existed else "GPU support was not installed")
    return existed
