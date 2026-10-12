# SPDX-License-Identifier: Apache-2.0
"""Streamed download with a SHA-256 check (standard library only; honours HTTPS_PROXY)."""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

CHUNK = 1 << 20
TIMEOUT_S = 60

Progress = Callable[[int, int | None], None]  # (bytes done, total or None)


class DownloadError(RuntimeError):
    """A download failed or its checksum does not match."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    progress: Progress | None = None,
    timeout: float = TIMEOUT_S,
) -> str:
    """Download ``url`` to ``dest`` (via ``<dest>.part``, replaced only when complete and,
    if ``sha256`` is given, matching). Returns the SHA-256 of the file."""
    from mskpipe import __version__

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": f"mskpipe/{__version__}"})
    digest = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as f:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            while block := response.read(CHUNK):
                f.write(block)
                digest.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"Download of {url} failed: {exc}") from exc
    if total is not None and done != total:
        part.unlink(missing_ok=True)
        raise DownloadError(f"Download of {url} incomplete: {done} of {total} bytes")
    value = digest.hexdigest()
    if sha256 and value != sha256.lower():
        part.unlink(missing_ok=True)
        raise DownloadError(f"Checksum mismatch for {url}: expected {sha256}, got {value}")
    part.replace(dest)
    return value
