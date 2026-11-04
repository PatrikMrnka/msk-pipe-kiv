# SPDX-License-Identifier: Apache-2.0
"""Make the attachment atlas asset of the release (run once per atlas version, locally).

    python packaging/make_atlas_zip.py data/reference-GT/lhdl_atlas --out dist

Writes ``dist/msk-pipe-atlas-lhdl-v1.zip`` with ``lhdl/atlas.json``, ``bones/*.obj``,
``muscles/**/*.vtk`` and the atlas' own ``LICENSE`` and ``README.md`` (CC BY-NC-SA 2.0 BE,
from ``packaging/atlas``), and prints its SHA-256 and the commands to publish it. The zip
is byte-for-byte reproducible (fixed order and timestamps).

The atlas is derived from LHDL data: it is never committed, only published as this
separately licensed release asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "packaging" / "atlas"
DATE = (2026, 1, 1, 0, 0, 0)
PATTERNS = ("atlas.json", "bones/*.obj", "muscles/**/*.vtk")


def atlas_files(folder: Path) -> list[Path]:
    if not (folder / "atlas.json").is_file():
        raise SystemExit(f"{folder} has no atlas.json")
    json.loads((folder / "atlas.json").read_text(encoding="utf-8"))  # must be valid
    files = sorted({p for pattern in PATTERNS for p in folder.glob(pattern) if p.is_file()})
    if not any(p.parent.name == "bones" for p in files):
        raise SystemExit(f"{folder} has no bones/*.obj")
    return files


def make_zip(folder: Path, archive: Path, name: str = "lhdl") -> str:
    """Write the zip; returns its SHA-256."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        (f"{name}/{p.relative_to(folder).as_posix()}", p.read_bytes()) for p in atlas_files(folder)
    ]
    for template in ("LICENSE", "README.md"):
        entries.append((f"{name}/{template}", (TEMPLATES / template).read_bytes()))
    with zipfile.ZipFile(archive, "w") as zf:
        for arcname, data in sorted(entries):
            info = zipfile.ZipInfo(arcname, DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("atlas", type=Path, help="atlas folder (atlas.json, bones/, muscles/)")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--tag", default="atlas-lhdl-v1")
    args = parser.parse_args(argv)
    archive = args.out / f"msk-pipe-{args.tag}.zip"
    digest = make_zip(args.atlas, archive)
    print(f"{archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
    print(f'sha256 = "{digest}"   -> packaging/release.toml [atlas]')
    print("Publish once:")
    print(
        f"  gh release create {args.tag} {archive.as_posix()} --latest=false "
        f'--title "Attachment atlas LHDL v1 (CC BY-NC-SA 2.0 BE)" '
        '--notes "Attachment atlas for msk-pipe, derived from LHDL Cadaver #1. '
        'Licence CC BY-NC-SA 2.0 BE (not Apache-2.0), see LICENSE and README.md in the zip."'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
