# SPDX-License-Identifier: Apache-2.0
"""Collect the licences of every package in an unpacked environment (release ``licenses/``).

    python packaging/collect_licenses.py --prefix build/test/env --out build/licenses

* conda packages: ``conda-meta/*.json`` (name, version, licence, source) and the licence
  files they install (paths containing LICENSE, LICENCE, COPYING, NOTICE, AUTHORS),
* PyPI packages: ``*.dist-info`` in site-packages (``License-Expression``/``License``/
  classifiers in METADATA, files in ``licenses/`` and LICENSE*/COPYING*/NOTICE*).

Output: ``packages.csv`` (one row per package) and ``<name>-<version>/`` with copies of the
licence files. Standard library only, so it runs with the environment's own Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from email.parser import Parser
from pathlib import Path

LICENSE_FILE = re.compile(r"(licen[cs]e|copying|notice|authors|copyright)", re.IGNORECASE)
FIELDS = ("name", "version", "source", "license", "url", "files")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name)


def conda_packages(prefix: Path) -> list[dict]:
    rows = []
    for meta in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        files = [prefix / f for f in data.get("files", []) if LICENSE_FILE.search(Path(f).name)]
        rows.append(
            {
                "name": data.get("name", meta.stem),
                "version": data.get("version", ""),
                "source": "conda",
                "license": data.get("license", ""),
                "url": data.get("url", ""),
                "paths": [f for f in files if f.is_file()],
            }
        )
    return rows


def _site_packages(prefix: Path) -> list[Path]:
    found = [prefix / "Lib" / "site-packages"]  # Windows layout
    found += sorted(prefix.glob("lib/python3*/site-packages"))  # POSIX layout
    return [p for p in found if p.is_dir()]


def pypi_packages(prefix: Path, conda_names: set[str]) -> list[dict]:
    rows = []
    for site in _site_packages(prefix):
        for info in sorted(site.glob("*.dist-info")):
            try:
                meta = Parser().parsestr((info / "METADATA").read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            name = meta.get("Name", info.name.split("-")[0])
            if (info / "INSTALLER").is_file():
                installer = (info / "INSTALLER").read_text(encoding="utf-8").strip()
                if installer == "conda" and name.lower() in conda_names:
                    continue  # described by its conda record
            licence = meta.get("License-Expression") or ""
            if not licence:
                short = (meta.get("License") or "").strip()
                licence = short if short and "\n" not in short and len(short) < 120 else ""
            if not licence:
                classifiers = [c for c in meta.get_all("Classifier") or [] if "License ::" in c]
                licence = "; ".join(c.split("::")[-1].strip() for c in classifiers)
            paths = [p for p in info.rglob("*") if p.is_file() and LICENSE_FILE.search(p.name)]
            rows.append(
                {
                    "name": name,
                    "version": meta.get("Version", ""),
                    "source": "pypi",
                    "license": licence,
                    "url": meta.get("Home-page") or _project_url(meta),
                    "paths": paths,
                }
            )
    return rows


def _project_url(meta) -> str:
    for item in meta.get_all("Project-URL") or []:
        _label, _, url = item.partition(",")
        if url.strip():
            return url.strip()
    return ""


def collect(prefix: Path, out: Path) -> list[dict]:
    if not (prefix / "conda-meta").is_dir():
        raise SystemExit(f"{prefix} is not an environment (no conda-meta)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    conda = conda_packages(prefix)
    rows = conda + pypi_packages(prefix, {r["name"].lower() for r in conda})
    for row in rows:
        folder = out / _safe(f"{row['name']}-{row['version']}")
        copied = []
        for path in row.pop("paths"):
            folder.mkdir(exist_ok=True)
            target = folder / _safe(path.name)
            n = 1
            while target.exists():
                n += 1
                target = folder / _safe(f"{path.stem}.{n}{path.suffix}")
            shutil.copy2(path, target)
            copied.append(target.name)
        row["files"] = " ".join(copied)
    with (out / "packages.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["name"].lower(), r["source"])))
    (out / "README.txt").write_text(
        "Licences of the packages bundled in environment.tar (one row per package in\n"
        "packages.csv; licence files copied into <name>-<version>/). msk-pipe itself is\n"
        "Apache-2.0 (../LICENSE); see ../THIRD_PARTY_LICENSES.md for the main components.\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", required=True, help="unpacked environment")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = collect(Path(args.prefix), Path(args.out))
    missing = [r["name"] for r in rows if not r["license"]]
    print(f"{len(rows)} packages, licence unknown for {len(missing)}: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
