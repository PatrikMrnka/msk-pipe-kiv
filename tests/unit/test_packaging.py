# SPDX-License-Identifier: Apache-2.0
"""Release tooling (packaging/): pins, wheel sources from pixi.lock, release folder and
archive (Windows zip, Linux tar.gz), licence collection, atlas asset, workflow, build info
(no network, on any system)."""

import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from mskpipe.core import build_info
from mskpipe.core.manifest import git_info, package_info

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging"
SHA = re.compile(r"^[0-9a-f]{64}$")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


br = _load("build_release")
licenses = _load("collect_licenses")
atlas_zip = _load("make_atlas_zip")


@pytest.fixture(scope="module")
def lock():
    return yaml.safe_load((ROOT / "pixi.lock").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- pins / lock


def test_pins(lock):
    pins = br.load_pins()
    version = pins["pixi-pack"]["version"]
    assert pins["environment"] == {"name": "cpu", "platforms": ["win-64", "linux-64"]}
    assert set(br.TARGETS) == set(pins["environment"]["platforms"])
    for platform in pins["environment"]["platforms"]:
        pp = pins["pixi-pack"][platform]
        assert SHA.match(pp["pack_sha256"]) and SHA.match(pp["unpack_sha256"])
        assert version in pp["pack_url"] and version in pp["unpack_url"]
        assert pp["pack_url"].endswith(".exe") == (platform == "win-64")
        # every release platform is locked (pixi.toml) and has its GPU wheels
        assert platform in lock["environments"]["cpu"]["packages"], "pixi.lock"
    assert len({pins["pixi-pack"][p]["pack_sha256"] for p in br.TARGETS}) == len(br.TARGETS)
    sha = pins["atlas"]["sha256"]
    assert sha == "" or SHA.match(sha)
    assert pins["atlas"]["file"] == f"msk-pipe-{pins['atlas']['tag']}.zip"


@pytest.mark.parametrize("platform", sorted(br.TARGETS))
def test_wheel_sources_from_lock(lock, platform):
    sources = br.wheel_sources(lock, "cpu", platform)
    assert sources[0] == str(ROOT) or str(ROOT) in sources
    gits = [s for s in sources if s.startswith("git+")]
    assert {s.split("@")[0] for s in gits} == {
        "git+https://github.com/MuscleMap/MuscleMap",
        "git+https://github.com/PatrikMrnka/pystaple",
    }
    assert all(re.search(r"@[0-9a-f]{40}$", s) for s in gits)
    assert all(not s.endswith(".whl") for s in sources)
    packages = br.git_packages(lock, "cpu", "win-64")
    assert set(packages) == {"scripts", "pystaple"}
    assert all(len(c) == 40 for c in packages.values())


def test_wheel_sources_rejects_unpinned_git():
    lock = {"environments": {"e": {"packages": {"p": [{"pypi": "git+https://x/y"}]}}}}
    with pytest.raises(br.BuildError, match="pinned revision"):
        br.wheel_sources(lock, "e", "p")


def test_wheel_version(tmp_path):
    (tmp_path / "mskpipe-0.1.0-py3-none-any.whl").write_bytes(b"")
    assert br.wheel_version(tmp_path) == "0.1.0"
    with pytest.raises(br.BuildError):
        br.wheel_version(tmp_path / "missing")


def test_fetch_checks_checksum(tmp_path):
    src = tmp_path / "tool.exe"
    src.write_bytes(b"binary")
    digest = br.sha256(src)
    assert br.fetch(src.as_uri(), tmp_path / "out" / "tool.exe", digest).is_file()
    with pytest.raises(br.BuildError, match="Checksum"):
        br.fetch(src.as_uri(), tmp_path / "out2" / "tool.exe", "0" * 64)
    with pytest.raises(br.BuildError, match="No SHA-256"):
        br.fetch(src.as_uri(), tmp_path / "out3" / "tool.exe", "")


# ---------------------------------------------------------------- atlas


def make_atlas(folder: Path) -> Path:
    (folder / "bones").mkdir(parents=True)
    (folder / "muscles" / "iliacus_r").mkdir(parents=True)
    (folder / "atlas.json").write_text('{"muscles": {}}', encoding="utf-8")
    (folder / "bones" / "femur_r.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (folder / "muscles" / "iliacus_r" / "Iliacus_Ori.vtk").write_text("x\n", encoding="utf-8")
    (folder / "report.txt").write_text("not shipped", encoding="utf-8")
    return folder


def test_atlas_zip_is_reproducible_and_licensed(tmp_path):
    src = make_atlas(tmp_path / "atlas")
    first = atlas_zip.make_zip(src, tmp_path / "a.zip")
    second = atlas_zip.make_zip(src, tmp_path / "b.zip")
    assert first == second
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert "lhdl/LICENSE" in names and "lhdl/README.md" in names
    assert "lhdl/muscles/iliacus_r/Iliacus_Ori.vtk" in names
    assert "lhdl/report.txt" not in names
    licence = zipfile.ZipFile(tmp_path / "a.zip").read("lhdl/LICENSE").decode()
    assert "CC BY-NC-SA 2.0 BE" in licence and "LHDL" in licence
    # the release build unpacks it and requires the licence files
    br.extract_atlas(tmp_path / "a.zip", tmp_path / "out")
    assert (tmp_path / "out" / "atlas.json").is_file() and (tmp_path / "out" / "LICENSE").is_file()


def test_extract_atlas_requires_licence_and_safe_paths(tmp_path):
    bare = tmp_path / "bare.zip"
    with zipfile.ZipFile(bare, "w") as zf:
        zf.writestr("lhdl/atlas.json", "{}")
    with pytest.raises(br.BuildError, match="LICENSE"):
        br.extract_atlas(bare, tmp_path / "x")
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../atlas.json", "{}")
    with pytest.raises(br.BuildError, match="Unsafe"):
        br.extract_atlas(evil, tmp_path / "y")


def test_atlas_missing_pin_is_allowed_only_on_request(monkeypatch, tmp_path, capsys):
    pins = br.load_pins()
    pins["atlas"]["sha256"] = ""
    monkeypatch.setattr(br, "load_pins", lambda path=br.PINS: pins)
    assert br.main(["atlas", "--out", str(tmp_path), "--allow-missing"]) == 0
    assert "building without it" in capsys.readouterr().out
    assert br.main(["atlas", "--out", str(tmp_path)]) == 1


# ---------------------------------------------------------------- release folder


def test_stage_and_zip(tmp_path):
    pack, unpack = tmp_path / "environment.tar", tmp_path / "pixi-unpack.exe"
    pack.write_bytes(b"tar")
    unpack.write_bytes(b"MZ")
    lic = tmp_path / "licenses"
    lic.mkdir()
    (lic / "packages.csv").write_text("name\n", encoding="utf-8")
    atlas = tmp_path / "atlas.zip"
    atlas_zip.make_zip(make_atlas(tmp_path / "atlas_src"), atlas)
    env = tmp_path / "env"
    (env / "Library" / "bin").mkdir(parents=True)
    (env / "VCRUNTIME140.dll").write_bytes(b"MZ")
    (env / "Library" / "bin" / "vcruntime140_1.dll").write_bytes(b"MZ")
    folder = br.stage(
        tmp_path / "stage" / "msk-pipe-0.1.0",
        runtime=br.runtime_dlls(env),
        version="0.1.0",
        pack=pack,
        unpack=unpack,
        licenses=lic,
        maxpath=180,
        atlas=atlas,
    )
    for name in (
        "msk-pipe.cmd",
        "mskpipe.cmd",
        "bin/_setup.cmd",
        "bin/pixi-unpack.exe",
        "bin/vcruntime140.dll",
        "bin/vcruntime140_1.dll",
        "bin/maxpath.txt",
        "environment.tar",
        "atlas/lhdl/atlas.json",
        "atlas/lhdl/LICENSE",
        "licenses/packages.csv",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_LICENSES.md",
        "CITATION.cff",
        "README.txt",
        "version.txt",
    ):
        assert (folder / name).is_file(), name
    for cmd in ("msk-pipe.cmd", "mskpipe.cmd", "bin/_setup.cmd"):
        data = (folder / cmd).read_bytes()
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b""), cmd
    assert "msk-pipe 0.1.0" in (folder / "README.txt").read_text(encoding="utf-8")
    archive = br.zip_folder(folder, tmp_path / "dist" / "msk-pipe-0.1.0-win64.zip")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "msk-pipe-0.1.0/msk-pipe.cmd" in names
        assert zf.getinfo("msk-pipe-0.1.0/environment.tar").compress_type == zipfile.ZIP_STORED
    sums = br.write_sums([archive], tmp_path / "dist" / "SHA256SUMS.txt").read_text("ascii")
    assert sums == f"{br.sha256(archive)}  {archive.name}\n"
    with pytest.raises(br.BuildError, match="2 GiB"):
        br.zip_folder(folder, tmp_path / "big.zip", limit=10)


def _linux_parts(tmp_path):
    pack, unpack = tmp_path / "environment.tar", tmp_path / "pixi-unpack"
    pack.write_bytes(b"tar")
    unpack.write_bytes(b"\x7fELF")
    lic = tmp_path / "licenses"
    lic.mkdir()
    (lic / "packages.csv").write_text("name\n", encoding="utf-8")
    atlas = tmp_path / "atlas.zip"
    atlas_zip.make_zip(make_atlas(tmp_path / "atlas_src"), atlas)
    env = tmp_path / "env"
    (env / "ssl").mkdir(parents=True)
    (env / "ssl" / "cacert.pem").write_text("-----BEGIN CERTIFICATE-----\n", encoding="ascii")
    return pack, unpack, lic, atlas, env


def test_stage_and_tar_linux(tmp_path):
    pack, unpack, lic, atlas, env = _linux_parts(tmp_path)
    folder = br.stage(
        tmp_path / "stage" / "msk-pipe-0.1.0",
        runtime=[br.ca_bundle(env)],
        version="0.1.0",
        pack=pack,
        unpack=unpack,
        licenses=lic,
        atlas=atlas,
        platform="linux-64",
    )
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert names >= {
        "msk-pipe",
        "mskpipe",
        "bin/_setup.sh",
        "bin/pixi-unpack",
        "bin/cacert.pem",
        "environment.tar",
        "atlas/lhdl/atlas.json",
        "atlas/lhdl/LICENSE",
        "licenses/packages.csv",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_LICENSES.md",
        "CITATION.cff",
        "README.txt",
        "version.txt",
    }
    assert not [n for n in names if n.endswith((".cmd", ".exe", ".dll")) or "maxpath" in n]
    for script in ("msk-pipe", "mskpipe", "bin/_setup.sh", "README.txt", "version.txt"):
        assert b"\r" not in (folder / script).read_bytes(), script
    assert (folder / "version.txt").read_bytes() == b"0.1.0\n"
    assert "msk-pipe 0.1.0 for Linux" in (folder / "README.txt").read_text(encoding="utf-8")
    assert br.executables("linux-64") == ("msk-pipe", "mskpipe", "bin/pixi-unpack")
    assert br.executables("win-64") == ()

    archive = tmp_path / "dist" / "msk-pipe-0.1.0-linux-x86_64.tar.gz"
    br.tar_folder(folder, archive, br.executables("linux-64"))
    with tarfile.open(archive) as tf:
        members = {m.name: m for m in tf.getmembers()}
    staged = {p.relative_to(folder.parent).as_posix() for p in folder.rglob("*")}
    assert set(members) == {"msk-pipe-0.1.0"} | staged
    for name, member in members.items():
        relative = name.removeprefix("msk-pipe-0.1.0").lstrip("/")
        runnable = member.isdir() or relative in br.executables("linux-64")
        assert member.mode == (0o755 if runnable else 0o644), name
        assert (member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "")
    with pytest.raises(br.BuildError, match="2 GiB"):
        br.tar_folder(folder, tmp_path / "big.tar.gz", limit=10)
    with pytest.raises(br.BuildError, match="Not in"):
        br.tar_folder(folder, tmp_path / "x.tar.gz", ["bin/missing"])


def test_ca_bundle_required(tmp_path):
    with pytest.raises(br.BuildError, match="ca-certificates"):
        br.ca_bundle(tmp_path)


def test_assemble_linux_and_checksum_list(tmp_path, monkeypatch, capsys):
    pack, unpack, lic, atlas, env = _linux_parts(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    # the list of the Windows build, which made the draft release
    (dist / "SHA256SUMS.txt").write_text(f"{'a' * 64}  msk-pipe-0.1.0-win64.zip\n", "ascii")
    argv = ["assemble", "--platform", "linux-64", "--version", "0.1.0", "--pack", str(pack)]
    argv += ["--unpack", str(unpack), "--licenses", str(lic), "--runtime-from", str(env)]
    argv += ["--atlas", str(atlas), "--work", str(tmp_path / "work"), "--out", str(dist)]
    assert br.main(argv) == 0
    package = dist / "msk-pipe-0.1.0-linux-x86_64.tar.gz"
    assert package.is_file() and (dist / "atlas.zip").is_file()
    lines = (dist / "SHA256SUMS.txt").read_text("ascii").splitlines()
    assert lines == [
        f"{br.sha256(dist / 'atlas.zip')}  atlas.zip",
        f"{br.sha256(package)}  {package.name}",
        f"{'a' * 64}  msk-pipe-0.1.0-win64.zip",
    ]
    # the workflow step: add one file to a downloaded list; a second time changes nothing
    listed = tmp_path / "sums" / "SHA256SUMS.txt"
    assert br.main(["sums", "--file", str(listed), str(package)]) == 0
    assert br.main(["sums", "--file", str(listed), str(package)]) == 0
    assert listed.read_text("ascii") == f"{br.sha256(package)}  {package.name}\n"
    # Windows needs the path limit of its environment
    argv[2] = "win-64"
    assert br.main(argv) == 1 and "--maxpath" in capsys.readouterr().err


def test_runtime_dlls_required(tmp_path):
    with pytest.raises(br.BuildError, match=r"vcruntime140\.dll"):
        br.runtime_dlls(tmp_path)


def test_longest_path(tmp_path):
    deep = tmp_path / "env" / "Lib" / "site-packages" / "torch" / "include"
    deep.mkdir(parents=True)
    (deep / "x.h").write_text("", encoding="utf-8")
    expected = len(str(Path("env/Lib/site-packages/torch/include/x.h")))
    assert br.longest_path(tmp_path) == expected


def test_launcher_references():
    setup = (PACKAGING / "launcher" / "bin" / "_setup.cmd").read_text(encoding="utf-8")
    for name in (
        "version.txt",
        "environment.tar",
        "pixi-unpack.exe",
        "maxpath.txt",
        "activate.bat",
        "MSKPIPE_ATLAS_DIR",
        "atlas\\lhdl\\atlas.json",
    ):
        assert name in setup, name  # fmt: skip
    gui = (PACKAGING / "launcher" / "msk-pipe.cmd").read_text(encoding="utf-8")
    assert "-m mskpipe.gui" in gui and "MSKPIPE_PYTHONW" in gui
    assert importlib.util.find_spec("mskpipe.gui.__main__") is not None


def test_linux_launcher_references():
    linux = PACKAGING / "launcher" / "linux"
    setup = (linux / "bin" / "_setup.sh").read_text(encoding="utf-8")
    for name in (
        "version.txt",
        "environment.tar",
        "bin/pixi-unpack",
        "bin/cacert.pem",
        "MSKPIPE_ATLAS_DIR",
        "atlas/lhdl/atlas.json",
        "PYTHONNOUSERSITE",
    ):
        assert name in setup, name
    # the activate.sh of pixi-unpack breaks on a path with a space: not sourced
    assert not re.search(r"^\s*(source|\.)\s.*activate\.sh", setup, re.MULTILINE)
    gui = (linux / "msk-pipe").read_text(encoding="utf-8")
    cli = (linux / "mskpipe").read_text(encoding="utf-8")
    assert "-m mskpipe.gui" in gui and '-m mskpipe "$@"' in cli
    for text in (gui, cli):
        assert text.startswith("#!/usr/bin/env bash\n") and "bin/_setup.sh" in text
        assert "readlink -f" in text  # started through a symbolic link
    for script in (linux / "msk-pipe", linux / "mskpipe", linux / "bin" / "_setup.sh"):
        assert b"\r" not in script.read_bytes(), script
    readme = (linux / "README.txt").read_text(encoding="utf-8")
    assert "{version}" in readme and "./mskpipe" in readme and "./msk-pipe" in readme


@pytest.mark.skipif(sys.platform == "win32", reason="bash")
def test_linux_launcher_runs(tmp_path):
    """The scripts with a fake pixi-unpack: first start, a path with a space, stdout only
    from the program, a moved folder, the start through a symbolic link."""
    pack, unpack, lic, atlas, env = _linux_parts(tmp_path)
    python = '#!/bin/sh\necho "python $* atlas=$MSKPIPE_ATLAS_DIR"\n'
    unpack.write_text(
        "#!/bin/sh\n"  # pixi-unpack ARCHIVE -o ROOT -e env -s bash -q
        'test -f "$SSL_CERT_FILE" || exit 3\n'
        'mkdir -p "$3/env/bin"\n'
        f"cat > \"$3/env/bin/python\" <<'EOF'\n{python}EOF\n"
        'chmod +x "$3/env/bin/python"\n'
        "echo unpacked\n",
        encoding="utf-8",
    )
    folder = br.stage(
        tmp_path / "stage" / "msk-pipe-0.1.0",
        runtime=[br.ca_bundle(env)],
        version="0.1.0",
        pack=pack,
        unpack=unpack,
        licenses=lic,
        atlas=atlas,
        platform="linux-64",
    )
    archive = br.tar_folder(folder, tmp_path / "p.tar.gz", br.executables("linux-64"))
    home = tmp_path / "a home"
    home.mkdir()
    with tarfile.open(archive) as tf:
        tf.extractall(home, filter="tar")
    root = home / "msk-pipe-0.1.0"

    def run(script, *args):
        return subprocess.run(
            [str(script), *args], capture_output=True, text=True, stdin=subprocess.DEVNULL
        )

    first = run(root / "mskpipe", "run", "a b.nii.gz")
    assert first.returncode == 0, first.stderr
    assert first.stdout == f"python -m mskpipe run a b.nii.gz atlas={root}/atlas/lhdl\n"
    assert "first start" in first.stderr and "unpacked" in first.stderr
    again = run(root / "mskpipe", "--version")
    assert again.stderr == "" and again.stdout.startswith("python -m mskpipe --version")
    link = tmp_path / "mskpipe-link"
    link.symlink_to(root / "mskpipe")
    assert run(link, "plugins").stderr == ""
    moved = home / "moved"
    root.rename(moved)
    assert "first start" in run(moved / "mskpipe", "--version").stderr
    assert (moved / "env" / ".mskpipe-unpacked").read_text() == f"0.1.0*{moved}\n"


# ---------------------------------------------------------------- licences


def test_collect_licenses(tmp_path):
    prefix = tmp_path / "env"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "info").mkdir()
    (prefix / "info" / "LICENSE.txt").write_text("PSF", encoding="utf-8")
    (prefix / "conda-meta" / "python-3.11.0-h.json").write_text(
        json.dumps({"name": "python", "version": "3.11.0", "license": "PSF-2.0",
                    "files": ["info/LICENSE.txt", "python.exe"]}),
        encoding="utf-8",
    )  # fmt: skip
    dist = prefix / "Lib" / "site-packages" / "vtk-9.4.2.dist-info"
    (dist / "licenses").mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: vtk\nVersion: 9.4.2\nLicense-Expression: BSD-3-Clause\n",
        encoding="utf-8",
    )
    (dist / "licenses" / "Copyright.txt").write_text("Kitware", encoding="utf-8")
    other = prefix / "Lib" / "site-packages" / "foo-1.0.dist-info"
    other.mkdir()
    (other / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n"
        "Classifier: License :: OSI Approved :: MIT License\n",
        encoding="utf-8",
    )
    rows = licenses.collect(prefix, tmp_path / "out")
    by_name = {r["name"]: r for r in rows}
    assert by_name["python"]["license"] == "PSF-2.0"
    assert by_name["vtk"]["license"] == "BSD-3-Clause"
    assert by_name["foo"]["license"] == "MIT License"
    assert (tmp_path / "out" / "vtk-9.4.2" / "Copyright.txt").is_file()
    assert (tmp_path / "out" / "python-3.11.0" / "LICENSE.txt").is_file()
    csv_text = (tmp_path / "out" / "packages.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "name,version,source,license,url,files"


# ---------------------------------------------------------------- workflow + metadata


def test_release_workflow():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))
    triggers = wf[True] if True in wf else wf["on"]  # YAML 1.1 reads "on" as True
    assert triggers["push"]["tags"] == ["v*.*.*"] and "workflow_dispatch" in triggers
    assert set(wf["jobs"]) == {"build", "smoke", "build-linux", "smoke-linux", "publish"}
    assert wf["jobs"]["publish"]["if"] == "github.ref_type == 'tag'"
    # nothing is published before both packages passed their tests
    assert set(wf["jobs"]["publish"]["needs"]) == set(wf["jobs"]) - {"publish"}
    assert wf["jobs"]["publish"]["permissions"] == {"contents": "write"}
    # the ~1 GB package goes from job to job through a draft release, not an Actions
    # artifact (the artifact storage quota of a private repository is smaller)
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "upload-artifact" not in text and "download-artifact" not in text
    build, smoke = wf["jobs"]["build"], wf["jobs"]["smoke"]
    assert build["permissions"] == smoke["permissions"] == {"contents": "write"}
    draft = next(s for s in build["steps"] if s.get("id") == "draft")
    assert "--draft" in draft["run"] and "--verify-tag" in draft["run"]
    assert any("gh release download" in s.get("run", "") for s in smoke["steps"])
    assert "--draft=false" in wf["jobs"]["publish"]["steps"][-1]["run"]
    assert wf["permissions"] == {"contents": "read"}
    assert not (ROOT / "ci.yml").exists()  # the workflow lives in .github/workflows
    # PowerShell: "$x = if (...) { @(...) }" yields a string for one item; splatting a
    # string passes its characters (broke the atlas step of the first test build)
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert not re.search(r"=\s*if\s*\(.*\)\s*\{\s*@\(", text)
    # Linux: built on a fixed Ubuntu, added to the draft of the Windows build, tested in
    # the Docker images of both LTS releases (nothing installed in them)
    linux, clean = wf["jobs"]["build-linux"], wf["jobs"]["smoke-linux"]
    assert linux["needs"] == "build" and linux["runs-on"] == "ubuntu-24.04"
    assert linux["permissions"] == clean["permissions"] == {"contents": "write"}
    runs = "\n".join(s.get("run", "") for s in linux["steps"])
    assert "--platform linux-64" in runs and "--runtime-from build/test/env" in runs
    assert "gh release upload" in runs and "--clobber" in runs and "sums --file" in runs
    assert clean["strategy"]["matrix"]["ubuntu"] == ["24.04", "26.04"]
    test = clean["steps"][-1]["run"]
    assert "docker run" in test and "packaging/linux/smoke_test.sh" in test
    assert (PACKAGING / "linux" / "smoke_test.sh").is_file()


def test_zenodo_and_citation_metadata():
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert zenodo["license"] == citation["license"] == "Apache-2.0"
    assert len(zenodo["creators"]) == len(citation["authors"])


def test_version_ignores_non_release_tags():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"--match", "v[0-9]*"' in text and "tag_regex" in text


# ---------------------------------------------------------------- build info


def test_build_info_fallback(tmp_path, monkeypatch):
    info = {"commit": "a" * 40, "packages": {"pystaple": "b" * 40}}
    monkeypatch.setattr(build_info, "load", lambda: info)
    assert build_info.commit() == "a" * 40
    assert build_info.package_commit("PyStaple") == "b" * 40
    assert build_info.package_commit("other") is None
    # an installed release has no git checkout: the commit comes from the build
    assert git_info(tmp_path).commit == "a" * 40
    if importlib.util.find_spec("pystaple") is None:
        pytest.skip("pystaple not installed")
    assert package_info("pystaple").commit is not None


def test_build_info_absent_in_source_tree():
    build_info.load.cache_clear()
    assert build_info.load() == {}
