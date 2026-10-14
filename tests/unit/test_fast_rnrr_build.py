# SPDX-License-Identifier: Apache-2.0
"""Fast-RNRR built on the user's computer (mskpipe.resources.fast_rnrr): pinned spec,
consent, download/unpack/install with local fake files, lookup order, setup status, CLI.
Nothing is downloaded and no compiler runs (compile and check are replaced)."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mskpipe.cli import app
from mskpipe.config import load_config
from mskpipe.geometry.fast_rnrr import FastRnrrError, find_executable
from mskpipe.resources import fast_rnrr as fr
from mskpipe.resources.status import collect

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_external(monkeypatch):
    monkeypatch.delenv(fr.FAST_RNRR_ENV, raising=False)
    monkeypatch.setenv("PATH", "")


# ---------------------------------------------------------------- spec


def _pins_tool():
    spec = importlib.util.spec_from_file_location(
        "fast_rnrr_pins", ROOT / "tools" / "fast_rnrr_pins.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spec_matches_pins_tool():
    spec, tool = fr.load_spec(), _pins_tool()
    assert spec.fast_rnrr.commit == tool.FAST_RNRR_COMMIT
    assert [f.path for f in spec.fast_rnrr.files] == tool.FAST_RNRR_FILES
    assert spec.notice == tool.NOTICE and "research purposes" in spec.notice
    assert {c.version for c in spec.compilers.values()} == {tool.ZIG_VERSION}
    assert set(spec.compilers) == {"win_amd64", "manylinux_x86_64"}
    assert spec.compilers["win_amd64"].target == "x86_64-windows-gnu"
    assert spec.eigen.sha256 == tool.EIGEN["sha256"]
    for item in [*spec.fast_rnrr.files, spec.openmesh, spec.eigen, *spec.compilers.values()]:
        assert len(item.sha256) == 64


def test_spec_check_files_are_pinned():
    spec = fr.load_spec()
    pinned = {f.path for f in spec.fast_rnrr.files}
    assert set(spec.fast_rnrr.check.values()) <= pinned
    assert {"source", "target", "expected"} == set(spec.fast_rnrr.check)
    assert not any(p.startswith("data/") and p not in spec.fast_rnrr.check.values() for p in pinned)
    assert "-DOM_STATIC_BUILD" in spec.flags and "-std=c++14" in spec.flags


def test_omp_stub_defines_wtime():
    assert "omp_get_wtime" in fr.OMP_STUB and "#pragma once" in fr.OMP_STUB


def test_compare_result():
    want = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0]])
    ok = fr.compare_result(want + 1e-5, want, 1e-4)
    assert ok["ok"] and ok["vertices"] == 3 and ok["max_rel"] < 1e-6
    assert not fr.compare_result(want + 0.1, want, 1e-4)["ok"]
    assert not fr.compare_result(want[:2], want, 1e-4)["ok"]


# ---------------------------------------------------------------- fake build


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def fake_spec(tmp_path, monkeypatch):
    """Spec whose URLs are local files; the compiler step writes a fake executable."""
    src = tmp_path / "upstream"
    files = {
        "src/main.cpp": b"int main(){}\n",
        "include/nanoflann.hpp": b"// header\n",
        "data/test/source.obj": b"v 0 0 0\n",
        "data/test/target_mesh.obj": b"v 0 0 0\n",
        "data/test/mesh_res.obj": b"v 0 0 0\n",
    }
    for name, data in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    om = _tar(tmp_path / "om.tar.gz", {"om-1/src/OpenMesh/Core/a.cc": b"// a\n"})
    eigen = _tar(tmp_path / "eigen.tar.gz", {"eigen-x/Eigen/Dense": b"// e\n"})
    zig_name = "zig.exe" if sys.platform == "win32" else "zig"
    wheel = tmp_path / "ziglang-0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(f"ziglang/{zig_name}", b"fake zig")
        zf.writestr("ziglang/lib/x.h", b"")
    base = fr.load_spec()
    compiler = {
        "name": "zig",
        "version": "0.0",
        "url": wheel.as_uri(),
        "sha256": _sha(wheel.read_bytes()),
        "target": "native",
    }
    spec = base.model_copy(
        update={
            "fast_rnrr": base.fast_rnrr.model_copy(
                update={
                    "raw_url": src.as_uri() + "/{path}",
                    "files": [fr.PinnedFile(path=k, sha256=_sha(v)) for k, v in files.items()],
                    "sources": ["src/*.cpp"],
                }
            ),
            "openmesh": base.openmesh.model_copy(
                update={"url": om.as_uri(), "sha256": _sha(om.read_bytes()), "root": "om-1/src"}
            ),
            "eigen": base.eigen.model_copy(
                update={
                    "url": eigen.as_uri(),
                    "sha256": _sha(eigen.read_bytes()),
                    "root": "eigen-x",
                }
            ),
            "compilers": {
                tag: fr.Compiler(**compiler) for tag in ("win_amd64", "manylinux_x86_64")
            },
        }
    )
    calls = {}

    def fake_compile(zig, target, flags, includes, units, obj_dir, out, log, stop, jobs, env):
        calls["compile"] = {"zig": zig, "units": [u.name for u in units], "includes": includes}
        assert (includes[0] / "omp.h").is_file()
        out.write_bytes(b"fake exe")
        return out

    def fake_check(exe, src_dir, spec_, work, log):
        calls["check"] = exe
        return {"ok": calls.get("check_ok", True), "max_rel": 0.0, "max_abs": 0.0, "vertices": 1}

    monkeypatch.setattr(fr, "_compile_and_link", fake_compile)
    monkeypatch.setattr(fr, "run_check", fake_check)
    monkeypatch.setattr(fr, "platform_tag", lambda: "manylinux_x86_64")
    monkeypatch.setattr(fr, "load_spec", lambda: spec)
    return spec, calls


def test_build_needs_consent(fake_spec):
    spec, calls = fake_spec
    with pytest.raises(fr.FastRnrrSetupError, match="research purposes") as err:
        fr.build_fast_rnrr(print, accepted=False, spec=spec)
    assert fr.ACCEPT_FLAG in str(err.value) and not calls


def test_build_unsupported_platform(fake_spec, monkeypatch):
    spec, _ = fake_spec
    monkeypatch.setattr(fr, "platform_tag", lambda: None)
    with pytest.raises(fr.FastRnrrSetupError, match="not here"):
        fr.build_fast_rnrr(print, accepted=True, spec=spec)


def test_build_installs_and_cleans_up(fake_spec):
    spec, calls = fake_spec
    lines: list[str] = []
    exe = fr.build_fast_rnrr(lines.append, accepted=True, spec=spec)
    assert exe == fr.build_dir(spec) / fr.exe_name() and exe.read_bytes() == b"fake exe"
    assert calls["compile"]["units"] == ["a.cc", "main.cpp"]  # OpenMesh, then Fast-RNRR
    marker = fr.read_marker(exe.parent)
    assert marker["commit"] == spec.fast_rnrr.commit and marker["sha256"] == _sha(b"fake exe")
    assert marker["accepted"]["flag"] == fr.ACCEPT_FLAG
    assert "research purposes" in (exe.parent / fr.NOTICE_FILE).read_text(encoding="utf-8")
    assert spec.notice in lines
    root = fr.build_root()
    assert sorted(p.name for p in root.iterdir()) == [spec.short]  # work, downloads gone
    assert fr.built_executable(spec) == exe
    assert fr.describe_executable(exe)["commit"] == spec.fast_rnrr.commit
    status = fr.fast_rnrr_status()
    assert status.built and status.path == str(exe) and status.supported
    # already built: nothing happens without --force
    calls.clear()
    assert fr.build_fast_rnrr(print, accepted=True, spec=spec) == exe and not calls
    assert fr.remove_fast_rnrr(print, spec) and fr.built_executable(spec) is None
    assert not fr.remove_fast_rnrr(print, spec)


def test_failed_check_installs_nothing(fake_spec):
    spec, calls = fake_spec
    calls["check_ok"] = False
    with pytest.raises(fr.FastRnrrSetupError, match="does not reproduce"):
        fr.build_fast_rnrr(print, accepted=True, spec=spec)
    assert fr.built_executable(spec) is None
    assert not fr.build_dir(spec).exists()


def test_checksum_mismatch_stops(fake_spec):
    spec, _ = fake_spec
    bad = spec.model_copy(update={"eigen": spec.eigen.model_copy(update={"sha256": "0" * 64})})
    with pytest.raises(Exception, match="Checksum mismatch"):
        fr.build_fast_rnrr(print, accepted=True, spec=bad)
    assert fr.built_executable(spec) is None


def test_cancel_before_compiling(fake_spec):
    from mskpipe.resources.tool import ToolCancelled

    spec, calls = fake_spec
    with pytest.raises(ToolCancelled):
        fr.build_fast_rnrr(print, accepted=True, spec=spec, cancelled=lambda: True)
    assert "compile" not in calls and fr.built_executable(spec) is None


def test_unpack_rejects_unsafe_paths(tmp_path):
    archive = _tar(tmp_path / "bad.tar.gz", {"root/../../evil.txt": b"x"})
    with pytest.raises(fr.FastRnrrSetupError, match="Unsafe"):
        fr._unpack_tar(archive, tmp_path / "out", "root")


# ---------------------------------------------------------------- lookup


def test_locate_order(fake_spec, tmp_path, monkeypatch):
    spec, _ = fake_spec
    assert fr.locate(None) == (None, "none")
    with pytest.raises(FastRnrrError, match="mskpipe setup fast-rnrr"):
        find_executable(None)
    built = fr.build_fast_rnrr(print, accepted=True, spec=spec)
    assert fr.locate(None) == (built, "build") and find_executable(None) == built
    own = tmp_path / "own.exe"
    own.write_bytes(b"x")
    monkeypatch.setenv(fr.FAST_RNRR_ENV, str(own))
    assert fr.locate(None) == (own, "env")
    assert fr.locate(str(built)) == (built, "config")
    monkeypatch.setenv(fr.FAST_RNRR_ENV, str(tmp_path / "missing.exe"))
    assert fr.locate(None) == (None, "env")  # a wrong setting is not skipped silently
    with pytest.raises(FastRnrrError, match="not found"):
        find_executable(None)


def test_setup_status(fake_spec):
    spec, _ = fake_spec
    config = load_config(None, ["attachments.params.nonrigid=fast_rnrr"])
    status = collect(config, ["ct"])
    assert status.fast_rnrr.needed and not status.fast_rnrr.ok and not status.ready
    assert not collect(load_config(None, []), ["ct"]).fast_rnrr.needed
    fr.build_fast_rnrr(print, accepted=True, spec=spec)
    status = collect(config, ["ct"])
    assert status.fast_rnrr.ok and status.fast_rnrr.source == "build"


# ---------------------------------------------------------------- CLI


def test_cli_requires_consent(fake_spec):
    result = runner.invoke(app, ["setup", "fast-rnrr"])
    assert result.exit_code == 2
    assert "research purposes" in result.output and "--accept-research-licence" in result.output
    assert fr.built_executable() is None


def test_cli_build_and_remove(fake_spec):
    result = runner.invoke(app, ["setup", "fast-rnrr", "--accept-research-licence", "-j", "2"])
    assert result.exit_code == 0, result.output
    assert "nonrigid=fast_rnrr" in result.output and fr.built_executable() is not None
    shown = runner.invoke(
        app, ["setup", "status", "--set", "attachments.params.nonrigid=fast_rnrr"]
    )
    assert "Fast-RNRR:        ok" in shown.output
    assert runner.invoke(app, ["setup", "fast-rnrr", "--remove"]).exit_code == 0
    assert fr.built_executable() is None


def test_cli_status_reports_missing(fake_spec):
    shown = runner.invoke(
        app, ["setup", "status", "--set", "attachments.params.nonrigid=fast_rnrr", "--json"]
    )
    data = json.loads(shown.stdout)
    assert data["fast_rnrr"]["needed"] and data["fast_rnrr"]["path"] is None
