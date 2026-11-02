# SPDX-License-Identifier: Apache-2.0
"""Muscle Wrapping 2.x demo asset (packaging/make_mw2_demo_zip.py, packaging/mw2_demo,
workflow mw2-demo.yml) on synthetic PE files and a fake MW2 tree (no Windows, no network)."""

import importlib.util
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from mskpipe.steps import export_mw2

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging"
SHA = re.compile(r"^[0-9a-f]{64}$")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


br = _load("build_release")
demo = _load("make_mw2_demo_zip")


def make_pe(path: Path, imports=(), delay=(), extra: bytes = b"") -> Path:
    """Minimal PE32+ file with an import and a delay-import table in one section."""
    va, raw = 0x1000, 0x200
    n_imp, n_del = len(imports), len(delay)
    names_at = 20 * (n_imp + 1) + 32 * (n_del + 1)
    blob, name_rvas = b"", []
    for name in (*imports, *delay):
        name_rvas.append(va + names_at + len(blob))
        blob += name.encode("ascii") + b"\0"
    section = b""
    for rva in name_rvas[:n_imp]:
        section += struct.pack("<IIIII", 0, 0, 0, rva, 0)
    section += b"\0" * 20
    for rva in name_rvas[n_imp:]:
        section += struct.pack("<IIIIIIII", 1, rva, 0, 0, 0, 0, 0, 0)
    section += b"\0" * 32 + blob + extra
    dirs = [(0, 0)] * 16
    dirs[1] = (va, 20 * (n_imp + 1)) if n_imp else (0, 0)
    dirs[13] = (va + 20 * (n_imp + 1), 32 * (n_del + 1)) if n_del else (0, 0)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, 108, 16)
    for i, (rva, size) in enumerate(dirs):
        struct.pack_into("<II", opt, 112 + 8 * i, rva, size)
    header = bytearray(raw)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 0x40)
    header[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", header, 0x44, 0x8664, 1, 0, 0, 0, len(opt), 0x2022)
    header[0x58 : 0x58 + len(opt)] = opt
    sec = struct.pack("<8sIIIIIIHHI", b".idata", len(section), va, len(section), raw, 0, 0, 0, 0, 0)
    header[0x58 + len(opt) : 0x58 + len(opt) + 40] = sec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + section)
    return path


# ---------------------------------------------------------------- PE imports / closure


def test_pe_imports_reads_import_and_delay_tables(tmp_path):
    pe = make_pe(tmp_path / "a.dll", ["KERNEL32.dll", "vtkCommonCore-9.4.dll"], ["d3d11.dll"])
    assert demo.pe_imports(pe) == ["KERNEL32.dll", "vtkCommonCore-9.4.dll", "d3d11.dll"]
    assert demo.pe_imports(make_pe(tmp_path / "b.exe")) == []
    (tmp_path / "c.dll").write_bytes(b"not a PE")
    with pytest.raises(br.BuildError, match="not a PE"):
        demo.pe_imports(tmp_path / "c.dll")


def test_closure_skips_system_and_reports_missing_and_debug():
    graph = {
        "root.exe": ["Plugin.dll", "KERNEL32.dll", "api-ms-win-crt-heap-l1-1-0.dll"],
        "plugin.dll": ["libomp140.x86_64.dll", "opengl32.dll"],
        "libomp140.x86_64.dll": ["VCRUNTIME140.dll"],
        "vcruntime140.dll": [],
    }
    paths = {n: Path(n) for n in graph}
    found = demo.closure(["root.exe"], paths.get, lambda p: graph[p.name])
    assert set(found) == set(graph)
    del paths["libomp140.x86_64.dll"]
    with pytest.raises(br.BuildError, match=r"libomp140\.x86_64\.dll \(needed by plugin\.dll\)"):
        demo.closure(["root.exe"], paths.get, lambda p: graph[p.name])
    graph["plugin.dll"] = ["MSVCP140D.dll"]
    with pytest.raises(br.BuildError, match="Debug runtime"):
        demo.closure(["root.exe"], {n: Path(n) for n in graph}.get, lambda p: graph[p.name])


def test_system_dlls():
    for name in ("KERNEL32.dll", "opengl32.dll", "api-ms-win-crt-math-l1-1-0.dll", "dbghelp.dll"):
        assert demo.is_system(name), name
    for name in (
        "vcruntime140.dll",
        "libomp140.x86_64.dll",
        "vtkCommonCore-9.4.dll",
        "msvcp140.dll",
    ):
        assert not demo.is_system(name), name


# ---------------------------------------------------------------- fake MW2 build


def fake_mw2(root: Path) -> Path:
    b = root / "Build" / "Bin"
    make_pe(b / "opensim-cmd.exe", ["osimCommon.dll", "VCRUNTIME140.dll", "KERNEL32.dll"])
    make_pe(b / "OsimMuscleGeneratorTool.dll", ["osimCommon.dll", "libomp140.x86_64.dll"])
    make_pe(b / "osimCommon.dll", ["vtkzlib-9.4.dll", "MSVCP140.dll"])
    make_pe(b / "vtkzlib-9.4.dll")
    make_pe(b / "vcruntime140.dll")  # replaced by the conda runtime
    make_pe(b / "unused.dll")
    make_pe(b / "osimCommon_d.dll", ["VCRUNTIME140D.dll"])
    (b / "osimCommon.pdb").write_bytes(b"pdb")
    d = root / demo.DWNLD
    for rel in (
        "../../../LICENSE",
        "OpenSim/LICENSE.txt",
        "OpenSim/NOTICE",
        "OpenSim/dependencies/simbody/LICENSE.txt",
        "OpenSim/dependencies/spdlog/LICENSE",
        "OpenSim/Vendors/docopt/LICENSE-MIT",
        "VTK/Copyright.txt",
        "VTK/ThirdParty/zlib/vtkzlib/LICENSE",
        "VTK/ThirdParty/zlib/Testing/LICENSE",
        "VTK/Utilities/GitSetup/LICENSE",
        "Discregrid/LICENSE",
        "vtkVisualDebugger/LICENSE",
        "Eigen3/COPYING.MPL2",
    ):
        path = (d / rel).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"licence {rel}\n", encoding="utf-8")
    (d / "OpenSim" / "CMakeLists.txt").write_text(
        "set(OPENSIM_MAJOR_VERSION 4)\nset(OPENSIM_MINOR_VERSION 5)\nset(OPENSIM_PATCH_VERSION 2)\n"
    )
    return root


def fake_example(folder: Path) -> Path:
    (folder / "Geometry").mkdir(parents=True, exist_ok=True)
    (folder / "muscles" / "iliacus_r").mkdir(parents=True, exist_ok=True)
    (folder / "analysis").mkdir(exist_ok=True)
    (folder / export_mw2.SETUP_FILE).write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<OpenSimDocument Version="20302">
  <MuscleGeneratorTool name="t">
    <model_file>bone_model.osim</model_file>
    <motion_file>hip_flexion_r.mot</motion_file>
    <output_model_file>bone_model_mw2.osim</output_model_file>
    <output_muscle_analysis_file_prefix>analysis/</output_muscle_analysis_file_prefix>
    <MuscleGeneratorSet><objects>
      <MuscleGenerator name="iliacus_r">
        <MuscleGeometry><Mesh name="iliacus_r">
          <mesh_file>muscles/iliacus_r/Iliacus.obj</mesh_file></Mesh>
          <attachment_areas><AttachmentArea name="o">
            <point_file>muscles/iliacus_r/Iliacus_Ori.vtk</point_file></AttachmentArea>
          </attachment_areas></MuscleGeometry>
        <num_of_lines>3</num_of_lines>
      </MuscleGenerator>
    </objects></MuscleGeneratorSet>
  </MuscleGeneratorTool>
</OpenSimDocument>
""",
        encoding="utf-8",
    )
    (folder / "bone_model.osim").write_text(
        "<OpenSimDocument><Model><Mesh><mesh_file>pelvis.obj</mesh_file></Mesh>"
        "<Mesh><mesh_file>Geometry/femur_r.obj</mesh_file></Mesh></Model></OpenSimDocument>",
        encoding="utf-8",
    )
    for rel in (
        "hip_flexion_r.mot",
        "Geometry/pelvis.obj",
        "Geometry/femur_r.obj",
        "muscles/iliacus_r/Iliacus.obj",
        "muscles/iliacus_r/Iliacus_Ori.vtk",
        "export.json",
        "bone_model_mw2.osim",  # an earlier result, not shipped
        "analysis/iliacus_r_Lengths.sto",
        "defects.json",
    ):
        (folder / rel).write_text("{}\n" if rel.endswith(".json") else "x\n", encoding="utf-8")
    return folder


def _stage(tmp_path, mw2=None):
    mw2 = mw2 or fake_mw2(tmp_path / "mw2")
    runtimes = {
        "vcruntime140.dll": make_pe(tmp_path / "rt" / "vcruntime140.dll").read_bytes(),
        "msvcp140.dll": make_pe(tmp_path / "rt" / "msvcp140.dll").read_bytes(),
        "vcamp140.dll": b"not imported, not shipped",
        "libomp140.x86_64.dll": make_pe(
            tmp_path / "rt" / "libomp.dll", ["KERNEL32.dll"]
        ).read_bytes(),
    }
    packages = {
        "vc14_runtime": {"licence": b"MICROSOFT", "record": {"url": "u", "sha256": "s"}},
        "llvm-openmp": {"licence": b"LLVM", "record": {"url": "u", "sha256": "s"}},
    }
    return demo.stage(
        tmp_path / "stage" / "msk-pipe-mw2-demo-v1",
        mw2=mw2,
        example=fake_example(tmp_path / "run" / "06_mw2_input"),
        runtimes=runtimes,
        packages=packages,
        pins=br.load_pins(),
        mw2_source="test",
    )


def test_stage_check_and_zip(tmp_path):
    folder = _stage(tmp_path)
    bin_files = sorted(p.name for p in (folder / "bin").iterdir())
    assert bin_files == [
        "OsimMuscleGeneratorTool.dll",
        "libomp140.x86_64.dll",
        "msvcp140.dll",
        "opensim-cmd.exe",
        "osimCommon.dll",
        "vcruntime140.dll",
        "vtkzlib-9.4.dll",
    ]
    # the conda runtime replaced the build's copy
    assert (folder / "bin" / "vcruntime140.dll").read_bytes() == (
        tmp_path / "rt" / "vcruntime140.dll"
    ).read_bytes()
    assert demo.check(folder) == []
    lic = folder / "licenses"
    assert (lic / "VTK" / "ThirdParty" / "zlib" / "vtkzlib" / "LICENSE").is_file()
    assert not (lic / "VTK" / "ThirdParty" / "zlib" / "Testing").exists()
    assert not (lic / "VTK" / "Utilities" / "GitSetup").exists()
    assert (lic / "llvm-openmp" / "LICENSE.txt").read_bytes() == b"LLVM"
    assert (lic / "gcc-runtime" / "COPYING.RUNTIME").is_file()
    assert (folder / "LICENSE").read_text() == "licence ../../../LICENSE\n"
    ex = sorted(
        p.relative_to(folder / "example").as_posix()
        for p in (folder / "example").rglob("*")
        if p.is_file()
    )
    assert "bone_model_mw2.osim" not in ex and "defects.json" not in ex
    assert not any(e.startswith("analysis/") for e in ex)
    assert {
        "LICENSE",
        "README.txt",
        "Geometry/pelvis.obj",
        "Geometry/femur_r.obj",
        "export.json",
    } <= set(ex)
    assert b"CC BY-NC-SA 2.0 BE" in (folder / "example" / "LICENSE").read_bytes()
    cmd = (folder / "run_demo.cmd").read_bytes()
    assert b"\r\n" in cmd and b"\n" not in cmd.replace(b"\r\n", b"")
    record = json.loads((folder / "demo.json").read_text())
    assert record["muscle_wrapping"]["opensim"] == "4.5.2"
    assert SHA.match(record["files"]["bin/OsimMuscleGeneratorTool.dll"])
    first = demo.make_zip(folder, tmp_path / "a.zip")
    assert first == demo.make_zip(folder, tmp_path / "b.zip")
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert "msk-pipe-mw2-demo-v1/run_demo.cmd" in names
    assert "msk-pipe-mw2-demo-v1/bin/opensim-cmd.exe" in names


def test_check_finds_problems(tmp_path):
    folder = _stage(tmp_path)
    make_pe(folder / "bin" / "stray.dll")
    (folder / "bin" / "vtkzlib-9.4.dll").unlink()
    (folder / "example" / "LICENSE").unlink()
    problems = demo.check(folder)
    assert any("vtkzlib-9.4.dll" in p for p in problems)
    make_pe(folder / "bin" / "vtkzlib-9.4.dll")
    problems = demo.check(folder)
    assert any("stray.dll" in p for p in problems)
    assert any("example/LICENSE" in p for p in problems)


def test_gpl_code_and_missing_licence_stop_the_build(tmp_path):
    mw2 = fake_mw2(tmp_path / "mw2")
    make_pe(mw2 / "Build" / "Bin" / "vtkzlib-9.4.dll", extra=b"ALGLIB: allocation error\0")
    with pytest.raises(br.BuildError, match="ALGLIB"):
        _stage(tmp_path, mw2)
    make_pe(mw2 / "Build" / "Bin" / "vtkzlib-9.4.dll")
    (mw2 / demo.DWNLD / "Discregrid" / "LICENSE").unlink()
    with pytest.raises(br.BuildError, match="Licence of Discregrid"):
        _stage(tmp_path, mw2)


def test_example_files_checks_references_and_paths(tmp_path):
    folder = fake_example(tmp_path / "06_mw2_input")
    (folder / "muscles" / "iliacus_r" / "Iliacus_Ori.vtk").unlink()
    with pytest.raises(br.BuildError, match=r"Iliacus_Ori\.vtk"):
        demo.example_files(folder)
    folder = fake_example(tmp_path / "other")
    (folder / "export.json").write_text('{"image": "C:\\\\data\\\\ct.nii.gz"}', encoding="utf-8")
    with pytest.raises(br.BuildError, match="local path"):
        demo.example_files(folder)


def test_verify_counts_fibres(tmp_path):
    work = fake_example(tmp_path / "w")
    fibres = "".join(f'<Thelen2003Muscle name="iliacus_r_{i}"/>' for i in range(3))
    (work / "bone_model_mw2.osim").write_text(
        f"<OpenSimDocument><Model><ForceSet>{fibres}</ForceSet></Model></OpenSimDocument>"
    )
    assert demo.verify(work) == []
    (work / "bone_model_mw2.osim").write_text(
        '<OpenSimDocument><Thelen2003Muscle name="iliacus_r_0"/></OpenSimDocument>'
    )
    assert demo.verify(work) == ["iliacus_r: 1 fibres in bone_model_mw2.osim, expected 3"]
    (work / "analysis" / "iliacus_r_Lengths.sto").unlink()
    (work / "bone_model_mw2.osim").unlink()
    assert len(demo.verify(work)) == 2


def test_conda_members(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    import io
    import tarfile

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        for name, payload in (
            ("Library/bin/libomp.dll", b"dll"),
            ("info/licenses/LICENSE.TXT", b"lic"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    with zipfile.ZipFile(tmp_path / "p.conda", "w") as zf:
        zf.writestr("pkg-p.tar.zst", zstandard.ZstdCompressor().compress(data.getvalue()))
    files = demo.conda_members(tmp_path / "p.conda", lambda n: n.endswith(".dll"))
    assert files == {"Library/bin/libomp.dll": b"dll"}


# ---------------------------------------------------------------- pins, workflow, docs


def test_pins():
    pins = br.load_pins()["mw2_demo"]
    assert pins["file"] == f"msk-pipe-{pins['tag']}-win64.zip"
    assert pins["sha256"] == "" or SHA.match(pins["sha256"])
    conda = {p["name"]: p for p in pins["conda"]}
    assert set(conda) == {"vc14_runtime", "llvm-openmp"}
    for p in conda.values():
        assert p["url"].startswith("https://conda.anaconda.org/conda-forge/win-64/")
        assert p["url"].endswith(".conda") and SHA.match(p["sha256"])
        assert p["licence_file"].startswith("info/licenses/")
    assert conda["llvm-openmp"]["rename"] == {"libomp.dll": "libomp140.x86_64.dll"}


def test_fetch_without_pin(monkeypatch, tmp_path, capsys):
    pins = br.load_pins()
    pins["mw2_demo"]["sha256"] = ""
    monkeypatch.setattr(demo, "load_pins", lambda: pins)
    assert demo.main(["fetch", "--out", str(tmp_path), "--allow-missing"]) == 0
    assert "not pinned" in capsys.readouterr().out
    assert demo.main(["fetch", "--out", str(tmp_path)]) == 1


def test_run_demo_matches_export():
    text = (PACKAGING / "mw2_demo" / "run_demo.cmd").read_text(encoding="utf-8")
    assert export_mw2.SETUP_FILE in text and export_mw2.SETUP_FILE == demo.SETUP
    assert export_mw2.MW2_COMMAND.split()[:3] == [
        "opensim-cmd",
        "-L",
        "OsimMuscleGeneratorTool.dll",
    ]
    for needle in (
        "bin\\opensim-cmd.exe",
        "bin\\OsimMuscleGeneratorTool.dll",
        "run-tool",
        "mw2.log",
    ):
        assert needle in text, needle
    assert "MW2_DEMO_NO_PAUSE" in text
    # licence texts that come neither from the MW2 sources nor from the conda packages
    for name in ("gcc-runtime", "lapack"):
        assert name in demo.REQUIRED_LICENCES
        assert any((PACKAGING / "mw2_demo" / "licenses" / name).iterdir()), name


def test_demo_workflow():
    path = ROOT / ".github" / "workflows" / "mw2-demo.yml"
    wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = wf[True] if True in wf else wf["on"]
    assert "workflow_dispatch" in triggers
    assert "packaging/release.toml" in triggers["push"]["paths"]
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["demo"]
    assert job["runs-on"] == "windows-latest"
    runs = "\n".join(s.get("run", "") for s in job["steps"])
    for sub in ("fetch", "check", "verify"):
        assert f"make_mw2_demo_zip.py {sub}" in runs, sub
    assert "run_demo.cmd" in runs and job["env"]["MW2_DEMO_NO_PAUSE"] == "1"


def test_readme_points_to_demo():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Muscle Wrapping demo" in text
    assert br.load_pins()["mw2_demo"]["file"] in text and "run_demo.cmd" in text
