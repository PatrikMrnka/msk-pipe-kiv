# SPDX-License-Identifier: Apache-2.0
"""GPU support on demand (mskpipe.resources.gpu): wheel list from pixi.lock, overlay
install from local fake wheels, device selection and manifest (no GPU, no network)."""

import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mskpipe.cli import app
from mskpipe.config import InputSpec, load_config
from mskpipe.core import device as dev
from mskpipe.core.device import DeviceReport, TorchInfo, probe_torch_cuda, resolve_device
from mskpipe.core.runner import run_pipeline
from mskpipe.core.step import Step
from mskpipe.resources import download, gpu

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


# ------------------------------------------------------------------ wheel list


def test_spec_matches_pixi_lock():
    """gpu_wheels.json = torch/torchvision of the locked gpu environment, and that
    environment differs from cpu only in them (otherwise the overlay is not the gpu env)."""
    lock = yaml.safe_load((ROOT / "pixi.lock").read_text(encoding="utf-8"))
    assert gpu.load_spec() == gpu.spec_from_lock(lock), "run tools/gpu_wheels.py"
    diff = gpu.env_differences(lock)
    assert set(diff) == set(gpu.OVERLAY_PACKAGES), diff
    for cpu_version, gpu_version in diff.values():
        assert cpu_version.split("+")[0] == gpu_version.split("+")[0]


def _lock(gpu_torch: str = "2.7.1+cu128", extra_gpu: list | None = None) -> dict:
    def wheel(name, version):
        tag = version.replace("+", "%2B")
        return f"https://example.org/{name}-{tag}-cp311-cp311-win_amd64.whl"

    pkgs = [
        {
            "pypi": wheel("torch", "2.7.1+cpu"),
            "name": "torch",
            "version": "2.7.1+cpu",
            "sha256": "a",
        },
        {"pypi": wheel("torch", gpu_torch), "name": "torch", "version": gpu_torch, "sha256": "b"},
        {
            "pypi": wheel("torchvision", "0.22.1+cu128"),
            "name": "torchvision",
            "version": "0.22.1+cu128",
            "sha256": "c",
        },
        {"pypi": "git+https://github.com/x/y?rev=1#1", "name": "y", "sha256": "d"},
    ]
    common = [
        {
            "conda": "https://conda.anaconda.org/conda-forge/noarch/colorama-0.4.6-pyhd8ed1ab_1.conda"
        },
        {"pypi": "git+https://github.com/x/y?rev=1#1"},
    ]
    gpu_env = [
        *common,
        {"pypi": wheel("torch", gpu_torch)},
        {"pypi": wheel("torchvision", "0.22.1+cu128")},
        *(extra_gpu or []),
    ]
    cpu_env = [*common, {"pypi": wheel("torch", "2.7.1+cpu")}]
    return {
        "environments": {
            "cpu": {"packages": {"win-64": cpu_env}},
            "gpu": {"packages": {"win-64": gpu_env}},
        },
        "packages": pkgs,
    }


def test_spec_from_synthetic_lock():
    spec = gpu.spec_from_lock(_lock())
    assert spec.tag == "torch-2.7.1+cu128"
    assert (spec.python, spec.platform) == ("cp311", "win_amd64")
    assert [w.name for w in spec.wheels] == ["torch", "torchvision"]
    assert spec.wheels[0].filename == "torch-2.7.1+cu128-cp311-cp311-win_amd64.whl"
    diff = gpu.env_differences(_lock())
    assert diff == {"torch": ("2.7.1+cpu", "2.7.1+cu128"), "torchvision": (None, "0.22.1+cu128")}
    extra = [{"conda": "https://conda.anaconda.org/conda-forge/noarch/colorama-0.4.7-py_0.conda"}]
    assert "colorama" in gpu.env_differences(_lock(extra_gpu=extra))
    bad = _lock()
    bad["environments"]["gpu"]["packages"]["win-64"] = bad["environments"]["gpu"]["packages"][
        "win-64"
    ][:2]
    with pytest.raises(gpu.GpuSetupError, match="no wheel"):
        gpu.spec_from_lock(bad)


# ------------------------------------------------------------------ fake wheels


def make_wheel(folder: Path, name: str, version: str, *, module_body: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{version}-cp311-cp311-win_amd64.whl"
    dist = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{name}/__init__.py", module_body or f"__version__ = {version!r}\n")
        zf.writestr(f"{name}/lib/fake.dll", b"\0" * 16)
        zf.writestr(
            f"{dist}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        zf.writestr(f"{dist}/RECORD", "")
        zf.writestr(f"{name}-{version}.data/platlib/{name}_extra.py", "X = 1\n")
        zf.writestr(f"{name}-{version}.data/scripts/{name}run.exe", b"MZ")
    return path


FAKE_TORCH = """
__version__ = "2.7.1+cu128"
class version:
    cuda = "12.8"
class cuda:
    @staticmethod
    def is_available():
        return False
"""


@pytest.fixture
def wheels(tmp_path):
    """Spec + local wheels of a fake CUDA torch/torchvision (installable on any OS)."""
    folder = tmp_path / "wheels"
    paths = [
        make_wheel(folder, "torch", "2.7.1+cu128", module_body=FAKE_TORCH),
        make_wheel(folder, "torchvision", "0.22.1+cu128"),
    ]
    spec = gpu.GpuSpec(
        tag="torch-2.7.1+cu128",
        python="cp311",
        platform="win_amd64",
        wheels=[
            gpu.Wheel(name=n, version=v, url=p.as_uri(), sha256=download.file_sha256(p))
            for (n, v), p in zip(
                [("torch", "2.7.1+cu128"), ("torchvision", "0.22.1+cu128")], paths, strict=True
            )
        ],
    )
    return spec, folder


@pytest.fixture
def supported(monkeypatch):
    monkeypatch.setattr(gpu, "platform_supported", lambda spec=None: True)


def test_extract_wheel(tmp_path, wheels):
    _, folder = wheels
    site = tmp_path / "site"
    gpu.extract_wheel(folder / "torchvision-0.22.1+cu128-cp311-cp311-win_amd64.whl", site)
    assert (site / "torchvision" / "__init__.py").is_file()
    assert (site / "torchvision_extra.py").is_file()  # .data/platlib merged
    assert not list(site.rglob("*.exe"))  # .data/scripts skipped
    assert gpu.overlay_packages(site) == {"torchvision": "0.22.1+cu128"}
    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.py", "")
    with pytest.raises(gpu.GpuSetupError, match="Unsafe"):
        gpu.extract_wheel(bad, site)


def test_install_from_wheel_dir_and_remove(wheels, supported, monkeypatch):
    spec, folder = wheels
    monkeypatch.setattr(gpu, "load_spec", lambda: spec)
    assert gpu.installed_overlay() is None
    site = gpu.install_gpu(print, wheel_dir=folder)
    assert site == gpu.overlay_dir(spec) / gpu.SITE
    assert gpu.installed_overlay() == site
    assert gpu.overlay_packages(site) == {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128"}
    marker = json.loads((site.parent / gpu.MARKER).read_text(encoding="utf-8"))
    assert marker["tag"] == spec.tag
    status = gpu.gpu_status()
    assert status.installed and status.source == "overlay" and status.packages["torch"]
    lines = []
    gpu.install_gpu(lines.append, wheel_dir=folder)
    assert "already installed" in lines[0]
    assert gpu.install_gpu(print, wheel_dir=folder, force=True) == site
    assert not (gpu.overlay_root() / f"{spec.tag}.partial").exists()
    assert gpu.remove_gpu(print) and gpu.installed_overlay() is None
    assert not gpu.remove_gpu(print)


def test_install_downloads_and_checks(wheels, supported, tmp_path):
    spec, _ = wheels
    gpu.install_gpu(print, spec=spec)  # file:// URLs
    assert (gpu.overlay_dir(spec) / gpu.SITE / "torch" / "__init__.py").is_file()
    assert not list((gpu.overlay_root() / "downloads").glob("*.whl"))
    broken = spec.model_copy(deep=True)
    broken.wheels[0].sha256 = "0" * 64
    with pytest.raises(download.DownloadError, match="Checksum"):
        gpu.install_gpu(print, spec=broken, force=True)


def test_wrong_wheel_content(tmp_path, wheels, supported):
    spec, _ = wheels
    other = make_wheel(tmp_path / "other", "torch", "2.6.0+cu128")
    broken = spec.model_copy(deep=True)
    broken.wheels[0] = gpu.Wheel(
        name="torch", version="2.7.1+cu128", url=other.as_uri(), sha256=download.file_sha256(other)
    )
    with pytest.raises(gpu.GpuSetupError, match=r"unpacked 2\.6\.0"):
        gpu.install_gpu(print, spec=broken)
    assert gpu.installed_overlay() is None


def test_install_refused(wheels, monkeypatch):
    spec, _ = wheels
    monkeypatch.setattr(gpu, "platform_supported", lambda spec=None: False)
    with pytest.raises(gpu.GpuSetupError, match="pixi run -e gpu"):
        gpu.install_gpu(print, spec=spec)
    monkeypatch.setattr(gpu, "platform_supported", lambda spec=None: True)
    monkeypatch.setattr(gpu, "MIN_FREE_BYTES", 1 << 62)
    with pytest.raises(gpu.GpuSetupError, match="disk space"):
        gpu.install_gpu(print, spec=spec)


def test_platform_supported(monkeypatch):
    spec = gpu.load_spec()
    monkeypatch.setattr(sys, "platform", "linux")
    assert not gpu.platform_supported(spec)
    monkeypatch.setattr(sys, "platform", "win32")
    assert gpu.platform_supported(spec) == (sys.version_info[:2] == (3, 11))


def test_overlay_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(gpu.OVERLAY_ENV, str(tmp_path))
    assert gpu.installed_overlay() == tmp_path
    assert gpu.gpu_status().source == "env"
    monkeypatch.setenv(gpu.OVERLAY_ENV, str(tmp_path / "missing"))
    assert gpu.installed_overlay() is None


def test_with_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing")
    env = gpu.with_overlay({"A": "1"}, tmp_path)
    assert env["PYTHONPATH"].split(gpu.os.pathsep) == [str(tmp_path), "existing"]
    assert gpu.with_overlay({"A": "1"}, None) == {"A": "1"}


# ------------------------------------------------------------------ device + manifest


def test_probe_uses_overlay(wheels, supported, monkeypatch):
    """The torch probe sees the overlay build (real child interpreter, fake torch)."""
    spec, folder = wheels
    monkeypatch.setattr(gpu, "load_spec", lambda: spec)
    site = gpu.install_gpu(print, wheel_dir=folder)
    info = probe_torch_cuda(overlay=site)
    assert (info.version, info.cuda, info.cuda_available) == ("2.7.1+cu128", "12.8", False)


GPU = {
    "detected_by": "nvml",
    "driver_version": "617.14",
    "driver_cuda": "13.4",
    "gpus": [dev.GpuInfo(index=0, name="RTX", compute_capability="12.0")],
}


def test_resolve_device_with_overlay(tmp_path, monkeypatch):
    seen = []

    def probe(python=None, overlay=None):
        seen.append(overlay)
        return TorchInfo(version="2.7.1+cu128", cuda="12.8", cuda_available=True, kernel_ok=True)

    monkeypatch.setattr(dev, "detect_gpus", lambda: dict(GPU))
    monkeypatch.setattr(dev, "probe_torch_cuda", probe)
    monkeypatch.setenv(gpu.OVERLAY_ENV, str(tmp_path))
    report = resolve_device("auto")
    assert seen == [tmp_path] and report.selected == "cuda"
    assert report.torch_overlay == str(tmp_path)
    assert report.env()["PYTHONPATH"].startswith(str(tmp_path))
    cpu = resolve_device("cpu")
    assert "PYTHONPATH" not in cpu.env() and cpu.torch_overlay is None


def test_cpu_build_hints_gpu_setup(monkeypatch):
    monkeypatch.setattr(dev, "detect_gpus", lambda: dict(GPU))
    monkeypatch.setattr(
        dev, "probe_torch_cuda", lambda python=None, overlay=None: TorchInfo(version="2.7.1+cpu")
    )
    report = resolve_device("auto")
    assert report.selected == "cpu" and "mskpipe setup gpu" in report.reason
    with pytest.raises(dev.DeviceError, match="mskpipe setup gpu"):
        resolve_device("gpu")


class _Noop(Step):
    name = "prepare"

    def run(self, ctx) -> None:
        assert ctx.device.env()["PYTHONPATH"].startswith(ctx.device.torch_overlay)


def test_manifest_records_overlay_packages(tmp_path, wheels, supported, monkeypatch):
    spec, folder = wheels
    monkeypatch.setattr(gpu, "load_spec", lambda: spec)
    site = gpu.install_gpu(print, wheel_dir=folder)
    image = tmp_path / "s.nii.gz"
    image.write_bytes(b"x")
    report = DeviceReport(
        requested="gpu", selected="cuda", gpu_index=0, reason="test", torch_overlay=str(site)
    )
    cfg = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    result = run_pipeline([_Noop()], InputSpec(image=image, modality="ct"), cfg, device=report)
    packages = result.manifest.environment.packages
    assert packages["torch"].version == "2.7.1+cu128"
    assert packages["torchvision"].version == "0.22.1+cu128"


# ------------------------------------------------------------------ CLI


def test_cli_setup_gpu(wheels, supported, monkeypatch):
    spec, folder = wheels
    monkeypatch.setattr(gpu, "load_spec", lambda: spec)
    result = runner.invoke(app, ["setup", "gpu", "--wheel-dir", str(folder)])
    assert result.exit_code == 0, result.output
    assert "GPU support installed" in result.output
    status = runner.invoke(app, ["setup", "status", "--json"])
    assert json.loads(status.output)["gpu"]["installed"] is True
    assert runner.invoke(app, ["setup", "gpu", "--remove"]).exit_code == 0
    assert gpu.installed_overlay() is None


def test_cli_setup_gpu_unsupported(monkeypatch):
    monkeypatch.setattr(gpu, "platform_supported", lambda spec=None: False)
    result = runner.invoke(app, ["setup", "gpu"])
    assert result.exit_code == 1 and "pixi run -e gpu" in result.output
