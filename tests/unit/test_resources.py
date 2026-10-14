# SPDX-License-Identifier: Apache-2.0
"""mskpipe.resources and `mskpipe setup`: per-user folder, downloads, licence key, model
weights and attachment atlas (no network, no GPU, no segmentation tools)."""

import ast
import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mskpipe.api import prefetch_weights
from mskpipe.cli import app
from mskpipe.config import PipelineConfig
from mskpipe.plugins.segmenters import musclemap as mm
from mskpipe.resources import atlas, download, licence, paths, weights
from mskpipe.resources.status import collect

REAL_PREFETCH = prefetch_weights  # imported before conftest replaces it in api

runner = CliRunner()
KEY = "aca_ABCDEFGHIJKLMN"


def _config(**overrides) -> PipelineConfig:
    from mskpipe.config.loader import load_config

    return load_config(None, [f"{k}={v}" for k, v in overrides.items()])


# ------------------------------------------------------------------ paths


def test_home_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MSKPIPE_HOME", str(tmp_path / "h"))
    assert paths.home() == tmp_path / "h"
    assert paths.weights_dir() == tmp_path / "h" / "weights"
    assert paths.atlas_root() == tmp_path / "h" / "atlas"


def test_home_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("MSKPIPE_HOME")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert paths.home() == tmp_path / "local" / "msk-pipe"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert paths.home() == tmp_path / "xdg" / "msk-pipe"


def test_totalseg_home(tmp_path, monkeypatch):
    monkeypatch.delenv("TOTALSEG_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "ts"))
    assert paths.totalseg_home() == tmp_path / "ts"
    assert paths.totalseg_weights_dir() == tmp_path / "ts" / "nnunet" / "results"
    monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(tmp_path / "w"))
    assert paths.totalseg_weights_dir() == tmp_path / "w"


# ------------------------------------------------------------------ download


def test_download_file_url_with_checksum(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 3_000_000)
    seen = []
    digest = download.download(
        src.as_uri(),
        tmp_path / "out" / "dst.bin",
        sha256=download.file_sha256(src),
        progress=lambda done, total: seen.append(done),
    )
    assert digest == download.file_sha256(src)
    assert (tmp_path / "out" / "dst.bin").read_bytes() == src.read_bytes()
    assert seen[-1] == 3_000_000


def test_download_checksum_mismatch_leaves_nothing(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"data")
    dst = tmp_path / "dst.bin"
    with pytest.raises(download.DownloadError, match="Checksum"):
        download.download(src.as_uri(), dst, sha256="0" * 64)
    assert not dst.exists()
    assert not dst.with_name("dst.bin.part").exists()


def test_download_missing_source(tmp_path):
    with pytest.raises(download.DownloadError, match="failed"):
        download.download((tmp_path / "nope").as_uri(), tmp_path / "dst")


# ------------------------------------------------------------------ licence


def _ts_home(tmp_path, monkeypatch, data=None) -> Path:
    home = tmp_path / "ts"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(home))
    monkeypatch.delenv("TOTALSEG_WEIGHTS_PATH", raising=False)
    if data is not None:
        (home / "config.json").write_text(json.dumps(data), encoding="utf-8")
    return home


def test_licence_number(tmp_path, monkeypatch):
    _ts_home(tmp_path, monkeypatch)
    assert licence.licence_number() is None
    _ts_home(tmp_path, monkeypatch, {"license_number": KEY})
    assert licence.licence_number() == KEY
    (tmp_path / "ts" / "config.json").write_text("{broken", encoding="utf-8")
    assert licence.licence_number() is None


@pytest.mark.parametrize(
    ("key", "problem"),
    [(KEY, None), ("abc_ABCDEFGHIJKLMN", "start with"), ("aca_SHORT", "18 characters")],
)
def test_licence_format(key, problem):
    result = licence.format_problem(key)
    assert (result is None) if problem is None else (problem in result)


def test_set_licence(tmp_path, monkeypatch):
    home = _ts_home(tmp_path, monkeypatch)
    calls = []

    def fake_run(argv, log, **_kw):
        calls.append(argv)
        (home / "config.json").write_text(json.dumps({"license_number": argv[4]}), "utf-8")
        log("License has been successfully saved.")
        return 0

    monkeypatch.setattr(licence, "run_tool", fake_run)
    lines = []
    licence.set_licence(f"  {KEY} ", lines.append)
    assert calls[0][1:] == ["-m", licence.SET_LICENCE_MODULE, "-l", KEY]
    assert licence.licence_number() == KEY
    assert lines == ["License has been successfully saved."]
    assert licence.set_licence_command(KEY, skip_validation=True)[-1] == "-sv"


def test_set_licence_errors(tmp_path, monkeypatch):
    _ts_home(tmp_path, monkeypatch)
    monkeypatch.setattr(licence, "run_tool", lambda argv, log, **_kw: 1)
    with pytest.raises(licence.LicenceError, match="must start"):
        licence.set_licence("wrong", print)
    with pytest.raises(licence.LicenceError, match="exit code 1"):
        licence.set_licence(KEY, print)
    monkeypatch.setattr(licence, "run_tool", lambda argv, log, **_kw: 0)
    with pytest.raises(licence.LicenceError, match="not stored"):
        licence.set_licence(KEY, print)


# ------------------------------------------------------------------ TotalSegmentator


@pytest.fixture
def fake_ts(tmp_path, monkeypatch):
    """A TotalSegmentator task table (``map_tasks_config``) without TotalSegmentator."""
    package = types.ModuleType("totalsegmentator")
    package.__path__ = []
    module = types.ModuleType("totalsegmentator.map_tasks_config")
    module.TASK_ID_WEIGHTS_CONFIGS = {
        i: {"foldername": f"Dataset{i}_model"} for ids in weights.TS_TASK_IDS.values() for i in ids
    }
    module.TASK_ID_WEIGHTS_CONFIGS[304]["rel_path"] = "extra"
    monkeypatch.setitem(sys.modules, "totalsegmentator", package)
    monkeypatch.setitem(sys.modules, "totalsegmentator.map_tasks_config", module)
    return _ts_home(tmp_path, monkeypatch) / "nnunet" / "results"


def _make_ts_weights(root: Path, task: str) -> None:
    for i in weights.TS_TASK_IDS[task]:
        rel = "extra/" if i == 304 else ""
        (root / f"{rel}Dataset{i}_model").mkdir(parents=True)


def test_required_ts_tasks():
    tasks = weights.required_ts_tasks(_config(), ["ct", "mri"])
    assert set(tasks) == {"total", "appendicular_bones", "total_mr", "appendicular_bones_mr"}
    no_key = _config(**{"segmentation.tibia_fibula": "musclemap"})
    assert weights.required_ts_tasks(no_key, ["ct"]) == ["total"]


def test_ts_status(fake_ts):
    item = weights.ts_status("total")
    assert item.status == "missing" and "Dataset291_model" in item.detail
    _make_ts_weights(fake_ts, "total")
    assert weights.ts_status("total").status == "ok"
    _make_ts_weights(fake_ts, "appendicular_bones")
    item = weights.ts_status("appendicular_bones")
    assert item.status == "ok" and item.licensed


def test_ts_status_without_totalsegmentator(monkeypatch):
    # None in sys.modules makes the import fail even if the module was imported before
    # (TotalSegmentator is installed in dev-cpu)
    monkeypatch.setitem(sys.modules, "totalsegmentator", None)
    monkeypatch.setitem(sys.modules, "totalsegmentator.map_tasks_config", None)
    assert weights.ts_status("total").status == "unknown"
    assert weights.ts_folders("unknown_task") is None


def test_download_ts(fake_ts, monkeypatch):
    calls = []

    def fake_run(argv, log, **_kw):
        calls.append(argv)
        _make_ts_weights(fake_ts, argv[-1])
        return 0

    monkeypatch.setattr(weights, "run_tool", fake_run)
    assert weights.download_ts("total", print).status == "ok"
    assert calls == [[sys.executable, "-m", weights.TS_DOWNLOAD_MODULE, "-t", "total"]]
    weights.download_ts("total", print)  # present: no second download
    assert len(calls) == 1
    with pytest.raises(weights.WeightsError, match="licence key"):
        weights.download_ts("appendicular_bones", print)
    monkeypatch.setattr(weights, "run_tool", lambda argv, log, **_kw: 0)
    with pytest.raises(weights.WeightsError, match="download failed"):
        weights.download_ts("total_mr", print)


def test_ts_task_ids_match_totalsegmentator():
    """The task table above matches TotalSegmentator's own (read without importing it)."""
    try:
        spec = importlib.util.find_spec("totalsegmentator.bin.totalseg_download_weights")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        pytest.skip("TotalSegmentator not installed")
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    table = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "TASK_TO_ID" for t in node.targets)
    )
    for task, ids in weights.TS_TASK_IDS.items():
        assert tuple(table[task]) == ids, task


# ------------------------------------------------------------------ MuscleMap


@pytest.fixture
def fake_mm(tmp_path, monkeypatch):
    """A MuscleMap package folder without MuscleMap; returns its model folder of v1.4."""
    pkg = tmp_path / "env" / "scripts"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(mm, "package_dir", lambda: pkg)
    return pkg / "models" / mm.REGION / "v1.4"


def _write_model(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / mm.MODEL_PTH).write_bytes(b"weights")
    (folder / mm.MODEL_JSON).write_text('{"labels": []}', encoding="utf-8")


def test_musclemap_sync(fake_mm):
    store = weights.mm_store_dir("1.4")
    assert weights.mm_status("1.4").status == "missing"
    assert weights.sync_musclemap("1.4") == "none"
    _write_model(fake_mm)
    assert weights.sync_musclemap("1.4") == "stored"
    assert (store / mm.MODEL_PTH).read_bytes() == b"weights"
    assert weights.sync_musclemap("1.4") == "none"
    # a new environment gets the model from the store
    for f in fake_mm.iterdir():
        f.unlink()
    item = weights.mm_status("1.4")
    assert item.status == "ok" and "store" in item.detail
    assert weights.sync_musclemap("1.4") == "restored"
    assert (fake_mm / mm.MODEL_JSON).read_text(encoding="utf-8") == '{"labels": []}'
    assert weights.mm_status("1.4").path == str(fake_mm)


def test_musclemap_not_installed(monkeypatch):
    def missing():
        raise mm.MuscleMapError("not installed")

    monkeypatch.setattr(mm, "package_dir", missing)
    assert weights.mm_status("1.4").status == "unknown"
    assert weights.sync_musclemap("1.4") == "none"
    with pytest.raises(weights.WeightsError, match="not installed"):
        weights.download_musclemap("1.4", print)


def test_download_musclemap(fake_mm, monkeypatch):
    calls = []

    def fake_run(argv, log, **_kw):
        calls.append(argv)
        _write_model(fake_mm)
        return 0

    monkeypatch.setattr(weights, "run_tool", fake_run)
    assert weights.download_musclemap("1.4", print).status == "ok"
    assert "ensure_model_downloaded" in calls[0][2] and "'1.4'" in calls[0][2]
    assert (weights.mm_store_dir("1.4") / mm.MODEL_PTH).is_file()
    weights.download_musclemap("1.4", print)
    assert len(calls) == 1
    monkeypatch.setattr(weights, "run_tool", lambda argv, log, **_kw: 0)
    for f in fake_mm.iterdir():
        f.unlink()
    for f in weights.mm_store_dir("1.4").iterdir():
        f.unlink()
    with pytest.raises(weights.WeightsError, match="download failed"):
        weights.download_musclemap("1.4", print)


def test_weights_status(fake_ts, fake_mm):
    items = weights.weights_status(_config(), ["ct"])
    assert {(i.tool, i.name) for i in items} == {
        ("totalsegmentator", "appendicular_bones"),
        ("totalsegmentator", "total"),
        ("musclemap", "wholebody v1.4"),
    }
    assert {i.status for i in items} == {"missing"}


# ------------------------------------------------------------------ atlas


def _atlas(folder: Path) -> Path:
    (folder / "bones").mkdir(parents=True)
    (folder / "atlas.json").write_text('{"muscles": {}}', encoding="utf-8")
    (folder / "bones" / "femur_r.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (folder / "LICENSE").write_text("CC BY-NC-SA 2.0 BE", encoding="utf-8")
    return folder


def _zip(folder: Path, archive: Path, prefix: str = "lhdl-atlas/") -> Path:
    with zipfile.ZipFile(archive, "w") as zf:
        for f in folder.rglob("*"):
            if f.is_file():
                zf.write(f, prefix + f.relative_to(folder).as_posix())
    return archive


def test_resolve_atlas_order(tmp_path, monkeypatch):
    monkeypatch.delenv(atlas.ATLAS_ENV, raising=False)
    assert atlas.resolve_atlas_dir(None) == (None, "none")
    _atlas(atlas.default_atlas_dir())
    assert atlas.resolve_atlas_dir(None) == (atlas.default_atlas_dir(), "home")
    monkeypatch.setenv(atlas.ATLAS_ENV, str(tmp_path / "env"))
    assert atlas.resolve_atlas_dir(None) == (tmp_path / "env", "env")
    assert atlas.resolve_atlas_dir(str(tmp_path / "cfg")) == (tmp_path / "cfg", "config")


def test_install_atlas_from_folder_and_zip(tmp_path):
    src = _atlas(tmp_path / "src")
    dest = atlas.install_atlas(str(src), print)
    assert dest == atlas.default_atlas_dir()
    assert (dest / "bones" / "femur_r.obj").is_file()
    with pytest.raises(atlas.AtlasError, match="--force"):
        atlas.install_atlas(str(src), print)
    archive = _zip(src, tmp_path / "atlas.zip")
    (src / "atlas.json").write_text('{"muscles": {"x": 1}}', encoding="utf-8")
    dest = atlas.install_atlas(str(archive), print, force=True)
    assert json.loads((dest / "atlas.json").read_text(encoding="utf-8")) == {"muscles": {}}
    assert not [p for p in paths.atlas_root().iterdir() if p.name.startswith(".install")]


def test_install_atlas_from_url_with_checksum(tmp_path):
    archive = _zip(_atlas(tmp_path / "src"), tmp_path / "atlas.zip", prefix="")
    digest = download.file_sha256(archive)
    dest = atlas.install_atlas(archive.as_uri(), print, name="v1", sha256=digest)
    assert dest == paths.atlas_root() / "v1" and (dest / "atlas.json").is_file()
    with pytest.raises(download.DownloadError, match="Checksum"):
        atlas.install_atlas(archive.as_uri(), print, name="v2", sha256="0" * 64)
    with pytest.raises(atlas.AtlasError, match="Checksum"):
        atlas.install_atlas(str(archive), print, name="v3", sha256="0" * 64)


def test_install_atlas_rejects_bad_input(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil/atlas.json", "{}")
    with pytest.raises(atlas.AtlasError, match="Unsafe"):
        atlas.install_atlas(str(bad), print)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(atlas.AtlasError, match=r"No atlas\.json"):
        atlas.install_atlas(str(empty), print)
    with pytest.raises(atlas.AtlasError, match="does not exist"):
        atlas.install_atlas(str(tmp_path / "missing"), print)
    with pytest.raises(atlas.AtlasError, match="Invalid atlas name"):
        atlas.install_atlas(str(empty), print, name="../x")


def test_bone_registration_uses_installed_atlas(tmp_path, monkeypatch):
    from mskpipe.plugins.attachments.bone_registration import BoneRegistration

    monkeypatch.delenv(atlas.ATLAS_ENV, raising=False)
    config = _config()
    assert "mskpipe setup atlas" in BoneRegistration.preflight(config, "ct")[0]
    atlas.install_atlas(str(_atlas(tmp_path / "src")), print)
    assert BoneRegistration.preflight(config, "ct") == []


# ------------------------------------------------------------------ status + CLI


def test_status_collect(tmp_path, monkeypatch, fake_ts, fake_mm):
    monkeypatch.delenv(atlas.ATLAS_ENV, raising=False)
    status = collect(_config(), ["ct"])
    assert status.licence.needed and not status.licence.present
    assert status.atlas.needed and not status.atlas.ok and status.atlas.source == "none"
    assert not status.ready
    _make_ts_weights(fake_ts, "total")
    _make_ts_weights(fake_ts, "appendicular_bones")
    _write_model(fake_mm)
    (fake_ts.parent.parent / "config.json").write_text(json.dumps({"license_number": KEY}))
    atlas.install_atlas(str(_atlas(tmp_path / "src")), print)
    status = collect(_config(), ["ct"])
    assert status.ready and status.atlas.source == "home"
    no_key = collect(_config(**{"segmentation.tibia_fibula": "musclemap"}), ["ct"])
    assert not no_key.licence.needed


def test_cli_setup_status(fake_ts, fake_mm, monkeypatch):
    monkeypatch.delenv(atlas.ATLAS_ENV, raising=False)
    result = runner.invoke(app, ["setup", "status", "-m", "ct", "--json"])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    assert data["atlas"]["source"] == "none"
    assert {w["name"] for w in data["weights"]} >= {"total", "wholebody v1.4"}
    result = runner.invoke(app, ["setup", "status", "-m", "ct"])
    assert "Licence key:" in result.output and "mskpipe setup" in result.output


def test_cli_setup_weights(fake_ts, fake_mm, monkeypatch):
    def fake_run(argv, log, **_kw):
        if "-t" in argv:
            _make_ts_weights(fake_ts, argv[-1])
        else:
            _write_model(fake_mm)
        return 0

    monkeypatch.setattr(weights, "run_tool", fake_run)
    result = runner.invoke(app, ["setup", "weights", "-m", "ct"])
    assert result.exit_code == 1  # appendicular_bones skipped: no licence key
    assert "needs the licence key" in result.output
    assert weights.ts_status("total").status == "ok"
    assert weights.mm_status("1.4").status == "ok"
    args = ["setup", "weights", "-m", "ct", "--set", "segmentation.tibia_fibula=musclemap"]
    assert runner.invoke(app, args).exit_code == 0


def test_cli_setup_licence_and_atlas(tmp_path, monkeypatch):
    _ts_home(tmp_path, monkeypatch)
    result = runner.invoke(app, ["setup", "licence", "bad"])
    assert result.exit_code == 1 and "license-academic" in result.output
    src = _atlas(tmp_path / "src")
    assert runner.invoke(app, ["setup", "atlas", str(src)]).exit_code == 0
    assert runner.invoke(app, ["setup", "atlas", str(src)]).exit_code == 1
    assert runner.invoke(app, ["setup", "atlas", str(src), "--force"]).exit_code == 0


# ------------------------------------------------------------------ weights before a run


def test_ensure_weights_downloads_missing_under_lock(fake_ts, fake_mm, monkeypatch, tmp_path):
    from mskpipe.resources import lock as lock_mod

    _ts_home(tmp_path, monkeypatch, {"license_number": KEY})
    calls = []

    def fake_run(argv, log, **_kw):
        calls.append(argv)
        with pytest.raises(lock_mod.BusyError):  # held while downloading
            lock_mod.setup_lock("other window").acquire()
        if "-t" in argv:
            _make_ts_weights(fake_ts, argv[-1])
        else:
            _write_model(fake_mm)
        return 0

    monkeypatch.setattr(weights, "run_tool", fake_run)
    lines = []
    done = weights.ensure_weights(_config(), "ct", lines.append)
    assert {(i.tool, i.name) for i in done} == {
        ("totalsegmentator", "total"),
        ("totalsegmentator", "appendicular_bones"),
        ("musclemap", "wholebody v1.4"),
    }
    assert len(calls) == 3
    assert weights.ensure_weights(_config(), "ct", lines.append) == []  # all present now
    assert len(calls) == 3
    with lock_mod.setup_lock("free again"):
        pass


def test_ensure_weights_skips_licensed_task_without_key(fake_ts, fake_mm, monkeypatch):
    def fake_run(argv, log, **_kw):
        if "-t" in argv:
            _make_ts_weights(fake_ts, argv[-1])
        else:
            _write_model(fake_mm)
        return 0

    monkeypatch.setattr(weights, "run_tool", fake_run)
    done = weights.ensure_weights(_config(), "ct", print)
    assert "appendicular_bones" not in {i.name for i in done}  # pre-run checks report it


def test_ensure_weights_offline(fake_ts, fake_mm, monkeypatch):
    monkeypatch.setattr(weights, "run_tool", lambda argv, log, **_kw: 1)
    with pytest.raises(weights.WeightsError, match="download failed"):
        weights.ensure_weights(_config(), "ct", print)


def test_run_tool_cancel_kills_the_download(tmp_path):
    import time

    from mskpipe.resources.tool import ToolCancelled, run_tool

    pid_file = tmp_path / "pid"
    script = (
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "print('downloading', flush=True)\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    lines = []
    with pytest.raises(ToolCancelled):
        run_tool(
            [sys.executable, "-c", script],
            lines.append,
            cancelled=lambda: time.monotonic() - start > 1.0,
        )
    assert time.monotonic() - start < 20 and lines == ["downloading"]
    import psutil

    pid = int(pid_file.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_prefetch_weights_errors(monkeypatch):
    """The real api.prefetch_weights (conftest replaces it for api.run in unit tests)."""
    import threading

    from mskpipe import api
    from mskpipe.resources.lock import LockCancelled
    from mskpipe.resources.tool import ToolCancelled

    prepared = SimpleNamespace(config=_config(), modality="ct")
    monkeypatch.setattr(weights, "ensure_weights", lambda *a, **k: [])
    assert REAL_PREFETCH(prepared) == 0.0

    def offline(*a, **k):
        raise weights.WeightsError("TotalSegmentator total: download failed (exit code 1)")

    monkeypatch.setattr(weights, "ensure_weights", offline)
    with pytest.raises(api.SetupError, match="first run needs internet"):
        REAL_PREFETCH(prepared)
    for exc in (ToolCancelled("x"), LockCancelled("y")):

        def cancelled(*a, _exc=exc, **k):
            raise _exc

        monkeypatch.setattr(weights, "ensure_weights", cancelled)
        with pytest.raises(api.SetupCancelled):
            REAL_PREFETCH(prepared, threading.Event())


def test_run_prefetches_only_when_segment_runs(monkeypatch, tmp_path):
    """api.run downloads before the first step, outside the step times."""
    from fake_pipeline import fake_steps, write_nifti

    from mskpipe import api

    calls = []
    monkeypatch.setattr(api, "prefetch_weights", lambda p, cancel=None: calls.append(p.steps))
    image = write_nifti(tmp_path / "s01.nii.gz")
    runs = tmp_path / "runs"
    req = api.RunRequest(image=image, modality="ct", runs_dir=runs)
    first = api.run(req, steps=fake_steps())
    assert first.status == "completed" and calls and "segment" in calls[0]
    calls.clear()
    api.run(api.RunRequest(resume=first.run_dir, from_step="mesh"), steps=fake_steps())
    assert calls == []
