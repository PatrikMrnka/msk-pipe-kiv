# SPDX-License-Identifier: Apache-2.0
"""Shared test setup.

* Every test gets its own empty per-user folder (``MSKPIPE_HOME``), so a developer's
  installed atlas or MuscleMap store never leaks into the tests.
* Unit tests never download model weights: ``api.run`` would otherwise fetch missing
  TotalSegmentator/MuscleMap weights (gigabytes) before a run wherever the tools are
  installed (dev-cpu). Tests of the download itself call the real functions directly.
* Linux: importing ``mskpipe.gui`` loads the X11 libraries of the environment that the
  system lacks, before a test module imports Qt (``pytest.importorskip`` would otherwise
  skip the GUI tests without a word; ``test_platform_libs`` checks that Qt imports).
"""

import pytest

import mskpipe.gui  # noqa: F401
from mskpipe import api


@pytest.fixture(autouse=True)
def _isolated_mskpipe_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MSKPIPE_HOME", str(tmp_path_factory.mktemp("mskpipe_home")))


@pytest.fixture(autouse=True)
def _no_weight_downloads(monkeypatch):
    monkeypatch.setattr(api, "prefetch_weights", lambda prepared, cancel=None: 0.0)
