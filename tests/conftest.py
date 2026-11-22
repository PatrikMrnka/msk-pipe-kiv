# SPDX-License-Identifier: Apache-2.0
"""Shared test setup.

* Every test gets its own empty per-user folder (``MSKPIPE_HOME``), so a developer's
  installed atlas or MuscleMap store never leaks into the tests.
* Unit tests never download model weights: ``api.run`` would otherwise fetch missing
  TotalSegmentator/MuscleMap weights (gigabytes) before a run wherever the tools are
  installed (dev-cpu). Tests of the download itself call the real functions directly.
"""

import pytest

from mskpipe import api


@pytest.fixture(autouse=True)
def _isolated_mskpipe_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MSKPIPE_HOME", str(tmp_path_factory.mktemp("mskpipe_home")))


@pytest.fixture(autouse=True)
def _no_weight_downloads(monkeypatch):
    monkeypatch.setattr(api, "prefetch_weights", lambda prepared, cancel=None: 0.0)
