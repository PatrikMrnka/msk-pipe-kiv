# SPDX-License-Identifier: Apache-2.0
"""Shared test setup: every test gets its own empty per-user folder (``MSKPIPE_HOME``),
so a developer's installed atlas or MuscleMap store never leaks into the tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_mskpipe_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MSKPIPE_HOME", str(tmp_path_factory.mktemp("mskpipe_home")))
