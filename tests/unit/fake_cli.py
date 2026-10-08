# SPDX-License-Identifier: Apache-2.0
"""``mskpipe`` CLI with the fake pipeline, for GUI tests that start a real child process.

    python tests/unit/fake_cli.py run IMAGE --modality ct ... --events jsonl

Behaviour of the fake steps comes from ``FAKE_FAIL``, ``FAKE_WAIT_CANCEL``, ``FAKE_HANG``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_pipeline import configure_from_env, fake_steps

from mskpipe import api
from mskpipe.cli import app

configure_from_env()
api.default_steps = lambda registry=None: fake_steps()
api.preflight = lambda prepared, registry=None: []

if __name__ == "__main__":
    app(prog_name="mskpipe")
