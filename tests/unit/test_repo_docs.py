# SPDX-License-Identifier: Apache-2.0
"""Repository documentation and workflow hygiene (README, CITATION, GitHub Actions)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# Oldest major versions running on Node.js 24 (Node.js 20 actions are deprecated):
# upload-artifact v6, download-artifact v7; checkout and setup-pixi already are.
MIN_MAJOR = {
    "actions/upload-artifact": 6,
    "actions/download-artifact": 7,
    "actions/checkout": 5,
}


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_has_both_install_paths():
    text = _readme()
    for heading in ("## a) Just run it", "## b) Build from source", "### ba)", "### bb)"):
        assert heading in text, heading
    assert "releases/latest" in text
    assert "msk-pipe.cmd" in text and "pixi run -e cpu gui" in text
    assert "skeleton_parity.py --fetch" in text


def test_readme_states_licences_and_citation():
    text = _readme()
    assert "Apache-2.0" in text and "CC BY-NC 4.0" in text and "CC BY-NC-SA 2.0 BE" in text
    assert "non-commercial" in text
    assert "CITATION.cff" in text


def test_readme_local_links_exist():
    for target in re.findall(r"\]\(((?!https?://|#)[^)\s]+)\)", _readme()):
        assert (ROOT / target).exists(), target


def test_readme_build_steps_match_release_workflow():
    """The 'build the release yourself' block uses the subcommands of the build job."""
    text = _readme()
    job = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    for sub in re.findall(r"build_release\.py (\w+)", job):
        assert f"build_release.py {sub}" in text, sub


def test_actions_run_on_node24():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in wf["jobs"].values():
            for step in job.get("steps", []):
                action, _, ref = step.get("uses", "").partition("@")
                if action in MIN_MAJOR:
                    major = int(re.match(r"v(\d+)", ref).group(1))
                    assert major >= MIN_MAJOR[action], f"{path.name}: {action}@{ref}"


def test_citation_version_is_semver():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(citation["version"]))
