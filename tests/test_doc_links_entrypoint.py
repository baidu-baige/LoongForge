# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the script path used by the upstream reusable CI workflow."""

from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/check_doc_links.py"


def test_workflow_entrypoint_exists_and_resolves_repository_root():
    workflow = (ROOT / ".github/workflows/doc-links.yml").read_text()
    assert "python3 .github/scripts/check_doc_links.py --changed-since" in workflow
    module = runpy.run_path(str(SCRIPT))
    assert module["repo_root"]() == ROOT


@pytest.fixture
def checkout(tmp_path):
    """A head checkout with no legacy ci/ package, like the migrated upstream."""
    script = tmp_path / ".github/scripts/check_doc_links.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    example = tmp_path / "examples/demo/run.sh"
    example.parent.mkdir(parents=True)
    example.write_text("#!/usr/bin/env bash\n")
    (tmp_path / "README.md").write_text("Initial documentation\n")
    commit(tmp_path, "initial")
    return tmp_path


def commit(repo, message):
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test",
         "-c", "user.email=test@example.invalid", "commit", "-qm", message],
        check=True,
    )


@pytest.mark.parametrize("exists, expected", [(True, 0), (False, 1)])
def test_changed_since_checks_real_links_from_workflow_path(checkout, exists, expected):
    target = "examples/demo/run.sh" if exists else "examples/missing/run.sh"
    (checkout / "README.md").write_text(f"Run [{target}]({target}).\n")
    commit(checkout, "update docs")
    result = subprocess.run(
        [sys.executable, str(checkout / ".github/scripts/check_doc_links.py"),
         "--changed-since", "HEAD^"],
        cwd=checkout.parent, capture_output=True, text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert "1 markdown file(s)" in result.stdout
    if not exists:
        assert "examples/missing/run.sh" in result.stdout


def test_changed_since_invalid_revision_still_fails(checkout):
    result = subprocess.run(
        [sys.executable, str(checkout / ".github/scripts/check_doc_links.py"),
         "--changed-since", "missing-revision"],
        cwd=checkout, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "could not determine changed Markdown files" in result.stderr
