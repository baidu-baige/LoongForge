# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tests.ci_test_utils import load_module


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "check_doc_links.py"
)
check_doc_links = load_module("check_doc_links", MODULE_PATH)

PLACEHOLDER = check_doc_links.PLACEHOLDER
changed_markdown_files = check_doc_links.changed_markdown_files
references = check_doc_links.references
repo_root = check_doc_links.repo_root
scan_documents = check_doc_links.scan_documents


def _references(text: str):
    return list(references(text, Path("docs/source")))


def test_finds_bare_relative_and_project_absolute_references():
    found = _references(
        """
        Run `examples/embodied/pi05/run.sh`.
        [config](../../configs/models/qwen3/model.yaml)
        https://github.com/baidu-baige/LoongForge/blob/master/examples/embodied/pi05/run.sh
        https://raw.githubusercontent.com/baidu-baige/LoongForge/master/configs/models/qwen3/model.yaml
        """
    )
    targets = {target for _, target in found}
    assert "examples/embodied/pi05/run.sh" in targets
    assert "configs/models/qwen3/model.yaml" in targets


def test_ignores_external_urls_and_bare_output_paths():
    found = _references(
        """
        `configs/generated/output.json`
        https://github.com/another-org/other-repo/blob/master/examples/missing.sh
        """
    )
    assert found == []


def test_placeholders_are_marked_for_main_to_skip():
    found = _references(
        "[run](../../examples/<model>/run.sh) and "
        "[config](../../configs/{name}.yaml)"
    )
    assert found
    assert all(PLACEHOLDER.search(raw) for raw, _ in found)


def test_missing_reference_is_reported_with_document_line(tmp_path):
    document = tmp_path / "README.md"
    document.write_text("[missing](examples/not-shipped.sh)\n", encoding="utf-8")
    findings, scanned = scan_documents(tmp_path, [Path("README.md")])
    assert scanned == 1
    assert findings == [(Path("README.md"), 1, "examples/not-shipped.sh")]


def test_changed_markdown_files_uses_git_revision(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    document = tmp_path / "README.md"
    document.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git", "-C", str(tmp_path), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "initial",
        ],
        check=True,
    )
    document.write_text("[missing](examples/not-shipped.sh)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git", "-C", str(tmp_path), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "changed",
        ],
        check=True,
    )
    changed = changed_markdown_files(tmp_path, "HEAD^")
    assert changed == [Path("README.md")]


def test_repo_root_matches_repository_parent():
    assert repo_root() == Path(__file__).resolve().parents[1]
