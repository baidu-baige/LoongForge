#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify repository paths referenced from Markdown actually exist.

The documentation commonly refers to launch scripts and configuration files
using bare repository paths, relative Markdown links, or links rendered by
GitHub. Those references are not checked by the Sphinx build, so this small
validator catches stale ``examples/`` and ``configs/`` paths in CI.

Usage::

    python3 .github/scripts/check_doc_links.py            # scan repository Markdown
    python3 .github/scripts/check_doc_links.py README.md  # scan selected files
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


SKIP_DIRS = {
    ".git",
    ".comate",
    "_build",
    "build",
    "images",
    "node_modules",
    "patches",
    "third_party",
    "website",
    "__pycache__",
}
ROOTS = ("examples/", "configs/")

# Restrict absolute URL scanning to this repository. Paths in links to other
# repositories are intentionally outside this check's ownership boundary.
ABSOLUTE_URL = re.compile(
    r"https?://(?:github\.com/[\w.-]+/LoongForge|"
    r"raw\.githubusercontent\.com/[\w.-]+/LoongForge)/[^\s)\]<>\"']+",
    re.IGNORECASE,
)
# Bare references are deliberately limited to shell scripts. A bare YAML or
# JSON path frequently denotes an output file that the reader is expected to
# create, rather than a file shipped in the repository.
BARE = re.compile(r"(?<![\w./-])((?:examples|configs)/[\w./-]+\.sh)")
RELATIVE_LINK = re.compile(r"\]\((?!https?://)([^)\s]+)\)", re.IGNORECASE)
URL_ROOT_REFERENCE = re.compile(r"((?:examples|configs)/[^?#\s)\]<>\"']+)")
PLACEHOLDER = re.compile(r"[<>{}$*]|\.\.\.")


def markdown_files(repo: Path, arguments: list[str]) -> list[Path]:
    """Return explicit Markdown files or all repository Markdown files."""
    if arguments:
        result = []
        for argument in arguments:
            path = (repo / argument).resolve()
            try:
                relative = path.relative_to(repo)
            except ValueError:
                raise ValueError(f"file is outside repository: {argument}")
            if path.suffix.lower() == ".md" and not path.is_symlink():
                result.append(relative)
        return result

    result = []
    for directory, dirnames, filenames in os.walk(repo):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() == ".md" and not path.is_symlink():
                result.append(path.relative_to(repo))
    return sorted(result)


def changed_markdown_files(repo: Path, revision: str) -> list[Path]:
    """Return Markdown files changed between ``revision`` and ``HEAD``."""
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{revision}...HEAD",
                "--",
                "*.md",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"could not determine changed Markdown files from {revision}: {exc}") from exc
    paths = [line for line in result.stdout.splitlines() if line]
    return markdown_files(repo, paths) if paths else []


def _clean_reference(reference: str) -> str:
    """Remove URL fragments/queries and decode escaped path characters."""
    return unquote(reference.split("#", 1)[0].split("?", 1)[0]).strip("<>").rstrip(".,;:")


def _url_target(url: str) -> str | None:
    parsed = urlsplit(url)
    # Find the first root marker in the URL path. This supports branch names
    # containing slashes without assuming that the branch is named ``master``.
    path = _clean_reference(parsed.path).lstrip("/")
    match = URL_ROOT_REFERENCE.search(path)
    return match.group(1) if match else None


def references(text: str, document_directory: Path):
    """Yield ``(raw_reference, repository_relative_target)`` pairs."""
    seen: set[tuple[str, str]] = set()

    for match in ABSOLUTE_URL.finditer(text):
        raw_url = match.group(0)
        target = _url_target(raw_url)
        if target is not None and (raw_url, target) not in seen:
            seen.add((raw_url, target))
            yield raw_url, target

    for match in BARE.finditer(text):
        raw = match.group(1)
        if (raw, raw) not in seen:
            seen.add((raw, raw))
            yield raw, raw

    for match in RELATIVE_LINK.finditer(text):
        raw = match.group(1)
        target = _clean_reference(raw)
        resolved = os.path.normpath(os.path.join(document_directory, target))
        if resolved.startswith(ROOTS) and (raw, resolved) not in seen:
            seen.add((raw, resolved))
            yield raw, resolved


def scan_documents(repo: Path, documents: list[Path]):
    """Return ``(findings, scanned_count)`` for repository-relative documents."""
    findings: list[tuple[Path, int, str]] = []
    scanned = 0
    for document in documents:
        path = repo / document
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"doc-links: skipped {document}: {exc}", file=sys.stderr)
            continue

        scanned += 1
        lines = text.splitlines()
        for raw, target in references(text, document.parent):
            if PLACEHOLDER.search(raw):
                continue
            if (repo / target).exists():
                continue
            line_number = next(
                (index for index, line in enumerate(lines, 1) if raw in line),
                0,
            )
            findings.append((document, line_number, raw))
    return findings, scanned


def repo_root() -> Path:
    """Return the repository root for this script's location."""
    return Path(__file__).resolve().parents[2]


def main(arguments: list[str]) -> int:
    repo = repo_root()
    changed_since = None
    paths = arguments
    if arguments[:1] == ["--changed-since"]:
        if len(arguments) < 2 or len(arguments) > 2:
            print(
                "doc-links: --changed-since requires exactly one revision",
                file=sys.stderr,
            )
            return 2
        changed_since = arguments[1]
        paths = []
    try:
        documents = (
            changed_markdown_files(repo, changed_since)
            if changed_since is not None
            else markdown_files(repo, paths)
        )
    except ValueError as exc:
        print(f"doc-links: {exc}", file=sys.stderr)
        return 2

    findings, scanned = scan_documents(repo, documents)

    if not findings:
        roots = " / ".join(ROOTS)
        print(
            f"doc-links: {scanned} markdown file(s) scanned, no broken {roots} references"
        )
        return 0

    print(
        f"doc-links: {len(findings)} broken reference(s) in {scanned} markdown file(s)\n"
    )
    for document, line_number, raw in findings:
        print(f"  {document}:{line_number}: {raw}")
    print(
        "\nThe referenced path does not exist. Update the reference, or restore the file it points to."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
