#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify that repository paths referenced from Markdown actually exist.

Docs point at `examples/` scripts and `configs/` YAML both as relative links
and as absolute github.com URLs. Neither form is validated by the Sphinx build,
so renaming a script silently breaks every doc that mentions it. This script
resolves every such reference and fails when the target is gone.

Usage:
    python3 ci/check_doc_links.py            # scan the whole repository
    python3 ci/check_doc_links.py README.md  # scan specific files
"""

from __future__ import annotations

import os
import re
import sys

# Directories whose Markdown is not ours to police (vendored code, build
# output, local-only scratch space).
SKIP_DIRS = {
    ".git", ".comate", "build", "_build", "node_modules", "third_party",
    "patches", "website", "images", "__pycache__",
}

# Reference roots we can verify. Anything else (docs/, loongforge/, ...) is
# fair game to add later; these two are the ones docs link to constantly.
ROOTS = ("examples/", "configs/")

# Absolute forms GitHub renders. Restricted to this project -- URLs pointing at
# other repositories legitimately contain `examples/` paths that do not exist
# here.
ABSOLUTE = re.compile(
    r"https?://(?:github\.com/[\w.-]+/LoongForge/(?:tree|blob)|"
    r"raw\.githubusercontent\.com/[\w.-]+/LoongForge)/[\w.-]+/"
    r"((?:examples|configs)/[^\s)\]\"'>]+)"
)

# Bare repo-root-relative mentions in shell commands. Limited to `.sh` on
# purpose: a script in a command is always meant to already exist, whereas a
# bare `.json` / `.yaml` is often an *output* path the reader is told to create
# (`--output configs/models/.../fp8_policy.json`).
BARE = re.compile(r"(?<![\w./-])((?:examples|configs)/[\w./-]+\.sh)")

# Relative links written from the containing document, e.g. ../../examples/x.sh
RELATIVE = re.compile(r"\]\((\.{1,2}/[^)\s]+\.(?:sh|ya?ml|py|json|md))")

# Placeholders are documentation, not paths.
PLACEHOLDER = re.compile(r"[<>{}$*]|\.\.\.")


def markdown_files(repo, argv):
    """Markdown to scan: explicit arguments, or every tracked-looking .md."""
    if argv:
        return [os.path.relpath(os.path.abspath(p), repo) for p in argv]
    found = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".md"):
                path = os.path.join(dirpath, name)
                # Symlinked docs point at a real file scanned elsewhere;
                # visiting both would report every finding twice.
                if not os.path.islink(path):
                    found.append(os.path.relpath(path, repo))
    return sorted(found)


def references(text, doc_dir):
    """Yield (raw_reference, repo_relative_path) pairs found in one document."""
    for pattern in (ABSOLUTE, BARE):
        for match in pattern.finditer(text):
            raw = match.group(1)
            yield raw, raw
    for match in RELATIVE.finditer(text):
        raw = match.group(1)
        resolved = os.path.normpath(os.path.join(doc_dir, raw))
        if resolved.startswith(ROOTS):
            yield raw, resolved


def main(argv):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    findings = []
    scanned = 0

    for doc in markdown_files(repo, argv):
        try:
            text = open(os.path.join(repo, doc), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"skipped {doc}: {exc}", file=sys.stderr)
            continue
        scanned += 1
        lines = text.splitlines()
        seen = set()
        for raw, target in references(text, os.path.dirname(doc)):
            if PLACEHOLDER.search(raw) or (raw, target) in seen:
                continue
            seen.add((raw, target))
            if os.path.exists(os.path.join(repo, target)):
                continue
            lineno = next(
                (i for i, line in enumerate(lines, 1) if raw in line), 0)
            findings.append((doc, lineno, raw))

    if not findings:
        print(f"doc-links: {scanned} markdown file(s) scanned, no broken "
              f"{' / '.join(ROOTS)} references")
        return 0

    print(f"doc-links: {len(findings)} broken reference(s) in {scanned} "
          f"markdown file(s)\n")
    for doc, lineno, raw in findings:
        print(f"  {doc}:{lineno}: {raw}")
    print("\nThe referenced path does not exist. Update the reference, or "
          "restore the file it points to.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
