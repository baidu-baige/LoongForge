#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Scan the repository for internal-only information before publishing.

Detects internal hostnames and package mirrors, private object-storage
locations, corporate emails, developer home directories, credential material,
RFC1918 addresses, and pre-rename internal project names. Rules live in
``sensitive_rules.py`` next to this file.

Usage:
    python3 .github/scripts/sensitive_scan.py                  # tracked + staged files
    python3 .github/scripts/sensitive_scan.py --history        # also git metadata
    python3 .github/scripts/sensitive_scan.py --strict         # warnings also fail
    python3 .github/scripts/sensitive_scan.py --strict --ci-summary  # CI-safe aggregate
    python3 .github/scripts/sensitive_scan.py --format json
    python3 .github/scripts/sensitive_scan.py --rule corp-email --rule internal-domain
    python3 .github/scripts/sensitive_scan.py --paths docker/ tests/

Exit codes:
    0  no blocking findings
    1  at least one error (or, with --strict, at least one warning)
    2  scanner could not run (not a git repo, bad rules, ...)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import NoReturn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sensitive_rules  # noqa: E402

SUPPRESS_RE = re.compile(r"sensitive-scan:\s*allow(?:\[([^\]]*)\])?")
MAX_SNIPPET = 160


@dataclass
class Finding:
    rule: str
    severity: str
    title: str
    location: str
    line: int
    snippet: str
    hint: str


def die(msg: str) -> NoReturn:
    print(f"sensitive-scan: {msg}", file=sys.stderr)
    raise SystemExit(2)


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ("git",) + args, capture_output=True, text=True, errors="replace"
    )
    if check and proc.returncode != 0:
        die(f"`git {' '.join(args)}` failed: {proc.stderr.strip()}")
    return proc.stdout


def repo_root() -> str:
    out = git("rev-parse", "--show-toplevel").strip()
    if not out:
        die("not inside a git repository")
    return out


def redact(text: str) -> str:
    """Keep enough context to locate the value without reprinting it."""
    if len(text) <= 12:
        return text[:3] + "*" * max(0, len(text) - 3)
    return f"{text[:6]}...{text[-4:]} [redacted, len={len(text)}]"


def compile_rules(selected: list[str] | None) -> list[dict]:
    rules = []
    for raw in sensitive_rules.RULES:
        if selected and raw["id"] not in selected:
            continue
        rule = dict(raw)
        try:
            rule["regex"] = re.compile(raw["pattern"])
        except re.error as exc:
            die(f"rule {raw['id']} has an invalid pattern: {exc}")
        rules.append(rule)
    if selected:
        unknown = set(selected) - {r["id"] for r in rules}
        if unknown:
            die(f"unknown rule id(s): {', '.join(sorted(unknown))}")
    return rules


def compile_allowlist() -> list[dict]:
    entries = []
    for raw in sensitive_rules.ALLOWLIST:
        if not raw.get("reason"):
            die(f"allowlist entry {raw!r} is missing a 'reason'")
        entry = dict(raw)
        entry["match_re"] = re.compile(raw["match"]) if raw.get("match") else None
        entries.append(entry)
    return entries


def is_allowlisted(allowlist: list[dict], rule_id: str, path: str, text: str) -> bool:
    for entry in allowlist:
        if entry["rule"] not in ("*", rule_id):
            continue
        if not fnmatch.fnmatch(path, entry["path"]):
            continue
        if entry["match_re"] is None or entry["match_re"].search(text):
            return True
    return False


def is_suppressed(line: str, rule_id: str) -> bool:
    m = SUPPRESS_RE.search(line)
    if not m:
        return False
    ids = m.group(1)
    if not ids:
        return True
    return rule_id in {part.strip() for part in ids.split(",") if part.strip()}


def excluded(path: str) -> bool:
    for pattern in sensitive_rules.CONFIG["exclude_paths"]:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*/{pattern}"):
            return True
    return False


def candidate_files(path_filters: list[str]) -> list[str]:
    """Tracked files, plus staged additions, plus untracked non-ignored files.

    Untracked files are included because they are what a developer is about to
    commit; ignored files are excluded because they never ship.
    """
    tracked = git("ls-files").splitlines()
    staged = git(
        "diff", "--cached", "--name-only", "--diff-filter=ACM"
    ).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    files = sorted({p for p in tracked + staged + untracked if p})
    if path_filters:
        files = [
            p for p in files
            if any(p == f or p.startswith(f.rstrip("/") + "/") for f in path_filters)
        ]
    return [p for p in files if not excluded(p)]


def read_lines(path: str) -> list[str] | None:
    """Return text lines, or None when the file is binary/unreadable."""
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    if b"\x00" in blob[:8192]:
        return None
    return blob.decode("utf-8", errors="replace").splitlines()


def scan_text(
    rules: list[dict],
    allowlist: list[dict],
    location: str,
    lines: list[str],
    allow_path: str,
) -> list[Finding]:
    out = []
    for lineno, line in enumerate(lines, start=1):
        for rule in rules:
            m = rule["regex"].search(line)
            if not m:
                continue
            hit = m.group(0)
            if is_suppressed(line, rule["id"]):
                continue
            if is_allowlisted(allowlist, rule["id"], allow_path, line):
                continue
            snippet = redact(hit) if rule.get("redact") else line.strip()
            out.append(
                Finding(
                    rule=rule["id"],
                    severity=rule["severity"],
                    title=rule["title"],
                    location=location,
                    line=lineno,
                    snippet=snippet[:MAX_SNIPPET],
                    hint=rule["hint"],
                )
            )
    return out


def scan_worktree(rules, allowlist, path_filters) -> list[Finding]:
    findings = []
    for path in candidate_files(path_filters):
        lines = read_lines(path)
        if lines is None:
            continue
        findings.extend(scan_text(rules, allowlist, path, lines, path))
    return findings


def scan_history(rules, allowlist) -> list[Finding]:
    """Scan git metadata and files that exist in HEAD but not on disk.

    Working-tree cleanliness says nothing about what a published repository
    exposes: author emails, commit messages, tag names and deleted-but-committed
    files all ship with the history.
    """
    findings = []
    sources = {
        "git:authors": sorted(set(
            git("log", "--all", "--format=%an <%ae>%n%cn <%ce>").splitlines()
        )),
        "git:commit-messages": git(
            "log", "--all", "--format=%s%n%b"
        ).splitlines(),
        "git:tags": git("tag").splitlines(),
        "git:branches": git(
            "branch", "-a", "--format=%(refname:short)"
        ).splitlines(),
    }
    for location, lines in sources.items():
        findings.extend(
            scan_text(rules, allowlist, location, [ln for ln in lines if ln], location)
        )

    on_disk = set(candidate_files([]))
    in_head = {p for p in git("ls-tree", "-r", "--name-only", "HEAD").splitlines() if p}
    for path in sorted(in_head - on_disk):
        if excluded(path):
            continue
        blob = git("show", f"HEAD:{path}", check=False)
        if not blob or "\x00" in blob[:8192]:
            continue
        findings.extend(
            scan_text(
                rules, allowlist, f"git:HEAD:{path}", blob.splitlines(), path
            )
        )
    return findings


def report_text(findings: list[Finding], rules: list[dict], strict: bool) -> None:
    cap = sensitive_rules.CONFIG["max_findings_per_rule"]
    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        by_rule.setdefault(f.rule, []).append(f)
    meta = {r["id"]: r for r in rules}

    for severity, label in (("error", "ERRORS (blocking)"), ("warn", "WARNINGS")):
        ids = [rid for rid in by_rule if meta[rid]["severity"] == severity]
        if not ids:
            continue
        total = sum(len(by_rule[rid]) for rid in ids)
        print(f"\n=== {label} — {total} finding(s) in {len(ids)} rule(s) ===")
        for rid in sorted(ids, key=lambda r: -len(by_rule[r])):
            hits = by_rule[rid]
            print(f"\n[{rid}] {meta[rid]['title']} — {len(hits)} finding(s)")
            print(f"  why : {meta[rid]['why']}")
            print(f"  fix : {meta[rid]['hint']}")
            for f in hits[:cap]:
                print(f"    {f.location}:{f.line}: {f.snippet}")
            if len(hits) > cap:
                print(f"    ... and {len(hits) - cap} more (use --format json for all)")

    errors = sum(1 for f in findings if f.severity == "error")
    warns = sum(1 for f in findings if f.severity == "warn")
    print(f"\nsummary: {errors} error(s), {warns} warning(s)")
    if not findings:
        print("clean: no internal information detected")
    elif errors == 0 and not strict:
        print("result: PASS (warnings do not block; re-run with --strict to enforce)")
    else:
        print("result: FAIL")


def report_ci_summary(findings: list[Finding], strict: bool) -> None:
    """Print an aggregate-only report suitable for public CI logs.

    Locations and snippets are deliberately omitted: either can contain the
    very value this scanner is intended to keep out of build logs.
    """
    counts: dict[tuple[str, str], int] = {}
    for finding in findings:
        key = (finding.severity, finding.rule)
        counts[key] = counts.get(key, 0) + 1

    errors = sum(
        count for (severity, _), count in counts.items() if severity == "error"
    )
    warns = sum(
        count for (severity, _), count in counts.items() if severity == "warn"
    )
    blocking = errors + (warns if strict else 0)
    print(
        f"sensitive-scan: {errors} error(s), {warns} warning(s), "
        f"{len(findings)} total"
    )
    if counts:
        for (severity, rule), count in sorted(counts.items()):
            print(f"sensitive-scan: {severity} {rule} ({count})")
    if blocking:
        print(f"sensitive-scan: FAIL ({blocking} blocking finding(s))")
    else:
        print("sensitive-scan: PASS")


def normalize_path_filters(path_filters: list[str], root: str) -> list[str]:
    """Convert path arguments to repository-relative paths without leaking them."""
    normalized = []
    for raw in path_filters:
        candidate = (
            os.path.abspath(raw)
            if os.path.isabs(raw)
            else os.path.abspath(os.path.join(root, raw))
        )
        try:
            relative = os.path.relpath(candidate, root)
        except ValueError:
            die("--paths must identify files inside the repository")
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            die("--paths must identify files inside the repository")
        if relative == os.curdir:
            continue
        normalized.append(relative.replace(os.sep, "/"))
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sensitive-scan",
        description="Detect internal-only information before publishing.",
    )
    parser.add_argument("--history", action="store_true",
                        help="also scan author emails, commit messages, tags, "
                             "branch names, and files deleted from the worktree "
                             "but still present in HEAD")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as blocking")
    parser.add_argument("--ci-summary", action="store_true",
                        help="print aggregate-only output safe for CI logs")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--rule", action="append", dest="rules", metavar="ID",
                        help="only run this rule (repeatable)")
    parser.add_argument("--paths", nargs="*", default=[], metavar="PATH",
                        help="restrict the worktree scan to these paths")
    parser.add_argument("--list-rules", action="store_true",
                        help="print the rule table and exit")
    args = parser.parse_args()

    if args.list_rules:
        for rule in sensitive_rules.RULES:
            print(f"{rule['severity']:<5} {rule['id']:<28} {rule['title']}")
        return 0

    root = repo_root()
    os.chdir(root)
    rules = compile_rules(args.rules)
    allowlist = compile_allowlist()

    findings = scan_worktree(
        rules, allowlist, normalize_path_filters(args.paths, root)
    )
    if args.history:
        findings.extend(scan_history(rules, allowlist))

    if args.ci_summary:
        report_ci_summary(findings, args.strict)
    elif args.format == "json":
        errors = sum(1 for f in findings if f.severity == "error")
        warns = len(findings) - errors
        json.dump(
            {
                "summary": {"errors": errors, "warnings": warns,
                            "total": len(findings), "strict": args.strict},
                "findings": [asdict(f) for f in findings],
            },
            sys.stdout,
            indent=2,
            ensure_ascii=False,
        )
        print()
    else:
        report_text(findings, rules, args.strict)

    blocking = [f for f in findings
                if f.severity == "error" or (args.strict and f.severity == "warn")]
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
