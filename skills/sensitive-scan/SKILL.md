---
name: sensitive-scan
description: >-
  Scan the repository for internal-only information before publishing or
  releasing: internal hostnames and package mirrors, internal container-image
  registries, private object-storage buckets, corporate email addresses,
  developer home directories, internal document references, credential
  material, RFC1918 addresses, and pre-rename internal project names. Use when
  the user asks to check for sensitive/internal information, prepare an
  open-source release, review a diff for leaks, or wire this check into CI.
---

# Sensitive information scan

Authoritative implementation: `ci/sensitive_scan.py`, rules in
`ci/sensitive_rules.py`. Always run the script rather than grepping by
hand — the rules encode findings from prior audits and the allowlist records
which matches are intentional.

## Running it

```bash
# Tracked + staged files. This is what CI runs on every PR.
python3 ci/sensitive_scan.py

# Add git metadata: author emails, commit messages, tag names, branch names,
# and files deleted from the worktree but still present in HEAD.
# ALWAYS use this before an actual release.
python3 ci/sensitive_scan.py --history

# Machine-readable output (contains every finding, not just the capped preview)
python3 ci/sensitive_scan.py --format json

# Narrow the scan
python3 ci/sensitive_scan.py --paths docker/ examples_xpu/
python3 ci/sensitive_scan.py --rule corp-email --rule internal-domain
python3 ci/sensitive_scan.py --list-rules
```

Exit codes: `0` clean, `1` blocking findings, `2` scanner could not run.

## Severity contract

- **error** — blocks CI. Names internal infrastructure, a person, or credential
  material. Fix these; do not allowlist them without a stated reason.
- **warn** — reported, does not block. Path and naming hygiene, high volume by
  nature. `--strict` promotes warnings to blocking; only use it on a tree that
  has already been cleaned, otherwise the cluster-path rule alone will fail.

The text report caps each rule at 20 examples and prints the true count. Use
`--format json` when you need the full list.

## Interpreting results

1. Read the `why` and `fix` lines the scanner prints — they say what the rule is
   actually protecting against.
2. Confirm each finding by reading the file. Do not act on the snippet alone.
3. `--history` findings **cannot be fixed by editing files**. Author emails, tag
   names, commit messages and deleted-but-committed files ship with the history.
   The remedies are squashing to a fresh initial commit, rewriting history, or
   deleting tags. Say so explicitly instead of proposing an edit.
4. A clean worktree scan does not mean the repository is clean. Content already
   deleted from the worktree still lives in `HEAD`; that is exactly what
   `--history` covers.

## Fixing findings

Prefer, in order:

1. Delete the reference if it serves no purpose externally.
2. Replace with a public equivalent (public registry, public mirror, public
   model URL).
3. Parameterize: `${VAR:-<generic default>}`. Generic defaults should be
   `/workspace/...` rather than a real cluster path.
4. Replace with an obvious placeholder (`bos:/path/to/artifact/`,
   `http://your-proxy-host:port`).

When deleting a file to remove a leak, check for dangling references first —
Sphinx `toctree` entries and README links break the docs build:

```bash
git ls-files -z | xargs -0 grep -nI "<deleted-basename>"
```

Prefer scrubbing a few lines over deleting a whole documented file.

## Suppressing a match

Inline, for one line:

```bash
BASE_IMAGE=example.internal/base:1  # sensitive-scan: allow[internal-domain]
```

`sensitive-scan: allow` with no brackets suppresses every rule on that line.

For a path or a recurring pattern, add an entry to `ALLOWLIST` in
`ci/sensitive_rules.py`. A `reason` is mandatory — the scanner exits with
code 2 if one is missing.

## Extending the rules

Add a dict to `RULES` in `ci/sensitive_rules.py` with `id`, `severity`,
`title`, `why`, `pattern`, `hint`. Set `"redact": True` for anything that could
match live credential material so the value is not reprinted in CI logs.

After changing a pattern, verify precision against real code before committing —
identifier-shaped false positives are the usual failure mode:

```bash
python3 ci/sensitive_scan.py --rule <new-rule-id>
```

## Scope boundary

This scanner finds internal *identifiers*. It is not a secret scanner — run
`gitleaks` alongside it for entropy-based credential detection. It also cannot
judge whether an unreleased model name, a chip codename, or a published
performance number is cleared for disclosure; those need a human decision.
