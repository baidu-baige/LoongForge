# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The three hand-written source rules, enforced instead of remembered.

``tools/embodied/code_checks.py`` holds the rules; this file points them at the
architecture package, its tests, and the tools themselves, and asserts zero
findings. Two of the three encode a standing project decision rather than a
style preference:

- comments and docstrings are English, while Chinese inside an error or log
  message is left alone, because tests match on those strings;
- an import is either used, re-exported through ``__all__``, or marked
  ``noqa`` to say it is kept for a registration side effect.

Half the cases here check the *checkers*. A source check that has quietly
stopped firing passes every file, which looks exactly like a clean tree, so
each rule is also handed a snippet it must reject.
"""

from __future__ import annotations

import pathlib

import pytest

from tools.embodied import code_checks

# ``pyproject.toml:282`` is the authority on line length for the repository.
LIMIT = code_checks.DEFAULT_LINE_LIMIT

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CHECKED = ("loongforge/embodied/model/arch",
           "tools/embodied")


def checked_paths() -> list[str]:
    """Return the directories plus the architecture test files themselves.

    The test files are listed by glob rather than by naming the directory,
    because ``tests/embodied/unit`` also holds the FP8 and optimizer cases,
    which predate these rules and are not in scope here.

    Returns:
        Absolute paths, directories included; ``code_checks.expand`` walks
        them.
    """
    here = pathlib.Path(__file__).parent
    out = [str(REPO_ROOT / path) for path in CHECKED]
    out.extend(str(path) for path in sorted(here.glob("test_arch_*.py")))
    out.append(str(here / "arch_parity.py"))
    return out


@pytest.mark.parametrize("rule", sorted(code_checks.RULES))
def test_the_architecture_sources_satisfy_every_rule(rule):
    """One case per rule, so a failure names which rule broke.

    Reporting every finding rather than the first keeps a single run enough
    to fix the tree.
    """
    findings = code_checks.check(checked_paths(), rules=[rule], limit=LIMIT)
    assert not findings, "\n".join(str(finding) for finding in findings)


def test_every_checked_path_exists_and_holds_python_files():
    """A path that stopped existing would make its rule pass vacuously."""
    expanded = code_checks.expand(checked_paths())
    assert len(expanded) > 60
    for path in CHECKED:
        assert (REPO_ROOT / path).is_dir(), path
        assert any(name.startswith(str(REPO_ROOT / path))
                   for name in expanded), path


def test_chinese_in_a_comment_is_a_finding_but_in_a_message_is_not(tmp_path):
    """The rule's whole point is the distinction between the two.

    The comment and the docstring are findings; the ``ValueError`` text is
    not, because callers and tests match on that string.
    """
    source = tmp_path / "sample.py"
    source.write_text(
        '"""\u6587\u6863."""\n\n\n'
        'def f():\n'
        '    # \u6ce8\u91ca\n'
        '    raise ValueError("\u9519\u8bef")\n',
        encoding="utf-8")
    findings = code_checks.check([str(source)], rules=["comment_language"])
    assert [finding.line for finding in findings] == [1, 5]


def test_a_noqa_import_and_an_all_entry_both_count_as_used(tmp_path):
    """Three imports, one genuinely unused, and only that one reported."""
    source = tmp_path / "sample.py"
    source.write_text(
        'import json\n'
        'import os  # noqa: F401  registers a codec\n'
        'import sys\n\n'
        '__all__ = ["json"]\n',
        encoding="utf-8")
    findings = code_checks.check([str(source)], rules=["unused_imports"])
    assert [(finding.line, finding.detail) for finding in findings] == [
        (3, "sys is never used")]


def test_a_multi_line_decorator_does_not_fake_a_blank_line_violation(tmp_path):
    """The reason the blank-line rule reads the AST and not the line above.

    ``second`` sits directly under a decorator's closing bracket, which every
    textual form of this rule reports as missing blank lines. ``third`` is a
    real violation and must still be caught.
    """
    source = tmp_path / "sample.py"
    source.write_text(
        'import pytest\n\n\n'
        '@pytest.mark.parametrize(\n'
        '    "x", [1, 2]\n'
        ')\n'
        'def second(x):\n'
        '    return x\n'
        'def third():\n'
        '    return 0\n',
        encoding="utf-8")
    findings = code_checks.check([str(source)], rules=["code_style"])
    assert [finding.line for finding in findings] == [9]


def test_the_line_limit_is_the_argument_and_not_a_constant(tmp_path):
    """Same file, two limits, two answers.

    The architecture package is held to the repository limit, while a subtree
    written to a narrower convention can be checked at that convention
    without editing the rule.
    """
    source = tmp_path / "sample.py"
    source.write_text(f'X = "{"a" * 90}"\n', encoding="utf-8")
    assert not code_checks.check([str(source)], rules=["code_style"],
                                 limit=120)
    narrow = code_checks.check([str(source)], rules=["code_style"], limit=79)
    assert [finding.line for finding in narrow] == [1]


def test_an_unknown_rule_name_is_refused_rather_than_skipped():
    """Mistyping a rule must not silently check nothing."""
    with pytest.raises(KeyError, match="unknown rule"):
        code_checks.check(checked_paths(), rules=["style"])
