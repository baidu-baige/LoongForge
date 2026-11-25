# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Source checks for hand-written rules that no installed linter enforces.

Three rules, each of which was found by hand often enough to be worth
automating:

1. ``comment_language`` -- Chinese in a comment or a docstring. Chinese inside
   any *other* string literal is left alone on purpose: those are error and
   log messages that callers and tests match on, so rewriting them is a
   behaviour change rather than a comment edit.
2. ``unused_imports`` -- an imported name that no code node references.
   ``__all__`` entries count as references (a re-exporting ``__init__`` is the
   normal case) and an import line carrying ``noqa`` is skipped (an import
   kept for its registration side effect is the normal case).
3. ``code_style`` -- over-long lines, trailing whitespace, and missing blank
   lines before a top-level definition. The blank-line rule is computed from
   the AST rather than from a regex over the previous line, because a
   multi-line decorator makes every textual version of that rule wrong.

Usage::

    PYTHONPATH=. python -m tools.embodied.code_checks loongforge/embodied
    PYTHONPATH=. python -m tools.embodied.code_checks \
        --rule code_style --limit 79 tools/embodied

The exit status is the number of findings, capped at 125, so the module is
usable as a shell gate.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# ``pyproject.toml:282`` sets ``line-length = 120`` for the whole repository.
# Pass ``--limit`` to check a subtree written to a narrower local convention.
DEFAULT_LINE_LIMIT = 120

_CJK = re.compile(r"[\u4e00-\u9fff]")

_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_DOCSTRING_OWNER = (ast.Module, ast.ClassDef, ast.FunctionDef,
                    ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Finding:
    """One rule violation, addressed the way an editor can jump to it.

    Attributes:
        path: Path as it was given on the command line.
        line: 1-based line number.
        rule: Name of the rule that fired.
        detail: What is wrong, short enough to read in a list.
    """

    path: str
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Return every line occupied by a module, class, or function docstring.

    Args:
        tree: Parsed module.

    Returns:
        The set of 1-based line numbers.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNER):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            end = first.end_lineno or first.lineno
            out.update(range(first.lineno, end + 1))
    return out


def _literal_lines(tree: ast.AST, docstrings: set[int]) -> set[int]:
    """Return every line occupied by a string literal that is not a docstring.

    Args:
        tree: Parsed module.
        docstrings: Output of ``_docstring_lines``, whose literals are
            comment-like and so must not be excused.

    Returns:
        The set of 1-based line numbers.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Constant, ast.JoinedStr)):
            continue
        if not hasattr(node, "lineno") or node.lineno in docstrings:
            continue
        end = node.end_lineno or node.lineno
        out.update(range(node.lineno, end + 1))
    return out


def comment_language(path: str, src: str) -> list[Finding]:
    """Report Chinese characters outside a runtime string literal.

    Args:
        path: Path used in the findings.
        src: File contents.

    Returns:
        One finding per offending line.
    """
    tree = ast.parse(src, path)
    docstrings = _docstring_lines(tree)
    literals = _literal_lines(tree, docstrings)
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        if not _CJK.search(line):
            continue
        if i in literals and i not in docstrings:
            continue
        out.append(Finding(path, i, "comment_language",
                           f"Chinese in a comment or docstring: "
                           f"{line.strip()[:72]}"))
    return out


def _exported_names(tree: ast.AST) -> set[str]:
    """Return the string entries of every ``__all__`` assignment.

    Args:
        tree: Parsed module.

    Returns:
        The exported names.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__"
                   for target in node.targets):
            continue
        for element in getattr(node.value, "elts", ()):
            if isinstance(element, ast.Constant) and isinstance(
                    element.value, str):
                out.add(element.value)
    return out


def unused_imports(path: str, src: str) -> list[Finding]:
    """Report imported names that no code node references.

    Args:
        path: Path used in the findings.
        src: File contents.

    Returns:
        One finding per unused import, ordered by line.
    """
    lines = src.split("\n")
    tree = ast.parse(src, path)
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        end = node.end_lineno or node.lineno
        if any("noqa" in lines[i - 1] for i in range(node.lineno, end + 1)):
            continue
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            if name != "annotations":
                imported[name] = node.lineno
    used = {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree)
             if isinstance(node, ast.Attribute)}
    used |= {node.value.id for node in ast.walk(tree)
             if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name)}
    used |= _exported_names(tree)
    return [Finding(path, line, "unused_imports", f"{name} is never used")
            for line, name in sorted((line, name)
                                     for name, line in imported.items()
                                     if name not in used)]


def code_style(path: str, src: str, *,
               limit: int = DEFAULT_LINE_LIMIT) -> list[Finding]:
    """Report over-long lines, trailing whitespace, and missing blank lines.

    The blank-line rule reads the first decorator's line number from the AST,
    so a definition behind a multi-line decorator is judged by where that
    decorator starts. Every textual form of the rule gets this wrong, because
    the line above such a definition is the decorator's closing bracket.

    Args:
        path: Path used in the findings.
        src: File contents.
        limit: Maximum line length, inclusive.

    Returns:
        One finding per violation, in line order.
    """
    lines = src.split("\n")
    out = []
    for i, line in enumerate(lines, 1):
        if len(line) > limit:
            out.append(Finding(path, i, "code_style",
                               f"line is {len(line)} columns, over {limit}"))
        if line != line.rstrip():
            out.append(Finding(path, i, "code_style", "trailing whitespace"))
    for node in ast.parse(src, path).body:
        if not isinstance(node, _DEFINITION):
            continue
        start = min([node.lineno]
                    + [d.lineno for d in node.decorator_list])
        if start > 3 and lines[start - 3:start - 1] != ["", ""]:
            out.append(Finding(path, start, "code_style",
                               "top-level definition needs two blank lines "
                               "above it"))
    return sorted(out, key=lambda finding: finding.line)


RULES = {"comment_language": comment_language,
         "unused_imports": unused_imports,
         "code_style": code_style}


def expand(paths: Iterable[str]) -> list[str]:
    """Expand directories into the ``.py`` files below them.

    Args:
        paths: Files or directories.

    Returns:
        Sorted paths, with each directory replaced by its Python files.
    """
    out: list[str] = []
    for raw in paths:
        path = pathlib.Path(raw)
        if path.is_dir():
            out.extend(str(child) for child in sorted(path.rglob("*.py")))
        else:
            out.append(str(path))
    return out


def check(paths: Iterable[str], *, rules: Sequence[str] = (),
          limit: int = DEFAULT_LINE_LIMIT) -> list[Finding]:
    """Run the selected rules over the selected files.

    Args:
        paths: Files or directories; directories are expanded.
        rules: Rule names to run, defaulting to all of them.
        limit: Line-length limit handed to ``code_style``.

    Returns:
        Every finding, grouped by file in path order.

    Raises:
        KeyError: A requested rule does not exist.
    """
    selected = tuple(rules) or tuple(RULES)
    unknown = [name for name in selected if name not in RULES]
    if unknown:
        raise KeyError(f"unknown rule(s) {unknown}; known rules are "
                       f"{sorted(RULES)}")
    out: list[Finding] = []
    for path in expand(paths):
        src = pathlib.Path(path).read_text(encoding="utf-8")
        for name in selected:
            rule = RULES[name]
            out.extend(rule(path, src, limit=limit)
                       if rule is code_style else rule(path, src))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Print every finding and return a shell-usable exit status.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        The number of findings, capped at 125 so it survives as an exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+",
                        help="files or directories to check")
    parser.add_argument("--rule", action="append", default=[],
                        choices=sorted(RULES),
                        help="run only this rule; repeatable")
    parser.add_argument("--limit", type=int, default=DEFAULT_LINE_LIMIT,
                        help=f"line-length limit "
                             f"(default {DEFAULT_LINE_LIMIT})")
    args = parser.parse_args(argv)
    findings = check(args.paths, rules=args.rule, limit=args.limit)
    for finding in findings:
        print(finding)
    print(f"--- {len(findings)} finding(s)")
    return min(len(findings), 125)


if __name__ == "__main__":
    sys.exit(main())
