# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check that every ``loongforge.*`` import and dotted path points at real code.

Runs without torch: it only parses files, so it catches stale paths left by
directory moves (imports, Hydra ``_target_``, registry strings).
"""

import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOTTED = re.compile(r"\bloongforge(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _module_file(parts):
    base = ROOT.joinpath(*parts)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return base if base.is_dir() else None  # namespace package


def _defines(module_file, name):
    if module_file.is_dir():
        return _module_file(module_file.relative_to(ROOT).parts + (name,)) is not None
    if _module_file(module_file.parent.relative_to(ROOT).parts + (name,)) and module_file.name == "__init__.py":
        return True  # submodule of a package
    tree = ast.parse(module_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            base = module_file.parent.relative_to(ROOT).parts
            if node.level:
                base = base[: len(base) - node.level + 1]
                star = _module_file(base + tuple((node.module or "").split(".")))
            else:
                star = _module_file(tuple(node.module.split(".")))
            if star is not None and star != module_file and _defines(star, name):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store):
            return True
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == name:
            return True
        if isinstance(node, ast.Constant) and node.value == name:
            return True  # lazy ``__getattr__`` export tables
    return False


def resolve(dotted):
    """Return True if ``dotted`` names a module, or an attribute of a module."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        module_file = _module_file(parts[:cut])
        if module_file is None:
            continue
        rest = parts[cut:]
        return not rest or _defines(module_file, rest[0])
    return False


def _python_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = path.relative_to(ROOT).parent.parts
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("loongforge."):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Resolve ``from ..x import y`` against the file's package.
                base = pkg[: len(pkg) - node.level + 1]
                module = ".".join(base + tuple(filter(None, (node.module or "").split("."))))
            else:
                module = node.module or ""
            if not module.startswith("loongforge"):
                continue
            for alias in node.names:
                yield node.lineno, module if alias.name == "*" else f"{module}.{alias.name}"


def _is_commented(path, lineno):
    """Return True if the import at lineno is inside a comment."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lineno <= len(lines):
        line = lines[lineno - 1].lstrip()
        return line.startswith("#")
    return False


def _files(*patterns):
    for pattern in patterns:
        yield from (p for p in ROOT.glob(pattern) if "__pycache__" not in p.parts)


class ImportTargetsTest(unittest.TestCase):
    def test_python_imports_resolve(self):
        missing = []
        for path in _files("loongforge/**/*.py", "tools/**/*.py", "tests/**/*.py"):
            for lineno, dotted in _python_imports(path):
                if not _is_commented(path, lineno):
                    if not resolve(dotted):
                        missing.append(f"{path.relative_to(ROOT)}:{lineno}: {dotted}")
        self.assertEqual(missing, [])

    def test_dotted_strings_resolve(self):
        missing = []
        for path in _files("loongforge/**/*.py", "configs/**/*.yaml", "tools/**/*.py", "examples/**/*.sh"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                for dotted in DOTTED.findall(line):
                    if not resolve(dotted):
                        missing.append(f"{path.relative_to(ROOT)}:{lineno}: {dotted}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
