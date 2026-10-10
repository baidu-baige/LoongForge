# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Scan loongforge/ for undefined names using pyflakes."""

import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def pyflakes_output():
    """Run pyflakes on loongforge/ and return the output."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", "loongforge"],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        pytest.skip("pyflakes not installed")
    return result.stdout + result.stderr


def test_no_undefined_names(pyflakes_output):
    """Ensure no 'undefined name' warnings in loongforge/."""
    # Match "undefined name 'x'" only; "import *" warnings also contain "undefined names".
    lines = [l for l in pyflakes_output.splitlines() if "undefined name '" in l]
    if lines:
        msg = f"{len(lines)} undefined name(s):\n" + "\n".join(lines)
        pytest.fail(msg)
