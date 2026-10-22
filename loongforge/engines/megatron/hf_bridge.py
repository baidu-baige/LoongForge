# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to the offline ``tools/mcore_bridge`` package.

HF checkpoint loading and export are engine checkpoint capabilities, but the
bridge is still an offline tool: it is not part of the installed package and it
imports the ``tools.mcore_checkpoint_convert`` tree.  Resolve it only when a run
actually loads HF weights or exports with ``--save-hf``.
"""

from __future__ import annotations

import os
import sys


def _ensure_tools_on_path() -> None:
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    for path in (repo_root, os.path.join(repo_root, "tools")):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_hf_checkpoint_online(*args, **kwargs):
    _ensure_tools_on_path()
    from mcore_bridge.checkpoint.hf_checkpoint_loader import (
        load_hf_checkpoint_online as _load,
    )

    return _load(*args, **kwargs)


def save_hf_checkpoint_online(*args, **kwargs):
    _ensure_tools_on_path()
    from mcore_bridge.checkpoint.hf_checkpoint_saver import (
        save_hf_checkpoint_online as _save,
    )

    return _save(*args, **kwargs)
