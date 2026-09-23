# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Generic ``sitecustomize`` entry point for spawn-launched DataLoader workers.
#
# This directory holds ONLY this file and is intentionally NOT a Python package
# (no ``__init__.py``). The lingbot_vla_v2 data pipeline puts this directory on
# ``PYTHONPATH`` (see ``worker_startup.arm_spawn_workers``) so CPython's
# ``site`` machinery imports this module at the start of every spawn worker
# interpreter -- before it unpickles any upstream object. The real work lives in
# the model package (``worker_startup.apply``); this file is a thin,
# gated dispatcher and a strict no-op in any process that did not set the marker.
import os

if os.environ.get("LOONGFORGE_SPAWN_WORKER_COMPAT") == "1":
    try:
        from loongforge.embodied.model.lingbot_vla_v2.worker_startup import (
            apply,
        )

        apply()
    except Exception:
        # Never let startup hardening break interpreter bring-up.
        pass
