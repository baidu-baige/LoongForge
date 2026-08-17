# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Test helper tools.

Intentionally empty: the task registry lives in `tasks/__init__.py` (SUPPORTED_TASKS).
Keeping this module import-free lets lightweight consumers (e.g. the embodied suite in
`tests/embodied/`) import `tools.color_logger` without pulling in torch and the whole
E2E task stack.
"""
