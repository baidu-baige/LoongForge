# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Developer tooling for the embodied training stack.

Two entry points, neither of which is imported by training code:

- ``code_checks``: source checks the ``test_arch_style`` case enforces and
  that can also be pointed at an arbitrary path from the shell.
- ``arch_probe``: read-only dumps of what a world-model spec resolves to
  (checkpoint keys, entry-stage shapes, per-token timesteps), for use when a
  ``strict=True`` load or a parity harness starts disagreeing.
"""
