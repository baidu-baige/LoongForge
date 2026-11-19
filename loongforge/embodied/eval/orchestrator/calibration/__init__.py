# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parallel-configuration calibration for the evaluation orchestrator.

The package groups the three pieces that only exist to pick ``N`` (sim workers
per GPU) and cache the decision:

- :mod:`.calibrate` -- the sweep driver (``python -m ...orchestrator.calibration.calibrate``)
- :mod:`.config`    -- validate a proposed parallel config and persist the cache
- :mod:`.report`    -- build ``report.json`` / ``report.md`` / ``env.json``

Nothing here is imported by the runtime evaluation path; the sweep drives
``orchestrator.run`` as a subprocess exactly like a user would.
"""

from loongforge.embodied.eval.orchestrator.calibration.config import (  # noqa: F401
    calibration_key,
    save_calibration,
    validate_parallel_config,
)
from loongforge.embodied.eval.orchestrator.calibration.report import (  # noqa: F401
    build_report,
    render_markdown,
)

__all__ = [
    "build_report",
    "calibration_key",
    "render_markdown",
    "save_calibration",
    "validate_parallel_config",
]
