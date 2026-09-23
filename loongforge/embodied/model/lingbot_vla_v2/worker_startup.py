# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Transformers-5.x startup compat for spawn-launched DataLoader workers.

The lingbot_vla_v2 model body is consumed from the upstream ``lingbotvla``
package. When the data pipeline runs with ``spawn`` workers, each worker is a
fresh interpreter that unpickles the dataset/collator; that unpickling triggers
``import lingbotvla.models`` -> ``lingbotvla.models.loader``, whose top-level
``from transformers import ... AutoModelForVision2Seq ...`` fails on Transformers
5.x (the symbol was renamed). The adapter's ``_ensure_transformers5_compat`` only
runs in the *main* process, so the worker dies before it ever runs.

We arrange for :func:`apply` to run at worker interpreter startup, before any
unpickling, via a ``sitecustomize`` hook (see ``spawn_bootstrap``). All the
model-specific logic lives here in the model package; the bootstrap directory
holds only the generic ``sitecustomize`` entry point.
"""

import os

# Marker the main process sets (and spawn children inherit) to gate the hook.
_SPAWN_COMPAT_ENV = "LOONGFORGE_SPAWN_WORKER_COMPAT"


def arm_spawn_workers() -> None:
    """From the main process, arm spawn DataLoader workers to run :func:`apply`.

    Puts the ``spawn_bootstrap`` directory (which holds ``sitecustomize.py``) on
    ``PYTHONPATH`` and sets the gate marker. Both are copied into ``os.environ``
    of every ``spawn``-started child, so the child's interpreter finds and runs
    our ``sitecustomize`` at startup. Idempotent; safe to call more than once.
    """
    here = os.path.dirname(os.path.abspath(__file__))  # .../lingbot_vla_v2
    bootstrap = os.path.join(here, "spawn_bootstrap")
    existing = os.environ.get("PYTHONPATH", "")
    if bootstrap not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            bootstrap + os.pathsep + existing if existing else bootstrap
        )
    os.environ[_SPAWN_COMPAT_ENV] = "1"


def apply() -> None:
    """Install the Transformers-5.x compat in the current (worker) interpreter.

    Idempotent and self-contained: reuses the adapter's compat helpers so the
    aliasing/guards stay defined in exactly one place.
    """
    from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import (
        _ensure_transformers5_compat,
        _import_upstream_data_guarded,
    )

    # 1. Re-expose Transformers-4.x names so the upstream loader import succeeds.
    _ensure_transformers5_compat()

    # 2. Pre-import the upstream data module that monkeypatches datasets, then
    #    restore the native ``generate_from_dict`` on datasets>=4.
    def _preimport_upstream_data():
        import lingbotvla.data.dataset  # noqa: F401
        return None

    _import_upstream_data_guarded(_preimport_upstream_data)
