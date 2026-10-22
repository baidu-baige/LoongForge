# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 VLA model package: configuration (GigaBrain07ModelConfig) and
model (GigaBrain07Policy).

Layer-A reshell: this package __init__ exposes the xvla/pi05-style names
(``GigaBrain07ModelConfig``, ``GigaBrain07Policy``) as a naming-alignment
surface, mirroring how ``model/xvla`` and ``model/pi05`` present their
config/model. The names are resolved **lazily** (PEP 562 ``__getattr__``) so
that merely importing a submodule (e.g. the slim config consumed by
``train/config_map.py``) does NOT trigger the heavy vendored model import —
keeping the training import chain and numerics byte-identical.

The real implementation is untouched and lives in the vendored package
``giga_models.models.vla.giga_brain_0.modeling_giga_brain_0`` (re-exported via
the naming-aligned ``modeling_giga_brain07`` module). Nothing here moves,
rewrites, or re-registers the model.
"""

from __future__ import annotations

__all__ = ["GigaBrain07ModelConfig", "GigaBrain07Policy", "from_pretrained"]


def __getattr__(name):
    # Lightweight, dependency-free dataclass — safe to resolve anywhere.
    if name == "GigaBrain07ModelConfig":
        from loongforge.embodied.model.giga_brain07.model_configuration_giga_brain07 import (
            GigaBrain07ModelConfig,
        )

        return GigaBrain07ModelConfig
    # Heavy: pulls the vendored model lib. Requires the giga PYTHONPATH
    # (model/giga_brain07 on sys.path) — the same one the trainer uses.
    if name in ("GigaBrain07Policy", "from_pretrained"):
        from loongforge.embodied.model.giga_brain07.modeling_giga_brain07 import (
            GigaBrain07Policy,
            from_pretrained,
        )

        return {"GigaBrain07Policy": GigaBrain07Policy, "from_pretrained": from_pretrained}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
