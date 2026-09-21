# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero (Wan-DiT-based Video-Language-Action) model package."""


def __getattr__(name: str):
    if name == "DreamZeroConfig":
        from .model_configuration_dreamzero import DreamZeroConfig

        return DreamZeroConfig
    if name == "DreamZeroPolicy":
        from .modeling_dreamzero import DreamZeroPolicy

        return DreamZeroPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DreamZeroConfig",
    "DreamZeroPolicy",
]
