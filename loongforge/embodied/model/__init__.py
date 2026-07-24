# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied Model - Unified model building entry point."""


from loongforge.embodied.model.registry import MODEL_REGISTRY, register_model, build_model

__all__ = ["build_model", "register_model", "MODEL_REGISTRY"]
