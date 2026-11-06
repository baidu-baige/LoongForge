# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""TorchAO-specific Float8Linear conversion."""

from .linear_conversion import build_torchao_config, convert_linear_to_torchao

__all__ = [
    "build_torchao_config",
    "convert_linear_to_torchao",
]
