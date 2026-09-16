# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Config dtype string to torch.dtype, shared by models and training engines."""


def resolve_dtype(name):
    """Map a config dtype string to the matching torch dtype."""
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
