# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from FastWAM (https://github.com/yuantianyuan01/FastWAM).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gradient checkpointing helpers for FastWAM modules."""

import torch


def create_custom_forward(module):
    """Wrap a module call so checkpoint can pass positional and keyword inputs."""
    def custom_forward(*inputs, **kwargs):
        """Run the wrapped module with checkpoint-provided inputs."""
        return module(*inputs, **kwargs)

    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    *args,
    **kwargs,
):
    """Run a model forward call with optional non-reentrant checkpointing."""
    if use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
