# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Hugging Face Transformers MiniCPM-V-4.6 under the Apache-2.0 License.
# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
#
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

"""MiniCPM-V-4.6 final visual merger."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from megatron.core.transformer.spec_utils import build_module

from loongforge.models.common import BaseMegatronModule
from loongforge.models.dispatch import multiacc_modules

from .minicpm_v_4_6_config import MiniCPMV46MergerConfig


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Ignore Transformer Engine FP8 metadata absent from HF-derived checkpoints."""
    del module
    for keys in incompatible_keys._asdict().values():
        for key in keys[::-1]:
            if key.endswith("._extra_state"):
                keys.remove(key)


class MiniCPMV46DownsampleMLP(nn.Module):
    """2x2 downsample MLP used by MiniCPM visual merger."""

    def __init__(self, config: MiniCPMV46MergerConfig, hidden_size: int, llm_embed_dim: int):
        super().__init__()
        self.config = config
        merged_hidden_size = hidden_size * 4
        self.pre_norm = build_module(
            multiacc_modules.TENorm,
            config=config,
            hidden_size=merged_hidden_size,
            eps=config.layernorm_epsilon,
        )
        linear_kwargs = {
            "config": config,
            "bias": config.add_bias_linear,
            "skip_bias_add": False,
            "parallel_mode": None,
            "skip_weight_param_allocation": False,
        }
        self.linear_1 = build_module(
            multiacc_modules.TELinear,
            merged_hidden_size,
            merged_hidden_size,
            init_method=config.init_method,
            **linear_kwargs,
        )
        self.linear_2 = build_module(
            multiacc_modules.TELinear,
            merged_hidden_size,
            llm_embed_dim,
            init_method=config.output_layer_init_method,
            **linear_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states).view(-1, self.linear_1.in_features)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.config.activation_func(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniCPMV46Merger(BaseMegatronModule):
    """MiniCPM visual merger from vision hidden size to LLM hidden size."""

    config_class = MiniCPMV46MergerConfig
    def __init__(
        self,
        config: MiniCPMV46MergerConfig,
        input_size: int,
        output_size: int,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__(config=config)
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.merger_times = config.merger_times
        mlps = [
            MiniCPMV46DownsampleMLP(config, input_size, input_size)
            for _ in range(self.merger_times - 1)
        ]
        mlps.append(MiniCPMV46DownsampleMLP(config, input_size, output_size))
        self.mlp = nn.ModuleList(mlps)
        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)
        if getattr(config, "freeze", False):
            self.freeze()

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sizes: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        if target_sizes is None:
            if image_grid_thw is None:
                raise ValueError("MiniCPM merger requires image_grid_thw or target_sizes.")
            target_sizes = image_grid_thw[:, 1:].to(dtype=torch.long)
        if hidden_states.dim() == 3:
            if hidden_states.shape[0] != 1:
                raise ValueError(
                    "MiniCPM merger expects packed vision features with shape [S, H] "
                    f"or [1, S, H], got {tuple(hidden_states.shape)}."
                )
            hidden_states = hidden_states[0]
        elif hidden_states.dim() != 2:
            raise ValueError(
                "MiniCPM merger expects packed vision features with shape [S, H] "
                f"or [1, S, H], got {tuple(hidden_states.shape)}."
            )

        merge_h, merge_w = self.merge_kernel_size

        start = 0
        processed_features = []
        for target_size in target_sizes:
            height, width = target_size
            height, width = int(height.item()), int(width.item())
            if height % merge_h != 0 or width % merge_w != 0:
                raise ValueError(
                    f"Patch grid ({height}, {width}) must be divisible by merge kernel size "
                    f"{self.merge_kernel_size}."
                )
            num_patches = height * width
            embed_dim = hidden_states.shape[-1]
            merged_h, merged_w = height // merge_h, width // merge_w
            hidden_state = (
                hidden_states[start: start + num_patches, :]
                .view(merged_h, merge_h, merged_w, merge_w, embed_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(merged_h * merged_w, merge_h * merge_w * embed_dim)
            )
            hidden_state = self.mlp[0](hidden_state)

            for i in range(1, self.merger_times):
                if height % merge_h != 0 or width % merge_w != 0:
                    raise ValueError(
                        f"Patch grid ({height}, {width}) must be divisible by merge kernel size "
                        f"{self.merge_kernel_size} at merge round {i}."
                    )
                height = height // merge_h
                width = width // merge_w

                inner_dim = hidden_state.shape[-1]
                merged_h, merged_w = height // merge_h, width // merge_w
                hidden_state = (
                    hidden_state.view(merged_h, merge_h, merged_w, merge_w, inner_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(merged_h * merged_w, merge_h * merge_w * inner_dim)
                )
                hidden_state = self.mlp[i](hidden_state)

            start += num_patches
            processed_features.append(hidden_state)
        if start != hidden_states.shape[0]:
            raise ValueError(
                f"MiniCPM merger received {hidden_states.shape[0]} vision tokens, "
                f"but target_sizes describes {start}."
            )
        return processed_features
