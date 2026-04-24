# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Patch Merger MLP Adapter"""
import torch
from typing import Union
from megatron.core.transformer.spec_utils import build_module
from loongforge.models.common import BaseMegatronModule
from loongforge.models.utils import import_module
from .moon_vision_config import PatchMergerMLPAdapterConfig


class PatchMergerMLP(BaseMegatronModule):
    """
    PatchMergerMLP is a MLP layer that merges patches of the same spatial location.
    """
    config_class = PatchMergerMLPAdapterConfig

    def __init__(
        self,
        config: PatchMergerMLPAdapterConfig,
        input_size: int,
        output_size: int,
        spatial_merge_size: Union[int, tuple] = 2,
        use_postshuffle_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        # Support both single int and tuple (kh, kw) for merge kernel size
        if isinstance(spatial_merge_size, (list, tuple)):
            self.merge_h, self.merge_w = spatial_merge_size
        else:
            self.merge_h = self.merge_w = spatial_merge_size
        self.hidden_size = input_size * self.merge_h * self.merge_w
        self.use_postshuffle_norm = use_postshuffle_norm

        if self.config.model_spec is None:
            model_spec = [
                "loongforge.models.encoder.qwen2_vl_vision_models.qwen2_vl_layer_spec",
                "get_adapeter_layer_with_te_spec",
            ]
        else:
            model_spec = self.config.model_spec
        submodules = import_module(model_spec, self.config)

        self.layernorm = build_module(
            submodules.layernorm,
            config=self.config,
            hidden_size=self.hidden_size if self.use_postshuffle_norm else input_size,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def forward(
        self, x: Union[torch.Tensor, list, tuple], window_index: torch.LongTensor = None
    ) -> torch.Tensor:
        """Forward pass."""

        # Support list/tuple input
        if isinstance(x, (list, tuple)):
            return [
                self._forward_single(item, window_index)
                for item in x
            ]

        return self._forward_single(x, window_index)

    def _forward_single(
        self, x: torch.Tensor, window_index: torch.LongTensor = None
    ) -> torch.Tensor:
        """Forward pass for single tensor."""
        B = x.shape[0]

        # Apply layernorm
        if self.use_postshuffle_norm:
            x = self.layernorm(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
        else:
            x = self.layernorm(x).view(B, -1, self.hidden_size)

        # MLP forward
        x, _ = self.linear_fc1(x)
        x = self.activation_func(x)
        x, _ = self.linear_fc2(x)

        # Restore order if needed
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            x = x[reverse_indices, :].contiguous()

        return x
