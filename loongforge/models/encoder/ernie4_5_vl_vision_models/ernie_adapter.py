# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from ERNIE (https://github.com/PaddlePaddle/ERNIE/)
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Ernie-VL adapter"""

import torch
import torch.nn as nn
from copy import deepcopy
from loongforge.models.common import BaseMegatronVisionModule
from loongforge.utils import get_args


class UniqueNameGuard:
    """name guard"""

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.counter = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_unique_name(self, name):
        """get unique name"""
        if name not in self.counter:
            self.counter[name] = 0
        else:
            self.counter[name] += 1
        return f"{self.prefix}{name}_{self.counter[name]}"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.

    RMSNorm is a simplified version of LayerNorm that focuses on the root mean square of inputs,
    omitting the mean-centering operation. This provides computational efficiency while maintaining
    good performance.

    """

    def __init__(self, hidden_size, rms_norm_eps=1e-05):
        """
        Initialize RMSNorm layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.get_default_dtype())
        )
        self.variance_epsilon = rms_norm_eps

    def forward(self, hidden_states):
        """
        Apply RMS normalization to input hidden states.

        Args:
            hidden_states (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Normalized output tensor of same shape as input

        Note:
            - computes RMSNorm manually:
                1. Compute variance of features
                2. Apply reciprocal square root normalization
                3. Scale by learned weight parameter
            - Maintains original dtype for numerical stability during computation
        """
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = torch.rsqrt(variance + self.variance_epsilon) * hidden_states
        return hidden_states.to(self.weight.dtype) * self.weight


class ErnieAdapter(BaseMegatronVisionModule):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, config, input_size, output_size):
        super().__init__(config)
        self.in_dim = input_size
        self.out_dim = output_size
        args = get_args()
        self.out_dtype = torch.bfloat16 if args.bf16 else torch.float16
        # using unique name space start with "mm_resampler_"
        with UniqueNameGuard("mm_resampler_") as guard:
            self.mlp = nn.Linear(self.in_dim, self.out_dim)
            out_config = deepcopy(config)
            out_config.hidden_size = self.out_dim
            self.after_norm = RMSNorm(out_config.hidden_size, out_config.rms_norm_eps)

    def forward(self, x, window_index=None, **kwargs):
        """
        x: image_features
        """
        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            return x

        image_features = fwd_mlp(x)

        if image_features.dim == 2:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C]).to(self.out_dtype)

        return image_features


