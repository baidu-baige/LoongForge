# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""megatron local norm"""

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
import torch
import torch.nn as nn

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as ApexFusedRMSNorm

    HAVE_FUSED_RMS_NORM = True
except:
    HAVE_FUSED_RMS_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as ApexFusedLayerNorm

    HAVE_FUSED_LAYER_NORM = True
except:
    HAVE_FUSED_LAYER_NORM = False


class FusedRMSNorm(ApexFusedRMSNorm):
    """Fused RMS Norm"""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps=1e-5,
        elementwise_affine=True,
    ):

        if not HAVE_FUSED_RMS_NORM:
            # TODO: Add pytorch only rms norm
            raise ValueError(
                f"Apex must currently be installed to use FusedRMSNorm op."
            )

        super().__init__(hidden_size, eps=eps, elementwise_affine=elementwise_affine)

        self.config = config

        self.sequence_parallel = self.config.sequence_parallel

        # set sequence parallelism flag on weight parameters
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)


class LocalNorm:
    """
    A conditional wrapper to initialize an instance of Megatron Local `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine=True,
    ):
        if config.normalization == "Apex_LayerNorm":
            if elementwise_affine:
                instance = FusedLayerNorm(
                    config=config,
                    hidden_size=hidden_size,
                    eps=eps,
                )
            else:
                assert (
                    HAVE_FUSED_LAYER_NORM
                ), "Apex must currently be installed to use FusedLayerNorm op."
                instance = ApexFusedLayerNorm(
                    hidden_size,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                )
        elif config.normalization == "Apex_RMSNorm":
            instance = FusedRMSNorm(
                config=config,
                hidden_size=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        elif config.normalization == "Torch_RMSNorm":
            instance = RMSNorm(
                config=config,
                hidden_size=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        else:
            raise Exception("Only LayerNorm and RMSNorm are curently supported")

        return instance


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.

    RMSNorm is a simplified version of LayerNorm that focuses on the root mean square of inputs,
    omitting the mean-centering operation. This provides computational efficiency while maintaining
    good performance.

    """

    def __init__(self, config, hidden_size: int, eps=1e-5, elementwise_affine=False):
        """
        Initialize RMSNorm layer.

        Args:
            config (Ernie4_5_Config): Model configuration.
            hidden_size (int): Hidden size of the model.
            eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-5.
            elementwise_affine (bool, optional): If True, applies a learnable per-element affine transformation
                (i.e., scaling and bias) to the normalized output. Defaults to False.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.get_default_dtype())
        )
        self.variance_epsilon = eps

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

