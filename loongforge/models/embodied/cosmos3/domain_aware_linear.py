# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain-aware linear layer for multi-embodiment robot learning.

This module provides a linear layer with domain-conditioned parameters,
where each domain (embodiment) has its own weight and bias vectors.

Based on the X-VLA implementation:
https://github.com/2toinf/X-VLA/blob/main/models/transformer.py
"""

import torch
from torch import nn


class DomainAwareLinear(nn.Module):
    """Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    During forward pass, weights are retrieved based on per-sample domain IDs.

    This enables learning domain-specific transformations for different robot
    embodiments while sharing the overall model architecture.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 50) -> None:
        """Initialize the domain-aware linear layer.

        Args:
            input_size: Dimension of input features.
            output_size: Dimension of output features.
            num_domains: Number of domains (embodiments) to support.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains

        # Store per-domain weights as embeddings: [num_domains, output_size * input_size]
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        # Store per-domain biases as embeddings: [num_domains, output_size]
        self.bias = nn.Embedding(num_domains, output_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """Forward pass with domain-specific weights.

        Args:
            x: Input tensor of shape [B, I] or [B, T, I] where B is batch size,
               T is sequence length, and I is input_size.
            domain_id: Domain indices of shape [B], one per sample in the batch.

        Returns:
            Output tensor of shape [B, O] or [B, T, O] where O is output_size.
        """
        B = domain_id.shape[0]

        # Retrieve per-sample weights: [B, input_size, output_size]
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)  # [B,input_size,output_size]

        # Retrieve per-sample biases: [B, output_size]
        b = self.bias(domain_id).view(B, self.output_size)  # [B,output_size]

        if x.dim() == 2:
            # 2D input: [B, I] @ [B, I, O] -> [B, O]
            return (
                torch.bmm(x.unsqueeze(1), W).squeeze(1) + b
            )  # [B,1,input_size] @ [B,input_size,output_size] -> [B,output_size]
        else:
            # 3D input: [B, T, I] @ [B, I, O] -> [B, T, O]
            # Bias [B, O] -> [B, 1, O] for broadcasting
            return torch.bmm(x, W) + b.unsqueeze(
                1
            )  # [B,T,input_size] @ [B,input_size,output_size] -> [B,T,output_size]
