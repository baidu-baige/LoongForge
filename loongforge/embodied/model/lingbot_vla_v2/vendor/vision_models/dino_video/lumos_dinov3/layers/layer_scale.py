"""Vendor module implementation."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Vendor interface."""

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        """Vendor interface."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        """Vendor interface."""
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        """Vendor interface."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
