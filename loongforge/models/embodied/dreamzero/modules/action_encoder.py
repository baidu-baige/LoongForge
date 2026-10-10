# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
### Implemented based on Pi0 Action Encoding
#
# Multi-embodiment action encoder.
# Helper functions are inlined here to keep runtime imports local.
#
# Pure PyTorch (no TE rewrite). One forward pass per training step routes
# each embodiment's action tensor through its own slice of the
# CategorySpecificLinear weights via cat_ids.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Inlined action-encoder helpers.
# ----------------------------------------------------------------------
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


# ----------------------------------------------------------------------
# Category-specific (per-embodiment) primitives
# ----------------------------------------------------------------------
class CategorySpecificLinear(nn.Module):
    """A per-category Linear: each `cat_id` selects its own (W, b) slice.

    Stored as `(num_categories, in, out)` weight + `(num_categories, out)` bias.
    Forward uses `bmm` for batched per-sample dispatch.
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer category-specific MLP with ReLU between layers."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Embodiment-routed action encoder used by CausalWanModel.

    For each (action, timestep, cat_id) triplet:
      a_emb  = W1[cat_id] @ action               # (B,T,w)
      tau    = sinusoidal_pos_enc(timestep)      # (B,T,w)
      x      = swish(W2[cat_id] @ cat([a_emb, tau]))  # (B,T,w)
      out    = W3[cat_id] @ x                    # (B,T,w)
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size
        )
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


__all__ = [
    "swish",
    "SinusoidalPositionalEncoding",
    "CategorySpecificLinear",
    "CategorySpecificMLP",
    "MultiEmbodimentActionEncoder",
]
