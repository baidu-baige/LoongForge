# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiMo-specific MTP layer override."""

import torch

from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.utils import make_viewless_tensor


class MimoMultiTokenPredictionLayer(MultiTokenPredictionLayer):
    """MiMo MTP layer with hidden-first concat before eh_proj."""

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """Concatenate [hidden, embed] before projection."""
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # MiMo official implementation uses hidden-first concatenation.
        hidden_states = torch.cat((hidden_states, decoder_input), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        return hidden_states
