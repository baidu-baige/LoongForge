# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X (https://github.com/X-Square-Robot/wall-x)
# under the Apache-2.0 License.

"""MoE routing operators (permute/unpermute).

These operators reorder tokens by expert assignment for efficient
Mixture-of-Experts processing.
"""

import logging

import torch

from wall_oss_05_op.base import OpsProxy

logger = logging.getLogger(__name__)


class PermuteOp(OpsProxy):
    """Reorder tokens by expert assignment for MoE processing.

    Signature: permute(tokens, indices, num_out_tokens=None, max_token_num=0) -> (permuted_tokens, sorted_indices)
    """

    def _get_cuda_kernel(self):
        """Return the CUDA-backed MoE permutation function if available."""
        try:
            from wall_oss_05_op._cuda_ext import is_available
            from wall_oss_05_op._cuda_wrappers import permute_kernel

            if not is_available():
                return None
            return permute_kernel
        except ImportError:
            return None
        except Exception as e:
            logger.warning("PermuteOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, tokens, indices, num_out_tokens=None, max_token_num=0):
        """Permute tokens by stable expert-index order with PyTorch."""
        del max_token_num  # unused, kept for API compatibility
        if indices.dim() == 1:
            indices = indices.view(-1, 1)
        expand_factor = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
        if num_out_tokens is not None:
            permuted_tokens = permuted_tokens[:num_out_tokens]
            sorted_indices = sorted_indices[:num_out_tokens]
        return permuted_tokens, sorted_indices


class UnpermuteOp(OpsProxy):
    """Restore tokens to original order after MoE processing.

    Signature: unpermute(permuted_tokens, sorted_indices, probs=None) -> restored_tokens
    """

    def _get_cuda_kernel(self):
        """Return the CUDA-backed MoE unpermutation function if available."""
        try:
            from wall_oss_05_op._cuda_ext import is_available
            from wall_oss_05_op._cuda_wrappers import unpermute_kernel

            if not is_available():
                return None
            return unpermute_kernel
        except ImportError:
            return None
        except Exception as e:
            logger.warning("UnpermuteOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, permuted_tokens, sorted_indices, probs=None):
        """Restore token order and optionally merge weighted top-k results."""
        if probs is not None:
            merge_factor = probs.size(1)
        else:
            merge_factor = 1
        unpermuted_tokens = torch.zeros_like(permuted_tokens)
        unpermuted_tokens.index_copy_(0, sorted_indices.long(), permuted_tokens)
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, merge_factor, permuted_tokens.size(-1)
        )
        if probs is not None:
            unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)
        return unpermuted_tokens


permute = PermuteOp("permute")
unpermute = UnpermuteOp("unpermute")
