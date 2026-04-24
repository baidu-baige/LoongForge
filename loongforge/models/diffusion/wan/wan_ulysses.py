# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DeepSpeed under the Apache-2.0 License.
# Copyright (c) Microsoft Corporation.

"""all to all operation"""

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import torch.distributed as dist

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


def _all_gather(
    group: dist.ProcessGroup,
    input_: torch.Tensor,
    gather_dim: int,
):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_.clone()  # Avoid in-place operation
    dist.all_gather(tensor_list, input_, group=group)

    return torch.cat(tensor_list, dim=gather_dim).contiguous()


def _all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
):
    world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _single_all_to_all(input, scatter_idx, gather_idx, group):
    """all_to_all operation"""
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1 :]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = (
            input.reshape(
                [-1, seq_world_size, inp_shape[scatter_idx]]
                + inp_shape[scatter_idx + 1 :]
            )
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_idx]
        + [
            inp_shape[gather_idx] * seq_world_size,
        ]
        + inp_shape[gather_idx + 1 :]
    ).contiguous()


class SeqAllToAll(torch.autograd.Function):
    """all to all funtion"""

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        single_all_to_all: bool,
    ) -> Tensor:
        """AllToAll  forward"""
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.single_all_to_all = single_all_to_all
        if single_all_to_all:
            return _single_all_to_all(input, scatter_idx, gather_idx, group)
        else:
            return _all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[None, Tensor, None, None, None]:
        """AllToAll  backward"""
        return (
            None,
            SeqAllToAll.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.single_all_to_all,
            ),
            None,
            None,
            None,
        )


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        recompute_num_layers: int = 0,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        pad_kv: bool = False,
        effective_length=None,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.recompute_num_layers = recompute_num_layers
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.pad_kv = pad_kv
        self.effective_length = effective_length

        def remove_extra_states_check(self, incompatible_keys):
            """
            Temporarily remove local_attn._extra_state as a missing key
            when loading older TransformerEngine checkpoints.

            refer to:
            https://github.com/NVIDIA/TransformerEngine/blob/8062ac503fa2a7419d7e1191fd328d76ce1e752a/
            transformer_engine/pytorch/attention.py#L4944
            """
            for key in incompatible_keys.missing_keys:
                if "local_attn._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor = None,
        attn_mask_type=None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        rotary_pos_emb: tuple = None,
        apply_rotary_fn=None,
        config: TransformerConfig = None,
        single_all_to_all: bool = False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # support packing sequence

        if packed_seq_params is not None:
            if packed_seq_params.qkv_format == "thd":
                assert len(query.shape) == 3
                self.scatter_idx = 1
                self.gather_idx = 0
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side
        # so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        query_layer = SeqAllToAll.apply(
            self.spg, query, self.scatter_idx, self.gather_idx, single_all_to_all
        )
        key_layer = SeqAllToAll.apply(
            self.spg, key, self.scatter_idx, self.gather_idx, single_all_to_all
        )
        value_layer = SeqAllToAll.apply(
            self.spg, value, self.scatter_idx, self.gather_idx, single_all_to_all
        )

        if self.pad_kv:
            # cat in cp dim, split muliti-head; we remove pads so that pads do not influence cross-attention
            key_layer = key_layer[: self.effective_length]
            value_layer = value_layer[: self.effective_length]

        if packed_seq_params is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # q_pos_emb and k_pos_emb shape: [max_s, 1, 1, d]
            q_pos_emb = _all_gather(self.spg, q_pos_emb, 0)
            k_pos_emb = _all_gather(self.spg, k_pos_emb, 0)
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            query_layer = apply_rotary_fn(
                query_layer,
                q_pos_emb,
                config=config,
                cu_seqlens=cu_seqlens_q,
            )
            key_layer = apply_rotary_fn(
                key_layer,
                k_pos_emb,
                config=config,
                cu_seqlens=cu_seqlens_kv,
            )

        # out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            *args,
        )

        output = SeqAllToAll.apply(
            self.spg,
            context_layer,
            self.gather_idx,
            self.scatter_idx,
            single_all_to_all,
        )
        # out e.g., [s/p::h]
        return output
