# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HfAttnConverter"""

import torch
import logging

logging.basicConfig(level=logging.INFO)

from convert_checkpoint.utils.utils import (
    transpose_shape0
)

class HfAttnQkvConverter():
    def __init__(self, c_config):
        self.c_config = c_config
        margs = self.c_config.get_args("mcore")
        cargs = self.c_config.get_args("common")

        self.name_map = self.c_config.get_name_map("huggingface")
        self.hargs = self.c_config.get_args("huggingface")

        hidden_size = cargs["hidden_size"]
        self.heads = cargs["num_attention_heads"]
        self.head_dim = cargs.get("head_dim", hidden_size // self.heads)
        self.transpose_query_key_value = margs.get("transpose_query_key_value", False)
        self.num_key_value_heads = cargs.get("num_key_value_heads", self.heads)
        self.num_padded_heads = cargs.get("num_padded_heads", 0)
        self.hidden_size_per_head = hidden_size // self.heads

    # common_to_hf attn qkv begin
    def split_attn_qkv(self, qkv_names, value):
        if value is None:
            return
        # transpose value in shape[0] for llama
        assert self.heads % self.num_key_value_heads == 0
        num_repeats = self.heads // self.num_key_value_heads
        num_splits = num_repeats + 2 # repeats*Q + K + V

        if self.num_padded_heads != 0:
            value = value[:self.heads * self.hidden_size_per_head * num_splits].contiguous()

        if not self.transpose_query_key_value:
            assert len(qkv_names) == 1
            value_list = [value]
        else:
            value = transpose_shape0(value, self.num_key_value_heads, num_splits)
            value_list = list(torch.chunk(value, num_splits, dim=0))
            q, k, v = torch.cat(value_list[:-2]), value_list[-2], value_list[-1]
            q = transpose_shape0(q, num_repeats, self.num_key_value_heads)
            if len(qkv_names) == 1:
                value_list = [torch.cat((q, k, v), dim=0)]
            else:
                assert len(qkv_names) == 3
                value_list = [q, k, v]

        return value_list
    # common_to_hf attn qkv end

    # hf_to_common attn_qkv begin
    def cat_attn_qkv(self, value_list):
        if not self.transpose_query_key_value:
            assert len(value_list) == 1
            value = torch.cat(value_list, dim=0)
        else:
            assert self.heads % self.num_key_value_heads == 0
            num_repeats = self.heads // self.num_key_value_heads
            num_splits = num_repeats + 2
            if len(value_list) == 1:
                value_list = list(torch.chunk(value_list[0], num_splits, dim=0))
                q, k, v = torch.cat(value_list[:-2]), value_list[-2], value_list[-1]
            else:
                assert len(value_list) == 3
                q, k, v = value_list[0], value_list[1], value_list[2]
            q = transpose_shape0(q, self.num_key_value_heads, num_repeats)
            value = torch.cat([q, k, v], dim=0)
            value = transpose_shape0(value, num_splits, self.num_key_value_heads)

        if self.num_padded_heads != 0:
            padded_dim = self.num_padded_heads * self.hidden_size_per_head * 3
            padded_tensor = torch.zeros((padded_dim, value.shape[-1]),
                                        dtype=value.dtype, device=value.device)
            padded_tensor[:value.shape[0], :] = value
            value = padded_tensor

        return value
    # hf_to_common attn_qkv end


class HfAttnGateQkvConverter():
    def __init__(self, c_config):
        self.c_config = c_config
        margs = self.c_config.get_args("mcore")
        cargs = self.c_config.get_args("common")

        self.name_map = self.c_config.get_name_map("huggingface")
        self.hargs = self.c_config.get_args("huggingface")

        self.heads = cargs["num_attention_heads"]
        self.hidden_size = cargs["hidden_size"]
        self.head_dim = cargs.get("head_dim", self.hidden_size // self.heads)
        self.num_key_value_heads = cargs.get("num_key_value_heads", self.heads)
        self.num_querys_per_group = self.heads // self.num_key_value_heads

        self.q_dim = 2 * self.num_querys_per_group * self.head_dim
        self.kv_dim = self.head_dim


    # common_to_hf gated_selfattn begin
    def split_attn_qgkv(self, qkv_names, value):
        attn_proj_weight = value.reshape((self.num_key_value_heads, -1, self.hidden_size))
        q = attn_proj_weight[:, :self.q_dim, :].reshape(-1, self.hidden_size).clone()
        k = attn_proj_weight[:, self.q_dim: -self.kv_dim, :].reshape(-1, self.hidden_size).clone()
        v = attn_proj_weight[:, -self.kv_dim:, :].reshape(-1, self.hidden_size).clone()

        return q, k, v
    # common_to_hf gated_selfattn end

    # hf_to_common gated_selfattn begin
    def cat_attn_qgkv(self, value_list):
        if value_list is None:
            return None
        assert len(value_list) == 3, "qgkv len must be 3."
        q = value_list[0]
        k = value_list[1]
        v = value_list[2]

        value = torch.cat([
            q.reshape((self.num_key_value_heads, -1, self.hidden_size)),
            k.reshape((self.num_key_value_heads, -1, self.hidden_size)),
            v.reshape((self.num_key_value_heads, -1, self.hidden_size)),
        ], dim=1).reshape((-1, self.hidden_size))

        return value
    # hf_to_common gated_selfattn end
