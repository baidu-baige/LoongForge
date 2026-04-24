# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Mcore_checkpoint converter for megatron lm. """

import io
import torch
import logging

logging.basicConfig(level=logging.INFO)

class McoreAttnGateQkvConverter():
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

    def chunk_gqkv(self, gqkv, tp):
        if tp == 1:
            return [gqkv]
        
        split_arg_list = [
            self.q_dim,
            self.kv_dim,
            self.kv_dim,
        ]

        q, k, v = torch.split(gqkv.reshape(self.num_key_value_heads, -1, self.hidden_size), split_arg_list, dim=1)
        q_s = torch.chunk(q, tp, dim=0)
        k_s = torch.chunk(k, tp, dim=0)
        v_s = torch.chunk(v, tp, dim=0)
        qkv_s = []
        for i in range(tp):
            qkv_s.append(torch.cat([q_s[i], k_s[i], v_s[i]], dim=1).flatten(0, 1))
        return qkv_s


class McoreMixerAttnConverter:
    """
        McoreBase
    """

    def __init__(self, c_config):
        self.c_config = c_config
        cargs = self.c_config.get_args("common")

        self.name_map = self.c_config.get_name_map("huggingface")

        # For attention
        hidden_size = cargs["hidden_size"]
        self.heads = cargs["num_attention_heads"]
        self.mixer_num_key_heads = cargs.get("mixer_num_key_heads", self.heads)
        self.mixer_num_value_heads = cargs.get("mixer_num_value_heads", self.heads)
        self.mixer_key_head_dim = cargs.get("mixer_key_head_dim", hidden_size // self.heads)
        self.mixer_value_head_dim = cargs.get("mixer_value_head_dim", hidden_size // self.heads)


    def chunk_mixer_in_proj_qkvz(self, qkvz, tp):
        if tp == 1:
            q, k, v, z = self.get_qkvz(qkvz, tp)
            return [qkvz]
        split_size_list = [
            self.mixer_key_head_dim, 
            self.mixer_key_head_dim, 
            self.mixer_value_head_dim * self.mixer_num_value_heads // self.mixer_num_key_heads, 
            self.mixer_value_head_dim * self.mixer_num_value_heads // self.mixer_num_key_heads
        ]
        q, k, v, z = torch.split(
            qkvz.reshape(self.mixer_num_key_heads, 2 * self.mixer_key_head_dim + 2 * self.mixer_value_head_dim * self.mixer_num_value_heads // self.mixer_num_key_heads, -1),
            split_size_list,
            dim=1
        )
        q_s = torch.chunk(q, tp, dim=0)
        k_s = torch.chunk(k, tp, dim=0)
        v_s = torch.chunk(v, tp, dim=0)
        z_s = torch.chunk(z, tp, dim=0)
        qkvz_s = []
        for i in range(tp):
            qkvz_s.append(torch.cat([q_s[i], k_s[i], v_s[i], z_s[i]], dim=1).flatten(0, 1))
        return qkvz_s

    def chunk_mixer_in_proj_ba(self, ba, tp):
        if tp == 1:
            return [ba]
        ba1, ba2 = ba.reshape(self.mixer_num_key_heads, 2 * self.mixer_num_value_heads // self.mixer_num_key_heads, -1).chunk(chunks=2, dim=1)
        ba1_s = torch.chunk(ba1, tp, dim=0)
        ba2_s = torch.chunk(ba2, tp, dim=0)
        ba_s = []
        for i in range(tp):
            ba_s.append(torch.cat([ba1_s[i], ba2_s[i]], dim=1).flatten(0, 1))
        return ba_s