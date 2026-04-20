# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" HfMixerAttnConverter """

import torch
import logging

logging.basicConfig(level=logging.INFO)

from omegaconf import ListConfig

from convert_checkpoint.common.common_checkpoint import (
    WEIGHT,
    BIAS
)

class HfMixerAttnConverter():
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


    # hf_to_common begin
    def cat_mixer_in_proj(self, value_list):
        if value_list is None:
            return None
        assert len(value_list) == 2, "qgkv len must be 2."
        in_proj_ba = value_list[0]
        in_proj_qkvz = value_list[1]
        return torch.cat([in_proj_qkvz, in_proj_ba], dim=0)
    # hf_to_common end

    # common_to_hf begin
    def split_mixer_in_proj(self, qkv_names, in_proj_qkvz):
        split_size_list = [
            2 * self.mixer_num_key_heads * self.mixer_key_head_dim + 2 * self.mixer_value_head_dim * self.mixer_num_value_heads,
            2 * self.mixer_num_value_heads
        ]
        hf_in_proj_qkvz, hf_in_proj_ba = torch.split(
            in_proj_qkvz,
            split_size_list,
            dim=0
        )
        return [hf_in_proj_ba, hf_in_proj_qkvz]
     # common_to_hf end

    # ====== qwen3.5 HF (separate qkv,z,b,a) <-> common (interleaved qkvz,ba) ======
    # HF qwen3.5 has contiguous layout:
    #   in_proj_qkv: [q_all, k_all, v_all]  shape (nk*dk*2 + nv*dv, H)
    #   in_proj_z:   [z_all]                 shape (nv*dv, H)
    #   in_proj_b:   [b_all]                 shape (nv, H)
    #   in_proj_a:   [a_all]                 shape (nv, H)
    #
    # GatedDeltaNet (qwen3_next) uses interleaved-by-key-head-group layout:
    #   in_proj_qkvz: per group g: [q_g(dk), k_g(dk), v_{g*r:g*r+r}(r*dv), z_{g*r:g*r+r}(r*dv)]
    #   in_proj_ba:   per group g: [b_{g*r:g*r+r}(r), a_{g*r:g*r+r}(r)]

    def cat_qkv_z_to_qkvz(self, value_list):
        """HF->Common: combine [in_proj_qkv, in_proj_z] into interleaved in_proj_qkvz."""
        if value_list is None:
            return None
        assert len(value_list) == 2, f"Expected [qkv, z], got {len(value_list)} tensors."
        qkv_weight = value_list[0]   # (nk*dk + nk*dk + nv*dv, H)
        z_weight = value_list[1]     # (nv*dv, H)

        nk = self.mixer_num_key_heads
        nv = self.mixer_num_value_heads
        dk = self.mixer_key_head_dim
        dv = self.mixer_value_head_dim
        r = nv // nk  # value heads per key head group
        H = qkv_weight.shape[1]

        # Split qkv into q, k, v (contiguous)
        q = qkv_weight[:nk * dk].reshape(nk, dk, H)
        k = qkv_weight[nk * dk:2 * nk * dk].reshape(nk, dk, H)
        v = qkv_weight[2 * nk * dk:].reshape(nk, r * dv, H)
        z = z_weight.reshape(nk, r * dv, H)

        # Interleave: per group [q, k, v, z]
        qkvz = torch.cat([q, k, v, z], dim=1).reshape(-1, H)
        return qkvz

    def split_qkvz_to_qkv_z(self, tag_names, qkvz_weight):
        """Common->HF: split interleaved in_proj_qkvz into [in_proj_qkv, in_proj_z]."""
        nk = self.mixer_num_key_heads
        nv = self.mixer_num_value_heads
        dk = self.mixer_key_head_dim
        dv = self.mixer_value_head_dim
        r = nv // nk
        H = qkvz_weight.shape[1]

        per_group = 2 * dk + 2 * r * dv
        grouped = qkvz_weight.reshape(nk, per_group, H)
        q = grouped[:, :dk, :]                          # (nk, dk, H)
        k = grouped[:, dk:2 * dk, :]                    # (nk, dk, H)
        v = grouped[:, 2 * dk:2 * dk + r * dv, :]       # (nk, r*dv, H)
        z = grouped[:, 2 * dk + r * dv:, :]              # (nk, r*dv, H)

        qkv = torch.cat([q.reshape(-1, H), k.reshape(-1, H), v.reshape(-1, H)], dim=0)
        z_out = z.reshape(-1, H)
        return [qkv, z_out]

    def cat_b_a_to_ba(self, value_list):
        """HF->Common: combine [in_proj_b, in_proj_a] into interleaved in_proj_ba."""
        if value_list is None:
            return None
        assert len(value_list) == 2, f"Expected [b, a], got {len(value_list)} tensors."
        b_weight = value_list[0]   # (nv, H)
        a_weight = value_list[1]   # (nv, H)

        nk = self.mixer_num_key_heads
        r = self.mixer_num_value_heads // nk
        H = b_weight.shape[1]

        b = b_weight.reshape(nk, r, H)
        a = a_weight.reshape(nk, r, H)

        # Interleave: per group [b, a]
        ba = torch.cat([b, a], dim=1).reshape(-1, H)
        return ba

    def split_ba_to_b_a(self, tag_names, ba_weight):
        """Common->HF: split interleaved in_proj_ba into [in_proj_b, in_proj_a]."""
        nk = self.mixer_num_key_heads
        r = self.mixer_num_value_heads // nk
        H = ba_weight.shape[1]

        grouped = ba_weight.reshape(nk, 2 * r, H)
        b = grouped[:, :r, :].reshape(-1, H)
        a = grouped[:, r:, :].reshape(-1, H)
        return [b, a]