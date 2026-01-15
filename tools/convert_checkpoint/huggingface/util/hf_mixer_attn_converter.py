"""
HfMixerAttnConverter
"""
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