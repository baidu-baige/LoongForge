# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Base utilities for converting common checkpoints to and from HuggingFace format."""

import torch
import logging

logging.basicConfig(level=logging.INFO)

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.common_checkpoint import CommonCheckpoint
from convert_checkpoint.utils.utils import (
    transpose_shape0
)

from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig

from convert_checkpoint.common.common_checkpoint import (
    WEIGHT,
    BIAS,
    WEIGHT_SCALE,
    ROTARY_EMB_INV_FREQ,
    ATTENTION_ROTARY_EMB_INV_FREQ,
    ATTENTION_QUERY_KEY_VALUE,
    ATTENTION_QUERY_GATE_KEY_VALUE,
    MIXER_ATT_IN_PROJ,
    MIXER_ATT_IN_PROJ_QKVZ,
    MIXER_ATT_IN_PROJ_BA,
    MIXER_ATT_CONV1D,
    ATTENTION_DENSE,
    ATTENTION_QNORM,
    ATTENTION_KNORM,
    MLP_DENSE_H_TO_4H,
    WORD_EMBEDDINGS_FOR_HEAD,
    WORD_EMBEDDINGS,
    MTP_WORD_EMBEDDING,
    TRANSFORMER,
    LAYER_PREFIX,
    MOE_SHARED_EXPERT,
    LAYER_NAME,
    LAYER_IS_DIRECT_NAME,
    LAYER_NO_LAYER_ID,
    LAYER_DEPEND_ON_KEY,
    LAYER_IS_DICT_FOR_EXPERT,
    LAYER_NEED_TRANSPOSE,
    LAYER_DTYPE
)

from convert_checkpoint.huggingface.util.hf_attn_converter import HfAttnQkvConverter, HfAttnGateQkvConverter
from convert_checkpoint.huggingface.util.hf_mixer_attn_converter import HfMixerAttnConverter


class HuggingfaceBase:
    """
        HuggingfaceBase
    """

    def __init__(self, c_config, args):
        self.c_config = c_config
        self.args = args
        self.hf_attn_converter = HfAttnQkvConverter(c_config)
        self.hf_attn_gate_converter = HfAttnGateQkvConverter(c_config)
        self.hf_mixer_attn_converter = HfMixerAttnConverter(c_config)
        margs = self.c_config.get_args("mcore")
        cargs = self.c_config.get_args("common")

        self.name_map = self.c_config.get_name_map("huggingface")
        self.hargs = self.c_config.get_args("huggingface")

        # For rotary
        self.use_rotary_position_embeddings = margs.get("use_rotary_position_embeddings", False)
        # For attention
        hidden_size = cargs["hidden_size"]
        self.heads = cargs["num_attention_heads"]
        self.head_dim = cargs.get("head_dim", hidden_size // self.heads)
        self.num_padded_heads = cargs.get("num_padded_heads", 0)
        self.hidden_size_per_head = hidden_size // self.heads
        self.rotary_base = self.hargs.get("rotary_base", 10000)

        self.transformer = self.name_map[TRANSFORMER]
        self.layer_prefix = self.name_map[LAYER_PREFIX]
        self.weight_scale_suffix = self.name_map.get("weight_scale_key", WEIGHT_SCALE)

    @staticmethod
    def get_hf_name_and_args(obj):
        # hf_name: name in huggingface
        # is_direct_name: whether the end of the path has '.weight'
        # is_dict_for_expert: whether the obj is for expert in the dict
        # need_transpose: whether the weight need to transpose
        # no_layer_id: whether the path has no layer id
        # depend_on_key: whether convert this key depend on one other key has value
        # dtype: dtype for the weight (e.g., bf16, fp8)
        if isinstance(obj, dict) or isinstance(obj, DictConfig):
            hf_name = obj[LAYER_NAME]
            is_direct_name = obj[LAYER_IS_DIRECT_NAME] if LAYER_IS_DIRECT_NAME in obj else False
            is_dict_for_expert = obj.get(LAYER_IS_DICT_FOR_EXPERT, False)
            need_transpose = obj.get(LAYER_NEED_TRANSPOSE, False)
            no_layer_id = obj[LAYER_NO_LAYER_ID] if LAYER_NO_LAYER_ID in obj else False
            depend_on_key = obj[LAYER_DEPEND_ON_KEY] if LAYER_DEPEND_ON_KEY in obj else None
            dtype = obj.get(LAYER_DTYPE, None)
        else:
            hf_name = obj
            is_direct_name = False
            is_dict_for_expert = False
            need_transpose = False
            no_layer_id = False
            depend_on_key = None
            dtype = None
        return hf_name, is_direct_name, is_dict_for_expert, need_transpose, no_layer_id, depend_on_key, dtype

    #========from commmon to hf===========
    def common_to_hf(self, name, c_ckpt, h_dict, layer_id=None, hf_layer_id=None,
                     layer_prefix=None, expert_name=None, transformer=None, spec_name=None):
        spec_name = name if spec_name is None or spec_name not in self.name_map else spec_name
        is_valid_name = spec_name in self.name_map and self.name_map[spec_name] is not None
        if not is_valid_name:
            return
        if name == MTP_WORD_EMBEDDING:
            layer_id = None
        common_key = CommonCheckpoint.get_key(name, layer_id=layer_id)
        weight, bias, weight_scale = c_ckpt.get(common_key)
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        transformer = self.transformer if transformer is None else transformer

        if name in [ROTARY_EMB_INV_FREQ, ATTENTION_ROTARY_EMB_INV_FREQ]:
            assert self.use_rotary_position_embeddings, \
                    f"mcore args.use_rotary_position_embeddings is required to be set to True \
                    since we capture the rotary_emb op"
        hf_name, is_direct_name, is_dict_for_expert, need_transpose, no_layer_id, depend_on_key, _ = \
                self.get_hf_name_and_args(self.name_map[spec_name])
        if hf_layer_id is None or no_layer_id:
            if is_direct_name:
                hf_weight_path = hf_name
            else:
                hf_weight_path = f"{hf_name}.{WEIGHT}"
            hf_bias_path = self.name_map[f"{spec_name}.{BIAS}"] \
                    if f"{spec_name}.{BIAS}" in self.name_map else f"{hf_name}.{BIAS}"
            hf_weight_scale_path = f"{hf_name}.{self.weight_scale_suffix}"
            self.update_tensor(h_dict, hf_weight_path, weight, hf_bias_path=hf_bias_path, bias=bias,
                    hf_weight_scale_path=hf_weight_scale_path, weight_scale=weight_scale)
        else:
            if name == ATTENTION_QUERY_KEY_VALUE:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                self.update_list_to_hf(h_dict, name, hf_prefix_path, weight, bias, weight_scale, self.hf_attn_converter.split_attn_qkv)
            elif name == ATTENTION_QUERY_GATE_KEY_VALUE:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                self.update_list_to_hf(h_dict, name, hf_prefix_path, weight, bias, weight_scale, self.hf_attn_gate_converter.split_attn_qgkv)
            elif name == MIXER_ATT_IN_PROJ:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                self.update_list_to_hf(h_dict, name, hf_prefix_path, weight, bias, weight_scale, self.hf_mixer_attn_converter.split_mixer_in_proj)
            elif name == MIXER_ATT_IN_PROJ_QKVZ and isinstance(self.name_map.get(name), (list, ListConfig)):
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                self.update_list_to_hf(h_dict, name, hf_prefix_path, weight, bias, weight_scale, self.hf_mixer_attn_converter.split_qkvz_to_qkv_z)
            elif name == MIXER_ATT_IN_PROJ_BA and isinstance(self.name_map.get(name), (list, ListConfig)):
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                self.update_list_to_hf(h_dict, name, hf_prefix_path, weight, bias, weight_scale, self.hf_mixer_attn_converter.split_ba_to_b_a)
            elif name == MLP_DENSE_H_TO_4H:
                hf_prefix_path= f"{transformer}.{layer_prefix}.{hf_layer_id}"
                self.update_h_to_4h(h_dict, name, hf_prefix_path, weight, bias, weight_scale)
            elif expert_name == MOE_SHARED_EXPERT:
                if expert_name not in self.name_map:
                    return
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}"
                self.update_h_to_4h(h_dict, spec_name, hf_prefix_path, weight, bias, weight_scale)
            else:
                if expert_name is None:
                    hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{hf_name}"
                else:
                    hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}"
                if is_direct_name:
                    hf_weight_path = hf_prefix_path
                else:
                    hf_weight_path = f"{hf_prefix_path}.{WEIGHT}"
                bias_name = f"{name}.{BIAS}"
                hf_bias_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_prefix_path}.{BIAS}"
                hf_weight_scale_path = f"{hf_prefix_path}.{self.weight_scale_suffix}"
                if self.num_padded_heads != 0:
                    if name == ATTENTION_DENSE:
                        weight = weight[:, :self.heads * self.hidden_size_per_head].contiguous()
                    elif name in [ATTENTION_QNORM, ATTENTION_KNORM]:
                        weight = weight[:self.heads * self.hidden_size_per_head].contiguous()
                if name == MTP_WORD_EMBEDDING:
                    weight = weight.clone()
                self.update_tensor(h_dict, hf_weight_path, weight, hf_bias_path=hf_bias_path, bias=bias,
                        hf_weight_scale_path=hf_weight_scale_path, weight_scale=weight_scale)


    # === update tensor to huggingface state_dict begin ===
    def update_tensor(self, h_dict, hf_weight_path, weight, hf_bias_path=None, bias=None,
                      hf_weight_scale_path=None, weight_scale=None):
        if weight is None:
            return
        h_dict[hf_weight_path] = weight
        if bias is not None and hf_bias_path is not None:
            h_dict[hf_bias_path] = bias
        if weight_scale is not None and hf_weight_scale_path is not None:
            h_dict[hf_weight_scale_path] = weight_scale

    def update_list_to_hf(self, h_dict, name, hf_prefix_path, weight, bias, weight_scale, func):
        if weight is None:
            return
        hf_name = self.name_map[name]
        tag_names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = func(tag_names, weight) if weight is not None else None
        bias_list = func(tag_names, bias) if bias is not None else None
        weight_scale_list = func(tag_names, weight_scale) if weight_scale is not None else None
        for i in range(len(tag_names)):
            tag_name = tag_names[i]
            hf_path= f"{hf_prefix_path}.{tag_name}"
            if weight_list is not None:
                h_dict[f"{hf_path}.{WEIGHT}"] = weight_list[i]
            if bias_list is not None:
                h_dict[f"{hf_path}.{BIAS}"] = bias_list[i]
            if weight_scale_list is not None:
                h_dict[f"{hf_path}.{self.weight_scale_suffix}"] = weight_scale_list[i]

    def update_h_to_4h(self, h_dict, name, hf_prefix_path, weight, bias, weight_scale, expert_id=None):
        if weight is None:
            return
        hf_name, is_direct_name, is_dict_for_expert, need_transpose, _, _, _ = self.get_hf_name_and_args(self.name_map[name])
        weight = weight.t() if need_transpose else weight
        names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = torch.chunk(weight, len(names), dim=0)
        bias_list = torch.chunk(bias, len(names), dim=0) if bias is not None else None
        weight_scale_list = torch.chunk(weight_scale, len(names), dim=0) if weight_scale is not None else None

        for i in range(len(names)):
            hf_path = f"{hf_prefix_path}.{names[i]}"
            hf_weight_path = f"{hf_path}.{WEIGHT}" if not is_direct_name else hf_path
            if is_dict_for_expert:
                assert expert_id is not None, "expert_id must be specified when is_dict_for_expert"
                h_dict[hf_weight_path] = {LAYER_IS_DICT_FOR_EXPERT: True} if hf_weight_path not in h_dict else h_dict[hf_weight_path]
                h_dict[hf_weight_path][expert_id] = weight_list[i] if weight_list is not None else None
            else:
                h_dict[hf_weight_path] = weight_list[i] if weight_list is not None else None

            if bias_list is not None:
                h_dict[f"{hf_path}.{BIAS}"] = bias_list[i]
            if weight_scale_list is not None:
                h_dict[f"{hf_path}.{self.weight_scale_suffix}"] = weight_scale_list[i]
    # === update tensor to huggingface state_dict end ===

    # ====== from hf to common ========
    def hf_to_common(self, name, c_ckpt, h_dict, layer_id=None, hf_layer_id=None, layer_prefix=None, expert_name=None, transformer=None, spec_name=None):
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        transformer = self.transformer if transformer is None else transformer
        spec_name = name if spec_name is None or spec_name not in self.name_map else spec_name
        is_valid_name = spec_name in self.name_map and self.name_map[spec_name] is not None
        if name == MTP_WORD_EMBEDDING:
            layer_id = None
        common_key = CommonCheckpoint.get_key(name, layer_id=layer_id)
        if is_valid_name:
            hf_name, is_direct_name, _, _, no_layer_id, depend_on_key, _ = \
                    self.get_hf_name_and_args(self.name_map[spec_name])
            if depend_on_key is not None:
                assert depend_on_key in self.name_map, f"depend_on_key {depend_on_key} is not in self.name_map"
                d_hf_name, is_direct_name_2, no_layer_id_2, _, _, _, _ = self.get_hf_name_and_args(self.name_map[depend_on_key])
                depend_weight, _, _ = self.get_weight(depend_on_key, h_dict, layer_id, hf_layer_id, layer_prefix,
                                                      expert_name, transformer, d_hf_name, is_direct_name_2, no_layer_id_2, True)
                if depend_weight is None:
                    return
        else:
            if name not in [WORD_EMBEDDINGS_FOR_HEAD, MTP_WORD_EMBEDDING] or WORD_EMBEDDINGS not in self.name_map:
                return
            else:
                layer_id = None
                hf_name = self.name_map[WORD_EMBEDDINGS]
            is_direct_name = False
            no_layer_id = False
        weight, bias, weight_scale = self.get_weight(name, h_dict, layer_id, hf_layer_id, layer_prefix,
                                                     expert_name, transformer, hf_name, is_direct_name,
                                                     no_layer_id, is_valid_name, spec_name=spec_name)
        c_ckpt.set(common_key, weight, bias, weight_scale)

    def get_weight(self, name, h_dict, layer_id, hf_layer_id, layer_prefix, expert_name,
                   transformer, hf_name, is_direct_name, no_layer_id, is_valid_name, spec_name=None):
        weight = None
        bias = None
        weight_scale = None
        if layer_id is None or no_layer_id:
            if is_valid_name:
                if is_direct_name:
                    hf_weight_path = hf_name
                else:
                    hf_weight_path = f"{hf_name}.{WEIGHT}"
                hf_bias_path = self.name_map[f"{name}.{BIAS}"] \
                        if f"{name}.{BIAS}" in self.name_map else f"{hf_name}.{BIAS}"
                hf_weight_scale_path = f"{hf_name}.{self.weight_scale_suffix}"
                weight, bias, weight_scale = self.get_from_state_dict(
                        h_dict, hf_weight_path, hf_bias_path=hf_bias_path, hf_weight_scale_path=hf_weight_scale_path)
            if (name == WORD_EMBEDDINGS_FOR_HEAD or name == MTP_WORD_EMBEDDING) and weight is None and WORD_EMBEDDINGS in self.name_map:
                hf_name, _, _, _, _, _, _ = self.get_hf_name_and_args(self.name_map[WORD_EMBEDDINGS])
                hf_weight_path = f"{hf_name}.{WEIGHT}"
                weight, bias, weight_scale = self.get_from_state_dict(h_dict, hf_weight_path)
        else:
            if name in [ROTARY_EMB_INV_FREQ, ATTENTION_ROTARY_EMB_INV_FREQ]:
                assert self.use_rotary_position_embeddings == True, \
                        f"mcore args.use_rotary_position_embeddings is required to be set to True \
                        since we capture the rotary_emb op"
            if name == ATTENTION_QUERY_KEY_VALUE:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                weight, bias, weight_scale = self.get_list_from_state_dict(name, h_dict, hf_prefix_path, self.hf_attn_converter.cat_attn_qkv)
            elif name == ATTENTION_QUERY_GATE_KEY_VALUE:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                weight, bias, weight_scale = self.get_list_from_state_dict(name, h_dict, hf_prefix_path, self.hf_attn_gate_converter.cat_attn_qgkv)
            elif name == MIXER_ATT_IN_PROJ:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                weight, bias, weight_scale = self.get_list_from_state_dict(name, h_dict, hf_prefix_path, self.hf_mixer_attn_converter.cat_mixer_in_proj)
            elif name == MIXER_ATT_IN_PROJ_QKVZ and isinstance(self.name_map.get(name), (list, ListConfig)):
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                weight, bias, weight_scale = self.get_list_from_state_dict(name, h_dict, hf_prefix_path, self.hf_mixer_attn_converter.cat_qkv_z_to_qkvz)
            elif name == MIXER_ATT_IN_PROJ_BA and isinstance(self.name_map.get(name), (list, ListConfig)):
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                weight, bias, weight_scale = self.get_list_from_state_dict(name, h_dict, hf_prefix_path, self.hf_mixer_attn_converter.cat_b_a_to_ba)
            elif name == MLP_DENSE_H_TO_4H:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}"
                weight, bias, weight_scale = self.get_h_to_4h_from_state_dict(name, h_dict, hf_prefix_path)
            elif expert_name == MOE_SHARED_EXPERT:
                if expert_name not in self.name_map:
                    return None, None, None
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}"
                weight, bias, weight_scale = self.get_h_to_4h_from_state_dict(spec_name, h_dict, hf_prefix_path)
            else:
                if expert_name is None:
                    hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{hf_name}"
                else:
                    hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}"
                if is_direct_name:
                    hf_weight_path = hf_prefix_path
                else:
                    hf_weight_path = f"{hf_prefix_path}.{WEIGHT}"
                bias_name = f"{name}.{BIAS}"
                hf_bias_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_prefix_path}.{BIAS}"
                hf_weight_scale_path = f"{hf_prefix_path}.{self.weight_scale_suffix}"
                weight, bias, weight_scale = self.get_from_state_dict(
                        h_dict, hf_weight_path, hf_bias_path=hf_bias_path, hf_weight_scale_path=hf_weight_scale_path)
                if ATTENTION_ROTARY_EMB_INV_FREQ == name and weight is None:
                    weight = 1.0 / (self.rotary_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
                # For attention padded heads
                if self.num_padded_heads != 0 and name in [ATTENTION_DENSE, ATTENTION_QNORM, ATTENTION_KNORM]:
                    weight = self.get_padded_head_weight(name, weight)
        return weight, bias, weight_scale

    def get_from_state_dict(self, h_dict, hf_weight_path, hf_bias_path=None, hf_weight_scale_path=None):
        weight = h_dict[hf_weight_path] if hf_weight_path in h_dict else None
        bias = h_dict[hf_bias_path] if hf_bias_path in h_dict else None
        weight_scale = h_dict[hf_weight_scale_path] if hf_weight_scale_path in h_dict else None
        return weight, bias, weight_scale

    def get_list_from_state_dict(self, name, h_dict, hf_prefix_path, func):
        hf_name = self.name_map[name]
        tag_names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = []
        bias_list = []
        weight_scale_list = []
        for tag_name in tag_names:
            hf_path= f"{hf_prefix_path}.{tag_name}"
            if f"{hf_path}.{WEIGHT}" in h_dict:
                weight_list.append(h_dict[f"{hf_path}.{WEIGHT}"])
            if f"{hf_path}.{BIAS}" in h_dict:
                bias_list.append(h_dict[f"{hf_path}.{BIAS}"])
            if f"{hf_path}.{self.weight_scale_suffix}" in h_dict:
                weight_scale_list.append(h_dict[f"{hf_path}.{self.weight_scale_suffix}"])
        weight = func(weight_list) if len(weight_list) > 0 else None
        bias = func(bias_list) if len(bias_list) > 0 else None
        weight_scale = func(weight_scale_list) if len(weight_scale_list) > 0 else None
        return weight, bias, weight_scale

    def get_h_to_4h_from_state_dict(self, name, h_dict, hf_prefix_path, expert_id=None):
        hf_name, is_direct_name, is_dict_for_expert, need_transpose, _, _, _ = self.get_hf_name_and_args(self.name_map[name])
        hf_names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = []
        bias_list = []
        weight_scale_list = []
        for hf_name in hf_names:
            hf_path= f"{hf_prefix_path}.{hf_name}"
            hf_weight_path = f"{hf_path}.{WEIGHT}" if not is_direct_name else hf_path
            hf_bias_path = f"{hf_path}.{BIAS}"
            if f"{name}.{BIAS}" in self.name_map:
                hf_bias_path = self.name_map[f"{name}.{BIAS}"]
            hf_weight_scale_path = f"{hf_path}.{self.weight_scale_suffix}"
            if hf_weight_path in h_dict:
                if is_dict_for_expert:
                    assert expert_id is not None, "expert_id must be specified when is_dict_for_expert is True"
                    weight = h_dict[hf_weight_path][expert_id]
                else:
                    weight = h_dict[hf_weight_path]
                weight_list.append(weight)
            if hf_bias_path in h_dict:
                bias_list.append(h_dict[hf_bias_path])
            if hf_weight_scale_path in h_dict:
                weight_scale_list.append(h_dict[hf_weight_scale_path])
        weight = torch.cat(weight_list, dim=0) if len(weight_list) > 0 else None
        if need_transpose and weight is not None:
            weight = weight.t()
        bias = torch.cat(bias_list, dim=0) if len(bias_list) > 0 else None
        weight_scale = torch.cat(weight_scale_list, dim=0) if len(weight_scale_list) > 0 else None
        return weight, bias, weight_scale

    def get_padded_head_weight(self, name, weight):
        padded_dim = self.num_padded_heads * self.hidden_size_per_head * 1
        if name == ATTENTION_DENSE:
            padded_tensor = torch.zeros((weight.shape[0], padded_dim), dtype=weight.dtype, device=weight.device)
            padded_tensor[:, :weight.shape[-1]] = weight
            weight = padded_tensor
        elif name in [ATTENTION_QNORM, ATTENTION_KNORM]:
            padded_tensor = torch.zeros(padded_dim, dtype=weight.dtype, device=weight.device)
            padded_tensor[:weight.shape[0]] = weight
            weight = padded_tensor
        return weight

