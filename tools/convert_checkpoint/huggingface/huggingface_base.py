""" Mcore_checkpoint converter for aiak megatron. """
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
    ATTENTION_DENSE,
    ATTENTION_QNORM,
    ATTENTION_KNORM,
    MLP_DENSE_H_TO_4H,
    WORD_EMBEDDINGS_FOR_HEAD,
    WORD_EMBEDDINGS,
    TRANSFORMER,
    LAYER_PREFIX,
    MOE_SHARED_EXPERT,
    LAYER_NAME,
    LAYER_IS_DIRECT_NAME
)


class HuggingfaceBase:
    """
        HuggingfaceBase
    """

    def __init__(self, c_config):
        self.c_config = c_config
        margs = self.c_config.get_args("mcore")
        cargs = self.c_config.get_args("common")

        self.name_map = self.c_config.get_name_map("huggingface")
        self.hargs = self.c_config.get_args("huggingface")

        # For rotary
        self.use_rotary_position_embeddings = margs.get("use_rotary_position_embeddings", False)
        # For attention
        hidden_size = cargs["hidden_size"]
        self.heads = cargs["num_attention_heads"]
        self.attn_dim = hidden_size // self.heads
        self.transpose_query_key_value = margs.get("transpose_query_key_value", False)
        num_key_value_heads = cargs.get("num_key_value_heads", self.heads)
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else self.heads
        self.num_padded_heads = cargs.get("num_padded_heads", 0)
        self.hidden_size_per_head = hidden_size // self.heads
        self.rotary_base = self.hargs.get("rotary_base", 10000)

        self.transformer = self.name_map[TRANSFORMER]
        self.layer_prefix = self.name_map[LAYER_PREFIX]

    def get_hf_name_and_args(self, obj):
        if isinstance(obj, dict) or isinstance(obj, DictConfig):
            hf_name = obj[LAYER_NAME]
            is_direct_name = obj[LAYER_IS_DIRECT_NAME] if LAYER_IS_DIRECT_NAME in obj else False
        else:
            hf_name = obj
            is_direct_name = False
        return hf_name, is_direct_name

    #========from commmon to hf===========
    def common_to_hf(self, name, c_ckpt, h_dict, layer_id=None, layer_prefix=None, expert_name=None):
        is_valid_name = name in self.name_map and self.name_map[name] is not None
        if not is_valid_name:
            return
        common_key = CommonCheckpoint.get_key(name, layer_id=layer_id)
        weight, bias, weight_scale = c_ckpt.get(common_key)
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix

        if name in [ROTARY_EMB_INV_FREQ, ATTENTION_ROTARY_EMB_INV_FREQ]:
            assert self.use_rotary_position_embeddings, \
                    f"mcore args.use_rotary_position_embeddings is required to be set to True \
                    since we capture the rotary_emb op"

        hf_name, is_direct_name = self.get_hf_name_and_args(self.name_map[name])
        if layer_id is None:
            if is_direct_name:
                hf_weight_path = hf_name
            else:
                hf_weight_path = f"{hf_name}.{WEIGHT}"
            hf_bias_path = self.name_map[f"{name}.{BIAS}"] \
                    if f"{name}.{BIAS}" in self.name_map else f"{hf_name}.{BIAS}"
            hf_weight_scale_path = f"{hf_name}.{WEIGHT_SCALE}"
            self.update_tensor(h_dict, hf_weight_path, weight, hf_bias_path=hf_bias_path, bias=bias,
                    hf_weight_scale_path=hf_weight_scale_path, weight_scale=weight_scale)
        else:
            if name == ATTENTION_QUERY_KEY_VALUE:
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}"
                self.update_attn_qkv(h_dict, name, hf_prefix_path, weight, bias)
            elif name == MLP_DENSE_H_TO_4H:
                hf_prefix_path= f"{self.transformer}.{layer_prefix}.{layer_id}"
                self.update_h_to_4h(h_dict, name, hf_prefix_path, weight, bias, weight_scale)
            elif expert_name == MOE_SHARED_EXPERT:
                if expert_name not in self.name_map:
                    return
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}"
                self.update_h_to_4h(h_dict, name, hf_prefix_path, weight, bias, weight_scale)
            else:
                if expert_name is None:
                    hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{hf_name}"
                else:
                    hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}"
                if is_direct_name:
                    hf_weight_path = hf_prefix_path
                else:
                    hf_weight_path = f"{hf_prefix_path}.{WEIGHT}"
                bias_name = f"{name}.{BIAS}"
                hf_bias_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_prefix_path}.{BIAS}"
                hf_weight_scale_path = f"{hf_prefix_path}.{WEIGHT_SCALE}"
                if self.num_padded_heads != 0:
                    if name == ATTENTION_DENSE:
                        weight = weight[:, :self.heads * self.hidden_size_per_head].contiguous()
                    elif name in [ATTENTION_QNORM, ATTENTION_KNORM]:
                        weight = weight[:self.heads * self.hidden_size_per_head].contiguous()
                self.update_tensor(h_dict, hf_weight_path, weight, hf_bias_path=hf_bias_path, bias=bias,
                        hf_weight_scale_path=hf_weight_scale_path, weight_scale=weight_scale)


    def get_attn_qkv_list(self, qkv_names, weight, bias):
        # transpose weight in shape[0] for llama
        assert self.heads % self.num_key_value_heads == 0
        num_repeats = self.heads // self.num_key_value_heads
        num_splits = num_repeats + 2 # repeats*Q + K + V

        if self.num_padded_heads != 0:
            weight = weight[:self.heads * self.hidden_size_per_head * num_splits].contiguous()
            if bias is not None:
                bias = bias[:self.heads * self.hidden_size_per_head * num_splits].contiguous()

        if weight is None:
            return None, None
        bias_list = None
        if not self.transpose_query_key_value:
            assert len(qkv_names) == 1
            weight_list = [weight]
            if bias is not None:
                bias_list = [bias]
        else:
            weight = transpose_shape0(weight, self.num_key_value_heads, num_splits)
            weight_list = list(torch.chunk(weight, num_splits, dim=0))
            q, k, v = torch.cat(weight_list[:-2]), weight_list[-2], weight_list[-1]
            q = transpose_shape0(q, num_repeats, self.num_key_value_heads)
            if len(qkv_names) == 1:
                weight_list = [torch.cat((q, k, v), dim=0)]
            else:
                assert len(qkv_names) == 3
                weight_list = [q, k, v]

            if bias is not None:
                bias = transpose_shape0(bias, self.num_key_value_heads, num_splits)
                bias_list = list(torch.chunk(bias, num_splits, dim=0))
                q, k, v = torch.cat(bias_list[:-2]), bias_list[-2], bias_list[-1]
                q = transpose_shape0(q, num_repeats, self.num_key_value_heads)
                if len(qkv_names) == 1:
                    bias_list = [torch.cat((q, k, v), dim=0)]
                else:
                    assert len(qkv_names) == 3
                    bias_list = [q, k, v]

        return weight_list, bias_list

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

    def update_attn_qkv(self, h_dict, name, hf_prefix_path, weight, bias):
        if weight is None:
            return
        hf_name = self.name_map[name]
        qkv_names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list, bias_list = self.get_attn_qkv_list(qkv_names, weight, bias)
        for i in range(len(qkv_names)):
            qkv_name = qkv_names[i]
            hf_path= f"{hf_prefix_path}.{qkv_name}"
            if weight_list is not None:
                h_dict[f"{hf_path}.{WEIGHT}"] = weight_list[i]
            if bias_list is not None:
                h_dict[f"{hf_path}.{BIAS}"] = bias_list[i]

    def update_h_to_4h(self, h_dict, name, hf_prefix_path, weight, bias, weight_scale):
        if weight is None:
            return
        hf_name = self.name_map[name]
        names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = torch.chunk(weight, len(names), dim=0)
        bias_list = torch.chunk(bias, len(names), dim=0) if bias is not None else None
        weight_scale_list = torch.chunk(weight_scale, len(names), dim=0) if weight_scale is not None else None

        for i in range(len(names)):
            hf_path = f"{hf_prefix_path}.{names[i]}"
            h_dict[f"{hf_path}.{WEIGHT}"] = weight_list[i] if weight_list is not None else None
            if bias_list is not None:
                h_dict[f"{hf_path}.{BIAS}"] = bias_list[i]
            if weight_scale_list is not None:
                h_dict[f"{hf_path}.{WEIGHT_SCALE}"] = weight_scale_list[i]
    # === update tensor to huggingface state_dict end ===

    # ====== from hf to common ========
    def hf_to_common(self, name, c_ckpt, h_dict, layer_id=None, layer_prefix=None, expert_name=None):
        is_valid_name = name in self.name_map and self.name_map[name] is not None
        if name != WORD_EMBEDDINGS_FOR_HEAD and not is_valid_name:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        common_key = CommonCheckpoint.get_key(name, layer_id=layer_id)
        if is_valid_name:
            hf_name, is_direct_name = self.get_hf_name_and_args(self.name_map[name])
        weight = None
        bias = None
        weight_scale = None
        if layer_id is None:
            if is_valid_name:
                if is_direct_name:
                    hf_weight_path = hf_name
                else:
                    hf_weight_path = f"{hf_name}.{WEIGHT}"
                hf_bias_path = self.name_map[f"{name}.{BIAS}"] \
                        if f"{name}.{BIAS}" in self.name_map else f"{hf_name}.{BIAS}"
                hf_weight_scale_path = f"{hf_name}.{WEIGHT_SCALE}"
                weight, bias, weight_scale = self.get_common_from_state_dict(
                        h_dict, hf_weight_path, hf_bias_path=hf_bias_path, hf_weight_scale_path=hf_weight_scale_path)
            if name == WORD_EMBEDDINGS_FOR_HEAD and weight is None and WORD_EMBEDDINGS in self.name_map:
                hf_name, _ = self.get_hf_name_and_args(self.name_map[WORD_EMBEDDINGS])
                hf_weight_path = f"{hf_name}.{WEIGHT}"
                weight, bias, weight_scale = self.get_common_from_state_dict(h_dict, hf_weight_path)
        else:
            if name in [ROTARY_EMB_INV_FREQ, ATTENTION_ROTARY_EMB_INV_FREQ]:
                assert self.use_rotary_position_embeddings == True, \
                        f"mcore args.use_rotary_position_embeddings is required to be set to True \
                        since we capture the rotary_emb op"
            if name == ATTENTION_QUERY_KEY_VALUE:
                weight, bias = self.get_attn_qkv_state_from_dict(name, h_dict, f"{self.transformer}.{layer_prefix}.{layer_id}")
            elif name == MLP_DENSE_H_TO_4H:
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}"
                weight, bias, weight_scale = self.get_h_to_4h_from_state_dict(name, h_dict, hf_prefix_path)
            elif expert_name == MOE_SHARED_EXPERT:
                if expert_name not in self.name_map:
                    return
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}"
                weight, bias, weight_scale = self.get_h_to_4h_from_state_dict(name, h_dict, hf_prefix_path)
            else:
                if expert_name is None:
                    hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{hf_name}"
                else:
                    hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}"
                if is_direct_name:
                    hf_weight_path = hf_prefix_path
                else:
                    hf_weight_path = f"{hf_prefix_path}.{WEIGHT}"
                bias_name = f"{name}.{BIAS}"
                hf_bias_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_prefix_path}.{BIAS}"
                hf_weight_scale_path = f"{hf_prefix_path}.{WEIGHT_SCALE}"
                weight, bias, weight_scale = self.get_common_from_state_dict(
                        h_dict, hf_weight_path, hf_bias_path=hf_bias_path, hf_weight_scale_path=hf_weight_scale_path)
                if ATTENTION_ROTARY_EMB_INV_FREQ == name and weight is None:
                    weight = 1.0 / (self.rotary_base ** (torch.arange(0, self.attn_dim, 2).float() / self.attn_dim))
                # For attention padded heads
                if self.num_padded_heads != 0 and name in [ATTENTION_DENSE, ATTENTION_QNORM, ATTENTION_KNORM]:
                    weight = self.get_padded_head_weight(name, weight)

        c_ckpt.set(common_key, weight, bias, weight_scale)

    def get_common_from_state_dict(self, h_dict, hf_weight_path, hf_bias_path=None, hf_weight_scale_path=None):
        weight = h_dict[hf_weight_path] if hf_weight_path in h_dict else None
        bias = h_dict[hf_bias_path] if hf_bias_path in h_dict else None
        weight_scale = h_dict[hf_weight_scale_path] if hf_weight_scale_path in h_dict else None
        return weight, bias, weight_scale

    def get_attn_qkv_state_from_dict(self, name, h_dict, hf_prefix_path):
        hf_name = self.name_map[name]
        qkv_names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = []
        bias_list = []
        for qkv_name in qkv_names:
            hf_path= f"{hf_prefix_path}.{qkv_name}"
            if f"{hf_path}.{WEIGHT}" in h_dict:
                weight_list.append(h_dict[f"{hf_path}.{WEIGHT}"])
            if f"{hf_path}.{BIAS}" in h_dict:
                bias_list.append(h_dict[f"{hf_path}.{BIAS}"])
        weight, bias = self.get_attn_qkv_from_list(weight_list, bias_list)
        return weight, bias

    def get_attn_qkv_from_list(self, weight_list, bias_list):
        weight = self.cat_attn_qkv(weight_list) if len(weight_list) > 0 else None
        bias = self.cat_attn_qkv(bias_list) if len(bias_list) > 0 else None

        return weight, bias

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

    def get_h_to_4h_from_state_dict(self, name, h_dict, hf_prefix_path):
        hf_name = self.name_map[name]
        hf_names = hf_name if isinstance(hf_name, (list, ListConfig)) else [hf_name]
        weight_list = []
        bias_list = []
        weight_scale_list = []
        for hf_name in hf_names:
            hf_path= f"{hf_prefix_path}.{hf_name}"
            hf_weight_name = f"{hf_path}.{WEIGHT}"
            hf_bias_name = f"{hf_path}.{BIAS}"
            if f"{name}.{BIAS}" in self.name_map:
                hf_bias_name = self.name_map[f"{name}.{BIAS}"]
            hf_weight_scale_name = f"{hf_path}.{WEIGHT_SCALE}"
            if hf_weight_name in h_dict:
                weight_list.append(h_dict[hf_weight_name])
            if hf_bias_name in h_dict:
                bias_list.append(h_dict[hf_bias_name])
            if hf_weight_scale_name in h_dict:
                weight_scale_list.append(h_dict[hf_weight_scale_name])
        weight = torch.cat(weight_list, dim=0) if len(weight_list) > 0 else None
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
