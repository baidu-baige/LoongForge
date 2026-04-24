# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Mcore_checkpoint converter for megatron lm. """

import logging

logging.basicConfig(level=logging.INFO)

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.common_checkpoint import LAYER_IS_DICT_FOR_EXPERT, CommonCheckpoint

from convert_checkpoint.huggingface.huggingface_base import HuggingfaceBase

from convert_checkpoint.common.common_checkpoint import (
    WEIGHT,
    BIAS,
    MOE_EXPERT_H_TO_4H,
    LAYER_IS_DICT_FOR_EXPERT,
)


class HuggingfaceMoe(HuggingfaceBase):
    """
        HuggingfaceMoe
    """

    def __init__(self, c_config, args):
        super().__init__(c_config, args)

    #========from commmon to hf===========
    def common_e_to_hf(self, expert_name, name, c_ckpt, h_dict, layer_id=None, hf_layer_id=None,
                       expert_id=None, layer_prefix=None, transformer=None, spec_name=None):
        spec_name = name if spec_name is None or spec_name not in self.name_map else spec_name
        if spec_name not in self.name_map or expert_name not in self.name_map:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        transformer = self.transformer if transformer is None else transformer
        common_key = CommonCheckpoint.get_key(f"{expert_name}.{name}", layer_id=layer_id, expert_id=expert_id)
        weight, bias, weight_scale = c_ckpt.get(common_key)
        hf_name, is_direct_name, is_dict_for_expert, need_transpose, _, _, _ = self.get_hf_name_and_args(self.name_map[spec_name])
        if name == MOE_EXPERT_H_TO_4H:
            if expert_id is None or is_dict_for_expert:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}"
            else:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}.{expert_id}"
            self.update_h_to_4h(h_dict, spec_name, hf_prefix_path, weight, bias, weight_scale, expert_id=expert_id)
        else:
            # MOE_EXPERT_4H_TO_H
            weight = weight.t() if weight is not None and need_transpose else weight
            if expert_id is None or is_dict_for_expert:
                hf_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{hf_name}"
            else:
                hf_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{hf_name}"
            hf_weight_path = f"{hf_path}.{WEIGHT}" if not is_direct_name else hf_path
            bias_name = f"{spec_name}.{BIAS}"
            if expert_id is None or is_dict_for_expert:
                hf_bias_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            else:
                hf_bias_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            hf_weight_scale_path = f"{hf_path}.{self.weight_scale_suffix}"
            self.update_tensor(h_dict, hf_weight_path, weight, hf_bias_path=hf_bias_path, bias=bias,
                    hf_weight_scale_path=hf_weight_scale_path, weight_scale=weight_scale,
                    expert_id=expert_id, is_dict_for_expert=is_dict_for_expert)

    def update_tensor(self, h_dict, hf_weight_path, weight, hf_bias_path=None, bias=None,
                      hf_weight_scale_path=None, weight_scale=None, expert_id=None, is_dict_for_expert=False):
        if weight is None:
            return
        if is_dict_for_expert:
            assert expert_id is not None, "expert_id must be specified when is_dict_for_expert"
            h_dict[hf_weight_path] = {LAYER_IS_DICT_FOR_EXPERT: True} if hf_weight_path not in h_dict else h_dict[hf_weight_path]
            h_dict[hf_weight_path][expert_id] = weight
        else:
            h_dict[hf_weight_path] = weight
        if bias is not None and hf_bias_path is not None:
            h_dict[hf_bias_path] = bias
        if weight_scale is not None and hf_weight_scale_path is not None:
            h_dict[hf_weight_scale_path] = weight_scale
    # ====== from hf to common ========

    def hf_e_to_common(self, expert_name, name, c_ckpt, h_dict, layer_id=None, hf_layer_id=None,
                       expert_id=None, layer_prefix=None, transformer=None, spec_name=None):
        spec_name = name if spec_name is None or spec_name not in self.name_map else spec_name
        if spec_name not in self.name_map or expert_name not in self.name_map:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        transformer = self.transformer if transformer is None else transformer
        common_key = CommonCheckpoint.get_key(f"{expert_name}.{name}", layer_id=layer_id, expert_id=expert_id)
        hf_name, is_direct_name, is_dict_for_expert, need_transpose, _, _, _ = self.get_hf_name_and_args(self.name_map[spec_name])
        if name == MOE_EXPERT_H_TO_4H:
            if expert_id is None or is_dict_for_expert:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}"
            else:
                hf_prefix_path = f"{transformer}.{layer_prefix}.{hf_layer_id}.{self.name_map[expert_name]}.{expert_id}"
            weight, bias, weight_scale = self.get_h_to_4h_from_state_dict(spec_name, h_dict, hf_prefix_path, expert_id=expert_id)
        else:
            # MOE_EXPERT_4H_TO_H
            if expert_id is None or is_dict_for_expert:
                hf_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{hf_name}"
            else:
                hf_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{hf_name}"
            hf_weight_path = f"{hf_path}.{WEIGHT}" if not is_direct_name else hf_path
            bias_name = f"{spec_name}.{BIAS}"
            if expert_id is None or is_dict_for_expert:
                hf_bias_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            else:
                hf_bias_path = f"{transformer}.{layer_prefix}.{hf_layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            hf_weight_scale_path = f"{hf_path}.{self.weight_scale_suffix}"
            weight, bias, weight_scale = self.get_from_state_dict(
                    h_dict, hf_weight_path, hf_bias_path=hf_bias_path, hf_weight_scale_path=hf_weight_scale_path,
                    expert_id=expert_id, is_dict_for_expert=is_dict_for_expert)
            weight = weight.t() if weight is not None and need_transpose else weight
        log_flag = (expert_id is None or expert_id == 0)
        c_ckpt.set(common_key, weight, bias, weight_scale, log_flag=log_flag)

    def get_from_state_dict(self, h_dict, hf_weight_path, hf_bias_path=None, hf_weight_scale_path=None,
                                   expert_id=None, is_dict_for_expert=False):
        if is_dict_for_expert:
            assert expert_id is not None, "expert_id must be specified when is_dict_for_expert"
            weight = h_dict[hf_weight_path][expert_id] if hf_weight_path in h_dict else None
        else:
            weight = h_dict[hf_weight_path] if hf_weight_path in h_dict else None
        bias = h_dict[hf_bias_path] if hf_bias_path in h_dict else None
        weight_scale = h_dict[hf_weight_scale_path] if hf_weight_scale_path in h_dict else None
        return weight, bias, weight_scale