""" Mcore_checkpoint converter for aiak megatron. """
import logging

logging.basicConfig(level=logging.INFO)

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.common_checkpoint import CommonCheckpoint

from convert_checkpoint.huggingface.huggingface_base import HuggingfaceBase

from convert_checkpoint.common.common_checkpoint import (
    WEIGHT,
    BIAS,
    WEIGHT_SCALE,
    MOE_EXPERT_H_TO_4H,
)


class HuggingfaceMoe(HuggingfaceBase):
    """
        HuggingfaceMoe
    """

    def __init__(self, c_config):
        super().__init__(c_config)

    #========from commmon to hf===========
    def common_e_to_hf(self, expert_name, name, c_ckpt, h_dict, layer_id=None, expert_id=None, layer_prefix=None):
        if name not in self.name_map:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        common_key = CommonCheckpoint.get_key(f"{expert_name}.{name}", layer_id=layer_id, expert_id=expert_id)
        weight, bias, weight_scale = c_ckpt.get(common_key)
        if name == MOE_EXPERT_H_TO_4H:
            if expert_id is None:
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}"
            else:
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}.{expert_id}"
            self.update_h_to_4h(h_dict, name, hf_prefix_path, weight, bias, weight_scale)
        else:
            # MOE_EXPERT_4H_TO_H
            if expert_id is None:
                hf_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}{self.name_map[name]}"
            else:
                hf_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{self.name_map[name]}"
            hf_weight_path = f"{hf_path}.{WEIGHT}"
            bias_name = f"{name}.{BIAS}"
            if expert_id is None:
                hf_bias_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            else:
                hf_bias_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            hf_weight_scale_path = f"{hf_path}.{WEIGHT_SCALE}"
            self.update_tensor(h_dict, hf_weight_path, weight, hf_bias_path=hf_bias_path, bias=bias,
                    hf_weight_scale_path=hf_weight_scale_path, weight_scale=weight_scale)

    # ====== from hf to common ========

    def hf_e_to_common(self, expert_name, name, c_ckpt, h_dict, layer_id=None, expert_id=None, layer_prefix=None):
        if name not in self.name_map:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        common_key = CommonCheckpoint.get_key(f"{expert_name}.{name}", layer_id=layer_id, expert_id=expert_id)
        if name == MOE_EXPERT_H_TO_4H:
            if expert_id is None:
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}"
            else:
                hf_prefix_path = f"{self.transformer}.{layer_prefix}.{layer_id}.{self.name_map[expert_name]}.{expert_id}"
            weight, bias, weight_scale = self.get_h_to_4h_from_state_dict(name, h_dict, hf_prefix_path)
        else:
            # MOE_EXPERT_4H_TO_H
            if expert_id is None:
                hf_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{self.name_map[name]}"
            else:
                hf_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{self.name_map[name]}"
            hf_weight_path = f"{hf_path}.{WEIGHT}"
            bias_name = f"{name}.{BIAS}"
            if expert_id is None:
                hf_bias_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            else:
                hf_bias_path = f"{self.transformer}.{layer_prefix}.{layer_id}."\
                        f"{self.name_map[expert_name]}.{expert_id}.{self.name_map[bias_name]}" \
                        if bias_name in self.name_map else f"{hf_path}.{BIAS}"
            hf_weight_scale_path = f"{hf_path}.{WEIGHT_SCALE}"
            weight, bias, weight_scale = self.get_common_from_state_dict(
                    h_dict, hf_weight_path, hf_bias_path=hf_bias_path, hf_weight_scale_path=hf_weight_scale_path)
        c_ckpt.set(common_key, weight, bias, weight_scale)
