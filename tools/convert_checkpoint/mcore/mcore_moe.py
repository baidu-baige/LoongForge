# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert MoE checkpoint parameters between common and Megatron Core formats."""

import io
import torch
import logging

logging.basicConfig(level=logging.INFO)

from convert_checkpoint.common.common_checkpoint import CommonCheckpoint

from convert_checkpoint.mcore.mcore_base import McoreBase

from convert_checkpoint.common.common_checkpoint import (
    WEIGHT,
    BIAS,
    EXTRA_DATA,
    MOE_EXPERT,
    MOE_GROUPED_GEMM_EXPERT,
    LORA_NAME_IN,
    LORA_NAME_OUT
)


class McoreMoe(McoreBase):
    """
        McoreMoe
    """

    def __init__(self, c_config, args):
        super().__init__(c_config, args)

    def common_e_to_mcore(self, expert_name, name, c_ckpt, m_dict, t_name, layer_id, m_layer_id,
                          ep_id=None, expert_id=None, layer_prefix=None, name_prefix=None):
        if name not in self.name_map or expert_name not in self.name_map:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        common_key = CommonCheckpoint.get_key(f"{expert_name}.{name}", layer_id=layer_id, expert_id=expert_id)
        (mcore_name, has_extra, is_layernorm), (is_fp8, fp8_ignore_tp), (is_direct_name, ignore_tp, dtype) = self.get_mcore_name_and_extra(self.name_map[name])
        weight, bias, weight_scale = c_ckpt.get(common_key)
        local_eid = self.expert_local_mapping[expert_id]
        if self.args.moe_grouped_gemm:
            m_name_prefix = self.name_map[MOE_GROUPED_GEMM_EXPERT] if name_prefix is None \
                    else f"{name_prefix}.{self.name_map[MOE_GROUPED_GEMM_EXPERT]}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}.{mcore_name}"
            mcore_weight_path = f"{mcore_path}.{WEIGHT}{local_eid}"
            mcore_bias_path = f"{mcore_path}.{BIAS}{local_eid}"
        else:
            m_name_prefix = self.name_map[MOE_EXPERT] if name_prefix is None \
                    else f"{name_prefix}.{self.name_map[MOE_EXPERT]}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}.{local_eid}.{mcore_name}"
            mcore_weight_path = f"{mcore_path}.{WEIGHT}"
            mcore_bias_path = f"{mcore_path}.{BIAS}"
        mcore_extra_path=f"{mcore_path}.{EXTRA_DATA}"
        etp_to_tp = self.etp_to_tp_mapping[ep_id] if self.etp is not None else None
        log_flag = (expert_id is None or expert_id == 0)
        if self.etp is None:
            weight_list, bias_list = self.get_chunked_weight(
                    name, self.tp, mcore_weight_path, mcore_bias_path,
                    weight, bias, weight_scale, is_fp8, fp8_ignore_tp, log_flag=log_flag,
                    ignore_tp=ignore_tp)
            self.update_mcore_expert_weight(
                    m_dict, t_name, mcore_weight_path, mcore_bias_path, weight_list,
                    m_tp=self.tp, has_extra=has_extra, mcore_extra_path=mcore_extra_path,
                    bias_list=bias_list)
        else:
            weight_list, bias_list = self.get_chunked_weight(
                    name, self.etp, mcore_weight_path, mcore_bias_path,
                    weight, bias, weight_scale, is_fp8, fp8_ignore_tp, log_flag=log_flag,
                    ignore_tp=ignore_tp)
            self.update_mcore_expert_weight(
                    m_dict, t_name, mcore_weight_path, mcore_bias_path, weight_list, 
                    m_tp=self.etp, has_extra=has_extra, mcore_extra_path=mcore_extra_path,
                    bias_list=bias_list, etp_to_tp=etp_to_tp)

    def update_mcore_expert_weight(
            self, m_dict, t_name, mcore_weight_path, mcore_bias_path,
            weight_list, m_tp, has_extra=False, mcore_extra_path=None, bias_list=None, etp_to_tp=None):
        if weight_list is None:
            return
        for mt in range(m_tp):
            if self.etp is None:
                t = mt
            else:
                et = mt
                t = etp_to_tp[et]
            if mt not in m_dict:
                continue
            m_dict[mt][t_name][f"{mcore_weight_path}"] = weight_list[mt]
            if bias_list is not None:
                m_dict[mt][t_name][f"{mcore_bias_path}"] = bias_list[mt]
            if has_extra:
                m_dict[mt][t_name][f"{mcore_extra_path}"] = None

    # =====mcore to common====
    def mcore_e_to_common(self, expert_name, name, c_ckpt, e_m_dict, t_name, layer_id,
                          m_layer_id, expert_id=None, layer_prefix=None, name_prefix=None):
        if name not in self.name_map or expert_name not in self.name_map:
            return
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        common_key = CommonCheckpoint.get_key(f"{expert_name}.{name}", layer_id=layer_id, expert_id=expert_id)
        (mcore_name, has_extra, is_layernorm), (is_fp8, fp8_ignore_tp), (is_direct_name, ignore_tp, dtype) = self.get_mcore_name_and_extra(self.name_map[name])
        local_eid = self.expert_local_mapping[expert_id]
        if self.args.moe_grouped_gemm:
            m_name_prefix = self.name_map[MOE_GROUPED_GEMM_EXPERT] if name_prefix is None \
                    else f"{name_prefix}.{self.name_map[MOE_GROUPED_GEMM_EXPERT]}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}.{mcore_name}"
            mcore_weight_path = f"{mcore_path}.{WEIGHT}{local_eid}"
            mcore_bias_path = f"{mcore_path}.{BIAS}{local_eid}"
            mcore_lora_in_path = f"{mcore_path}.{LORA_NAME_IN}.{WEIGHT}{local_eid}"
            mcore_lora_out_path = f"{mcore_path}.{LORA_NAME_OUT}.{WEIGHT}{local_eid}"
        else:
            m_name_prefix = self.name_map[MOE_EXPERT] if name_prefix is None \
                    else f"{name_prefix}.{self.name_map[MOE_EXPERT]}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}.{local_eid}.{mcore_name}"
            mcore_weight_path = f"{mcore_path}.{WEIGHT}"
            mcore_bias_path = f"{mcore_path}.{BIAS}"
            mcore_lora_in_path = f"{mcore_path}.{LORA_NAME_IN}.{WEIGHT}"
            mcore_lora_out_path = f"{mcore_path}.{LORA_NAME_OUT}.{WEIGHT}"
        weight_list, bias_list, weight_scale_list = self.get_mcore_e_weight_list(
                e_m_dict, t_name, mcore_weight_path, mcore_bias_path)
        lora_in_weight_list, _, _ = self.get_mcore_e_weight_list(e_m_dict, t_name, mcore_lora_in_path, None)
        lora_out_weight_list, _, _ = self.get_mcore_e_weight_list(e_m_dict, t_name, mcore_lora_out_path, None)

        m_tp = self.etp if self.etp is not None else self.tp
        weight, bias, weight_scale = self.get_cat_weight(
                name, m_tp, weight_list, bias_list, weight_scale_list, is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp)
        if lora_in_weight_list is not None and lora_out_weight_list is not None:
            # Merge lora weight
            lora_out_weight, _, _ = self.get_cat_weight(
                name, self.tp, lora_out_weight_list, None, None, is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp, chunk_dim=0)
            lora_in_weight, _, _ = self.get_cat_weight(
                name, self.tp, lora_in_weight_list, None, None, is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp)
            weight = self.lora_merge(weight, lora_out_weight, lora_in_weight, self.lora_alpha, self.lora_dim)

        log_flag = (expert_id is None or expert_id == 0)
        c_ckpt.set(common_key, weight, bias=bias, weight_scale=weight_scale, log_flag=log_flag)

    def get_mcore_e_weight_list(self, e_m_dict, t_name, mcore_weight_path, mcore_bias_path):
        # e_m_dict: 
        #   etp is None: t->dict
        #   etp is not None: et->dict
        m_tp = self.tp if self.etp is None else self.etp
        weight_list = [None] * m_tp
        bias_list = [None] * m_tp
        weight_scale_list = [None] * m_tp
        for mt in range(m_tp):
            assert mt in e_m_dict, f"tp={mt} not found in m_dict. {e_m_dict.keys()=}"
            weight_list[mt], bias_list[mt], weight_scale_list[mt] = \
                    self.get_weight_by_tp(e_m_dict[mt], t_name, mcore_weight_path, mcore_bias_path)
        weight_list = None if all(x is None for x in weight_list) else weight_list
        bias_list = None if all(x is None for x in bias_list) else bias_list
        weight_scale_list = None if all(x is None for x in weight_scale_list) else weight_scale_list
        return weight_list, bias_list, weight_scale_list
