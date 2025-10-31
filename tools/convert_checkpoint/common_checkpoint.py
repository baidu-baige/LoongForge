#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

from convert_checkpoint.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.utils import get_element_from_dict_by_path
from convert_checkpoint.arguments import parse_args


""" Name of common checkpoint layers, matching key of name_map in {model}.json """

WORD_EMBEDDINGS = "word_embeddings"
WORD_POSITION_EMBEDDINGS = "word_position_embeddings"
WORD_BLOCK_POSITION_EMBEDDINGS = "word_block_position_embeddings"
LAYER_PREFIX = "layer_prefix"
MTP_LAYER_PREFIX = "mtp_layer_prefix"
TRANSFORMER = "transformer"
INPUT_LAYERNORM = "input_layernorm"
ROTARY_EMB_INV_FREQ = "rotary_emb.inv_freq"
ATTENTION_ROTARY_EMB_INV_FREQ = "attention.rotary_emb.inv_freq"
ATTENTION_QUERY_KEY_VALUE = "attention.query_key_value"
ATTENTION_QKV_MAP = "attention.qkv_map"
ATTENTION_DENSE = "attention.dense"
ATTENTION_QNORM = "attention.q_norm"
ATTENTION_KNORM = "attention.k_norm"
POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
POST_ATTENTION_LAYERSCALE = "post_attention_layerscale"
MOE_GATE = "moe.gate"
MOE_GATE_BIAS = "moe.gate.bias"
MOE_MLP = "moe.mlp"
MOE_EXPERT = "moe.expert"
MOE_GROUPED_GEMM_EXPERT = "moe.groupedgemm.expert"
MOE_SHARED_EXPERT = "moe.shared_expert"
MLP_DENSE_H_TO_4H = "mlp.dense_h_to_4h"
MLP_DENSE_4H_TO_H = "mlp.dense_4h_to_h"
MOE_SHARED_EXPERT_DENSE_H_TO_4H = "mlp.shared_expert.dense_h_to_4h"
POST_MLP_LAYERNORM = "post_mlp_layernorm"
POST_MLP_LAYERSCALE = "post_mlp_layerscale"
FINAL_LAYERNORM = "final_layernorm"
WORD_EMBEDDINGS_FOR_HEAD = "word_embeddings_for_head"

WORD_EMBEDDINGS_TPL = "word_embeddings_tpl"
WORD_POSITION_EMBEDDINGS_TPL = "word_position_embeddings_tpl"
WORD_BLOCK_POSITION_EMBEDDINGS_TPL = "word_block_position_embeddings_tpl"
TRANSFORMER_TPL = "transformer_tpl"
WORD_EMBEDDINGS_FOR_HEAD_TPL = "word_embeddings_for_head_tpl"

MTP_WORD_EMBEDDING = "mtp_word_embeddings"
MTP_ENORM = "mtp_enorm"
MTP_HNORM = "mtp_hnorm"
MTP_EH_PROJ = "mtp_eh_proj"
MTP_SHARED_HEAD_NORM = "mtp_shared_head_norm"
MTP_SHARED_HEAD_HEAD = "mtp_shared_head_head"


class CommonCheckpoint(AbstractCheckpoint):
    """
       CommonCheckpoint
    """
    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.other_args = {}
        self.args = parse_args()

    @staticmethod
    def convert_from_common(*args, **kwargs):
        raise NotImplementedError()
    
    def convert_to_common(self, *args, **kwargs):
        raise NotImplementedError()

    def convert(self, ckpt_class, *args, **kwargs):
        return ckpt_class.convert_from_common(self, *args, **kwargs)
          
    def set_word_embedding(self, weight):
        self._set(WORD_EMBEDDINGS, "weight", weight)

    def clear_word_embedding(self):
        self._clear(WORD_EMBEDDINGS, "weight")

    def get_word_embedding(self):
        return self._get(WORD_EMBEDDINGS, "weight")

    def set_word_position_embedding(self, weight):
        self._set(WORD_POSITION_EMBEDDINGS, "weight", weight)

    def clear_word_position_embedding(self):
        self._clear(WORD_POSITION_EMBEDDINGS, "weight")

    def get_word_position_embedding(self):
        return self._get(WORD_POSITION_EMBEDDINGS, "weight")

    def set_word_block_position_embedding(self, weight):
        self._set(WORD_BLOCK_POSITION_EMBEDDINGS, "weight", weight)

    def clear_word_block_position_embedding(self):
        self._clear(WORD_BLOCK_POSITION_EMBEDDINGS, "weight")

    def get_word_block_position_embedding(self):
        return self._get(WORD_BLOCK_POSITION_EMBEDDINGS, "weight")

    def set_layer_input_layernorm(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{INPUT_LAYERNORM}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_input_layernorm(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{INPUT_LAYERNORM}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_input_layernorm_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{INPUT_LAYERNORM}"
        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def get_layer_input_layernorm_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{INPUT_LAYERNORM}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_attention_rotary_emb_inv_freq(self, index, inv_freq, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_ROTARY_EMB_INV_FREQ}"
        if one_layer_weights is None:
            self._set(path, "value", inv_freq)
        else:
            one_layer_weights[path] = inv_freq

    def clear_layer_attention_rotary_emb_inv_freq(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_ROTARY_EMB_INV_FREQ}"
        if one_layer_weights is None:
            self._clear(path, "value")
        else:
            one_layer_weights[path] = None

    def get_layer_attention_rotary_emb_inv_freq(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_ROTARY_EMB_INV_FREQ}"
        if one_layer_weights is None:
            return self._get(path, "value")
        else:
            return one_layer_weights[path]

    # ====attention qkv a b=====
    def set_layer_attention_by_name(self, name, index, weight, bias, one_layer_weights=None, weight_scale_inv=None):
        """
        Sets the attention weights and biases of the Transformer layer by name.
        Args:
            name (str): The name of the attention parameter to be set, which can be one of "query_weight", "key_weight", or "value_weight".
            index (int): The index number of the Transformer layer, counted from 0.
            weight (Tensor): The weight matrix of the attention parameter, with a shape of (hidden_size, hidden_size).
            bias (Tensor, optional): The bias vector of the attention parameter, with a shape of (hidden_size,), defaults to None.
        """
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{name}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
            if weight_scale_inv is not None:
                self._set(path, "weight_scale_inv", weight_scale_inv)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias
            if weight_scale_inv is not None:
                one_layer_weights[path + ".weight_scale_inv"] = weight_scale_inv

    def clear_layer_attention_by_name(self, name, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{name}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
            self._clear(path, "weight_scale_inv")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None
            one_layer_weights[path + ".weight_scale_inv"] = None

    def get_layer_attention_weight_by_name(self, name, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{name}"
        if one_layer_weights is None:
            weight = self._get(path, "weight")
            weight_scale_inv = self._get(path, "weight_scale_inv")
        else:
            weight = one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None
            weight_scale_inv = one_layer_weights[path + ".weight_scale_inv"] \
                if path + ".weight_scale_inv" in one_layer_weights else None
        return weight, weight_scale_inv

    def get_layer_attention_bias_by_name(self, name, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{name}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None
    # ====attention qkv a b end=====

    def set_layer_attention_query_key_value(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QUERY_KEY_VALUE}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_attention_query_key_value(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QUERY_KEY_VALUE}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_attention_query_key_value_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QUERY_KEY_VALUE}"
        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None
    
    def get_layer_attention_query_key_value_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QUERY_KEY_VALUE}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_attention_dense(self, index, weight, bias, one_layer_weights=None, weight_scale_inv=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_DENSE}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
            if weight_scale_inv is not None:
                self._set(path, "weight_scale_inv", weight_scale_inv)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias
            if weight_scale_inv is not None:
                one_layer_weights[path + ".weight_scale_inv"] = weight_scale_inv

    def clear_layer_attention_dense(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_DENSE}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
            self._clear(path, "weight_scale_inv")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None
            one_layer_weights[path + ".weight_scale_inv"] = None

    def get_layer_attention_dense_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_DENSE}"
        if one_layer_weights is None:
            weight = self._get(path, "weight")
            weight_scale_inv = self._get(path, "weight_scale_inv")
        else:
            weight = one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None
            weight_scale_inv = one_layer_weights[path + ".weight_scale_inv"] \
                if path + ".weight_scale_inv" in one_layer_weights else None
        return weight, weight_scale_inv

    def get_layer_attention_dense_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_DENSE}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_post_attention_layernorm(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERNORM}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_post_attention_layernorm(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERNORM}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_post_attention_layernorm_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERNORM}"
        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def get_layer_post_attention_layernorm_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERNORM}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_attention_q_layernorm(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QNORM}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_attention_q_layernorm(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QNORM}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_attention_q_layernorm_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QNORM}"

        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def get_layer_attention_q_layernorm_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_QNORM}"

        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_attention_k_layernorm(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_KNORM}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_attention_k_layernorm(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_KNORM}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_attention_k_layernorm_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_KNORM}"

        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def get_layer_attention_k_layernorm_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{ATTENTION_KNORM}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_post_attention_layerscale(self, index, weight, bias=None, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERSCALE}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
        else:
            one_layer_weights[path + ".weight"] = weight

    def clear_layer_post_attention_layerscale(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERSCALE}"
        if one_layer_weights is None:
            self._clear(path, "weight")
        else:
            one_layer_weights[path + ".weight"] = None

    def get_layer_post_attention_layerscale(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_ATTENTION_LAYERSCALE}"

        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def set_layer_post_mlp_layerscale(self, index, weight, bias=None, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERSCALE}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
        else:
            one_layer_weights[path + ".weight"] = weight

    def clear_layer_post_mlp_layerscale(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERSCALE}"
        if one_layer_weights is None:
            self._clear(path, "weight")
        else:
            one_layer_weights[path + ".weight"] = None

    def get_layer_post_mlp_layerscale(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERSCALE}"
        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def set_layer_mlp_dense_h_to_4h(self, index, weight, bias, is_moe_mlp=False, expert_id=None, is_shared=False,
                                    one_layer_weights=None, weight_scale_inv=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_H_TO_4H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_H_TO_4H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_H_TO_4H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_H_TO_4H}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
            if weight_scale_inv is not None:
                self._set(path, "weight_scale_inv", weight_scale_inv)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias
            if weight_scale_inv is not None:
                one_layer_weights[path + ".weight_scale_inv"] = weight_scale_inv

    def clear_layer_mlp_dense_h_to_4h(self, index, is_moe_mlp = False, expert_id = None, is_shared = False,
                                      one_layer_weights=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_H_TO_4H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_H_TO_4H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_H_TO_4H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_H_TO_4H}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
            self._clear(path, "weight_scale_inv")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None
            one_layer_weights[path + ".weight_scale_inv"] = None

    def get_layer_mlp_dense_h_to_4h_weight(self, index, is_moe_mlp = False, expert_id = None, is_shared = False,
                                           one_layer_weights=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_H_TO_4H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_H_TO_4H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_H_TO_4H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_H_TO_4H}"
        if one_layer_weights is None:
            weight = self._get(path, "weight")
            weight_scale_inv = self._get(path, "weight_scale_inv")
        else:
            weight = one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None
            weight_scale_inv = one_layer_weights[path + ".weight_scale_inv"] \
                if path + ".weight_scale_inv" in one_layer_weights else None
        return weight, weight_scale_inv

    def get_layer_mlp_dense_h_to_4h_bias(self, index, is_moe_mlp = False, expert_id = None, is_shared = False,
                                         one_layer_weights=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_H_TO_4H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_H_TO_4H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_H_TO_4H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_H_TO_4H}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_mlp_dense_4h_to_h(self, index, weight, bias, is_moe_mlp=False, expert_id=None, is_shared=False,
                                    one_layer_weights=None, weight_scale_inv=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_4H_TO_H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_4H_TO_H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_4H_TO_H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_4H_TO_H}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
            if weight_scale_inv is not None:
                self._set(path, "weight_scale_inv", weight_scale_inv)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias
            if weight_scale_inv is not None:
                one_layer_weights[path + ".weight_scale_inv"] = weight_scale_inv

    def clear_layer_mlp_dense_4h_to_h(self, index, is_moe_mlp = False, expert_id = None, is_shared = False,
                                      one_layer_weights=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_4H_TO_H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_4H_TO_H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_4H_TO_H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_4H_TO_H}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
            self._clear(path, "weight_scale_inv")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None
            one_layer_weights[path + ".weight_scale_inv"] = None

    def get_layer_mlp_dense_4h_to_h_weight(self, index, is_moe_mlp = False, expert_id = None, is_shared = False,
                                           one_layer_weights=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_4H_TO_H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_4H_TO_H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_4H_TO_H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_4H_TO_H}"
        if one_layer_weights is None:
            weight = self._get(path, "weight")
            weight_scale_inv = self._get(path, "weight_scale_inv")
        else:
            weight = one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None
            weight_scale_inv = one_layer_weights[path + ".weight_scale_inv"] \
                if path + ".weight_scale_inv" in one_layer_weights else None
        return weight, weight_scale_inv

    def get_layer_mlp_dense_4h_to_h_bias(self, index, is_moe_mlp = False, expert_id = None, is_shared = False,
                                         one_layer_weights=None):
        if is_moe_mlp:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_MLP}.{MLP_DENSE_4H_TO_H}"
        elif is_shared:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_SHARED_EXPERT}.{MLP_DENSE_4H_TO_H}"
        elif expert_id is None:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MLP_DENSE_4H_TO_H}"
        else:
            path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_EXPERT}.{expert_id}.{MLP_DENSE_4H_TO_H}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_post_mlp_layernorm(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERNORM}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_post_mlp_layernorm(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERNORM}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_post_mlp_layernorm_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERNORM}"
        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def get_layer_post_mlp_layernorm_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{POST_MLP_LAYERNORM}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_moe_gate(self, index, weight, bias, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_GATE}"
        if one_layer_weights is None:
            self._set(path, "weight", weight)
            self._set(path, "bias", bias)
        else:
            one_layer_weights[path + ".weight"] = weight
            one_layer_weights[path + ".bias"] = bias

    def clear_layer_moe_gate(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_GATE}"
        if one_layer_weights is None:
            self._clear(path, "weight")
            self._clear(path, "bias")
        else:
            one_layer_weights[path + ".weight"] = None
            one_layer_weights[path + ".bias"] = None

    def get_layer_moe_gate_weight(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_GATE}"
        if one_layer_weights is None:
            return self._get(path, "weight")
        else:
            return one_layer_weights[path + ".weight"] if path + ".weight" in one_layer_weights else None

    def get_layer_moe_gate_bias(self, index, one_layer_weights=None):
        path = f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MOE_GATE}"
        if one_layer_weights is None:
            return self._get(path, "bias")
        else:
            return one_layer_weights[path + ".bias"] if path + ".bias" in one_layer_weights else None

    def set_layer_mtp_weight(self, index, mtp_word_embedding, mtp_enorm, mtp_hnorm, mtp_eh_proj, \
                             mtp_shared_head_norm, mtp_shared_head_head, one_layer_weights=None):
        if one_layer_weights is None:
            self._set(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_WORD_EMBEDDING}", "weight", mtp_word_embedding)
            self._set(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_ENORM}", "weight", mtp_enorm)
            self._set(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_HNORM}", "weight", mtp_hnorm)
            self._set(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_EH_PROJ}", "weight", mtp_eh_proj)
            self._set(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_NORM}", "weight", mtp_shared_head_norm)
            self._set(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_HEAD}", "weight", mtp_shared_head_head)
        else:
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_WORD_EMBEDDING}.weight"] = mtp_word_embedding
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_ENORM}.weight"] = mtp_enorm
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_HNORM}.weight"] = mtp_hnorm
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_EH_PROJ}.weight"] = mtp_eh_proj
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_NORM}.weight"] = mtp_shared_head_norm
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_HEAD}.weight"] = mtp_shared_head_head

    def clear_layer_mtp_weight(self, index, one_layer_weights=None):
        if one_layer_weights is None:
            self._clear(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_WORD_EMBEDDING}", "weight")
            self._clear(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_ENORM}", "weight")
            self._clear(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_HNORM}", "weight")
            self._clear(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_EH_PROJ}", "weight")
            self._clear(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_NORM}", "weight")
            self._clear(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_HEAD}", "weight")
        else:
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_WORD_EMBEDDING}.weight"] = None
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_ENORM}.weight"] = None
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_HNORM}.weight"] = None
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_EH_PROJ}.weight"] = None
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_NORM}.weight"] = None
            one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_HEAD}.weight"] = None

    def get_layer_mtp_weight(self, index, one_layer_weights=None):
        if one_layer_weights is None:
            mtp_word_embedding = self._get(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_WORD_EMBEDDING}", "weight")
            mtp_enorm = self._get(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_ENORM}", "weight")
            mtp_hnorm = self._get(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_HNORM}", "weight")
            mtp_eh_proj = self._get(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_EH_PROJ}", "weight")
            mtp_shared_head_norm = self._get(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_NORM}", "weight")
            mtp_shared_head_head = self._get(f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_HEAD}", "weight")
        else:
            mtp_word_embedding = one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_WORD_EMBEDDING}.weight"]
            mtp_enorm = one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_ENORM}.weight"]
            mtp_hnorm = one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_HNORM}.weight"]
            mtp_eh_proj = one_layer_weights[f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_EH_PROJ}.weight"]
            mtp_shared_head_norm = one_layer_weights[
                f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_NORM}.weight"]
            mtp_shared_head_head = one_layer_weights[
                f"{TRANSFORMER}.{LAYER_PREFIX}.{index}.{MTP_SHARED_HEAD_HEAD}.weight"]

        return (mtp_word_embedding, mtp_enorm, mtp_hnorm, mtp_eh_proj), (mtp_shared_head_norm, mtp_shared_head_head)

    def set_final_layernorm(self, weight, bias):
        self._set(FINAL_LAYERNORM, "weight", weight)
        self._set(FINAL_LAYERNORM, "bias", bias)

    def clear_final_layernorm(self):
        self._clear(FINAL_LAYERNORM, "weight")
        self._clear(FINAL_LAYERNORM, "bias")

    def get_final_layernorm_weight(self):
        return self._get(FINAL_LAYERNORM, "weight")

    def get_final_layernorm_bias(self):
        return self._get(FINAL_LAYERNORM, "bias")

    def set_word_embeddings_for_head(self, weight):
        self._set(WORD_EMBEDDINGS_FOR_HEAD, "weight", weight)

    def clear_word_embeddings_for_head(self):
        self._clear(WORD_EMBEDDINGS_FOR_HEAD, "weight")

    def get_word_embeddings_for_head_weight(self):
        return self._get(WORD_EMBEDDINGS_FOR_HEAD, "weight")

    def _set(self, path, key, val):
        if val is not None:
            element = get_element_from_dict_by_path(
                self.state_dict, path
            )
            element[key] = val

    def _clear(self, path, key):
        element = get_element_from_dict_by_path(
            self.state_dict, path
        )
        element[key] = None

    def _get(self, path, key):
        element = get_element_from_dict_by_path(
            self.state_dict, path
        )
        return element[key] if key in element else None

    def has_optimizer(self):
        return "optimizer" in self.state_dict
