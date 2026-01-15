#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

from convert_checkpoint.common.abstact_checkpoint import AbstractCheckpoint

import logging

logging.basicConfig(level=logging.INFO)


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

# MLA begin
ATTENTION_Q_DOWN = "attention.q_down"
ATTENTION_Q_UP = "attention.q_up"
ATTENTION_Q_UP_LAYERNORM = "attention.q_up_layernorm"
ATTENTION_KV_DOWN = "attention.kv_down"
ATTENTION_KV_UP = "attention.kv_up"
ATTENTION_KV_UP_LAYERNORM = "attention.kv_up_layernorm"
ATTENTION_Q = "attention.q"

ATTENTION_LIGHTNING_INDEXER_K_NORM = "attention.lightning_indexer.k_norm"
ATTENTION_LIGHTNING_INDEXER_WEIGHTS_PROJ = "attention.lightning_indexer.weights_proj"
ATTENTION_LIGHTHING_INDEXER_WK = "attention.lightning_indexer.wk"
ATTENTION_LIGHTHING_INDEXER_WQ_B = "attention.lightning_indexer.wq_b"
# MLA end

ATTENTION_QUERY_KEY_VALUE = "attention.query_key_value"

ATTENTION_DENSE = "attention.dense"
ATTENTION_QNORM = "attention.q_a_layernorm"
ATTENTION_KNORM = "attention.kv_a_layernorm"
POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
POST_ATTENTION_LAYERSCALE = "post_attention_layerscale"

# Dense begin
MLP_DENSE_H_TO_4H = "mlp.dense_h_to_4h"
MLP_DENSE_4H_TO_H = "mlp.dense_4h_to_h"
POST_MLP_LAYERNORM = "post_mlp_layernorm"
POST_MLP_LAYERSCALE = "post_mlp_layerscale"
# Dense end

# ====MOE begin====
MOE_GATE = "moe.gate"

# expert
MOE_EXPERT = "moe.expert"
MOE_GROUPED_GEMM_EXPERT = "moe.groupedgemm.expert"
MOE_EXPERT_H_TO_4H = "moe.expert_h_to_4h"
MOE_EXPERT_4H_TO_H = "moe.expert_4h_to_h"

# shared expert
MOE_SHARED_EXPERT = "moe.shared_expert"
# ====MOE end====

FINAL_LAYERNORM = "final_layernorm"
WORD_EMBEDDINGS_FOR_HEAD = "word_embeddings_for_head"

WORD_EMBEDDINGS_TPL = "word_embeddings_tpl"
WORD_POSITION_EMBEDDINGS_TPL = "word_position_embeddings_tpl"
TRANSFORMER_TPL = "transformer_tpl"
WORD_EMBEDDINGS_FOR_HEAD_TPL = "word_embeddings_for_head_tpl"

MTP_WORD_EMBEDDING = "mtp_word_embeddings"
MTP_ENORM = "mtp_enorm"
MTP_HNORM = "mtp_hnorm"
MTP_EH_PROJ = "mtp_eh_proj"
MTP_SHARED_HEAD_NORM = "mtp_shared_head_norm"
MTP_SHARED_HEAD_HEAD = "mtp_shared_head_head"

FIRST_LAYER_NAMES = [WORD_EMBEDDINGS, WORD_POSITION_EMBEDDINGS, WORD_BLOCK_POSITION_EMBEDDINGS] # in the first layer
BASE_NAMES = [INPUT_LAYERNORM, ATTENTION_ROTARY_EMB_INV_FREQ, ROTARY_EMB_INV_FREQ, ATTENTION_QUERY_KEY_VALUE,
            ATTENTION_Q_DOWN, ATTENTION_Q_UP, ATTENTION_Q_UP_LAYERNORM, ATTENTION_KV_DOWN, ATTENTION_KV_UP,
            ATTENTION_KV_UP_LAYERNORM, ATTENTION_Q, ATTENTION_DENSE,
            ATTENTION_LIGHTNING_INDEXER_K_NORM, ATTENTION_LIGHTNING_INDEXER_WEIGHTS_PROJ,
            ATTENTION_LIGHTHING_INDEXER_WK, ATTENTION_LIGHTHING_INDEXER_WQ_B,
            POST_ATTENTION_LAYERNORM, POST_ATTENTION_LAYERSCALE, ATTENTION_QNORM, ATTENTION_KNORM,
            POST_MLP_LAYERNORM, POST_MLP_LAYERSCALE, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, MOE_GATE]
MOE_EXPERT_PROJS = [MOE_EXPERT_H_TO_4H, MOE_EXPERT_4H_TO_H]
LAST_LAYER_NAMES = [FINAL_LAYERNORM, WORD_EMBEDDINGS_FOR_HEAD] # in the last layer
MTP_NAMES = [MTP_WORD_EMBEDDING, MTP_ENORM, MTP_HNORM, MTP_EH_PROJ, MTP_SHARED_HEAD_NORM, MTP_SHARED_HEAD_HEAD]


WEIGHT = "weight"
BIAS = "bias"
WEIGHT_SCALE = "weight_scale_inv"
LAYERNORM_WEIGHT = "layer_norm_weight"
LAYERNORM_BIAS = "layer_norm_bias"
EXTRA_DATA = "_extra_state"

# The member names for each key in one layer
LAYER_NAME = "name"
LAYER_EXTRA_DATA = "extra"
LAYER_IS_LAYERNORM = "is_layernorm"
LAYER_IS_FP8 = "fp8"
LAYER_FP8_IGNORE_TP = "fp8_ignore_tp"
LAYER_IS_DIRECT_NAME = "is_direct_name"
LAYER_IS_DICT_FOR_EXPERT = "is_dict_for_expert"
LAYER_NEED_TRANSPOSE = "need_transpose"

class CommonCheckpoint(AbstractCheckpoint):
    """
       CommonCheckpoint
    """
    def __init__(self, c_config):
        super().__init__(c_config)
        self.other_args = {}
        self.model_dict = {}

    @staticmethod
    def convert_from_common(*args, **kwargs):
        raise NotImplementedError()
    
    def convert_to_common(self, *args, **kwargs):
        raise NotImplementedError()

    def get(self, key):
        weight = self.model_dict[f"{key}.{WEIGHT}"] if f"{key}.{WEIGHT}" in self.model_dict else None
        bias = self.model_dict[f"{key}.{BIAS}"] if f"{key}.{BIAS}" in self.model_dict else None
        weight_scale = self.model_dict[f"{key}.{WEIGHT_SCALE}"] if f"{key}.{WEIGHT_SCALE}" in self.model_dict else None
        return weight, bias, weight_scale

    def clear(self):
        del self.model_dict

    def has_optimizer(self):
        return "optimizer" in self.state_dict

    @staticmethod
    def get_key(name, layer_id=None, expert_id=None):
        key = name
        if layer_id is not None:
            key += f".{LAYER_PREFIX}.{layer_id}"
            if expert_id is not None:
                key += f".{MOE_EXPERT}.{expert_id}"
        return key

    def set(self, common_key, weight, bias=None, weight_scale=None):
        if weight is not None:
            self.model_dict[f"{common_key}.{WEIGHT}"] = weight
            logging.info(f"> common set {common_key}, weight shape: {weight.shape}")

        # bias
        if bias is not None:
            self.model_dict[f"{common_key}.{BIAS}"] = bias
            logging.info(f"> common set {common_key}, bias shape: {bias.shape}")

        # weight scale inv for fp8
        if weight_scale is not None:
            self.model_dict[f"{common_key}.{WEIGHT_SCALE}"] = weight_scale
            logging.info(f"> common set {common_key}, weight_scale, shape: {weight_scale.shape}")
