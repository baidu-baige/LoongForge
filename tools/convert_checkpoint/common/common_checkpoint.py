# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Common checkpoint structures and shared checkpoint key definitions."""

from convert_checkpoint.common.abstact_checkpoint import AbstractCheckpoint

import logging

logging.basicConfig(level=logging.INFO)


# Name of common checkpoint layers, matching key of name_map in {model}.json

WORD_EMBEDDINGS = "word_embeddings"
WORD_POSITION_EMBEDDINGS = "word_position_embeddings"
WORD_BLOCK_POSITION_EMBEDDINGS = "word_block_position_embeddings"
LAYER_PREFIX = "layer_prefix"
MTP_LAYER_PREFIX = "mtp_layer_prefix"
TRANSFORMER = "transformer"
MTP_TRANSFORMER = "mtp_transformer"
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

ATTENTION_INDEXER_K_NORM = "attention.indexer.k_norm"
ATTENTION_INDEXER_WEIGHTS_PROJ = "attention.indexer.weights_proj"
ATTENTION_INDEXER_WK = "attention.indexer.wk"
ATTENTION_INDEXER_WQ_B = "attention.indexer.wq_b"

# Backward-compatible aliases for older conversion configs.
ATTENTION_LIGHTNING_INDEXER_K_NORM = ATTENTION_INDEXER_K_NORM
ATTENTION_LIGHTNING_INDEXER_WEIGHTS_PROJ = ATTENTION_INDEXER_WEIGHTS_PROJ
ATTENTION_LIGHTHING_INDEXER_WK = ATTENTION_INDEXER_WK
ATTENTION_LIGHTHING_INDEXER_WQ_B = ATTENTION_INDEXER_WQ_B
# MLA end

# DeepSeek V4 grouped output projection begin
ATTENTION_WO_A = "attention.wo_a"
ATTENTION_WO_B = "attention.wo_b"
# DeepSeek V4 grouped output projection end

# DeepSeek V4 attention sink begin
ATTENTION_SINK = "attention.attn_sink"
# DeepSeek V4 attention sink end

# DeepSeek V4 KV/Q norm begin
ATTENTION_KV_NORM = "attention.kv_norm"
ATTENTION_Q_NORM = "attention.q_norm"
# DeepSeek V4 KV/Q norm end

# DeepSeek V4 compressor begin
ATTENTION_COMPRESSOR_WKV = "attention.compressor.wkv"
ATTENTION_COMPRESSOR_WGATE = "attention.compressor.wgate"
ATTENTION_COMPRESSOR_APE = "attention.compressor.ape"
ATTENTION_COMPRESSOR_NORM = "attention.compressor.norm"
# DeepSeek V4 compressor end

# DeepSeek V4 indexer compressor begin
ATTENTION_INDEXER_COMPRESSOR_WKV = "attention.indexer.compressor.wkv"
ATTENTION_INDEXER_COMPRESSOR_WGATE = "attention.indexer.compressor.wgate"
ATTENTION_INDEXER_COMPRESSOR_APE = "attention.indexer.compressor.ape"
ATTENTION_INDEXER_COMPRESSOR_NORM = "attention.indexer.compressor.norm"
# DeepSeek V4 indexer compressor end

# DeepSeek V4 Hyper-Connection begin
HC_ATTN_FN = "hc_attn_fn"
HC_ATTN_BASE = "hc_attn_base"
HC_ATTN_SCALE = "hc_attn_scale"
HC_ATTN_ALPHA_PRE = "hc_attn_alpha_pre"
HC_ATTN_ALPHA_POST = "hc_attn_alpha_post"
HC_ATTN_ALPHA_RES = "hc_attn_alpha_res"
HC_FFN_FN = "hc_ffn_fn"
HC_FFN_BASE = "hc_ffn_base"
HC_FFN_SCALE = "hc_ffn_scale"
HC_FFN_ALPHA_PRE = "hc_ffn_alpha_pre"
HC_FFN_ALPHA_POST = "hc_ffn_alpha_post"
HC_FFN_ALPHA_RES = "hc_ffn_alpha_res"
# DeepSeek V4 Hyper-Connection end

# DeepSeek V4 HyperHead begin
HC_HEAD_FN = "hc_head_fn"
HC_HEAD_BASE = "hc_head_base"
HC_HEAD_SCALE = "hc_head_scale"
HC_HEAD_ALPHA_PRE = "hc_head_alpha_pre"
HC_HEAD_ALPHA_POST = "hc_head_alpha_post"
HC_HEAD_ALPHA_RES = "hc_head_alpha_res"
# DeepSeek V4 HyperHead end

# DeepSeek V4 MTP specific begin
MTP_E_PROJ = "mtp_e_proj"
MTP_H_PROJ = "mtp_h_proj"
MTP_NORM = "mtp_norm"
MTP_HC_ATTN_FN = "mtp_hc_attn_fn"
MTP_HC_ATTN_BASE = "mtp_hc_attn_base"
MTP_HC_ATTN_SCALE = "mtp_hc_attn_scale"
MTP_HC_ATTN_ALPHA_PRE = "mtp_hc_attn_alpha_pre"
MTP_HC_ATTN_ALPHA_POST = "mtp_hc_attn_alpha_post"
MTP_HC_ATTN_ALPHA_RES = "mtp_hc_attn_alpha_res"
MTP_HC_FFN_FN = "mtp_hc_ffn_fn"
MTP_HC_FFN_BASE = "mtp_hc_ffn_base"
MTP_HC_FFN_SCALE = "mtp_hc_ffn_scale"
MTP_HC_FFN_ALPHA_PRE = "mtp_hc_ffn_alpha_pre"
MTP_HC_FFN_ALPHA_POST = "mtp_hc_ffn_alpha_post"
MTP_HC_FFN_ALPHA_RES = "mtp_hc_ffn_alpha_res"
MTP_HC_HEAD_FN = "mtp_hc_head_fn"
MTP_HC_HEAD_BASE = "mtp_hc_head_base"
MTP_HC_HEAD_SCALE = "mtp_hc_head_scale"
MTP_HC_HEAD_ALPHA_PRE = "mtp_hc_head_alpha_pre"
MTP_HC_HEAD_ALPHA_POST = "mtp_hc_head_alpha_post"
MTP_HC_HEAD_ALPHA_RES = "mtp_hc_head_alpha_res"
MTP_HEAD = "mtp_head"
# DeepSeek V4 MTP specific end

ATTENTION_QUERY_KEY_VALUE = "attention.query_key_value"
ATTENTION_QUERY_GATE_KEY_VALUE = "attention.query_gate_key_value"
MIXER_ATT_LOG = "mixer_att.log"
MIXER_ATT_DT = "mixer_att.dt_bias"
MIXER_ATT_CONV1D = "mixer_att.conv1d"
MIXER_ATT_NORM = "mixer_att.norm"
MIXER_ATT_OUT_PROJ = "mixer_att.out_proj"
MIXER_ATT_IN_PROJ = "mixer_att.in_proj"
MIXER_ATT_IN_PROJ_QKVZ = "mixer_att.in_proj_qkvz"
MIXER_ATT_IN_PROJ_BA = "mixer_att.in_proj_ba"
MIXER_INPUT_LAYERNORM = "mixer_input_layernorm"

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
MOE_SHARED_EXPERT_H_TO_4H = "moe.shared_expert_h_to_4h"
MOE_SHARED_EXPERT_4H_TO_H = "moe.shared_expert_4h_to_h"

# shared expert
MOE_SHARED_EXPERT = "moe.shared_expert"
MOE_SHARED_EXPERT_GATE = "moe.shared_expert_gate"

# DeepSeek V4 Hash Router tid2eid per-layer name
MOE_GATE_TID2EID = "moe.gate.tid2eid"
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
MTP_NAME_PREFIX_FOR_LAYER = "mtp_name_prefix_for_layer"

MTP_MOE_EXPERT_H_TO_4H = "mtp_moe.expert_h_to_4h"
MTP_MOE_EXPERT_4H_TO_H = "mtp_moe.expert_4h_to_h"
MTP_MOE_SHARED_EXPERT_H_TO_4H = "mtp_moe.shared_expert_h_to_4h"
MTP_MOE_SHARED_EXPERT_4H_TO_H = "mtp_moe.shared_expert_4h_to_h"

# =========quant_type==========
QUANT_DTYPE_FP8 = "fp8"
QUANT_DTYPE_BF16 = "bf16"
QUANT_HF_BF16_AND_MCORE_FP8 = "hf_bf16_and_mcore_fp8"

# =========vision=========
VISION_MAP = "vision_map"
VISION_WORD_EMBEDDINGS = "vision_word_embeddings"

FIRST_LAYER_NAMES = [WORD_EMBEDDINGS, WORD_POSITION_EMBEDDINGS, WORD_BLOCK_POSITION_EMBEDDINGS, VISION_WORD_EMBEDDINGS]
BASE_NAMES = [INPUT_LAYERNORM, ATTENTION_ROTARY_EMB_INV_FREQ, ROTARY_EMB_INV_FREQ, ATTENTION_QUERY_KEY_VALUE,
            ATTENTION_QUERY_GATE_KEY_VALUE, MIXER_ATT_LOG, MIXER_ATT_DT, MIXER_INPUT_LAYERNORM, MIXER_ATT_CONV1D,
            MIXER_ATT_NORM, MIXER_ATT_OUT_PROJ, MIXER_ATT_IN_PROJ, MIXER_ATT_IN_PROJ_QKVZ, MIXER_ATT_IN_PROJ_BA,
            ATTENTION_Q_DOWN, ATTENTION_Q_UP, ATTENTION_Q_UP_LAYERNORM, ATTENTION_KV_DOWN, ATTENTION_KV_UP,
            ATTENTION_KV_UP_LAYERNORM, ATTENTION_Q, ATTENTION_DENSE,
            ATTENTION_INDEXER_K_NORM, ATTENTION_INDEXER_WEIGHTS_PROJ,
            ATTENTION_INDEXER_WK, ATTENTION_INDEXER_WQ_B, MOE_SHARED_EXPERT_GATE,
            POST_ATTENTION_LAYERNORM, POST_ATTENTION_LAYERSCALE, ATTENTION_QNORM, ATTENTION_KNORM,
            POST_MLP_LAYERNORM, POST_MLP_LAYERSCALE, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, MOE_GATE,
            # DeepSeek V4 additions
            ATTENTION_WO_A, ATTENTION_WO_B, ATTENTION_SINK, ATTENTION_KV_NORM, ATTENTION_Q_NORM,
            ATTENTION_COMPRESSOR_WKV, ATTENTION_COMPRESSOR_WGATE, ATTENTION_COMPRESSOR_APE, ATTENTION_COMPRESSOR_NORM,
            ATTENTION_INDEXER_COMPRESSOR_WKV, ATTENTION_INDEXER_COMPRESSOR_WGATE,
            ATTENTION_INDEXER_COMPRESSOR_APE, ATTENTION_INDEXER_COMPRESSOR_NORM,
            # DeepSeek V4 Hyper-Connection per-layer
            HC_ATTN_FN, HC_ATTN_BASE, HC_ATTN_ALPHA_PRE, HC_ATTN_ALPHA_POST, HC_ATTN_ALPHA_RES, HC_FFN_FN, HC_FFN_BASE, HC_FFN_ALPHA_PRE, HC_FFN_ALPHA_POST, HC_FFN_ALPHA_RES,
            # DeepSeek V4 Hash Router tid2eid
            MOE_GATE_TID2EID]
MOE_EXPERT_PROJS = [MOE_EXPERT_H_TO_4H, MOE_EXPERT_4H_TO_H]
LAST_LAYER_NAMES = [FINAL_LAYERNORM, WORD_EMBEDDINGS_FOR_HEAD,
                    # DeepSeek V4 HyperHead global parameters
                    HC_HEAD_FN, HC_HEAD_BASE, HC_HEAD_SCALE] # in the last layer

# DeepSeek V4 Hyper-Connection per-layer names (not standard BASE_NAMES, handled separately)
HC_NAMES = [HC_ATTN_FN, HC_ATTN_BASE, HC_ATTN_ALPHA_PRE, HC_ATTN_ALPHA_POST, HC_ATTN_ALPHA_RES, HC_FFN_FN, HC_FFN_BASE, HC_FFN_ALPHA_PRE, HC_FFN_ALPHA_POST, HC_FFN_ALPHA_RES]

# DeepSeek V4 HyperHead global names (like FINAL_LAYERNORM, not per-layer)
HC_HEAD_NAMES = [HC_HEAD_FN, HC_HEAD_BASE, HC_HEAD_SCALE]

MTP_NAMES = [MTP_WORD_EMBEDDING, MTP_ENORM, MTP_HNORM, MTP_EH_PROJ, MTP_SHARED_HEAD_NORM, MTP_SHARED_HEAD_HEAD,
             # DeepSeek V4 MTP additions
             MTP_E_PROJ, MTP_H_PROJ, MTP_NORM,
             MTP_HC_ATTN_FN, MTP_HC_ATTN_BASE, MTP_HC_ATTN_ALPHA_PRE, MTP_HC_ATTN_ALPHA_POST, MTP_HC_ATTN_ALPHA_RES,
             MTP_HC_FFN_FN, MTP_HC_FFN_BASE, MTP_HC_FFN_ALPHA_PRE, MTP_HC_FFN_ALPHA_POST, MTP_HC_FFN_ALPHA_RES,
             MTP_HC_HEAD_FN, MTP_HC_HEAD_BASE, MTP_HC_HEAD_SCALE,
             MTP_HEAD]

EMBED_NAMES = [WORD_EMBEDDINGS, MTP_WORD_EMBEDDING, WORD_EMBEDDINGS_FOR_HEAD, VISION_WORD_EMBEDDINGS, MTP_SHARED_HEAD_HEAD]

WEIGHT = "weight"
BIAS = "bias"
WEIGHT_SCALE = "weight_scale_inv"
LAYERNORM_WEIGHT = "layer_norm_weight"
LAYERNORM_BIAS = "layer_norm_bias"
EXTRA_DATA = "_extra_state"
LORA_NAME_IN = "adapter.linear_in"
LORA_NAME_OUT = "adapter.linear_out"

# The member names for each key in one layer
LAYER_NAME = "name"
LAYER_EXTRA_DATA = "extra"
LAYER_IS_LAYERNORM = "is_layernorm"
LAYER_IS_FP8 = "fp8"
LAYER_FP8_IGNORE_TP = "fp8_ignore_tp"
LAYER_IGNORE_TP = "ignore_tp"
LAYER_IS_DIRECT_NAME = "is_direct_name"
LAYER_NO_LAYER_ID = "no_layer_id"
LAYER_DEPEND_ON_KEY = "depend_on_key"
LAYER_IS_DICT_FOR_EXPERT = "is_dict_for_expert"
LAYER_NEED_TRANSPOSE = "need_transpose"
LAYER_DTYPE = "dtype"

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
        if layer_id is None:
            key = name
        else:
            if expert_id is not None:
                key = f"{LAYER_PREFIX}.{layer_id}.{MOE_EXPERT}.{expert_id}.{name}"
            else:
                name = "moe.shared_expert_h_to_4h" if name == MOE_EXPERT_H_TO_4H else name
                name = "moe.shared_expert_4h_to_h" if name == MOE_EXPERT_4H_TO_H else name
                key = f"{LAYER_PREFIX}.{layer_id}.{name}"
        return key

    def set(self, common_key, weight, bias=None, weight_scale=None, log_flag=True):
        if weight is not None:
            self.model_dict[f"{common_key}.{WEIGHT}"] = weight
            if log_flag:
                logging.info(f"> common set {common_key}, weight shape: {weight.shape}")

        # bias
        if bias is not None:
            self.model_dict[f"{common_key}.{BIAS}"] = bias
            if log_flag:
                logging.info(f"> common set {common_key}, bias shape: {bias.shape}")

        # weight scale inv for fp8
        if weight_scale is not None:
            self.model_dict[f"{common_key}.{WEIGHT_SCALE}"] = weight_scale
            if log_flag:
                logging.info(f"> common set {common_key}, weight_scale, shape: {weight_scale.shape}")
