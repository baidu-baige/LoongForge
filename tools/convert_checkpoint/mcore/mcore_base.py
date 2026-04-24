# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Base utilities for converting common checkpoints to and from Megatron Core format."""

import io
from typing import Any
from convert_checkpoint.huggingface.huggingface_base import HuggingfaceBase
import torch
import logging
from omegaconf.dictconfig import DictConfig

logging.basicConfig(level=logging.INFO)

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common.common_checkpoint import QUANT_DTYPE_BF16, QUANT_DTYPE_FP8, QUANT_HF_BF16_AND_MCORE_FP8, VISION_WORD_EMBEDDINGS, CommonCheckpoint
from convert_checkpoint.utils.utils import (
    add_embedding_padding, cut_embedding_padding,
    transpose_shape0,
    convert_fp8_to_bf16,
    convert_bf16_to_fp8,
    is_power_of_two
)

from convert_checkpoint.utils.utils import (
    get_ep_map,
    get_etp_map,
    get_quantizer_with_weight_scale_inv
)

from convert_checkpoint.common.common_checkpoint import (
    WEIGHT, BIAS, WEIGHT_SCALE, LAYERNORM_WEIGHT, LAYERNORM_BIAS, LORA_NAME_IN, LORA_NAME_OUT, MIXER_ATT_IN_PROJ_QKVZ, MIXER_ATT_IN_PROJ_BA,
    ATTENTION_QUERY_GATE_KEY_VALUE, WORD_EMBEDDINGS, WORD_EMBEDDINGS_FOR_HEAD, MTP_SHARED_HEAD_HEAD, MLP_DENSE_H_TO_4H,
    MLP_DENSE_4H_TO_H, MOE_EXPERT_H_TO_4H, MTP_WORD_EMBEDDING, LAYER_IS_DIRECT_NAME,
    LAYER_PREFIX, MTP_NAME_PREFIX_FOR_LAYER, EXTRA_DATA, LAYER_NAME, LAYER_EXTRA_DATA,
    LAYER_IS_LAYERNORM, LAYER_IS_FP8, LAYER_FP8_IGNORE_TP, LAYER_IGNORE_TP,
    LAYER_DTYPE
)

from convert_checkpoint.mcore.util.mcore_attn_converter import McoreAttnGateQkvConverter, McoreMixerAttnConverter

TENSOR_PARALLEL_DIM = {
    "word_embeddings.weight": 0,
    "attention.query_key_value.weight": 0,
    "attention.query_key_value.bias": 0,
    "attention.q_down.weight": 0,
    "attention.q_up.weight": 0,
    "attention.kv_down.weight": 0,
    "attention.kv_up.weight": 0,
    "attention.q.weight": 0,
    "attention.dense.weight" : 1,
    "attention.query_gate_key_value.weight": 0,
    "mixer_att.log.weight": 0,
    "mixer_att.dt_bias.weight": 0,
    "mixer_att.conv1d.weight": 0,
    "mixer_att.out_proj.weight": 1,
    "mlp.dense_h_to_4h.weight": 0,
    "mlp.dense_h_to_4h.bias": 0,
    "mlp.dense_4h_to_h.weight": 1,
    "moe.expert_h_to_4h.weight": 0,
    "moe.expert_h_to_4h.bias": 0,
    "moe.expert_4h_to_h.weight": 1,
    "word_embeddings_for_head.weight": 0,
    "mtp_word_embeddings.weight": 0,
    "mtp_shared_head_head.weight": 0,
    "mtp_eh_proj.weight": 0,
    "vision_word_embeddings.weight": 0
}


class McoreBase:
    """
        McoreBase
    """

    def __init__(self, c_config, args):
        self.c_config = c_config
        self.args = args
        args.lora_alpha = self.args.lora_alpha
        args.lora_dim = self.args.lora_dim
        args.load_lora_ckpt_path = None
        margs = c_config.get_args("mcore")
        self.cargs = self.c_config.get_args("common")
        self.hf_name_map = self.c_config.get("name_map")["huggingface"]
        self.name_map = self.c_config.get("name_map")["mcore"]
        self.mcore_mixer_attn_converter = McoreMixerAttnConverter(c_config)
        self.mcore_attn_gqkv_converter = McoreAttnGateQkvConverter(c_config)

        self.tp = args.tensor_model_parallel_size
        self.pp = args.pipeline_model_parallel_size
        self.ep = args.expert_parallel_size
        self.etp = args.expert_tensor_parallel_size

        self.save_path = self.args.save_ckpt_path
        self.load_path = self.args.load_ckpt_path
        self.lora_alpha = self.args.lora_alpha
        self.lora_dim = self.args.lora_dim
        self.transpose_mlp_dense = margs.get("transpose_mlp_dense", False)
        self.tensor_parallel_dim = TENSOR_PARALLEL_DIM
        if c_config.get("tensor_parallel_dim", None) is not None:
            for k, v in c_config.get("tensor_parallel_dim").items():
                if k not in TENSOR_PARALLEL_DIM or (k in TENSOR_PARALLEL_DIM and TENSOR_PARALLEL_DIM[k] != v):
                    self.tensor_parallel_dim[k] = v
        self.layer_prefix = self.name_map[LAYER_PREFIX]
        self.name_prefix_for_layer = self.name_map[MTP_NAME_PREFIX_FOR_LAYER] if MTP_NAME_PREFIX_FOR_LAYER in self.name_map else None
        self.add_embed_padding = margs.get("add_embedding_padding", False)
        self.untie_embeddings_and_output_weights = margs.get("untie_embeddings_and_output_weights", False)

        self.divisible_by = margs.get("make_vocab_size_divisible_by", 128)
        self.vocab_size = self.cargs.get("vocab_size", None)
        self.padded_vocab_size = margs.get("pad_vocab_size_to", None)
        self.convert_to_fp8 = self.args.convert_to_fp8

        num_experts = self.cargs.get("num_experts", None)
        self.dtype = c_config.get_dtype()

        self.expert_local_mapping, _, _ = get_ep_map(num_experts, self.ep)
        self.etp_to_tp_mapping, _ = get_etp_map(self.tp, self.ep, self.etp)


    def get_mcore_name_and_extra(self, obj):
        if isinstance(obj, dict) or isinstance(obj, DictConfig):
            mcore_name = obj[LAYER_NAME]
            has_extra = obj[LAYER_EXTRA_DATA] if LAYER_EXTRA_DATA in obj else False
            is_layernorm = obj[LAYER_IS_LAYERNORM] if LAYER_IS_LAYERNORM in obj else False
            is_fp8 = obj[LAYER_IS_FP8] if LAYER_IS_FP8 in obj else False
            fp8_ignore_tp = obj[LAYER_FP8_IGNORE_TP] if LAYER_FP8_IGNORE_TP in obj else False
            is_direct_name = obj[LAYER_IS_DIRECT_NAME] if LAYER_IS_DIRECT_NAME in obj else False
            ignore_tp = obj[LAYER_IGNORE_TP] if LAYER_IGNORE_TP in obj else False
            dtype = obj.get(LAYER_DTYPE, None)
        else:
            mcore_name = obj
            has_extra = False
            is_layernorm = False
            is_fp8 = False
            fp8_ignore_tp = False
            is_direct_name = False
            ignore_tp = False
            dtype = None
        return (mcore_name, has_extra, is_layernorm), (is_fp8, fp8_ignore_tp), (is_direct_name, ignore_tp, dtype)

    #========to mcore===========
    def common_to_mcore(self, name, c_ckpt, m_dict, t_name, layer_id=None, m_layer_id=None,
                        layer_prefix=None, ep_id=None, expert_name=None, name_prefix=None):
        if name not in self.name_map:
            return
        if name == WORD_EMBEDDINGS_FOR_HEAD and (not self.untie_embeddings_and_output_weights and self.pp == 1):
            return
        common_key = CommonCheckpoint.get_key(name, layer_id=layer_id)
        if name == MTP_WORD_EMBEDDING:
            layer_id = None
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        (mcore_name, has_extra, is_layernorm), (is_fp8, fp8_ignore_tp), (is_direct_name, ignore_tp, mcore_dtype) = \
                self.get_mcore_name_and_extra(self.name_map[name])
        # ======weight need quantization when dtype is not equal begin =======
        quant_type = None
        if mcore_dtype is not None:
            _, _, _, _, _, _, hf_dtype = HuggingfaceBase.get_hf_name_and_args(self.hf_name_map[name])
            if hf_dtype is not None and hf_dtype != mcore_dtype:
                quant_type = QUANT_HF_BF16_AND_MCORE_FP8 \
                        if hf_dtype == QUANT_DTYPE_BF16 and mcore_dtype == QUANT_DTYPE_FP8 else None                
        # ======weight need quantization when dtype is not equal end =======
        if layer_id is None:
            mcore_path = mcore_name
        elif expert_name is not None:
            if expert_name not in self.name_map:
                return
            m_name_prefix = self.name_map[expert_name] if name_prefix is None \
                    else f"{name_prefix}.{self.name_map[expert_name]}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}.{mcore_name}"
        else:
            m_name_prefix = mcore_name if name_prefix is None else f"{name_prefix}.{mcore_name}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}"
        if is_direct_name:
            mcore_weight_path = mcore_path
        elif is_layernorm:
            mcore_weight_path = f"{mcore_path}.{LAYERNORM_WEIGHT}"
        else:
            mcore_weight_path = f"{mcore_path}.{WEIGHT}"
        if is_layernorm:
            mcore_bias_path = f"{mcore_path}.{LAYERNORM_BIAS}"
        else:
            mcore_bias_path = f"{mcore_path}.{BIAS}"
        bias_name = f"{name}.{BIAS}"
        if bias_name in self.name_map:
            mcore_bias_name = self.name_map[bias_name]
            m_bias_name = mcore_bias_name if name_prefix is None else f"{name_prefix}.{mcore_bias_name}"
            mcore_bias_path = f"{layer_prefix}.{m_layer_id}.{m_bias_name}"
        has_extra_path = f"{mcore_path}.{EXTRA_DATA}"

        weight, bias, weight_scale = c_ckpt.get(common_key)
        if weight is None:
            return
        if self.add_embed_padding and name in [WORD_EMBEDDINGS, MTP_WORD_EMBEDDING, WORD_EMBEDDINGS_FOR_HEAD, VISION_WORD_EMBEDDINGS]:
            weight = add_embedding_padding(weight, self.divisible_by, self.vocab_size, self.tp, self.padded_vocab_size)
        weight_list, bias_list = self.get_chunked_weight(
                name, self.tp, mcore_weight_path, mcore_bias_path, weight, bias, weight_scale,
                is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp, quant_type=quant_type)
        etp_to_tp = self.etp_to_tp_mapping[ep_id] if self.etp is not None and ep_id is not None else None
        self.update_mcore_weight(
                m_dict, t_name, mcore_weight_path, mcore_bias_path, has_extra_path,
                weight_list, bias_list=bias_list, etp_to_tp=etp_to_tp, has_extra=has_extra)

    def update_mcore_weight(self, m_dict, t_name, mcore_weight_path, mcore_bias_path, has_extra_path,
                            weight_list, bias_list=None, etp_to_tp=None, has_extra=False):
        # m_dict:
        #   no etp: tp -> {layer_name -> weight}
        #   etp: etp -> {layer_name -> weight}
        # weight_list: tp -> [weight]
        if weight_list is None:
            return
        if self.etp is None:
            m_tp = self.tp
        else:
            m_tp = self.etp
        for mt in range(m_tp):
            if self.etp is None or etp_to_tp is None:
                t = mt
            else:
                et = mt
                t = etp_to_tp[et]
            if mt not in m_dict:
                continue
            m_dict[mt][t_name] = {} if t_name not in m_dict[mt] else m_dict[mt][t_name]
            m_dict[mt][t_name][mcore_weight_path] = weight_list[t]
            if mcore_bias_path is not None and bias_list is not None:
                m_dict[mt][t_name][mcore_bias_path] = bias_list[t]
            if has_extra:
                m_dict[mt][t_name][has_extra_path] = None

    def get_tp_chunk_list(self, name, m_tp, chunk_dim, source, need_transpose=False):
        if source is None:
            return None
        if chunk_dim is None or m_tp == 1:
            source_list = [source] * m_tp
        else:
            if need_transpose:
                source = transpose_shape0(source, 2, m_tp)
            source_list = torch.chunk(source, m_tp, dim=chunk_dim)
            source_list = [s.clone() for s in source_list]

        return source_list

    def convert_bf16_to_fp8s(self, name, m_tp, chunk_dim, weight_bf16, fp8_ignore_tp, ignore_tp, need_transpose=False):
        if fp8_ignore_tp or ignore_tp:
            weight_bf16_s = [weight_bf16] * m_tp
        else:
            weight_bf16_s = self.get_tp_chunk_list(name, m_tp, chunk_dim, weight_bf16, need_transpose=need_transpose)
        weight_s = []
        weight_scale_s = []
        for w_bf16 in weight_bf16_s:
            w, w_scale = convert_bf16_to_fp8(
                    w_bf16, method=self.args.quant_method, amax_epsilon=self.args.amax_epsilon,
                    force_pow_2_scales=self.args.force_pow_2_scales)
            weight_s.append(w)
            weight_scale_s.append(w_scale)
        return weight_s, weight_scale_s

    def get_chunked_weight(self, name, m_tp, weight_path, bias_path, weight, bias=None,
                           weight_scale=None, is_fp8=False, fp8_ignore_tp=False, log_flag=True,
                           ignore_tp=False, quant_type=None):
        if weight is None:
            return None, None
        need_transpose = (m_tp > 1 and self.transpose_mlp_dense and \
                name in [MLP_DENSE_H_TO_4H, MOE_EXPERT_H_TO_4H])
        chunk_dim = self.tensor_parallel_dim.get(f"{name}.{WEIGHT}", None)

        if weight_scale is None and quant_type is None:
            if ignore_tp:
                weight_list = [weight] * m_tp
            else:
                weight_list = self.get_tp_chunk_list(name, m_tp, chunk_dim, weight, need_transpose=need_transpose)
        bias_list = None
        if bias is not None:
            bias_chunk_dim = self.tensor_parallel_dim.get(f"{name}.{BIAS}", None)
            bias_list = self.get_tp_chunk_list(name, m_tp, bias_chunk_dim, bias, need_transpose=need_transpose)
        if weight_scale is None:
            if quant_type == QUANT_HF_BF16_AND_MCORE_FP8:
                # ======weight need quantization when dtype is not equal =======
                weight_s, weight_scale_s = self.convert_bf16_to_fp8s(
                        name, m_tp, chunk_dim, weight, fp8_ignore_tp, ignore_tp, need_transpose=need_transpose)
                weight_list = []
                for w, w_scale in zip(weight_s, weight_scale_s):
                    weight_list.append(get_quantizer_with_weight_scale_inv(w, w_scale, self.dtype, amax_epsilon=self.args.amax_epsilon))
        else:
            # fp8 chunk
            if (self.args.fp8_force_no_requant \
                    or is_power_of_two(weight_scale) == self.args.force_pow_2_scales):
                if fp8_ignore_tp or ignore_tp:
                    weight_s = [weight] * m_tp
                    weight_scale_s = [weight_scale] * m_tp
                else:
                    weight_s = self.get_tp_chunk_list(name, m_tp, chunk_dim, weight, need_transpose=need_transpose)
                    weight_scale_s = self.get_tp_chunk_list(
                            name, m_tp, chunk_dim, weight_scale, need_transpose=need_transpose)
            else:
                # First do dequantization then re-quantize back to FP8
                weight_bf16 = convert_fp8_to_bf16(weight, weight_scale, dtype=torch.float32)
                weight_s, weight_scale_s = self.convert_bf16_to_fp8s(
                        name, m_tp, chunk_dim, weight_bf16, fp8_ignore_tp, ignore_tp, need_transpose=need_transpose)
            weight_list = []
            for w, w_scale in zip(weight_s, weight_scale_s):
                weight_list.append(get_quantizer_with_weight_scale_inv(w, w_scale, self.dtype, amax_epsilon=self.args.amax_epsilon))

        weight_shapes = [obj.shape for obj in weight_list]
        if log_flag:
            logging.info(f"Chunk weight({name=}) {weight_path}, {m_tp=}, {chunk_dim=}, ori_weight: {weight.shape}, {weight_shapes=}")
        if bias is not None:
            bias_shapes = [obj.shape for obj in bias_list]
            if log_flag:
                logging.info(f"Chunk bias {bias_path}, {m_tp=}, {bias_shapes=}")
        return weight_list, bias_list

    #========from mcore===========
    def mcore_to_common(self, name, c_ckpt, m_dict, t_name, layer_id=None, m_layer_id=None,
                        layer_prefix=None, expert_name=None, name_prefix=None):
        # m_dict: t->dict
        # ep_mcore_state_dict: 
        #   etp is None: ep_id->t->dict
        #   etp is not None: ep_id->et->dict
        if m_dict is None:
            return
        if name not in self.name_map:
            return
        if name == MTP_WORD_EMBEDDING:
            layer_id = None
        common_key = CommonCheckpoint.get_key(name, layer_id=layer_id)
        if name == WORD_EMBEDDINGS_FOR_HEAD and not self.untie_embeddings_and_output_weights and self.pp == 1:
            name = WORD_EMBEDDINGS
        layer_prefix = self.layer_prefix if layer_prefix is None else layer_prefix
        need_cut_padding = self.add_embed_padding and \
                name in [WORD_EMBEDDINGS, MTP_WORD_EMBEDDING, WORD_EMBEDDINGS_FOR_HEAD, VISION_WORD_EMBEDDINGS]
        if name == MTP_SHARED_HEAD_HEAD:
            assert WORD_EMBEDDINGS_FOR_HEAD in self.name_map, \
                    f"{WORD_EMBEDDINGS_FOR_HEAD} is needed in name_map"
            (mcore_name, has_extra, is_layernorm), (is_fp8, fp8_ignore_tp), (is_direct_name, ignore_tp, mcore_dtype) = self.get_mcore_name_and_extra(self.name_map[WORD_EMBEDDINGS_FOR_HEAD])
            mcore_path = mcore_name
            need_cut_padding = True
        else:
            (mcore_name, has_extra, is_layernorm), (is_fp8, fp8_ignore_tp), (is_direct_name, ignore_tp, mcore_dtype) = self.get_mcore_name_and_extra(self.name_map[name])
            mcore_path = None
        # ======weight need quantization when dtype is not equal begin =======
        quant_type = None
        if mcore_dtype is not None:
            _, _, _, _, _, _, hf_dtype = HuggingfaceBase.get_hf_name_and_args(self.hf_name_map[name])
            if hf_dtype is not None and hf_dtype != mcore_dtype:
                quant_type = QUANT_HF_BF16_AND_MCORE_FP8 \
                        if hf_dtype == QUANT_DTYPE_BF16 and mcore_dtype == QUANT_DTYPE_FP8 else None                
        # ======weight need quantization when dtype is not equal end =======
        if mcore_path is not None:
            pass
        elif layer_id is None:
            mcore_path = mcore_name
        elif expert_name is not None:
            if expert_name not in self.name_map:
                return
            m_name_prefix = self.name_map[expert_name] if name_prefix is None \
                    else f"{name_prefix}.{self.name_map[expert_name]}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}.{mcore_name}"
        else:
            m_name_prefix = mcore_name if name_prefix is None else f"{name_prefix}.{mcore_name}"
            mcore_path = f"{layer_prefix}.{m_layer_id}.{m_name_prefix}"
        mcore_lora_in_path = None
        mcore_lora_out_path = None
        if is_direct_name:
            mcore_weight_path = mcore_path
        elif is_layernorm:
            mcore_weight_path = f"{mcore_path}.{LAYERNORM_WEIGHT}"
        else:
            mcore_weight_path = f"{mcore_path}.{WEIGHT}"
            mcore_lora_in_path = f"{mcore_path}.{LORA_NAME_IN}.{WEIGHT}"
            mcore_lora_out_path = f"{mcore_path}.{LORA_NAME_OUT}.{WEIGHT}"
        if is_layernorm:
            mcore_bias_path = f"{mcore_path}.{LAYERNORM_BIAS}"
        else:
            mcore_bias_path = f"{mcore_path}.{BIAS}"
        bias_name = f"{name}.{BIAS}"
        if bias_name in self.name_map:
            mcore_bias_name = self.name_map[bias_name]
            m_bias_name = mcore_bias_name if name_prefix is None else f"{name_prefix}.{mcore_bias_name}"
            mcore_bias_path = f"{layer_prefix}.{m_layer_id}.{m_bias_name}"

        weight_list, bias_list, weight_scale_list = self.get_mcore_weight_list(
                m_dict, t_name, mcore_weight_path, mcore_bias_path)
        if mcore_lora_in_path is None:
            lora_in_weight_list = None
        else:
            lora_in_weight_list, _, _ = self.get_mcore_weight_list(m_dict, t_name, mcore_lora_in_path, None)
        if mcore_lora_out_path is None:
            lora_out_weight_list = None
        else:
            lora_out_weight_list, _, _ = self.get_mcore_weight_list(m_dict, t_name, mcore_lora_out_path, None)

        weight, bias, weight_scale = self.get_cat_weight(
            name, self.tp, weight_list, bias_list, weight_scale_list, is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp, quant_type=quant_type)
        if lora_in_weight_list is not None and lora_out_weight_list is not None:
            # Merge lora weight
            lora_out_weight, _, _ = self.get_cat_weight(
                name, self.tp, lora_out_weight_list, None, None, is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp, chunk_dim=0)
            lora_in_weight, _, _ = self.get_cat_weight(
                name, self.tp, lora_in_weight_list, None, None, is_fp8, fp8_ignore_tp, ignore_tp=ignore_tp)
            weight = self.lora_merge(weight, lora_out_weight, lora_in_weight, self.lora_alpha, self.lora_dim)

        if need_cut_padding:
            weight = cut_embedding_padding(weight, self.vocab_size)
        c_ckpt.set(common_key, weight, bias=bias, weight_scale=weight_scale)

    def get_weight_by_tp(self, m_state, t_name, mcore_weight_path, mcore_bias_path):
        # weight
        if mcore_weight_path in m_state[t_name]:
            weight = m_state[t_name][mcore_weight_path]
        else:
            return None, None, None
        # bias
        if mcore_bias_path is not None and mcore_bias_path in m_state[t_name]:
            bias = m_state[t_name][mcore_bias_path]
        else:
            bias = None
        # weight_scale_inv
        weight_scale = None
        try:
            from transformer_engine.pytorch.tensor.float8_blockwise_tensor \
                    import Float8BlockwiseQTensor
            if isinstance(weight, Float8BlockwiseQTensor):
                temp_weight = weight
                weight = temp_weight._rowwise_data.view(torch.float8_e4m3fn)
                weight_scale = temp_weight._rowwise_scale_inv
        except:
            if self.args.pretrain_as_fp8:
                raise Exception("Please install Float8BlockwiseQTensor first when pretrain_as_fp8 is true.")
        return weight, bias, weight_scale

    def get_mcore_weight_list(self, m_dict, t_name, mcore_weight_path, mcore_bias_path):
        # m_dict: t->dict
        weight_list = [None] * self.tp
        bias_list = [None] * self.tp
        weight_scale_list = [None] * self.tp
        for t in range(self.tp):
            assert t in m_dict, f"tp={t} not found in m_dict. {m_dict.keys()=}"
            weight_list[t], bias_list[t], weight_scale_list[t] = \
                    self.get_weight_by_tp(m_dict[t], t_name, mcore_weight_path, mcore_bias_path)
        weight_list = None if all(x is None for x in weight_list) else weight_list
        bias_list = None if all(x is None for x in bias_list) else bias_list
        weight_scale_list = None if all(x is None for x in weight_scale_list) else weight_scale_list
        return weight_list, bias_list, weight_scale_list

    def get_tp_cat_source(self, name, m_tp, chunk_dim, source_list, need_transpose=False):
        if source_list is None:
            return None
        if chunk_dim is None or m_tp == 1:
            source = source_list[0]
        else:
            source = torch.cat(source_list, dim=chunk_dim)
        source = transpose_shape0(source, m_tp, 2) if need_transpose else source
        return source

    def convert_fp8s_to_bf16(self, name, m_tp, chunk_dim, weight_list, weight_scale_list, need_transpose=False):
        if weight_scale_list is not None:
            # fp8 need quantization.
            weight_bf16_s = []
            for i in range(len(weight_list)):
                weight = weight_list[i]
                weight_scale = weight_scale_list[i]
                weight_bf16 = convert_fp8_to_bf16(weight, weight_scale, dtype=torch.float32)
                weight_bf16_s.append(weight_bf16)
        else:
            # bf16 convert_to_fp8
            weight_bf16_s = weight_list
        return self.get_tp_cat_source(name, m_tp, chunk_dim, weight_bf16_s, need_transpose=need_transpose)

    def get_cat_weight(self, name, m_tp, weight_list, bias_list, weight_scale_list,
                       is_fp8, fp8_ignore_tp, ignore_tp=False, chunk_dim=None, quant_type=None):
        need_transpose = (m_tp > 1 and self.transpose_mlp_dense and \
                name in [MLP_DENSE_H_TO_4H, MOE_EXPERT_H_TO_4H])
        chunk_dim = self.tensor_parallel_dim.get(f"{name}.{WEIGHT}", None) if chunk_dim is None else chunk_dim
        if chunk_dim is None or m_tp == 1 or ignore_tp:
            # need not chunk
            weight = weight_list[0] if weight_list is not None else None
            weight_scale = weight_scale_list[0] if weight_scale_list is not None else None
            if weight_scale is not None and quant_type == QUANT_HF_BF16_AND_MCORE_FP8:
                # ======weight need quantization when dtype is not equal =======
                weight = convert_fp8_to_bf16(weight, weight_scale, dtype=torch.float32)
                weight_scale = None
        elif weight_scale_list is not None and quant_type == QUANT_HF_BF16_AND_MCORE_FP8:
            # ======weight need cat and need quantization when dtype is not equal =======
            weight = self.convert_fp8s_to_bf16(name, m_tp, chunk_dim, weight_list, weight_scale_list, need_transpose=need_transpose)
            weight_scale = None
        elif weight_scale_list is None and not (is_fp8 and self.convert_to_fp8):
            # bf16 and not convert to fp8
            weight = self.get_tp_cat_source(name, m_tp, chunk_dim, weight_list, need_transpose=need_transpose)
            weight_scale = None
        elif weight_scale_list is not None and (self.args.fp8_force_no_requant \
                or is_power_of_two(weight_scale_list[0]) == self.args.force_pow_2_scales):
            # fp8 and no quantization
            if (is_fp8 and fp8_ignore_tp) or ignore_tp:
                weight = weight_list[0] if weight_list is not None else None
                weight_scale = weight_scale_list[0] if weight_scale_list is not None else None
            else:
                BLOCK_SIZE = 128  # TODO: quantization block size fixs to 128 for now, generalize to others for compatibility
                for i in range(len(weight_list)):
                    _w_dim0, _w_dim1 = weight_list[i].size()
                    _ws_dim0, _ws_dim1 = _w_dim0 // BLOCK_SIZE, _w_dim1 // BLOCK_SIZE
                    weight_scale_list[i] = weight_scale_list[i][:_ws_dim0, :_ws_dim1]
                weight = self.get_tp_cat_source(name, m_tp, chunk_dim, weight_list, need_transpose=need_transpose)
                weight_scale = self.get_tp_cat_source(name, m_tp, chunk_dim, weight_scale_list, need_transpose=need_transpose)
        else:
            # need quantization. fp8 or bf16 convert_to_fp8
            weight_bf16 = self.convert_fp8s_to_bf16(name, m_tp, chunk_dim, weight_list, weight_scale_list, need_transpose=need_transpose)
            weight, weight_scale = convert_bf16_to_fp8(
                    weight_bf16, method=self.args.quant_method, amax_epsilon=self.args.amax_epsilon,
                    force_pow_2_scales=self.args.force_pow_2_scales)
            weight = weight.view(torch.float8_e4m3fn)

        bias_chunk_dim = self.tensor_parallel_dim.get(f"{name}.{BIAS}", None)
        if bias_chunk_dim is None:
            # Don't chunk bias
            bias = bias_list[0] if bias_list is not None else None
        elif bias_list is not None:
            bias = self.get_tp_cat_source(name, m_tp, bias_chunk_dim, bias_list, need_transpose=need_transpose)
        else:
            bias = None
        
        return weight, bias, weight_scale

    def lora_merge(
        self,
        base_weight: torch.Tensor,
        linear_out: torch.Tensor,
        linear_in: torch.Tensor,
        alpha: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Merges the LoRA adapter weights with the base model weights.

        Args:
            base_weight (torch.Tensor): The base model weights.
            linear_out (torch.Tensor): LoRA's B matrix.
            linear_in (torch.Tensor): LoRA's A matrix.
            alpha (int): Weighting factor for the low-rank projection.
            dim (int): Dimension of the low-rank projection space.

        Returns:
            torch.Tensor: The merged weights.
        """
        lora_weight = alpha / dim * (linear_out @ linear_in)
        return base_weight + lora_weight

