#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import shutil
import torch
import types
from tqdm import tqdm

from convert_checkpoint.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.common_checkpoint import CommonCheckpoint
from convert_checkpoint.megatron_optimizer import MegatronOptimizer, merge_optimizer_by_pp_tp, merge_optimizer_by_dp
from convert_checkpoint.utils import (
    check_path_in_dict,
    get_element_from_dict_by_path,
    add_embedding_padding, cut_embedding_padding,
    transpose_shape0,
    partition_balanced
)

from convert_checkpoint.common_checkpoint import (
    WORD_EMBEDDINGS,
    WORD_POSITION_EMBEDDINGS,
    WORD_BLOCK_POSITION_EMBEDDINGS,
    TRANSFORMER,
    LAYER_PREFIX,
    INPUT_LAYERNORM,
    ATTENTION_ROTARY_EMB_INV_FREQ,
    ATTENTION_QUERY_KEY_VALUE,
    ATTENTION_DENSE,
    POST_ATTENTION_LAYERNORM,
    POST_ATTENTION_LAYERSCALE,
    MLP_DENSE_H_TO_4H,
    MLP_DENSE_4H_TO_H,
    POST_MLP_LAYERSCALE,
    FINAL_LAYERNORM,
    WORD_EMBEDDINGS_FOR_HEAD,
    WORD_EMBEDDINGS_TPL,
    WORD_POSITION_EMBEDDINGS_TPL,
    WORD_BLOCK_POSITION_EMBEDDINGS_TPL,
    TRANSFORMER_TPL,
    WORD_EMBEDDINGS_FOR_HEAD_TPL,
)


def get_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


class MegatronCheckpoint(AbstractCheckpoint):
    """
        MegatronCheckpoint
    """

    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.version = 3.0
        self.iteration = 0
        self.args = types.SimpleNamespace()
        self.rng_state = None

    def init_pipeline_size(self, pp, tp, dp, tensor_parallel_dim, stage):
        "Initialize tp pp dp"
        self.pp = pp
        self.tp = tp
        self.dp = dp
        self.state_dict = []
        self.tensor_parallel_dim = tensor_parallel_dim
        self.num_stages = stage or 1
        assert self.num_layers // self.pp % self.num_stages == 0
        for p in range(pp):
            self.state_dict.append([])
            for t in range(tp):
                self.state_dict[p].append({})

    def init_optimizer(self, use_distributed_optimizer):
        assert self.pp > 0 and self.tp > 0
        self.use_distributed_optimizer = use_distributed_optimizer
        self.optim_state_dict = []
        for p in range(self.pp):
            self.optim_state_dict.append([])
            for t in range(self.tp):
                opt = MegatronOptimizer.generate_optimizer(self, self.num_layers // self.pp, p)
                self.optim_state_dict[p].append(opt)
    
    def init_layers(self):
        self.layers = []
        for p in range(self.pp):
            self.layers.append(self.get_layers(p))

    def init_named_parameters_shape(self):
        self.named_parameters_shape = self._get_named_parameters_shape()
        self.named_parameters_shape_by_p = [None] * self.pp
        self.named_parameters_shape_by_t = [None] * self.tp
        self.named_parameters_shape_by_pt = []
        for p in range(self.pp):
            self.named_parameters_shape_by_p[p] = self._get_named_parameters_shape(pp_rank=p)
        for t in range(self.tp):
            self.named_parameters_shape_by_t[t] = self._get_named_parameters_shape(tp_rank=t)
        for p in range(self.pp):
            self.named_parameters_shape_by_pt.append([None] * self.tp)
            for t in range(self.tp):
                self.named_parameters_shape_by_pt[p][t] =  self._get_named_parameters_shape(p, t)

    @staticmethod
    def convert_from_common(c_ckpt, c_config):
        """
        Convert common checkpoint to megatron checkpoint.

        Args:
            c_ckpt: CommonCheckpoint
            c_config: CommonConfig
        """

        print("\n==================== Common -> Megatron ====================")

        name_map = c_config.get("name_map")["megatron"]
        cargs = c_config.get_args("common")
        margs = c_config.get_args("megatron")
        dtype = c_config.get_dtype()
        tensor_parallel_dim = c_config.get("tensor_parallel_dim")

        tp = margs["tensor_model_parallel_size"]
        pp = margs["pipeline_model_parallel_size"]
        dp = margs["data_parallel_size"]
        num_layers = cargs["num_layers"]
        hidden_size = cargs["hidden_size"]
        use_distributed_optimizer = margs["use_distributed_optimizer"]
        num_layers_per_stage = margs["num_layers_per_virtual_pipeline_stage"] or (num_layers // pp)
        stage = num_layers // pp // num_layers_per_stage

        m_ckpt = MegatronCheckpoint(num_layers)
        m_ckpt.set_dtype(dtype)
        m_ckpt.init_pipeline_size(pp, tp, dp, tensor_parallel_dim, stage)
        m_ckpt.set_name_map(name_map)

        layer_prefix = name_map[LAYER_PREFIX]
        num_layers_in_vp, _ = partition_balanced(num_layers, pp * stage)

        def _convert(layer_name, transpose_shape=None):
            _get_weight_func = {
                INPUT_LAYERNORM: c_ckpt.get_layer_input_layernorm_weight,
                ATTENTION_QUERY_KEY_VALUE: c_ckpt.get_layer_attention_query_key_value_weight,
                ATTENTION_DENSE: c_ckpt.get_layer_attention_dense_weight,
                POST_ATTENTION_LAYERNORM: c_ckpt.get_layer_post_attention_layernorm_weight,
                MLP_DENSE_H_TO_4H: c_ckpt.get_layer_mlp_dense_h_to_4h_weight,
                MLP_DENSE_4H_TO_H: c_ckpt.get_layer_mlp_dense_4h_to_h_weight,
                POST_ATTENTION_LAYERSCALE: c_ckpt.get_layer_post_attention_layerscale,
                POST_MLP_LAYERSCALE: c_ckpt.get_layer_post_mlp_layerscale
            }[layer_name]
            _get_bias_func = {
                INPUT_LAYERNORM: c_ckpt.get_layer_input_layernorm_bias,
                ATTENTION_QUERY_KEY_VALUE: c_ckpt.get_layer_attention_query_key_value_bias,
                ATTENTION_DENSE: c_ckpt.get_layer_attention_dense_bias,
                POST_ATTENTION_LAYERNORM: c_ckpt.get_layer_post_attention_layernorm_bias,
                MLP_DENSE_H_TO_4H: c_ckpt.get_layer_mlp_dense_h_to_4h_bias,
                MLP_DENSE_4H_TO_H: c_ckpt.get_layer_mlp_dense_4h_to_h_bias,
                POST_ATTENTION_LAYERSCALE: lambda x: None,
                POST_MLP_LAYERSCALE: lambda x: None
            }[layer_name]
            weight_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.weight")
            bias_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.bias")
            for virtual_p in range(pp * stage):
                p = virtual_p % pp
                stage_index = virtual_p // pp
                layer_offset = sum(num_layers_in_vp[:virtual_p])
                transformer = m_ckpt.get_transformer_name(stage_index)
                for layer_index in range(num_layers_in_vp[virtual_p]):
                    name = f"{layer_prefix}.{layer_index}.{name_map[layer_name]}"
                    layer_id = layer_index + layer_offset
                    # weight
                    weight = _get_weight_func(layer_id)
                    if transpose_shape is not None:
                        weight = transpose_shape0(weight, *transpose_shape)
                    m_ckpt.update_tensor(p, weight, transformer, name + ".weight", weight_chunk_dim)
                    # bias
                    bias = _get_bias_func(layer_id)
                    if transpose_shape is not None and bias is not None:
                        bias = transpose_shape0(bias, *transpose_shape)
                    m_ckpt.update_tensor(p, bias, transformer, name + ".bias", bias_chunk_dim)
            print(f"> {layer_name} chunk weight dim {weight_chunk_dim}, bias dim {bias_chunk_dim}")

        # 1.1 word embeddings with paddding
        weight = c_ckpt.get_word_embedding()
        if margs.get("add_embedding_padding", False):
            divisible_by = margs["make_vocab_size_divisible_by"]
            vocab_size = cargs["vocab_size"]
            padded_vocab_size = margs.get("pad_vocab_size_to")
            if padded_vocab_size is None:
                assert vocab_size == weight.shape[0]
            weight = add_embedding_padding(weight, divisible_by, vocab_size, tp, padded_vocab_size)

        name = m_ckpt.get_word_embedding_name()
        chunk_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS}.weight")
        m_ckpt.update_tensor(0, weight, name, "weight", chunk_dim)
        print(f"> {name} weight {weight.shape}")

        # 1.2 position embedding
        if margs.get("add_position_embedding", False):
            name = m_ckpt.get_position_embedding_name()
            weight = c_ckpt.get_word_position_embedding()
            chunk_dim = tensor_parallel_dim.get(f"{WORD_POSITION_EMBEDDINGS}.weight")
            m_ckpt.update_tensor(0, weight, name, "weight", chunk_dim)
            print(f"> {name} weight {weight.shape}")

        if margs.get("add_block_position_embedding", False):
            name =  m_ckpt.get_block_position_embedding_name()
            weight = c_ckpt.get_word_block_position_embedding()
            chunk_dim = tensor_parallel_dim.get(f"{WORD_BLOCK_POSITION_EMBEDDINGS}.weight")
            m_ckpt.update_tensor(0, weight, name, "weight", chunk_dim)
            print(f"> {name} weight {weight.shape}")

        # 2. transformer layers
        # 2.1 input_layernorm
        _convert(INPUT_LAYERNORM)

        # 2.2 rotary_emb.inv_freqs
        if ATTENTION_ROTARY_EMB_INV_FREQ in name_map:
            chunk_dim = tensor_parallel_dim.get(ATTENTION_ROTARY_EMB_INV_FREQ)
            for virtual_p in range(pp * stage):
                p = virtual_p % pp
                stage_index = virtual_p // pp
                layer_offset = sum(num_layers_in_vp[:virtual_p])
                transformer = m_ckpt.get_transformer_name(stage_index)
                for layer_index in range(num_layers_in_vp[virtual_p]):
                    name = f"{layer_prefix}.{layer_index}.{name_map[ATTENTION_ROTARY_EMB_INV_FREQ]}"
                    layer_id = layer_index + layer_offset
                    inv_freq = c_ckpt.get_layer_attention_rotary_emb_inv_freq(layer_id)
                    m_ckpt.update_tensor(p, inv_freq, transformer, name, chunk_dim)
            print(f"> rotary_emb.inv_freqs chunk dim {chunk_dim}")

        # 2.3 self attention query_key_value
        _convert(ATTENTION_QUERY_KEY_VALUE)

        # 2.4 self attention dense
        _convert(ATTENTION_DENSE)

        # 2.5 post attention layernorm
        _convert(POST_ATTENTION_LAYERNORM)

        # post attention layescale
        _convert(POST_ATTENTION_LAYERSCALE)

        # 2.6 mlp dense h_to_4h
        if margs.get("transpose_mlp_dense", False):
            _convert(MLP_DENSE_H_TO_4H, transpose_shape=(2, tp))
        else:
            _convert(MLP_DENSE_H_TO_4H)

        # 2.7 mlp dense 4h_to_h
        _convert(MLP_DENSE_4H_TO_H)

        # post mlp layerscale
        _convert(POST_MLP_LAYERSCALE)

        # 2.8 final_layernorm
        transformer = m_ckpt.get_transformer_name(m_ckpt.num_stages - 1)
        name = name_map[FINAL_LAYERNORM]
        # weight
        chunk_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.weight")
        weight = c_ckpt.get_final_layernorm_weight()
        m_ckpt.update_tensor(pp-1, weight, transformer, name + ".weight", chunk_dim)
        # bias
        chunk_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.bias")
        bias = c_ckpt.get_final_layernorm_bias()
        m_ckpt.update_tensor(pp-1, bias, transformer, name + ".bias", chunk_dim)
        print(f"> {name} weight {weight.shape}")

        # 3 word embedding for head
        if margs.get("untie_embeddings_and_output_weights", False) or m_ckpt.pp > 1:
            chunk_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS_FOR_HEAD}.weight")
            name = m_ckpt.get_word_embedding_for_head_name()
            weight = c_ckpt.get_word_embeddings_for_head_weight()
            if margs.get("add_embedding_padding", False):
                divisible_by = margs["make_vocab_size_divisible_by"]
                orig_vocab_size = cargs["vocab_size"]
                padded_vocab_size = margs.get("pad_vocab_size_to")
                weight = add_embedding_padding(weight, divisible_by, orig_vocab_size, tp, padded_vocab_size)
            m_ckpt.update_tensor(pp - 1, weight, name, "weight", chunk_dim)
            print(f"> {name} weight {weight.shape}")

        # 4. optimizer
        m_ckpt.init_layers()
        m_ckpt.init_named_parameters_shape()
        m_ckpt.init_optimizer(use_distributed_optimizer)
        if c_ckpt.has_optimizer():
            opt = MegatronOptimizer.generate_optimizer(m_ckpt, c_ckpt.num_layers)
            opt.load(c_ckpt.state_dict["optimizer"])
            if margs.get("add_embedding_padding", False):
                divisible_by = margs["make_vocab_size_divisible_by"]
                vocab_size = cargs["vocab_size"]
                padded_vocab_size = margs.get("pad_vocab_size_to")
                opt.add_embedding_padding(divisible_by, vocab_size, tp, hidden_size, padded_vocab_size)
            if m_ckpt.pp == 1 and not margs.get("untie_embeddings_and_output_weights", False):
                opt.remove_word_embedding_for_head()
            opt.build_param_map(m_ckpt.get_named_parameters_shape())
            if stage > 1:
                opt.interleave(stage, pp)
            opts = opt.chunk_by_pp_tp(pp, tp, margs)
            for p in range(pp):
                for t in range(tp):
                    named_parameters_shape = m_ckpt.get_named_parameters_shape(p, t)
                    opts[p][t].build_param_map(named_parameters_shape)
            m_ckpt.optim_state_dict = opts
        else:
            print("> optimizer empty")

        # 5.others
        m_ckpt.iteration = c_ckpt.other_args.get("iteration", m_ckpt.iteration)
        m_ckpt.version = c_ckpt.other_args.get("version", m_ckpt.version)
        m_ckpt.args = c_ckpt.other_args.get("args", m_ckpt.args)
        m_ckpt.rng_state = c_ckpt.other_args.get("rng_state", m_ckpt.rng_state)

        m_ckpt.debug("Finish common -> megatron")
        return m_ckpt

    def update_tensor(self, pp, source, layer, key, dim=None):
        if source is not None:
            if dim is not None:
                source = torch.chunk(source, self.tp, dim=dim)
            for t in range(self.tp):
                element = get_element_from_dict_by_path(
                    self.state_dict[pp][t], layer
                )
                element[key] = (source if dim is None else source[t]) \
                        .clone().to(self.dtype)

    def get_tensor(self, pp, layer, key, dim=None):
        if dim is not None:
            tp_state_dict = []
            for t in range(self.tp):
                element = get_element_from_dict_by_path(
                    self.state_dict[pp][t], layer
                )
                if key not in element:
                    return None
                tp_state_dict.append(element[key])
            return torch.cat(tp_state_dict, dim=dim)
        else:
            element = get_element_from_dict_by_path(
                self.state_dict[pp][0], layer
            )
            return element[key] if key in element else None
        
    def convert_to_common(self, c_config):
        """
        Convert Megatron checkpoint to common checkpoint.
            Args:
                c_config: CommonConfig
        """

        print("\n==================== Megatron -> Common ====================")

        tensor_parallel_dim = self.tensor_parallel_dim
        name_map = c_config.get_name_map("megatron")
        cargs = c_config.get_args("common")
        margs = c_config.get_args("megatron")
        num_layers = cargs["num_layers"]
        hidden_size = cargs["hidden_size"]
        num_attention_heads = cargs["num_attention_heads"]
        num_layers_per_stage = self.num_layers // self.pp // self.num_stages

        c_ckpt = CommonCheckpoint(num_layers)
        c_ckpt.set_dtype(self.dtype)

        layer_prefix = name_map[LAYER_PREFIX]
        num_layers_in_vp, _ = partition_balanced(num_layers, self.pp * self.num_stages)

        def _convert(layer_name, transpose_shape=None):
            _set_func = {
                INPUT_LAYERNORM: c_ckpt.set_layer_input_layernorm,
                ATTENTION_QUERY_KEY_VALUE: c_ckpt.set_layer_attention_query_key_value,
                ATTENTION_DENSE: c_ckpt.set_layer_attention_dense,
                POST_ATTENTION_LAYERNORM: c_ckpt.set_layer_post_attention_layernorm,
                MLP_DENSE_H_TO_4H: c_ckpt.set_layer_mlp_dense_h_to_4h,
                MLP_DENSE_4H_TO_H: c_ckpt.set_layer_mlp_dense_4h_to_h,
                POST_ATTENTION_LAYERSCALE: c_ckpt.set_layer_post_attention_layerscale,
                POST_MLP_LAYERSCALE: c_ckpt.set_layer_post_mlp_layerscale
            }[layer_name]
            weight_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.weight")
            bias_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.bias")
            for virtual_p in range(self.pp * self.num_stages):
                p = virtual_p % self.pp
                stage_index = virtual_p // self.pp
                layer_offset = sum(num_layers_in_vp[:virtual_p])
                transformer = self.get_transformer_name(stage_index)
                for layer_index in range(num_layers_in_vp[virtual_p]):
                    name = f"{layer_prefix}.{layer_index}.{name_map[layer_name]}"
                    layer_id = layer_index + layer_offset
                    weight = self.get_tensor(p, transformer, name + ".weight", weight_chunk_dim)
                    bias = self.get_tensor(p, transformer, name + ".bias", bias_chunk_dim)
                    if transpose_shape is not None:
                        weight = transpose_shape0(weight, *transpose_shape)
                        if bias is not None:
                            bias = transpose_shape0(bias, *transpose_shape)
                    _set_func(layer_id, weight, bias)
            print(f"> {layer_name} chunk weight dim {weight_chunk_dim}, bias dim {bias_chunk_dim}")

        # 1.1 word_embeddings
        parallel_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS}.weight")
        name = self.get_word_embedding_name()
        weight = self.get_tensor(0, name, 'weight', parallel_dim)
        if margs.get("add_embedding_padding", False):
            orig_vocab_size = cargs["vocab_size"]
            weight = cut_embedding_padding(weight, orig_vocab_size)
        c_ckpt.set_word_embedding(weight)
        print(f"> word_embeddings weight: {weight.shape}")

        # 1.2 position embedding
        if margs.get("add_position_embedding", False):
            name = self.get_position_embedding_name()
            parallel_dim = tensor_parallel_dim.get(f"{WORD_POSITION_EMBEDDINGS}.weight")
            weight = self.get_tensor(0, name, 'weight', parallel_dim)
            c_ckpt.set_word_position_embedding(weight)
            print(f"add position embedding weight: {weight.shape}")

        if margs.get("add_block_position_embedding", False):
            name =  self.get_block_position_embedding_name()
            parallel_dim = tensor_parallel_dim.get(f"{WORD_BLOCK_POSITION_EMBEDDINGS}.weight")
            weight =  self.get_tensor(0, name, 'weight', parallel_dim)
            c_ckpt.set_word_block_position_embedding(weight)
            print(f"> add block position embedding weight: {weight.shape}")

        # 2. transformer layers

        # 2.1 input_layernorm
        _convert(INPUT_LAYERNORM)

        # 2.2 rotary_emb.inv_freqs
        if ATTENTION_ROTARY_EMB_INV_FREQ in name_map:
            chunk_dim = tensor_parallel_dim.get(ATTENTION_ROTARY_EMB_INV_FREQ)
            for virtual_p in range(self.pp * self.num_stages):
                p = virtual_p % self.pp
                stage_index = virtual_p // self.pp
                layer_offset = sum(num_layers_in_vp[:virtual_p])
                transformer = self.get_transformer_name(stage_index)
                for layer_index in range(num_layers_in_vp[virtual_p]):
                    name = f"{layer_prefix}.{layer_index}.{name_map[ATTENTION_ROTARY_EMB_INV_FREQ]}"
                    layer_id = layer_index + layer_offset
                    inv_freq = self.get_tensor(p, transformer, name, chunk_dim)
                    c_ckpt.set_layer_attention_rotary_emb_inv_freq(layer_id, inv_freq)
            print(f"> rotary_emb.inv_freqs chunk dim {chunk_dim}")
        elif margs.get('use_rotary_position_embeddings', False):
            dim = hidden_size // num_attention_heads
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            for virtual_p in range(self.pp * self.num_stages):
                p = virtual_p % self.pp
                stage_index = virtual_p // self.pp
                layer_offset = sum(num_layers_in_vp[:virtual_p])
                for layer_index in range(num_layers_in_vp[virtual_p]):
                    layer_id = layer_index + layer_offset
                    c_ckpt.set_layer_attention_rotary_emb_inv_freq(layer_id, inv_freq)
            print(f"> rotary_emb.inv_freqs, created by dim {dim}")

   
        # 2.3 self attention query_key_value
        _convert(ATTENTION_QUERY_KEY_VALUE)

        # 2.4 self attention dense
        _convert(ATTENTION_DENSE)

        # 2.5 post attention layernorm
        _convert(POST_ATTENTION_LAYERNORM)

        # post attention layerscale
        _convert(POST_ATTENTION_LAYERSCALE)

        # 2.6 mlp dense h_to_4h
        if margs.get("transpose_mlp_dense", False):
            _convert(MLP_DENSE_H_TO_4H, transpose_shape=(self.tp, 2))
        else:
            _convert(MLP_DENSE_H_TO_4H)

        # 2.7 mlp dense 4h_to_h
        _convert(MLP_DENSE_4H_TO_H)

        # post mlp layerscale
        _convert(POST_MLP_LAYERSCALE)

        # 2.8 final_layernorm
        transformer = self.get_transformer_name(self.num_stages - 1)
        name = name_map[FINAL_LAYERNORM]

        parallel_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.weight")
        weight =  self.get_tensor(self.pp-1, transformer, name + ".weight", parallel_dim)

        parallel_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.bias")
        bias =  self.get_tensor(self.pp-1, transformer, name + ".bias", parallel_dim)

        c_ckpt.set_final_layernorm(weight, bias)
        print(f"> final_layernorm weight {weight.shape}")

        # 3 word embedding for head
        if margs.get("untie_embeddings_and_output_weights", False) or self.pp > 1:
            parallel_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS_FOR_HEAD}.weight")
            name = self.get_word_embedding_for_head_name()
        else:
            parallel_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS}.weight")
            name = self.get_word_embedding_name()
        weight =  self.get_tensor(self.pp - 1, name, 'weight', parallel_dim)
        if margs.get("add_embedding_padding", False):
            orig_vocab_size = cargs["vocab_size"]
            weight = cut_embedding_padding(weight, orig_vocab_size)
        c_ckpt.set_word_embeddings_for_head(weight)
        print(f"> word embedding for head weight {weight.shape}")

        # 4. optimizer
        if self.has_optimizer():
            opt = merge_optimizer_by_pp_tp(self.optim_state_dict, margs)
            opt.build_param_map(self.get_named_parameters_shape())
            if margs.get("add_embedding_padding", False):
                orig_vocab_size = cargs["vocab_size"]
                opt.cut_embedding_padding(orig_vocab_size)
            if self.num_stages > 1:
                opt.interleave(self.pp, self.num_stages)
            if self.pp == 1 and not margs.get("untie_embeddings_and_output_weights", False):
                opt.add_word_embedding_for_head()
            c_ckpt.state_dict["optimizer"] = opt.to_dict()
            print("> optimizer params: ", opt.get_param_num())
        else:
            print("> optimizer empty")


        # 5.others
        c_ckpt.other_args["iteration"] = self.iteration
        c_ckpt.other_args["version"] = self.version
        c_ckpt.other_args["args"] = self.args
        c_ckpt.other_args["rng_state"] = self.rng_state

        return c_ckpt

    def set_name_map(self, name_map):
        """ set name_map """
        self.name_map = name_map

    def get_word_embedding_name(self):
        """ get word_embedding name """
        if self.num_stages > 1:
            if WORD_EMBEDDINGS_TPL not in self.name_map:
                return None
            return self.name_map[WORD_EMBEDDINGS_TPL] % 0
        else:
            return self.name_map.get(WORD_EMBEDDINGS)

    def get_position_embedding_name(self):
        """ get position_embedding name """
        if self.num_stages > 1 :
            if WORD_POSITION_EMBEDDINGS_TPL not in self.name_map:
                return None
            return self.name_map[WORD_POSITION_EMBEDDINGS_TPL] % 0
        else:
            return self.name_map.get(WORD_POSITION_EMBEDDINGS)

    def get_block_position_embedding_name(self):
        """ get block_position_embedding name """
        if self.num_stages > 1 :
            if WORD_BLOCK_POSITION_EMBEDDINGS_TPL not in self.name_map:
                return None
            return self.name_map[WORD_BLOCK_POSITION_EMBEDDINGS_TPL] % 0
        else:
            return self.name_map.get(WORD_BLOCK_POSITION_EMBEDDINGS)

    def get_transformer_name(self, stage_index):
        """ get transformer name """
        if self.num_stages > 1:
            return self.name_map[TRANSFORMER_TPL] % stage_index 
        else:
            return self.name_map[TRANSFORMER]

    def get_word_embedding_for_head_name(self):
        """ get word_embedding for head name """
        if self.num_stages > 1:
            return self.name_map[WORD_EMBEDDINGS_FOR_HEAD_TPL] % (self.num_stages - 1)
        else:
            return self.name_map[WORD_EMBEDDINGS_FOR_HEAD]

    def has_optimizer(self):
        """ whether has optimizer """
        for p in range(self.pp):
            for t in range(self.tp):
                if self.optim_state_dict[p][t].empty():
                    return False
        return True

    def has_bias(self, pp_rank=None):
        for p in range(self.pp) if pp_rank is None else [pp_rank]:
            for prefix, name, _ in self.layers[p]:
                if name.endswith("bias"):
                    return True
        return False

    def has_word_embeddings(self, p=None):
        """ whether has word embeddings """
        p = 0 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if prefix == self.get_word_embedding_name() and "weight" == name:
                return True
        return False
    def has_position_embeddings(self, p=None):
        """ whether has position embeddings """
        p = 0 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if prefix == self.get_position_embedding_name() and "weight" == name:
                return True
        return False

    def has_block_position_embeddings(self, p=None):
        """ whether has block position embeddings """
        p = 0 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if prefix == self.get_block_position_embedding_name() and "weight" == name:
                return True
        return False

    def has_word_embeddings_for_head(self, p=None):
        p = self.pp-1 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if self.get_word_embedding_for_head_name() == prefix and "weight" == name:
                return True
        return False

    def has_final_layernorm(self, key, p=None):
        p = self.pp-1 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if f"{self.name_map[FINAL_LAYERNORM]}.{key}" == name:
                return True
        return False

    def get_layers(self, pp_rank=None):
        """ get all layers """
        layers = []
        tensor_parallel_dim = self.tensor_parallel_dim

        # embedding
        if pp_rank == 0 or pp_rank is None:
            keys = (WORD_EMBEDDINGS, \
                    WORD_POSITION_EMBEDDINGS, \
                    WORD_BLOCK_POSITION_EMBEDDINGS)
            names = [self.get_word_embedding_name(), \
                    self.get_position_embedding_name(), \
                    self.get_block_position_embedding_name()]
            for i in range(3):
                if names[i] is not None and check_path_in_dict(self.state_dict[0][0], names[i]):
                    chunk_dim = tensor_parallel_dim.get(f"{keys[i]}.weight", -1)
                    layers.append((names[i], "weight", chunk_dim))

        # transformers
        TRANSFORMER_LAYERS = [INPUT_LAYERNORM, ATTENTION_QUERY_KEY_VALUE, ATTENTION_DENSE, \
                POST_ATTENTION_LAYERNORM, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, ]
        num_layers_per_pp = self.num_layers // self.pp
        num_layers_per_stage = num_layers_per_pp // self.num_stages
        for p in range(self.pp) if pp_rank is None else [pp_rank]:
            layer_prefix = self.name_map.get(LAYER_PREFIX)
            for layer_id in range(num_layers_per_pp):
                layer_prefix = self.name_map.get(LAYER_PREFIX)
                stage_index = layer_id // num_layers_per_stage
                layer_index = layer_id % num_layers_per_stage
                transformer_name = self.get_transformer_name(stage_index)
                transformer = get_element_from_dict_by_path(self.state_dict[p][0], transformer_name)
                for layer in TRANSFORMER_LAYERS:
                    for w in ("weight", "bias"):
                        layer_name = f"{layer_prefix}.{layer_index}.{self.name_map[layer]}.{w}"
                        if layer_name in transformer.keys():
                            chunk_dim = tensor_parallel_dim.get(f"{layer}.{w}", -1)
                            layers.append((transformer_name, layer_name, chunk_dim))
                for layer in [ATTENTION_ROTARY_EMB_INV_FREQ]:
                    if layer not in self.name_map:
                        continue
                    layer_name = f"{layer_prefix}.{layer_index}.{self.name_map[layer]}"
                    if layer_name in transformer.keys():
                        chunk_dim = self.tensor_parallel_dim.get(f"{layer}", -1)
                        layers.append((transformer_name, layer_name, chunk_dim))

            transformer_name = self.get_transformer_name(self.num_stages - 1)
            transformer = get_element_from_dict_by_path(self.state_dict[p][0], transformer_name)
            for layer in [FINAL_LAYERNORM]:
                for w in ("weight", "bias"):
                    layer_name = f"{self.name_map[layer]}.{w}"
                    if layer_name in transformer.keys():
                        chunk_dim = tensor_parallel_dim.get(f"{layer}.{w}", -1)
                        layers.append((transformer_name, layer_name, chunk_dim))

        # emebdding for head
        if pp_rank == self.pp-1 or pp_rank is None:
            layer = self.get_word_embedding_for_head_name()
            chunk_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS_FOR_HEAD}.weight", -1)
            layers.append((layer, "weight", chunk_dim))

        return layers

    def get_transformer_layers(self, pp_rank, layer_index):
        """ get transformer layers """
        layers = []
        pp_rank = 0 if pp_rank is None else pp_rank
        transformer_name = self.get_transformer_name(0)
        for meta in self.get_named_parameters_shape(pp_rank):
            layer_name = meta[0]
            key = f"{transformer_name}.{self.name_map[LAYER_PREFIX]}.{layer_index}."
            if key in layer_name:
                layers.append(meta)
        # print(pp_rank, layer_index, layers)
        return layers

    def get_named_parameters_shape(self, p=None, t=None):
        if p == None and t == None:
            return self.named_parameters_shape
        if p != None and t != None:
            return self.named_parameters_shape_by_pt[p][t]
        if p != None:
            return self.named_parameters_shape_by_p[p]
        if t != None:
            return self.named_parameters_shape_by_t[t]

    def _get_named_parameters_shape(self, pp_rank=None, tp_rank=None):
        """ return list of (layer_name, tensor_shape, parallel_dim) """
        result = []
        for p in range(self.pp) if pp_rank is None else [pp_rank]:
            for layer, key, parallel_dim in self.layers[p]:
                if key.endswith("weight") or key.endswith("bias"):
                    element = get_element_from_dict_by_path(
                        self.state_dict[p][0], layer
                    )
                    if key in element:
                        shape = element[key].shape
                        if tp_rank is None and parallel_dim >= 0:
                            shape = list(shape)
                            shape[parallel_dim] *= self.tp
                            shape = torch.Size(shape)
                        result.append((f"{layer}.{key}", shape, parallel_dim))
        return result

    def load(self, load_path, m_config, name_map, load_optimizer=True):
        """
        Load megatron checkpoint from checkpoints folder.

            Args:
                load_path (str): the path to checkpoint
                m_config: megatron m_config loaded from ckpt
        """

        tp = m_config.get("tensor_model_parallel_size")
        pp = m_config.get("pipeline_model_parallel_size")
        dp = m_config.get("data_parallel_size")
        dtype = m_config.get("dtype")
        tensor_parallel_dim = m_config.get("tensor_parallel_dim")
        num_layers_per_stage = m_config.get('num_layers_per_virtual_pipeline_stage')
        stage = None if num_layers_per_stage is None else (self.num_layers // pp // num_layers_per_stage)
        use_distributed_optimizer = m_config.get("use_distributed_optimizer", False)
        self.set_dtype(dtype)
        self.init_pipeline_size(pp, tp, dp, tensor_parallel_dim, stage)
        self.set_name_map(name_map)

        # weight and bias
        pbar = tqdm(range(self.tp * self.pp), desc='Loading Megatron-LM Checkpoint', leave=False)
        for p in range(self.pp):
            for t in range(self.tp):
                sub_dir_name = f"mp_rank_{t:02d}" if self.pp == 1 \
                        else f"mp_rank_{t:02d}_{p:03d}"
                checkpoint_name = os.listdir(os.path.join(load_path, sub_dir_name))[0]
                checkpoint_path = os.path.join(load_path, sub_dir_name, checkpoint_name)
                self.state_dict[p][t] = torch.load(checkpoint_path, map_location="cpu")
                pbar.update(1)
        self.iteration = self.state_dict[0][0].get('iteration', 0)
        self.version = self.state_dict[0][0]['checkpoint_version']
        self.args = self.state_dict[0][0]['args']
        self.rng_state = self.state_dict[0][0].get('rng_state', None)
        self.init_layers()
        self.init_named_parameters_shape()
        self.init_optimizer(use_distributed_optimizer)

        # optimizer
        if load_optimizer:
            if use_distributed_optimizer:
                for p in range(self.pp):
                    for t in range(self.tp):
                        opts = []
                        named_parameters_shape = self.get_named_parameters_shape(p, t)
                        for d in range(self.dp):
                            if self.pp == 1:
                                checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}"
                            else:
                                checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}"
                            checkpoint_dir = os.path.join(load_path, checkpoint_dir)
                            checkpoint_path = os.path.join(checkpoint_dir, "optim.pt")
                            optim_state_dict = torch.load(checkpoint_path, map_location="cpu")
                            opt = MegatronOptimizer.generate_optimizer(self, self.num_layers // self.pp, p)
                            opt.load(optim_state_dict)
                            opts.append(opt)
                            opt.debug(f"tp/pp/dp rank: {t}/{p}/{d}, load from: {checkpoint_path}")
                        self.optim_state_dict[p][t] = merge_optimizer_by_dp(opts, named_parameters_shape)
                        self.optim_state_dict[p][t].debug(f"merge by dp {self.dp} in pp/tp rank {p}/{t}")

            else:
                for p in range(self.pp):
                    for t in range(self.tp):
                        if "optimizer" in self.state_dict[p][t]:
                            named_parameters_shape = self.get_named_parameters_shape(p, t)
                            self.optim_state_dict[p][t].load(self.state_dict[p][t], named_parameters_shape)
        self.debug("==================== megatron checkpoint loaded ================================")

    def save(self, save_path, m_config=None, save_optim=True):
        """  save megatron checkpoint """
        os.makedirs(save_path, exist_ok=True)
        # Saving the tracker file
        tracker_filepath = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        with open(tracker_filepath, "w") as f:
            f.write(str(self.iteration or "release"))

        # create `release` dir in args.load_path
        folder_name = f"iter_{self.iteration:07d}" if self.iteration > 0 else "release"
        release_dir = os.path.join(save_path, folder_name)
        os.makedirs(release_dir, exist_ok=True)

        # megatron config
        margs = self.args
        if m_config is not None:
            for k, v in m_config.data.items():
                setattr(margs, k, v)
        print(f"Saving megatron args {margs}")

        # weight and bias
        for p in range(self.pp):
            for t in range(self.tp):
                self.state_dict[p][t]["checkpoint_version"] = self.version
                checkpoint_dir = (
                    f"mp_rank_{t:02d}"
                    if self.pp == 1
                    else f"mp_rank_{t:02d}_{p:03d}"
                )

                if self.use_distributed_optimizer:
                    checkpoint_name = "model_rng.pt"
                else:
                    self.state_dict[p][t].update(self.optim_state_dict[p][t].to_dict())
                    checkpoint_name = "model_optim_rng.pt"
                if margs is not None:
                    self.state_dict[p][t]['args'] = margs
                if self.rng_state is not None:
                    self.state_dict[p][t]['rng_state'] = self.rng_state
                self.state_dict[p][t]["iteration"] = self.iteration
                checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(self.state_dict[p][t], checkpoint_path)
                print(f"Saving megatron checkpoint {self.state_dict[p][t].keys()} to: {checkpoint_path}")

        # optimizer
        if self.use_distributed_optimizer and save_optim:
            for p in range(self.pp):
                for t in range(self.tp):
                    chunk_optimers = self.optim_state_dict[p][t].chunk_by_dp(self.dp, self.num_stages)
                    for d in range(self.dp):
                        if self.pp == 1:
                            checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}"
                        else:
                            checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}"

                        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(checkpoint_dir, "optim.pt")
                        torch.save(
                            chunk_optimers[d].to_dict(),
                            checkpoint_path,
                        )
                        chunk_optimers[d].debug(f"tp/pp/dp rank {t}/{p}/{d}, saved to: {checkpoint_path}")

    def debug(self, title):
        """ debbug """
        print(f"\n【Megatron】{title}")
        print(f"-> tp/pp/dp size: {self.tp}/{self.pp}/{self.dp}")
        if self.has_optimizer():
            for t in range(self.tp):
                for p in range(self.pp):
                    self.optim_state_dict[p][t].debug(f"tp/pp rank {t}/{p}")
        print("\n")

if __name__ == "__main__":
    pass
