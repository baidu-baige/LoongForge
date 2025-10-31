#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os, sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from convert_checkpoint.megatron_optimizer import MegatronOptimizer, merge_optimizer_by_pp_tp, merge_optimizer_by_dp
from convert_checkpoint.megatron_checkpoint import MegatronCheckpoint
from convert_checkpoint.megatron_config import MegatronConfig
from convert_checkpoint.common_config import CommonConfig
from convert_checkpoint.arguments import parse_args


def load_optim_by_dp(ckpt, load_path, p, t):
    "load optimizer by dp strategy"
    named_parameters_shape = ckpt.get_named_parameters_shape(p, t)
    opts = []
    if ckpt.use_distributed_optimizer:
        for d in range(ckpt.dp):
            if ckpt.pp == 1:
                checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}"
            else:
                checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}"
            checkpoint_dir = os.path.join(load_path, checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, "optim.pt")
            _state_dict = torch.load(checkpoint_path, map_location="cpu")
            opt = MegatronOptimizer.generate_optimizer(ckpt, ckpt.num_layers // ckpt.pp, p)
            opt.load(_state_dict)
            opts.append(opt)
            opt.debug(f"tp/pp/dp rank: {t}/{p}/{d}, load from: {checkpoint_path}")
        return merge_optimizer_by_dp(opts, named_parameters_shape)
    else:
        opt = MegatronOptimizer.generate_optimizer(ckpt, ckpt.num_layers // ckpt.pp, p)
        if ckpt.pp == 1:
            checkpoint_dir = f"mp_rank_{t:02d}"
        else:
            checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}"
        checkpoint_dir = os.path.join(load_path, checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model_optim_rng.pt")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        opt.load(state_dict, named_parameters_shape)
        opt.build_param_map(named_parameters_shape)
        opt.debug(f"tp/pp rank: {t}/{p}, load from: {checkpoint_path}")
        return opt


def save_optim_by_dp(optim, ckpt, save_path, p, t):
    """ save optimizer by dp strategy """
    os.makedirs(save_path, exist_ok=True)
    folder_name = f"iter_{ckpt.iteration:07d}" if ckpt.iteration > 0 else "release"
    release_dir = os.path.join(save_path, folder_name)
    os.makedirs(release_dir, exist_ok=True)

    if ckpt.use_distributed_optimizer:
        chunk_optimers = optim.chunk_by_dp(ckpt.dp, ckpt.num_stages)
        del optim
        for d in range(ckpt.dp):
            if ckpt.pp == 1:
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
    else:
        checkpoint_dir = (
            f"mp_rank_{t:02d}"
            if ckpt.pp == 1
            else f"mp_rank_{t:02d}_{p:03d}"
        )
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        checkpoint_name = "model_optim_rng.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        state_dict = torch.load(checkpoint_path)
        state_dict.update(optim.to_dict())
        torch.save(state_dict, checkpoint_path)
        optim.debug(f"tp/pp rank {t}/{p}, saved to: {checkpoint_path}")
    

def change_pp(s_ckpt, t_ckpt, load_path, save_path, config):
    """ change pp """
    margs = config.get_args("megatron")
    for t in range(s_ckpt.tp):
        opts = [[load_optim_by_dp(s_ckpt, load_path, p, t)] \
                    for p in range(s_ckpt.pp)]
        optim = merge_optimizer_by_pp_tp(opts, margs)
        del opts
        optim.build_param_map(s_ckpt.get_named_parameters_shape(t=t))
        if s_ckpt.num_stages > 1:
            optim.interleave(s_ckpt.pp, s_ckpt.num_stages)
        if s_ckpt.pp == 1 and not margs.get("untie_embeddings_and_output_weights", False):
            optim.add_word_embedding_for_head()
        if t_ckpt.pp == 1 and not margs.get("untie_embeddings_and_output_weights", False):
            optim.remove_word_embedding_for_head()
        if t_ckpt.num_stages > 1:
            optim.interleave(t_ckpt.num_stages, t_ckpt.pp)
        opts = optim.chunk_by_pp_tp(t_ckpt.pp, 1, margs)
        del optim
        for p, opt in enumerate(opts):
            opt[0].build_param_map(t_ckpt.get_named_parameters_shape(p, t))
            save_optim_by_dp(opt[0], t_ckpt, save_path, p, t)
        del opts


def change_tp(s_ckpt, t_ckpt, load_path, save_path, config):
    """ change tp """
    cargs = config.get_args("common")
    margs = config.get_args("megatron")
    for p in range(s_ckpt.pp):
        opts = [[ load_optim_by_dp(s_ckpt, load_path, p, t) \
                    for t in range(s_ckpt.tp) ]]
        optim = merge_optimizer_by_pp_tp(opts, margs)
        del opts
        optim.build_param_map(s_ckpt.get_named_parameters_shape(p=p))
        if (p == 0 or p == s_ckpt.pp - 1) and margs.get("add_embedding_padding", False):
            divisible_by = margs["make_vocab_size_divisible_by"]
            padded_vocab_size = margs.get("pad_vocab_size_to")
            vocab_size = cargs["vocab_size"]
            hidden_size = cargs["hidden_size"]
            optim.cut_embedding_padding(vocab_size)
            optim.add_embedding_padding(divisible_by, vocab_size, t_ckpt.tp, hidden_size, padded_vocab_size)

        optim.use_distributed_optimizer = t_ckpt.use_distributed_optimizer
        optim.build_param_map(t_ckpt.get_named_parameters_shape(p=p))
        opts = optim.chunk_by_pp_tp(1, t_ckpt.tp, margs)
        del optim
        for t, opt in enumerate(opts[0]):
            opt.build_param_map(t_ckpt.get_named_parameters_shape(p, t))
            save_optim_by_dp(opt, t_ckpt, save_path, p, t)
        del opts


def change_dp(s_ckpt, t_ckpt, load_path, save_path):
    """ change dp """
    for p in range(s_ckpt.pp):
        for t in range(s_ckpt.tp):
            opt = load_optim_by_dp(s_ckpt, load_path, p, t)
            save_optim_by_dp(opt, t_ckpt, save_path, p, t)
            del opt


if __name__ == "__main__":
    args = parse_args()

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    args = parse_args()
    ckpt_path = args.load_ckpt_path
    save_path = args.save_ckpt_path
    config_path = args.common_config_path

    config = CommonConfig()
    config.load(config_path)
    for group in ["common", "megatron", "huggingface"]:
        args = parse_args(group)
        config.update_args(vars(args), group)
    dtype = config.get_dtype()

    cargs = config.get_args()
    num_layers = cargs['num_layers']
    name_map = config.get_name_map("megatron")

    # source ckpt
    s_config = MegatronConfig()
    s_config.load(ckpt_path)
    s_config.update({"tensor_parallel_dim": config.get("tensor_parallel_dim")})
    s_ckpt = MegatronCheckpoint(num_layers)
    s_ckpt.load(ckpt_path, s_config, name_map, load_optimizer=False)
    s_ckpt.set_dtype(dtype)
    s_ckpt.state_dict = []

    # target ckpt
    assert s_ckpt.iteration > 0
    folder_name = f"iter_{s_ckpt.iteration:07d}"
    release_dir = os.path.join(save_path, folder_name)
    t_config = MegatronConfig()
    t_config.load(release_dir)
    t_config.update({"tensor_parallel_dim": config.get("tensor_parallel_dim")})
    t_ckpt = MegatronCheckpoint(num_layers)
    t_ckpt.load(release_dir, t_config, name_map, load_optimizer=False)
    t_ckpt.set_dtype(dtype)
    t_ckpt.state_dict = []

    is_change_pp = (s_ckpt.pp != t_ckpt.pp) or (s_ckpt.num_stages != t_ckpt.num_stages)
    is_change_tp = (s_ckpt.tp != t_ckpt.tp)
    assert not (is_change_pp and is_change_tp), "cann't change tp and pp at the same time"
    if is_change_tp:
        change_tp(s_ckpt, t_ckpt, ckpt_path, save_path, config)
    elif is_change_pp :
        change_pp(s_ckpt, t_ckpt, ckpt_path, save_path, config)
    else:
        change_dp(s_ckpt, t_ckpt, ckpt_path, save_path)
