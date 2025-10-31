#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import torch
import numpy
import math
from copy import deepcopy

from convert_checkpoint.utils import add_embedding_padding, cut_embedding_padding, transpose_shape0


def get_param_size_per_dp(total, dp):
    """ split params to each dp """
    size = math.ceil(total / dp)
    return [size] * (dp - 1) + [total - (dp - 1) * size]


def get_group_id(name, shape):
    return 1 if name.endswith('bias') or len(shape) == 1 else 0


def split_optimer_param(params, index):
    """
        param {
            key1: tensor 
            key2: tensor
        }
    """
    result = ({}, {})
    for key, value in params.items():
        result[0][key] = value[:index].clone()
        result[1][key] = value[index:].clone()
    return result


def cat_optimer_param(params):
    """
        param {
            key1: tensor 
            key2: tensor
        }
    """
    result = {}
    for k in params[0].keys():
        result[k] = torch.cat([param[k] for param in params], dim=0)
    return result


def merge_optimizer_by_pp_tp(optimizers, args):
    """ merge optimizers by pp tp """
    num_layers = sum([opts[0].num_layers for opts in optimizers])
    result = optimizers[0][0].clone_without_param(num_layers, end_offset=optimizers[-1][0].end_offset)

    # state
    if optimizers[0][0].get_state_num() > 0:
        state_keys = optimizers[0][0].get_keys()
        print("merge_optimizer by pp tp, state keys: ", state_keys)
        for group_id in range(2):
            for opts in optimizers:
                param_ids = opts[0].get_param_ids_in_groups()[group_id]
                for param_id in param_ids:
                    name, shape, parallel_dim, _ = opts[0].param_map[param_id]
                    if parallel_dim < 0:
                        param = opts[0].get_state(param_id)
                    else:
                        param = {}
                        for key in state_keys:
                            data = []
                            for opt in opts:
                                value = opt.get_state(param_id, key).view(shape)
                                data.append(value)
                            data = torch.cat(data, dim=parallel_dim)
                            if "dense_h_to_4h" in name and args.get("transpose_mlp_dense", False):
                                tp = len(opts)
                                data = transpose_shape0(data, tp, 2)
                            if opts[0].use_distributed_optimizer:
                                param[key] = data.view(-1)
                            else:
                                param[key] = data
                    result.add_state_in_group(param, group_id)

    # shard_fp32_from_float16_groups
    if optimizers[0][0].get_shard_fp32_num() > 0:
        print("merge_optimizer by pp tp, shard_fp32_from_float16_groups ")
        for opts in optimizers:
            param_groups = opts[0].get_param_ids_in_groups()
            for group_id, param_ids in enumerate(param_groups):
                for index, param_id in enumerate(param_ids):
                    name, shape, parallel_dim, _ = opts[0].param_map[param_id]
                    if parallel_dim < 0:
                        param = opts[0].get_shard_fp32_in_group(group_id, index)
                    else:
                        data = []
                        for opt in opts:
                            value = opt.get_shard_fp32_in_group(group_id, index).view(shape)
                            data.append(value)
                        data = torch.cat(data, dim=parallel_dim)
                        if "dense_h_to_4h" in name and args.get("transpose_mlp_dense", False):
                            tp = len(opts)
                            data = transpose_shape0(data, tp, 2)
                        if opts[0].use_distributed_optimizer:
                            param = data.view(-1)
                        else:
                            param = data
                    result.add_shard_fp32(param, group_id)
        result.update_param_groups_by_shard()

    result.debug(f"merge by pp{len(optimizers)} tp{len(optimizers[0])}")
    return result


def check_split_group(named_parameters_shape, params_per_dp):
    """ check which params group will be split when zero1 enable
        returns:
            0: split in group0
            1: split in group1
            -1: no split needed
    """
    result = []
    cnt = 0
    d = len(params_per_dp) - 1
    for name, shape, _ in named_parameters_shape:
        param_size = numpy.prod(shape)
        while True:
            if cnt + param_size > params_per_dp[d]:
                group_id = get_group_id(name, shape)
                result.append(group_id)
                param_size -= params_per_dp[d] - cnt
                cnt = 0
                d -= 1
            elif cnt + param_size == params_per_dp[d]:
                result.append(-1)
                cnt = 0
                d -= 1
                break
            else:
                cnt += param_size
                break
    assert cnt == 0 and d == -1
    return list(reversed(result))


def get_named_parameters_shape_by_stage(named_parameters_shape, stage):
    """ get named_parameters_shape by stage """
    return [meta for meta in named_parameters_shape if meta[0].startswith(f"model{stage}")]


def get_param_shape_in_groups(named_parameters_shape, params_per_dp):
    """ split named_parameters_shape by dp """
    result = [[[], []]]
    cnt = 0
    d = len(params_per_dp) - 1
    for meta in named_parameters_shape:
        name = meta[0]
        shape = meta[1]
        param_size = numpy.prod(shape)
        group_id = get_group_id(name, shape)
        while True:
            if cnt + param_size > params_per_dp[d]:
                result[-1][group_id].append(params_per_dp[d] - cnt)
                param_size -= params_per_dp[d] - cnt
                cnt = 0
                d -= 1
                result.append([[], []])
            elif cnt + param_size == params_per_dp[d]:
                result[-1][group_id].append(param_size)
                cnt = 0
                d -= 1
                result.append([[], []])
                break
            else:
                result[-1][group_id].append(param_size)
                cnt += param_size
                break
    assert cnt == 0 and d == -1
    return list(reversed(result[:-1]))


def merge_optimizer_by_dp(optimizers, named_parameters_shape):
    """ merging optimizers by dp """
    new_optimizer = next(opt for opt in optimizers if opt.get_param_groups_num() == 2).clone_without_param()
    opt_params_size = sum([opt.get_param_size() for opt in optimizers])
    model_params = sum([numpy.prod(shape) for name, shape, dim in named_parameters_shape])
    if opt_params_size == 0:
        return new_optimizer
    else:
        assert opt_params_size == model_params
    dp = len(optimizers)
    num_stages = len(set([meta[0].split('.')[0] for meta in named_parameters_shape]))
    cnt = [[0, 0] for _ in range(dp)]
    new_param_id = 0

    for group_id in range(2):
        for stage in range(num_stages):
            if num_stages > 1:
                _named_parameters_shape = get_named_parameters_shape_by_stage(named_parameters_shape, stage)
            else:
                _named_parameters_shape = named_parameters_shape
            total_params_size = sum([numpy.prod(shape) for name, shape, dim in _named_parameters_shape])
            params_per_dp = get_param_size_per_dp(total_params_size, dp)
            which_group_split = check_split_group(_named_parameters_shape, params_per_dp)
            assert len(which_group_split) == dp

            shape_in_group = get_param_shape_in_groups(_named_parameters_shape, params_per_dp)

            for d in range(dp - 1, -1, -1):
                param_groups = optimizers[d].get_param_ids_in_groups()
                if len(param_groups) <= group_id:
                    continue
                start = cnt[d][group_id]
                end = start + len(shape_in_group[d][group_id])
                for index, param_id in enumerate(param_groups[group_id][start:end]):
                    # state
                    state = optimizers[d].get_state(param_id)
                    assert list(state.values())[0].shape == shape_in_group[d][group_id][index]
                    if new_optimizer.has_param(new_param_id):
                        last_state = new_optimizer.get_state(new_param_id)
                        state = cat_optimer_param([state, last_state])
                        new_optimizer.update_state(state, group_id, new_param_id)
                    else:
                        new_optimizer.add_state_in_group(state, group_id)
                    # shard_fp32
                    shard_fp32 = optimizers[d].get_shard_fp32_by_param_id(param_id)
                    last_shard_fp32 = new_optimizer.get_shard_fp32_by_param_id(new_param_id)
                    if last_shard_fp32 is None:
                        new_optimizer.add_shard_fp32(shard_fp32, group_id)
                    else:
                        shard_fp32 = torch.cat([shard_fp32, last_shard_fp32], dim=0)
                        new_optimizer.update_shard_fp32(shard_fp32, new_param_id)
                    new_param_id += 1
                # If the param has been split, it need to be merged with next dp rank
                if group_id == which_group_split[d]:
                    new_param_id -= 1
                cnt[d][group_id] = end
    new_optimizer.build_param_map(named_parameters_shape)
    return new_optimizer


class MegatronOptimizer():
    """ Megatron optimizer """
    def __init__(self, num_layers, start_offset, end_offset, params_num_per_layer):
        """ 
            num_laerners (int): number of layers in pp rank
            start_offset (int): start offset of attension params
            end_offset (int): end offset of attension params
            params_num_per_layer (list): number of group params per layer
        """
        self.num_layers = num_layers
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.params_num_per_layer = params_num_per_layer
        self.state = {}
        self.param_groups = [{}, {}]
        self.grad_scaler = {}
        self.shard_fp32 = [[], []]
        self.param_scheduler = {}
        self.use_distributed_optimizer = False

    @ staticmethod
    def generate_optimizer(model, num_layers, pp_rank=None):
        """ geneate megatron optimizer """
        start_offset = 0
        end_offset = 0
        params_num_per_layer = [0, 0] # [group0, group1]
        for name, shape, _ in model.get_transformer_layers(pp_rank, 0):
            if name.endswith('weight'):
                if len(shape) == 1:
                    params_num_per_layer[1] += 1
                else:
                    params_num_per_layer[0] += 1
            elif name.endswith('bias'):
                params_num_per_layer[1] += 1
            else:
                pass
        if model.has_word_embeddings(pp_rank):
            start_offset += 1
        if model.has_position_embeddings(pp_rank):
            start_offset += 1
        if model.has_block_position_embeddings(pp_rank):
            start_offset += 1
        if model.has_final_layernorm("weight", pp_rank):
            end_offset += 1
        if model.has_final_layernorm("bias", pp_rank):
            end_offset += 1
        if model.has_word_embeddings_for_head(pp_rank):
            end_offset += 1
        opt = MegatronOptimizer(num_layers, start_offset, end_offset, params_num_per_layer)
        opt.use_distributed_optimizer = model.use_distributed_optimizer
        return opt
    
    def clone_without_param(self, num_layers=None, start_offset=None, end_offset=None):
        """ clone without params """
        result = MegatronOptimizer(self.num_layers, self.start_offset, self.end_offset, self.params_num_per_layer)
        result.use_distributed_optimizer = self.use_distributed_optimizer
        result.param_groups = deepcopy(self.param_groups)
        for group in result.param_groups:
            group["params"] = []
        result.shard_fp32 = [[] for group in self.shard_fp32]
        result.grad_scaler = deepcopy(self.grad_scaler)
        result.param_scheduler = deepcopy(self.param_scheduler)
        if num_layers is not None:
            result.num_layers = num_layers
        if start_offset is not None:
            result.start_offset = start_offset
        if end_offset is not None:
            result.end_offset = end_offset
        return result
    
    def add_word_embedding_for_head(self):
        """ input: [0,1,2,3] [4,5,6,7,8,9] """
        """ output: [0,1,2,3,4] [5,6,7,8,9,10] """
        len1 = len(self.param_groups[0]['params'])
        len2 = len(self.param_groups[1]['params'])
        self.param_groups[0]['params'] = list(range(len1 + 1))
        self.param_groups[1]['params'] = list(range(len1 + 1, len1 + len2 + 1))
        for i in range(len2 + len1, len1, -1):
            self.state[i] = self.state[i - 1]
        self.state[len1] = deepcopy(self.state[0])
        self.shard_fp32[0].append(self.shard_fp32[0][0].clone())

    def remove_word_embedding_for_head(self):
        """ input: [0,1,2,3,4] [5,6,7,8,9,10] """
        """ output: [0,1,2,3] [4,5,6,7,8,9] """
        len1 = len(self.param_groups[0]['params'])
        len2 = len(self.param_groups[1]['params'])
        self.param_groups[0]['params'] = list(range(len1 - 1))
        self.param_groups[1]['params'] = list(range(len1 - 1, len1 + len2 - 1))
        for i in range(len1 - 1, len1 + len2 - 1):
            self.state[i] = self.state[i + 1]
        del self.state[len1 + len2 - 1]
        self.shard_fp32[0].pop()

    def get_state_in_model_order(self):
        """ Optimizer parameters are grouped by shape and name, such as:
            group 0: emb_w、qkv_w、dense_w、h_to_4h_w、4h_to_h_w
            group 1: norm_w、norm_b、qkv_b、dense_b、norm_w、norm_b、h_to_4h_b、4h_to_h_b、final_norm_w、final_norm_b
    
            return list of (param_id, group_id, name, size) in the order of model structure
        """

        result = [None] * len(self.param_map)
        for param_id, meta in self.param_map.items():
            name, shape, _, index = meta
            group_id = get_group_id(name, shape)
            result[index] = (param_id, group_id, name, numpy.prod(shape))
        assert None not in result
        return result

    def chunk_by_dp(self, dp, num_stages):
        """ chuck by dp """
        if self.empty():
            return [deepcopy(self) for i in range(dp)]

        result = [self.clone_without_param() for i in range(dp)]

        if self.get_state_num() > 0:
            assert self.get_state_num() == self.get_param_num()
            for stage in range(num_stages):
                param_groups = [[], []]
                cnt = 0
                d = dp - 1
                model_prefix = f"model{stage}" if num_stages > 1 else "model"
                state = [meta for meta in self.get_state_in_model_order() if meta[2].startswith(model_prefix)]
                total_params_size = sum([meta[3] for meta in state])
                params_size_per_dp = get_param_size_per_dp(total_params_size, dp)
                for param_id, group_id, name, _ in state:
                    param = self.get_state(param_id)

                    while True:
                        param_size = numpy.prod(list(param.values())[0].shape)
                        if cnt + param_size > params_size_per_dp[d]:
                            offset = params_size_per_dp[d] - cnt
                            param, _ = split_optimer_param(param, -offset)
                            param_groups[group_id].append(_)
                            result[d].add_param_groups(param_groups)
                            param_groups = [[], []]
                            cnt = 0
                            d -= 1
                        elif cnt + param_size == params_size_per_dp[d]:
                            param_groups[group_id].append(param)
                            result[d].add_param_groups(param_groups)
                            param_groups = [[], []]
                            cnt = 0
                            d -= 1
                            break
                        else:
                            param_groups[group_id].append(param)
                            cnt += param_size
                            break
                assert d == -1 and cnt == 0, f"{d} {cnt}"

        if self.get_shard_fp32_num() > 0:
            assert self.get_shard_fp32_num() == self.get_param_num()
            assert self.get_shard_fp32_size() == self.get_param_size(), f"{self.get_shard_fp32_size()} {self.get_param_size()}"
            for stage in range(num_stages):
                cnt = 0
                d = dp - 1
                model_prefix = f"model{stage}" if num_stages > 1 else "model"
                state = [meta for meta in self.get_state_in_model_order() if meta[2].startswith(model_prefix)]
                total_params_size = sum([meta[3] for meta in state])
                params_size_per_dp = get_param_size_per_dp(total_params_size, dp)
                for param_id, group_id, name, _ in state:
                    shard_fp32 = self.get_shard_fp32_by_param_id(param_id)
                    while True:
                        shard_fp32_size = numpy.prod(shard_fp32.shape)
                        if cnt + shard_fp32_size > params_size_per_dp[d]:
                            offset = params_size_per_dp[d] - cnt
                            result[d].add_shard_fp32(shard_fp32[-offset:], group_id)
                            shard_fp32 = shard_fp32[:-offset]
                            cnt = 0
                            d -= 1
                        elif cnt + shard_fp32_size == params_size_per_dp[d]:
                            result[d].add_shard_fp32(shard_fp32, group_id)
                            cnt = 0
                            d -= 1
                            break
                        else:
                            result[d].add_shard_fp32(shard_fp32, group_id)
                            cnt += shard_fp32_size
                            break
                assert d == -1 and cnt == 0

            for opt in result:
                opt.update_param_groups_by_shard()

        return result
            
    def chunk_by_pp_tp(self, pp, tp, args):
        """ chunk by pp tp """
        self.debug(f"chunk by pp{pp} tp{tp}")
        num_layers_per_pp = self.num_layers // pp
        opts = []
        for p in range(pp):
            opts.append([])
            for t in range(tp):
                start_offset = self.start_offset if p == 0 else 0
                end_offset = self.end_offset if p == pp-1 else 0
                opt = self.clone_without_param(num_layers_per_pp, start_offset, end_offset)
                opts[p].append(opt)

        if self.get_state_num() > 0:
            assert self.get_state_num() == self.get_param_num()
            state_range = self.get_state_range(pp)
            print(f"chunk optimizer by pp{pp} tp{tp}, state range: {state_range}")
            for p in range(pp):
                for group_id, param_list in enumerate(state_range[p]):
                    for param_id in param_list:
                        chunk_param  = [{} for t in range(tp)]
                        for key, value in self.state[param_id].items():
                            name, shape, parallel_dim, _ = self.param_map[param_id]
                            if "dense_h_to_4h" in name and args.get("transpose_mlp_dense", False):
                                value = transpose_shape0(value.view(shape), 2, tp)
                                if self.use_distributed_optimizer:
                                    value = value.view(-1)
                            if parallel_dim < 0:
                                for t in range(tp):
                                    chunk_param[t][key] = value
                            else:
                                chunk_state = value.view(shape).chunk(tp, dim=parallel_dim)
                                del value
                                for t in range(tp):
                                    chunk_param[t][key] = chunk_state[t].clone()
                                    if self.use_distributed_optimizer:
                                        chunk_param[t][key] = chunk_param[t][key].reshape(-1)
                        for t in range(tp):
                            opts[p][t].add_state(chunk_param[t])
                    for t in range(tp):
                        assert opts[p][t].get_state_num() == opts[p][t].get_param_num()

        if self.get_shard_fp32_num() > 0:
            assert self.get_shard_fp32_num() == self.get_param_num()
            param_id = 0
            for group_id, shard in enumerate(self.shard_fp32):
                for index, data in enumerate(shard):
                    p = (index-self.start_offset) // (self.params_num_per_layer[0]*num_layers_per_pp) if group_id == 0 \
                            else index // (self.params_num_per_layer[1] * num_layers_per_pp)
                    p = max(p, 0)
                    p = min(p, pp-1)

                    name, shape, parallel_dim, _ = self.param_map[param_id]
                    if "dense_h_to_4h" in name and args.get("transpose_mlp_dense", False):
                        data = transpose_shape0(data.view(shape), 2, tp)
                        if self.use_distributed_optimizer:
                            data = data.view(-1)
                    if parallel_dim < 0:
                        for t in range(tp):
                            if len(opts[p][t].shard_fp32) <= group_id:
                                opts[p][t].shard_fp32.append([])
                            opts[p][t].shard_fp32[group_id].append(data)
                    else:
                        chunk_shard_fp32 = data.view(shape).chunk(tp, dim=parallel_dim)
                        del data
                        for t in range(tp):
                            if len(opts[p][t].shard_fp32) <= group_id:
                                opts[p][t].shard_fp32.append([])
                            data = chunk_shard_fp32[t].clone()
                            if self.use_distributed_optimizer:
                                data = data.reshape(-1)
                            opts[p][t].shard_fp32[group_id].append(data)
                    param_id += 1

            for p in range(pp):
                for t in range(tp):
                    opts[p][t].update_param_groups_by_shard()
        
        for p in range(pp):
            for t in range(tp):
                assert opts[p][t].get_param_size() == opts[p][t].get_shard_fp32_size()

        return opts

    def get_shard_fp32_by_param_id(self, param_id):
        """ get  shard fp32 by param id """
        shard_fp32 = sum(self.shard_fp32, [])
        return shard_fp32[param_id] if len(shard_fp32) > param_id else None

    def get_shard_fp32_in_group(self, group_id, index):
        """" get  shard fp32 by group id and index"""
        return self.shard_fp32[group_id][index]

    def update_shard_fp32(self, param, param_id):
        """" update shard fp32 """
        group_id = 0
        for group in self.shard_fp32:
            if param_id >= len(group):
                param_id -= len(group)
                group_id += 1
            else:
                break
        self.shard_fp32[group_id][param_id] = param
        
    def add_shard_fp32(self, param, group_id):
        """ add optimizer shard fp32 """
        self.shard_fp32[group_id].append(param.clone())

    def set_param_groups(self, param_groups):
        """ set param_groups """
        self.state = {}
        for idx, param_group in enumerate(param_groups):
            offset = len(self.state)
            self.param_groups[idx]["params"] = list(range(offset, offset + len(param_group)))
            for i, param in enumerate(param_group):
                self.state[offset+i] = param

    def get_param_groups(self):
        """ get param_groups """
        return [[self.state[param_id] for param_id in param_group.get("params", [])] \
                for param_group in self.param_groups]
                

    def add_param_groups(self, param_groups):
        """ add param_group """
        assert len(param_groups) <= len(self.param_groups)
        new_param_groups = [origin_param_group + param_groups[idx] \
                for idx, origin_param_group in enumerate(self.get_param_groups())]
        self.set_param_groups(new_param_groups)

    def update_param_groups_by_shard(self):
        """ update parms group by shard """
        offset = 0
        for group_id, group in enumerate(self.shard_fp32):
            self.param_groups[group_id]["params"] = list(range(offset, offset + len(group)))
            offset += len(group)
        
    def add_state(self, param):
        """ add optimizer param """
        param_id, group_id = self.new_param()
        self.param_groups[group_id]["params"].append(param_id)
        self.state[param_id] = param

    def add_state_in_group(self, param, group_id):
        """ add optimizer param with group id """
        param_id, _ = self.new_param()
        self.param_groups[group_id]["params"].append(param_id)
        self.state[param_id] = param

    def update_state(self, param, group_id, param_id):
        """ update optimizer param """
        assert param_id in self.state
        assert param_id in self.param_groups[group_id]["params"]
        self.state[param_id] = param

    def get_state_num(self):
        """ get optimizer state num """
        s = 0
        for param_id, param in self.state.items():
            if len(param) > 0:
                s += 1
        return s

    def get_state_size(self):
        """ sum all state' size """
        s = 0
        for param_id, param in self.state.items():
            for key, data in param.items():
                s += numpy.prod(data.shape)
                break
        return s

    def get_param_size(self):
        """ get  param size """
        if self.get_state_num() > 0:
            return self.get_state_size()
        elif self.get_shard_fp32_num() > 0:
            return self.get_shard_fp32_size()
        else:
            return 0

    def get_param_num(self):
        """ count all params """
        return sum([len(group.get("params", [])) \
                for group in  self.param_groups])

    def get_param_groups_num(self):
        """ get param groups num """
        return len(self.param_groups)

    def get_shard_fp32_num(self):
        return len(sum(self.shard_fp32, []))

    def get_shard_fp32_size(self):
        """ sum all shard_fp32' size """
        s = 0
        for group in self.shard_fp32:
            for data in group:
                s += numpy.prod(data.shape)
        return s

    def get_state(self, param_id, key = None):
        """ get state """
        if key == None:
            return self.state.get(param_id, {})
        else:
            return self.state[param_id][key]

    def get_param_ids_in_groups(self):
        """" get param ids in groups """
        return [param_group["params"] for param_group in self.param_groups]

    def get_keys(self):
        """ return keys of optimizer state """
        assert not self.empty()
        return list(self.state[0].keys())

    def get_state_range(self, pp):
        """ divide state among various pp rank, and return list of (start, end)
        """
        state_range = [[[], []] for i in range(pp)]
        step0 = self.num_layers // pp * self.params_num_per_layer[0]
        step1 = self.num_layers // pp * self.params_num_per_layer[1]
        param_groups = self.get_param_ids_in_groups()
        state_range[0][0] += range(self.start_offset)
        i = self.start_offset
        j = 0
        for p in range(pp-1):
            state_range[p][0] += param_groups[0][i: i + step0]
            state_range[p][1] += param_groups[1][j: j + step1]
            i += step0
            j += step1
        state_range[-1][0] += param_groups[0][i:]
        state_range[-1][1] += param_groups[1][j:]
        return state_range

    def need_reshape(self):
        """ whether need reshape """
        if self.empty():
            return False
        return self.use_distributed_optimizer ^ (self.get_shard_fp32_by_param_id(0).dim == 1)

    def build_param_map(self, named_parameters_shape):
        param_map = {}
        group0, group1 = [], []
        model_param_id = 0
        for name, shape, parallel_dim in named_parameters_shape:
            group_id = get_group_id(name, shape)
            (group0, group1)[group_id].append((
                name, 
                shape, 
                parallel_dim, 
                model_param_id))
            model_param_id += 1
        for meta in group0 + group1:
            param_id = len(param_map)
            param_map[param_id] = meta

        need_reshape = self.need_reshape()
        for param_id, meta in param_map.items():
            state = self.get_state(param_id)
            for k, param in state.items():
                assert numpy.prod(param.shape) == numpy.prod(meta[1]), f"{param_map}, {param.shape} {meta}, {param_id}"
                if need_reshape:
                    new_shape = -1 if self.use_distributed_optimizer else meta[1]
                    self.state[param_id][k] = param.view(new_shape)
                    shard_fp32 = self.get_shard_fp32_by_param_id(param_id)
                    self.update_shard_fp32(shard_fp32.view(new_shape), param_id)
        self.param_map = param_map

    def cut_embedding_padding(self, vocab_size):
        """ cut embedding padding """
        ids, _ = self.get_param_ids_in_groups()
        embedding_param_ids = []
        if self.start_offset > 0:
            embedding_param_ids.append(0)
        if self.end_offset > 0:
            embedding_param_ids.append(len(ids)-1)
        for i in embedding_param_ids:
            name, shape, parallel_dim, _ = self.param_map[i]
            for k, v in self.state[i].items():
                data = cut_embedding_padding(
                        v.view(shape), 
                        vocab_size
                )
                if self.use_distributed_optimizer:
                    data = data.view(-1)
                self.state[i][k] = data

        if self.get_shard_fp32_num() > 0:
            for i in embedding_param_ids:
                name, shape, parallel_dim, _ = self.param_map[i]
                data = cut_embedding_padding(
                        self.shard_fp32[0][i].view(shape), 
                        vocab_size
                )
                if self.use_distributed_optimizer:
                    data = data.view(-1)
                self.shard_fp32[0][i] = data

    def add_embedding_padding(self, divisible_by, vocab_size, tp, hidden_size, padded_vocab_size=None):
        """ add embedding padding """
        ids, _ = self.get_param_ids_in_groups()
        embedding_param_ids = []
        if self.start_offset > 0:
            embedding_param_ids.append(0)
        if self.end_offset > 0:
            embedding_param_ids.append(len(ids)-1)
        for i in embedding_param_ids:
            for k, v in self.state[i].items():
                data = add_embedding_padding(
                        v.view(-1, hidden_size), 
                        divisible_by, 
                        vocab_size, 
                        tp,
                        padded_vocab_size
                )
                if self.use_distributed_optimizer:
                    data = data.view(-1)
                self.state[i][k] = data
        if self.get_shard_fp32_num() > 0:
            for i in embedding_param_ids:
                data = add_embedding_padding(
                        self.shard_fp32[0][i].view(-1, hidden_size), 
                        divisible_by,
                        vocab_size, 
                        tp,
                        padded_vocab_size
                )
                if self.use_distributed_optimizer:
                    data = data.view(-1)
                self.shard_fp32[0][i] = data
    
    def has_param(self, param_id):
        """ check if optimizer has this param """
        for group in self.param_groups:
            if param_id in group["params"]:
                return True
        return False

    def new_param(self):
        """ generate new (param_id, group_id) """
        boundary = self.num_layers  * self.params_num_per_layer[0] + self.start_offset
        param_id = self.get_param_num()
        group_id = 0 if param_id < boundary else 1
        return param_id, group_id

    def empty(self):
        return self.get_state_num() == 0 and self.get_shard_fp32_num() == 0
    
    def to_dict(self):
        """ to dict """
        fp32_key = "shard_fp32_from_float16_groups" if self.use_distributed_optimizer \
                else "fp32_from_fp16_params"
        return { \
            "optimizer": { \
                "optimizer": { \
                    "state": self.state, \
                    "param_groups": self.param_groups \
                }, \
                "grad_scaler": self.grad_scaler, \
                fp32_key: self.shard_fp32 \
            }, \
            "opt_param_scheduler": self.param_scheduler \
        }
    def interleave(self, stage, pp):
        """ interleave """
        origin_params = list(range(self.get_param_num()))
        origin_transformer_params = origin_params[self.start_offset : -self.end_offset]
        new_transformer_params = transpose_shape0(torch.tensor(origin_transformer_params), stage, pp).tolist()
        new_params = origin_params[:self.start_offset] + new_transformer_params + origin_params[-self.end_offset:]

        _map = {} # model_param_index => optim_param_id
        for param_id, meta in self.param_map.items():
            model_param_index = meta[3]
            _map[model_param_index] = param_id

        new_state =  {}
        for old_param_id in range(self.get_param_num()):
            model_param_index = self.param_map[old_param_id][3]
            new_param_id = _map[new_params[model_param_index]]
            new_state[old_param_id] = self.get_state(new_param_id)
        self.state = new_state

        old_shard_fp32 = sum(self.shard_fp32, [])
        self.shard_fp32 = [[None] * len(group) for group in self.shard_fp32]
        for old_param_id in range(self.get_shard_fp32_num()):
            model_param_index = self.param_map[old_param_id][3]
            new_param_id = _map[new_params[model_param_index]]
            self.update_shard_fp32(old_shard_fp32[new_param_id], old_param_id)

    def load(self, state_dict, named_parameters_shape=None):
        """ load optimizer from dict """
        self.state = state_dict["optimizer"]["optimizer"]["state"]
        self.param_groups = state_dict["optimizer"]["optimizer"]["param_groups"]
        self.grad_scaler = state_dict["optimizer"].get("grad_scaler")
        for k in ["shard_fp32_from_float16_groups", "fp32_from_fp16_params"]:
            if k in state_dict["optimizer"]:
                self.shard_fp32 = state_dict["optimizer"][k]
                break
        self.param_scheduler = state_dict.get("opt_param_scheduler")

        if named_parameters_shape is not None and self.get_param_num() > 0:
            self.build_param_map(named_parameters_shape)

    def debug(self, msg):
        """ debug """
        print(f"\n【Optimizer】{msg}")
        print(f"-> layers: {self.num_layers}, params offset: [{self.start_offset}: -{self.end_offset}], params_per_layer: {self.params_num_per_layer}")
        print(f"-> param num: {self.get_param_num()}, param size: {self.get_param_size()}")
        print(f"-> state num: {self.get_state_num()}, state size: {self.get_state_size()}")
        print(f"-> shard_fp32 num: {self.get_shard_fp32_num()}, shard_fp32 size: {self.get_shard_fp32_size()}\n")
