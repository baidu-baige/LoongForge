#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""General utilities."""

import os
import json
import torch
from bisect import bisect_left
from math import floor, ceil


LOADED_STATE_DICT = None


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def check_path_in_dict(d, path):
    """
    check path exists in dictionary
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            return False
        d = d[k]
    return True


def vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by, tp):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tp
    while (after % multiple) != 0:
        after += 1
    return after


def add_embedding_padding(weight, divisible_by, orig_vocab_size, tp, padded_vocab_size=None):
    """ add embedding padding """
    if padded_vocab_size is None:
        padded_vocab_size = vocab_size_with_padding(orig_vocab_size, divisible_by, tp)
        padding_size = padded_vocab_size - orig_vocab_size
    else:
        padding_size = padded_vocab_size - weight.shape[0]
    if padding_size < 0:
        return weight[0:padded_vocab_size, :]
    elif padding_size > 0:
        return torch.cat((weight, weight[-1].unsqueeze(0).expand(padding_size, -1)))
    else:
        return weight


def cut_embedding_padding(weight, orig_vocab_size):
    """ cut embedding padding """
    #TODO
    return weight[0:orig_vocab_size, :]


def transpose_shape0(param, m, n):
    """ transpose on shape 0 """
    _shape = param.size()
    current_shape = (m, n, _shape[0] // (m * n)) + _shape[1:]
    return param.view(*current_shape) \
            .transpose(0, 1).contiguous() \
            .view(*_shape)

def uneven_vpp_partition(num_layers, pp, vp, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage):
    assert num_layers is not None and num_layers > 0, "num_layers must be provided."
    assert pp is not None and pp > 1, "pipeline model parallel size must be greater than 1."
    assert vp is not None and vp == 2, "virtual pipeline must be 2."
    assert num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None, \
        "num_layers_in_first_pipeline_stage or num_layers_in_last_pipeline_stage must be provided."
    # Number of layers to distribute over rest of pipeline stages
    layers_to_distribute = num_layers
    # Number of pipeline stages left for distributing transformer layers
    pipeline_stages_left = pp
    parts_count = [0 for _ in range(pp*vp)]
    # If the uneven first (last) pipeline stage is enabled, remove the specified number
    # of layers to calculate the number of layers on each middle pipeline stage.
    if num_layers_in_first_pipeline_stage is not None:
        layers_to_distribute -= num_layers_in_first_pipeline_stage
        parts_count[0] = ceil(num_layers_in_first_pipeline_stage / vp)
        parts_count[pp * vp - 1] = num_layers_in_first_pipeline_stage - parts_count[0]
        pipeline_stages_left -= 1

    if num_layers_in_last_pipeline_stage is not None:
        layers_to_distribute -= num_layers_in_last_pipeline_stage
        parts_count[pp-1] = ceil(num_layers_in_last_pipeline_stage / vp)
        parts_count[pp] = num_layers_in_last_pipeline_stage - parts_count[pp-1]
        pipeline_stages_left -= 1
    num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
    for i in range(1, pp-1):
        parts_count[i] = num_layers_per_pipeline_rank // vp
        parts_count[2 * pp - 1 - i] = num_layers_per_pipeline_rank - parts_count[i]
    if num_layers_in_first_pipeline_stage is None:
        parts_count[0] = num_layers_per_pipeline_rank // vp
        parts_count[2 * pp - 1] = num_layers_per_pipeline_rank - parts_count[0]
    if num_layers_in_last_pipeline_stage is None:
        parts_count[pp-1] = num_layers_per_pipeline_rank // vp
        parts_count[pp] = num_layers_per_pipeline_rank - parts_count[pp-1]
    return parts_count


def custom_partition_imbalanced(num_layers, num_parts, custom_layers):
    """
    custom partition imbalanced.
    first and last stages contain less layers,
    other stages contain more layers
    """
    splits = []
    if custom_layers.find(',') != -1:
        splits = [int(s) for s in custom_layers.split(',')]
    if len(splits) != num_parts:
        raise ValueError(
            f'the argments of custom_pipeline_layers must be equal to pipeline size {num_parts}.'
        )
    assert num_layers == sum(splits), f'the sum of custom_pipeline_layers must be equal to num_layers {num_layers}.'
    # First check for the trivial edge case
    if num_layers <= num_parts:
        parts = partition_uniform(num_layers, num_parts)
    else:
        parts = [0] * (num_parts + 1)
        parts_count = [num_layers // num_parts] * num_parts
        for i in range(num_parts):
            parts_count[i] = splits[i]
        for i in range(1, len(parts_count) + 1):
            parts[i] = parts[i - 1] + parts_count[i - 1]

    return parts_count, parts


def partition_balanced(num_layers, num_parts, eps=1e-3):
    """
    partition balanced.
    """
    # First check for the trivial edge case
    if num_layers <= num_parts:
        parts = partition_uniform(num_layers, num_parts)
    else:
        weights = [1] * num_layers
        weights_ = prefix_sum_inc(weights)

        # Find the smallest bottleneck (weight of heaviest partition)
        bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

        # Now compute that partitioning
        parts, success = _lprobe(weights_, num_parts, bottleneck)
        assert success

    parts_count = [0] * num_parts

    for i in range(1, len(parts)):
        parts_count[i - 1] = parts[i] - parts[i - 1]

    return parts_count, parts


def partition_uniform(num_items, num_parts):
    """
    partition uniform.
    """
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
    return parts


def prefix_sum_inc(weights):
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_


def _rb_partition_balanced(weights, num_parts, eps):
    total_weight = weights[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def _lprobe(weights, num_parts, bottleneck):
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(weights, bsum, lo=step - chunksize, hi=min(step, num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight

def touch_file(done_dir, p_id, ep_id=None):
    if ep_id is None:
        fname = f'{p_id}.done'
    else:
        fname = f'{p_id}_{ep_id}.done'
    done_file_name = os.path.join(done_dir, fname)
    with open(done_file_name, 'w'):
        os.utime(done_file_name, None)

def check_all_done(done_dir, p, ep):
    fnames = []
    if ep is None:
        for p_id in range(p):
            fname = f'{p_id}.done'
            fnames.append(fname)
    else:
        for p_id in range(p):
            for ep_id in range(ep):
                fname = f'{p_id}_{ep_id}.done'
                fnames.append(fname)
    all_done = True
    for fname in fnames:
        done_file_name = os.path.join(done_dir, fname)
        if not os.path.exists(done_file_name):
            all_done = False
            break
    return all_done

def get_done_keys(done_dir, p, cur_ep_ids=None):
    done_keys = []
    if cur_ep_ids is None:
        fname = f'{p}.done'
        if os.path.exists(os.path.join(done_dir, fname)):
            done_keys.append((p, None))
    else:
        for ep_id in cur_ep_ids:
            fname = f'{p}_{ep_id}.done'
            if os.path.exists(os.path.join(done_dir, fname)):
                done_keys.append((p, ep_id))
    return done_keys


def make_hf_sub_checkpoints(base_path):
    # 初始化全局计数器
    global_file_count = 0
    sum_sub_count = 0

    path = f'{base_path}/sub_checkpoint/'
    # 假设文件列表是已知的，这里用一个列表模拟
    temp_paths = []
    for sub_dir_name in os.listdir(path):
        if sub_dir_name.isdigit():
            temp_paths.append(sub_dir_name)
    sorted_path_list = sorted(temp_paths, key=int)

    for index in sorted_path_list:
        if index.isdigit():  # 检查是否为数字目录
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.startswith('model-') and filename.endswith('.safetensors'):
                        parts = filename.split('-of-')
                        if len(parts) == 2:
                            global_file_count += 1
    # 遍历所有子目录
    all_dict = {}
    for index in sorted_path_list:
        if index.isdigit():  # 检查是否为数字目录
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                # 初始化子目录计数器
                local_file_count = 0

                # 遍历子目录中的所有文件
                one_dict = {}
                print(f"{subdir_path=}")
                for filename in os.listdir(subdir_path):
                    if filename.startswith('model-') and filename.endswith('.safetensors'):
                        # 解析文件名，提取 i 和 sub_count
                        parts = filename.split('-of-')
                        if len(parts) == 2:
                            file_base, file_count = parts
                            i_str = file_base.split('-')[-1]
                            sub_count_str = file_count.split('.')[0]
                            i = int(i_str)
                            sub_count = int(sub_count_str)

                            # 更新全局计数器
                            local_file_count += 1

                            # 计算新的文件名
                            new_i = sum_sub_count + i
                            new_filename = f'model-{new_i:05d}-of-{global_file_count:05d}.safetensors'

                            # 重命名文件
                            old_filepath = os.path.join(subdir_path, filename)
                            new_filepath = os.path.join(subdir_path, new_filename)
                            one_dict[filename] = new_filename
                all_dict[subdir_path] = one_dict

                # 更新累计子文件个数
                sum_sub_count += local_file_count


    # 用于存储合并后的metadata和weight_map
    merged_metadata = {"total_size": 0}
    merged_weight_map = {}

    # 遍历文件列表，合并metadata和weight_map
    for index in sorted_path_list:
        if index.isdigit():  # 检查是否为数字目录
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                file_name = f"{subdir_path}/model.safetensors.index.json"
                with open(file_name, 'r') as f:
                    file_content = json.load(f)
                # 合并metadata
                merged_metadata["total_size"] += file_content["metadata"]["total_size"]
                # 合并weight_map，并使用one_dict进行替换
                subdir_path = os.path.join(path, index)
                one_dict = all_dict[subdir_path]
                for key, value in file_content["weight_map"].items():
#                    print(f"{key=}, {value=}, {one_dict=}")
#                    print(f"{one_dict[value]=}")
                    if value in one_dict:
                        # 替换成one_dict中对应的值
                        merged_weight_map[key] = one_dict[value]
                    else:
                        # 如果没有找到替换项，则保留原值（这里可以根据需求调整）
                        # 注意：这里保留原值可能没有意义，因为通常我们不会想要保留文件名作为权重名
                        # 但为了保持示例的完整性，我保留了这一行
                        # 在实际应用中，你可能想要抛出一个错误或者记录一个警告
                        merged_weight_map[key] = value  # 这通常不是期望的行为，仅用于示例

    # 构建新的dict
    new_dict = {
        "metadata": merged_metadata,
        "weight_map": merged_weight_map
    }

    # 将新的dict写回到model.safetensors.index.json文件中
    with open(f'{base_path}/model.safetensors.index.json', 'w') as f:
        json.dump(new_dict, f, indent=4)
    for index in sorted_path_list:
        if index.isdigit():  # 检查是否为数字目录
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                one_dict = all_dict[subdir_path]
                for filename, new_filename in one_dict.items():
                    old_filepath = os.path.join(subdir_path, filename)
                    new_filepath = os.path.join(base_path, new_filename)
                    os.rename(old_filepath, new_filepath)
                    print(f'Renamed: {old_filepath} -> {new_filepath}')

    print(f"合并和替换完成，新的model.safetensors.index.json文件已生成。"
          f"{base_path}/model.safetensors.index.json")
    import shutil
    shutil.rmtree(f'{base_path}/sub_checkpoint')

def get_num_layers_in_vp_map(stage, num_layers, pp,
                           num_nextn_predict_layers=0,
                           custom_pipeline_layers=None,
                           num_layers_in_first_pipeline_stage=None,
                           num_layers_in_last_pipeline_stage=None):
    ori_num_layers = num_layers - num_nextn_predict_layers
    if custom_pipeline_layers is not None:
        assert num_layers_in_first_pipeline_stage is None and num_layers_in_last_pipeline_stage is None, \
            "custom_pipeline_layers need not num_layers_in_first_pipeline_stage or in_last_pipeline_stage"
        num_layers_in_vp, _ = custom_partition_imbalanced(ori_num_layers, pp * stage, custom_pipeline_layers)
        num_layers_in_vp[-1] += num_nextn_predict_layers
    elif num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
        num_layers_in_vp = uneven_vpp_partition(
            num_layers, pp, stage, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage)
        num_layers_in_vp[-1] += num_nextn_predict_layers
    else:
        num_layers_in_vp, _ = partition_balanced(num_layers, pp * stage)
    return num_layers_in_vp

def get_virtual_partition(dualpipev, stage_index, p, pp, num_layers_in_vp):
    if dualpipev:
        if stage_index == 0:
            virtual_p = p
        else:
            virtual_p = pp * stage_index + (pp - 1 - p)
    else:
        virtual_p = p + pp * stage_index
    layer_offset = sum(num_layers_in_vp[:virtual_p])
    return virtual_p, layer_offset

def get_layer_ids(c_config, args, p):
    cargs = c_config.get_args("common")
    hargs = c_config.get_args("huggingface")
    margs = c_config.get_args("mcore")

    ori_num_layers = cargs["num_layers"]
    num_nextn_predict_layers = hargs.get("num_nextn_predict_layers", 0)
    num_layers = ori_num_layers + num_nextn_predict_layers
    num_layers_per_stage = margs["num_layers_per_virtual_pipeline_stage"]
    if num_layers_per_stage:
        stage = num_layers // pp // num_layers_per_stage
    else:
        stage = args.num_virtual_stages_per_pipeline_rank or 1
    dualpipev = args.vpp_scheduler == 'dualpipev'
    pp = args.pipeline_model_parallel_size
    custom_pipeline_layers = args.custom_pipeline_layers
    num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
    num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
    num_layers_in_vp = get_num_layers_in_vp_map(
            stage, num_layers, pp, num_nextn_predict_layers=num_nextn_predict_layers,
            custom_pipeline_layers=custom_pipeline_layers,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage)

    layer_ids = []
    for stage_index in range(stage):
        virtual_p, layer_offset, = get_virtual_partition(dualpipev, stage_index, p, pp, num_layers_in_vp)
        for layer_index in range(num_layers_in_vp[virtual_p]):
            layer_id = layer_index + layer_offset
            layer_ids.append(layer_id)
    return layer_ids

def get_pipeline_by_rank_id(rank_id, world_size, pp, ep=None):
    p_dict = {}
    if pp >= world_size or ep is None:
        assert pp % world_size == 0, f"pp must be divisible by world_size: {pp=}, {world_size=}"
        p_count_per_rank = pp // world_size
        for p in range(pp):
            if p // p_count_per_rank == rank_id:
                if ep is None:
                    p_dict[p] = None
                else:
                    p_dict[p] = []
                    for e in range(ep):
                        p_dict[p].append(e)
    else:
        assert pp * ep % world_size == 0, f"pp * ep = {pp * ep}, world_size = {world_size}"
        rank_count_per_pp = world_size // pp
        p = rank_id // rank_count_per_pp
        p_dict[p] = []
        ep_count_per_rank = ep // rank_count_per_pp
        for e in range(ep):
            if e // ep_count_per_rank == rank_id % rank_count_per_pp:
                p_dict[p].append(e)
    return p_dict


def get_ep_map(num_experts, ep, num_experts_for_test=None):
    if num_experts_for_test is None:
        experts_ids = [x for x in range(num_experts)]
        chunks = [experts_ids[x:x + num_experts // ep]
            for x in range(0, len(experts_ids), num_experts // ep)] # ep_id -> [expert_ids]
    else:
        experts_ids = [x for x in range(num_experts_for_test)]
        chunks = [experts_ids[x:x + num_experts_for_test // ep]
            for x in range(0, len(experts_ids), num_experts_for_test // ep)] # ep_id -> [expert_ids]

    expert_local_mapping = {}
    expert_ep_mapping = {}
    ep_expert_mapping = {}
    for ep_id, chunk in enumerate(chunks):
        ep_expert_mapping[ep_id] = chunk # ep_id -> [expert_ids]
        for idx, ele in enumerate(chunk):
            expert_local_mapping[ele] = idx # expert_id -> local_ep_id
            expert_ep_mapping[ele] = ep_id # expert_id -> ep_id
    print(f"expert_local_mapping: {expert_local_mapping}")
    print(f"expert_ep_mapping: {expert_ep_mapping}")
    print(f"ep_expert_mapping: {ep_expert_mapping}")
    return expert_local_mapping, expert_ep_mapping, ep_expert_mapping