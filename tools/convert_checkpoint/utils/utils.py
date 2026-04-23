# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""General utilities."""

import os
import re
import json
import torch
from typing import List
from bisect import bisect_left
from math import floor, ceil

from typing import Tuple, Literal
from typing import List


import logging

logging.basicConfig(level=logging.INFO)


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
    if weight is None:
        return None
    if padded_vocab_size is None:
        padded_vocab_size = vocab_size_with_padding(orig_vocab_size, divisible_by, tp)
        padding_size = padded_vocab_size - orig_vocab_size
    else:
        padding_size = padded_vocab_size - weight.shape[0]
    if orig_vocab_size > weight.shape[0]:
        padding_size += (orig_vocab_size - weight.shape[0])
    if padding_size < 0:
        return weight[0:padded_vocab_size, :]
    elif padding_size > 0:
        return torch.cat((weight, weight[-1].unsqueeze(0).expand(padding_size, -1)))
    else:
        return weight


def cut_embedding_padding(weight, orig_vocab_size):
    """ cut embedding padding """
    if weight is None:
        return None
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

def touch_file(done_dir, p, ep_id=None):
    if ep_id is None:
        fname = f'{p}.done'
    else:
        fname = f'{p}_{ep_id}.done'
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
    # Initialize global counter
    global_file_count = 0
    sum_sub_count = 0

    path = f'{base_path}/sub_checkpoint/'
    # Assume file list is known, use a list to simulate
    temp_paths = []
    for sub_dir_name in os.listdir(path):
        if sub_dir_name.isdigit():
            temp_paths.append(sub_dir_name)
    sorted_path_list = sorted(temp_paths, key=int)

    for index in sorted_path_list:
        if index.isdigit():  # Check if it is a numeric directory
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.startswith('model-') and filename.endswith('.safetensors'):
                        parts = filename.split('-of-')
                        if len(parts) == 2:
                            global_file_count += 1
    # Iterate through all subdirectories
    all_dict = {}
    for index in sorted_path_list:
        if index.isdigit():  # Check if it is a numeric directory
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                # Initialize subdirectory counter
                local_file_count = 0

                # Iterate through all files in subdirectory
                one_dict = {}
                logging.info(f"{subdir_path=}")
                for filename in os.listdir(subdir_path):
                    if filename.startswith('model-') and filename.endswith('.safetensors'):
                        # Parse filename, extract i and sub_count
                        parts = filename.split('-of-')
                        if len(parts) == 2:
                            file_base, file_count = parts
                            i_str = file_base.split('-')[-1]
                            sub_count_str = file_count.split('.')[0]
                            i = int(i_str)
                            sub_count = int(sub_count_str)

                            # Update global counter
                            local_file_count += 1

                            # Calculate new filename
                            new_i = sum_sub_count + i
                            new_filename = f'model-{new_i:05d}-of-{global_file_count:05d}.safetensors'

                            # Rename file
                            old_filepath = os.path.join(subdir_path, filename)
                            new_filepath = os.path.join(subdir_path, new_filename)
                            one_dict[filename] = new_filename
                all_dict[subdir_path] = one_dict

                # Update cumulative sub-file count
                sum_sub_count += local_file_count


    # Used to store merged metadata and weight_map
    merged_metadata = {"total_size": 0}
    merged_weight_map = {}

    # Iterate through file list, merge metadata and weight_map
    for index in sorted_path_list:
        if index.isdigit():  # Check if it is a numeric directory
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                file_name = f"{subdir_path}/model.safetensors.index.json"
                with open(file_name, 'r') as f:
                    file_content = json.load(f)
                # Merge metadata
                merged_metadata["total_size"] += file_content["metadata"]["total_size"]
                # Merge weight_map, and use one_dict for replacement
                subdir_path = os.path.join(path, index)
                one_dict = all_dict[subdir_path]
                for key, value in file_content["weight_map"].items():
#                    logging.info(f"{key=}, {value=}, {one_dict=}")
#                    logging.info(f"{one_dict[value]=}")
                    if value in one_dict:
                        # Replace with corresponding value in one_dict
                        merged_weight_map[key] = one_dict[value]
                    else:
                        # If no replacement item found, keep original value (can be adjusted as needed)
                        # Note: Keeping original value here may not have meaning, as typically we don't want to keep filename as weight name
                        # But for completeness of the example, I kept this line
                        # In actual application, you may want to throw an error or log a warning
                        merged_weight_map[key] = value  # This is usually not expected behavior, only for example

    # Build new dict
    new_dict = {
        "metadata": merged_metadata,
        "weight_map": merged_weight_map
    }

    # Write new dict back to model.safetensors.index.json file
    with open(f'{base_path}/model.safetensors.index.json', 'w') as f:
        json.dump(new_dict, f, indent=4)
    for index in sorted_path_list:
        if index.isdigit():  # Check if it is a numeric directory
            subdir_path = os.path.join(path, index)
            if os.path.isdir(subdir_path):
                one_dict = all_dict[subdir_path]
                for filename, new_filename in one_dict.items():
                    old_filepath = os.path.join(subdir_path, filename)
                    new_filepath = os.path.join(base_path, new_filename)
                    os.rename(old_filepath, new_filepath)
                    logging.info(f'Renamed: {old_filepath} -> {new_filepath}')

    logging.info(f"Merge and replace completed, new model.safetensors.index.json file generated. "
          f"{base_path}/model.safetensors.index.json")
    old_filepath = f"{base_path}/model-00001-of-00001.safetensors"
    if os.path.exists(old_filepath):
        new_filepath = f"{base_path}/model.safetensors"
        os.rename(old_filepath, new_filepath)
        os.remove(f"{base_path}/model.safetensors.index.json")
    import shutil
    shutil.rmtree(f'{base_path}/sub_checkpoint')

def get_num_layers_in_vp_map(stage, num_layers, pp,
                           mtp_num_layers=0,
                           custom_pipeline_layers=None,
                           num_layers_in_first_pipeline_stage=None,
                           num_layers_in_last_pipeline_stage=None):
    if custom_pipeline_layers is not None:
        assert num_layers_in_first_pipeline_stage is None and num_layers_in_last_pipeline_stage is None, \
            "custom_pipeline_layers need not num_layers_in_first_pipeline_stage or in_last_pipeline_stage"
        num_layers_in_vp, _ = custom_partition_imbalanced(num_layers, pp * stage, custom_pipeline_layers)
    elif num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
        num_layers_in_vp = uneven_vpp_partition(
            num_layers, pp, stage, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage)
    else:
        num_layers_in_vp, _ = partition_balanced(num_layers, pp * stage)
    num_layers_in_vp[-1] += mtp_num_layers
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
    cargs = c_config.get_args("common")  # Get model common config parameters

    # Get model layer count related parameters
    num_layers = cargs["num_layers"]  # Total number of model layers
    mtp_num_layers = args.mtp_num_layers if args.mtp_num_layers is not None else cargs.get("mtp_num_layers", 0)  # MTP additional layers, default 0
    num_layers_per_stage = args.num_layers_per_virtual_pipeline_stage
    # Calculate number of virtual pipeline stages
    if num_layers_per_stage:
        stage = num_layers // pp // num_layers_per_stage
    else:
        stage = args.num_virtual_stages_per_pipeline_rank or 1
    
    dualpipev = args.vpp_scheduler == 'dualpipev'  # Check whether to use dualpipev scheduler
    pp = args.pipeline_model_parallel_size  # Pipeline parallel size
    custom_pipeline_layers = args.custom_pipeline_layers  # Custom pipeline layer assignment
    num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers  # Number of layers in the first pipeline stage
    num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers  # Number of layers in the last pipeline stage

    # Get the layer count distribution across virtual pipeline stages
    num_layers_in_vp = get_num_layers_in_vp_map(
            stage, num_layers, pp, mtp_num_layers=mtp_num_layers,
            custom_pipeline_layers=custom_pipeline_layers,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage)

    layer_ids = []  # Store layer id list for the current pipeline rank
    # Iterate over all virtual pipeline stages
    for stage_index in range(stage):
        # Get the virtual partition and layer offset for the current stage
        virtual_p, layer_offset, = get_virtual_partition(dualpipev, stage_index, p, pp, num_layers_in_vp)
        # Iterate over all layers in the current virtual partition
        for layer_index in range(num_layers_in_vp[virtual_p]):
            layer_id = layer_index + layer_offset  # Compute global layer id
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


def get_ep_map(num_experts, ep):
    if num_experts is None or ep is None:
        return None, None, None
    experts_ids = [x for x in range(num_experts)]
    chunks = [experts_ids[x:x + num_experts // ep]
        for x in range(0, len(experts_ids), num_experts // ep)] # ep_id -> [expert_ids]

    expert_local_mapping = {}
    expert_ep_mapping = {}
    ep_expert_mapping = {}
    for ep_id, chunk in enumerate(chunks):
        ep_expert_mapping[ep_id] = chunk # ep_id -> [expert_ids]
        for idx, ele in enumerate(chunk):
            expert_local_mapping[ele] = idx # expert_id -> local_ep_id
            expert_ep_mapping[ele] = ep_id # expert_id -> ep_id
    logging.info(f"expert_local_mapping: {expert_local_mapping}")
    logging.info(f"expert_ep_mapping: {expert_ep_mapping}")
    logging.info(f"ep_expert_mapping: {ep_expert_mapping}")
    return expert_local_mapping, expert_ep_mapping, ep_expert_mapping

def get_etp_map(tp, ep, etp):
    if etp is None:
        return None, None
    assert tp % (etp * ep) == 0 or (etp * ep) % tp == 0, f"tp: {tp}, etp: {etp}, ep: {ep}, tp % (etp * ep) != 0"
    etp_to_tp_mapping = {}
    v_tp = tp
    if tp < etp * ep:
        v_tp = etp * ep
    tp_to_ep = {}
    for t in range (v_tp):
        etp_id = t % etp
        ep_id = (t // etp) % ep
        if ep_id not in etp_to_tp_mapping:
            etp_to_tp_mapping[ep_id] = {}
        etp_to_tp_mapping[ep_id][etp_id] = t % tp
        tp_to_ep[t] = ep_id if t not in tp_to_ep or ep_id < tp_to_ep[t] else tp_to_ep[t]

    logging.info(f"{etp_to_tp_mapping=}, {tp_to_ep=}")
    return etp_to_tp_mapping, tp_to_ep

def is_power_of_two(x):
    return bool(((x.view(torch.int32) & 0x007FFFFF) == 0).all())

def get_quantizer_with_weight_scale_inv(weight, weight_scale_inv, dtype, amax_epsilon=False):
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.constants import TE_DType
    assert weight.dtype in (torch.float8_e4m3fn, torch.uint8)
    q = Float8BlockQuantizer(fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                rowwise=True, columnwise=False,
                                amax_epsilon=amax_epsilon,
                                force_pow_2_scales=is_power_of_two(weight_scale_inv),
                                block_scaling_dim=2)
    qx = q.make_empty(weight.shape, dtype=dtype, device='cpu')
    qx._rowwise_data.copy_(weight.view(torch.uint8))
    qx._rowwise_scale_inv[:weight_scale_inv.size(0), :weight_scale_inv.size(1)].copy_(weight_scale_inv)
    return qx

def convert_fp8_to_bf16(fp8_blocks: torch.Tensor,
                        scales: torch.Tensor,
                        dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Dequantizes a tensor from FP8, assuming `fp8_blocks` has the original, unpadded shape.

    Args:
        fp8_blocks: The FP8 quantized tensor with its original shape.
        scales: The per-block scales used for quantization.
        dtype: The desired output data type (e.g., torch.bfloat16).

    Returns:
        The dequantized tensor with the same shape as `fp8_blocks`.
    """

    if fp8_blocks.dtype == torch.uint8:
        fp8_blocks = fp8_blocks.view(torch.float8_e4m3fn)

    # Get the original shape directly from the input tensor.
    m, n = fp8_blocks.shape

    # Calculate the padded dimensions required for 128x128 block operations.
    m_pad = ceil_div(m, 128) * 128
    n_pad = ceil_div(n, 128) * 128

    # Trim the scales tensor to match the padded dimensions of the input.
    scales = scales[:ceil_div(m, 128), :ceil_div(n, 128)].contiguous()

    # Create a padded version of the input tensor if its dimensions are not
    # a multiple of the block size.
    if m_pad == m and n_pad == n:
        fp8_padded = fp8_blocks
    else:
        fp8_padded = torch.zeros((m_pad, n_pad), dtype=fp8_blocks.dtype, device=fp8_blocks.device)
        fp8_padded[:m, :n] = fp8_blocks

    # Dequantize block by block.
    fp8_view = fp8_padded.view(-1, 128, n_pad // 128, 128)
    scale_expanded = scales.view(-1, 1, scales.size(1), 1)
    x_recon_view = fp8_view.to(scale_expanded.dtype) * scale_expanded

    # Reshape the dequantized blocks into a 2D tensor of the padded size,
    # then slice it back to the original dimensions.
    x_recon = x_recon_view.view(m_pad, n_pad)[:m, :n].contiguous()
    x_recon = x_recon.to(dtype)

    return x_recon

def convert_bf16_to_fp8(
    x: torch.Tensor,
    method: Literal["te", "pt"] = 'te',
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: 2d tensor, the model parameter what will be block-wise quantize to fp8.
        method: one of the ["te", "pt"], means using TransformerEngine or naive PyTorch
            to do the quantization respectively. Defaults to "te".
        fp8_dtype: the dtype of the output fp8 tensor. Defaults to torch.float8_e4m3fn.
        **kwargs: kwargs for quantization. Supported args:
            `amax_epsilon`: defaults to 0. Minimum value for amax (clamp floor).
            `force_pow_2_scales`: defaults to True. When True, uses power-of-2 scaling
                (matching DeepGEMM's get_e4m3_sf_and_sf_inv).
    Returns:
        x_scaled: 2d tensor, the quantized tensor.
        weight_scale_inv: 2d tensor, the scale_inv of the quantized tensor.
    """

    # Always do the quantization on device
    x = x.cuda()
    
    amax_epsilon = kwargs.get("amax_epsilon", 0.0)
    force_pow_2 = kwargs.get("force_pow_2_scales", True)

    if method == "pt":
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(min=amax_epsilon)
        if force_pow_2:
            # Power-of-2 scaling
            scaled = x_amax / 448.0
            exp = torch.ceil(torch.log2(scaled))
            scale = torch.pow(2.0, exp)
            scale_inv = torch.pow(2.0, -exp)
            x_scaled = (x_view * scale_inv).to(fp8_dtype)
        else:
            # Linear scaling
            scale = x_amax / 448.0
            scale_inv = 448.0 / x_amax
            x_scaled = (x_view * scale_inv).to(fp8_dtype)
        # scale returned is the scale factor (for dequantization: x = x_scaled * scale)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous().cpu(), \
            scale.view(x_view.size(0), x_view.size(2)).cpu()

    elif method == "te":
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
        from transformer_engine.pytorch.constants import TE_DType
        quantizer = Float8BlockQuantizer(
            fp8_dtype=TE_DType[fp8_dtype],
            rowwise=True,
            columnwise=False,
            amax_epsilon=amax_epsilon,
            force_pow_2_scales=force_pow_2,
            block_scaling_dim=2,
        )
        xq = quantizer(x)
        fp8_weight = xq._rowwise_data.view(fp8_dtype)
        fp8_scale_inv = xq._rowwise_scale_inv
        return fp8_weight.cpu(), fp8_scale_inv.cpu()
    else:
        raise ValueError(f"invalid quantization method: {method}")

def convert_layout_to_custom_pipeline_layers(layout_str: str) -> str:
    """
    Convert pipeline-model-parallel-layout to custom-pipeline-layers format.
    
    Each stage in the layout becomes one value in custom-pipeline-layers,
    counting only decoder layers (ignoring E, m, L).
    
    Args:
        layout_str: Pipeline layout string, e.g., "Et*5|t*8|t*6|t*8L"
    
    Returns:
        Comma-separated string of decoder layer counts per stage.
        Example: "Et*5|t*8|t*6|t*8L" -> "5,8,6,8"
    """
    # copy from megatron/core/transformer/pipeline_parallel_layer_layout.py
    def _parse_str_to_list(layout_str: str) -> List[List[str]]:
        """Parse a layout string to a list of lists.
        Example: "Ettt|(tt|)*29,m|L" will be parsed to
        [["E","t","t","t"]]+[["t","t"]]*29+[["m"],["L"]]"""

        layout_str = layout_str.replace(",", "")  # remove purely cosmetic commas

        # unroll multiplications in the expression
        patterns = [
            # unroll expression in parentheses ()*n. Examples:
            # xy(ab|cd|ef)*2,pq -> xyab|cd|efab|cd|efpq
            # (ab)*3 -> ababab
            # ab,(cd|)*2 -> abcd|cd|
            # (|ab)*2,cd -> |ab|abcd
            r'\(([^)]+)\)\*(\d+)',
            r'(.)\*(\d+)',  # unroll x*n to n xs
        ]
        for pattern in patterns:
            layout_str = re.sub(pattern, lambda x: x.group(1) * int(x.group(2)), layout_str)

        char2layer_type = {
            "E": "embedding",
            "L": "loss",
            "t": "decoder",  # t denotes "transformer"
            "m": "mtp",
        }

        # parse the layout string
        layout_list = []
        for stage in layout_str.split('|'):
            layout_list.append([])
            for layer_char in stage:
                assert layer_char in char2layer_type, (
                    f"Invalid layer character: {layer_char} ({stage=}, {layout_str=}),"
                    f" known layer characters: {list(char2layer_type.keys())}"
                )

                layout_list[-1].append(char2layer_type[layer_char])
        return layout_list
    
    # Parse layout string to list
    layout_list = _parse_str_to_list(layout_str)
    
    # Count decoder layers for each stage
    custom_pipeline_layers = []
    
    for stage in layout_list:
        # Count decoder layers (LayerType.decoder), ignoring E, m, L
        decoder_count = sum(1 for layer in stage if layer == "decoder")
        custom_pipeline_layers.append(decoder_count)
    
    # Convert to comma-separated string
    return ','.join(map(str, custom_pipeline_layers))
