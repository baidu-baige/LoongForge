# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
HF Checkpoint Online Saving for Training
Implements online saving of model to HF checkpoint format based on tools/dist_checkpoint modules
"""
import os
import sys
from typing import Optional
import shutil

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.distributed as dist
from megatron.training import print_rank_0

# Import existing dist_checkpoint modules
from tools.dist_checkpoint.core.parser import Parser
from tools.dist_checkpoint.core.topo_sharder import TopoSharder
from tools.dist_checkpoint.core.tp_gather import TPGather
from tools.dist_checkpoint.checkpoint.hf_checkpoint_converter import HfCheckpointConverter
from tools.dist_checkpoint.utils import time_checkpoint_operation
# Import the utility function for merging checkpoints
from tools.convert_checkpoint.utils.utils import make_hf_sub_checkpoints, get_etp_map, check_all_done
from tools.convert_checkpoint.utils.config_utils import get_yaml_config


def _consolidate_pp_checkpoints(save_hf_path: str, pp_size: int, original_hf_path: Optional[str] = None) -> None:
    """
    Consolidate checkpoint files from per-PP-rank directories to final checkpoint.

    This function is called by global rank 0 after all TP rank 0's have saved their checkpoints
    to save_hf_path/sub_checkpoint/{pp_rank}/ subdirectories. It:
    1. Uses make_hf_sub_checkpoints to merge and rename safetensors files
    2. Copies config/tokenizer files from original HF checkpoint

    Args:
        save_hf_path: Base checkpoint directory
        pp_size: Number of pipeline stages (for logging, not strictly used by make_hf_sub_checkpoints)
        original_hf_path: Path to original HF checkpoint to copy config/tokenizer files from
    """
    print_rank_0(f"Starting checkpoint consolidation from {pp_size} PP stages...")

    if not os.path.exists(save_hf_path):
        print_rank_0(f"Error: save_hf_path {save_hf_path} does not exist")
        return

    sub_checkpoint_base = os.path.join(save_hf_path, "sub_checkpoint")
    if not os.path.exists(sub_checkpoint_base):
        print_rank_0(f"Error: sub_checkpoint directory {sub_checkpoint_base} does not exist")
        return

    # Step 1: Use make_hf_sub_checkpoints to merge and rename files
    print_rank_0("Merging and renaming checkpoint files from all PP ranks...")
    try:
        make_hf_sub_checkpoints(save_hf_path)
        print_rank_0("Checkpoint files merged and consolidated successfully")
    except Exception as e:
        print_rank_0(f"Error during checkpoint consolidation: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 2: Copy non-weight HF checkpoint files from original HF checkpoint
    # Copy all files except safetensors shards and index files
    if original_hf_path and os.path.exists(original_hf_path):
        files_copied = 0
        for filename in os.listdir(original_hf_path):
            src_file = os.path.join(original_hf_path, filename)
            dst_file = os.path.join(save_hf_path, filename)

            # Skip directories and already existing files
            if os.path.isdir(src_file):
                continue
            if os.path.exists(dst_file):
                continue
            # Skip safetensors files (we already have consolidated versions)
            if filename.endswith('.safetensors'):
                continue
            # Skip index files (we already created merged index)
            if filename.endswith('.index.json'):
                continue

            try:
                shutil.copy2(src_file, dst_file)
                print_rank_0(f"Copied {filename}")
                files_copied += 1
            except Exception as e:
                print_rank_0(f"Warning: Failed to copy {filename}: {e}")

        if files_copied > 0:
            print_rank_0(f"Copied {files_copied} additional file(s)")
    else:
        if original_hf_path:
            print_rank_0(f"Warning: original_hf_path {original_hf_path} does not exist, skipping file copy")
        else:
            print_rank_0("Note: original_hf_path not provided, skipping additional file copy")

    # Step 3: Clean up dones/ subdirectory
    dones_dir = os.path.join(save_hf_path, "dones")
    if os.path.exists(dones_dir):
        try:
            shutil.rmtree(dones_dir)
            print_rank_0("Removed dones/ directory")
        except Exception as e:
            print_rank_0(f"Warning: Failed to remove dones/ directory: {e}")

    print_rank_0("Checkpoint consolidation completed!")


@time_checkpoint_operation
def save_hf_checkpoint_online(
    model,
    args,
    iters=None,
) -> None:
    """
    Save model to HF checkpoint online with distributed gathering

    Uses tools/dist_checkpoint modules:
    1. Parser to parse config
    2. TopoSharder to initialize parallel topology
    3. TPGather to gather TP shards to TP rank 0 (NCCL backend with CPU offload)
    4. HfCheckpointConverter to convert and save to pp{pp_rank}/ subdirectory
    5. Global rank 0 consolidates all PP checkpoints to final location

    Args:
        model: Megatron distributed model
        args: Command line arguments with save_hf_path, yaml_file
        iters: Iteration (optional)

    Returns:
        None
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tp_size: int = args.tensor_model_parallel_size
    pp_size: int = args.pipeline_model_parallel_size
    ep_size = getattr(args, 'expert_model_parallel_size', None)
    etp_size = getattr(args, 'expert_tensor_parallel_size', None)

    # Set GPU device for NCCL backend
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)

    if iters is None:
        save_hf_path = args.save_hf_path
    else:
        save_hf_path = f"{args.save_hf_path}/iter_{iters}"
    print_rank_0("="*80)
    print_rank_0("Saving HF checkpoint with online gathering")
    print_rank_0("="*80)
    print_rank_0(f"Save path: {save_hf_path}")
    print_rank_0(f"World size: {world_size}")
    if ep_size is not None and ep_size > 1:
        print_rank_0(f"Parallel config: TP={tp_size}, PP={pp_size}, EP={ep_size}, ETP={etp_size}")
    else:
        print_rank_0(f"Parallel config: TP={tp_size}, PP={pp_size}")

    # Step 1: Parse args to get config
    print_rank_0("Parsing config from args")
    parser = Parser(args)

    # Get parallel config
    parallel_config = parser.get_parallel_config()

    print_rank_0(f"Model type: {parser.type}")

    # Step 2: Initialize TopoSharder (parallel_state already initialized by training)
    # Note: parallel_state is already set up by training initialization
    print_rank_0("Initializing TopoSharder...")
    topo_sharder = TopoSharder(parallel_config)
    tp_rank, pp_rank, ep_rank, etp_rank, dp_rank = topo_sharder.get_current_rank_coordinates()

    # Step 3: Extract model state_dict
    print_rank_0("Extracting model state_dict...")
    from megatron.training.utils import unwrap_model

    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

    unwrapped_model = unwrap_model(model)
    num_vpp_stages = len(unwrapped_model)

    # Extract state_dict(s): non-VPP uses single dict, VPP uses list of dicts
    if num_vpp_stages == 1:
        model_state_dict = unwrapped_model[0].state_dict()
    else:
        model_state_dict = [m.state_dict() for m in unwrapped_model]

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        print_rank_0(f"State_dict extracted. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")

    c_config = get_yaml_config(parser.config_file, parser.convert_file, for_vlm=(parser.vision_patch_convert_file is not None))
    c_vision_patch_config = get_yaml_config(
        parser.config_file, parser.vision_patch_convert_file,
        adapter_convert_file=parser.adapter_convert_file) if parser.vision_patch_convert_file is not None else None

    # Step 4: Initialize TPGather
    print_rank_0("Initializing TPGather...")
    tp_gather = TPGather(topo_sharder, c_config, args, c_vision_patch_config=c_vision_patch_config)

    # Step 5: Gather state_dicts within TP group to TP rank 0
    print_rank_0("Gathering state_dicts within TP group...")
    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

    if num_vpp_stages == 1:
        # Non-VPP: single gather
        gathered_state_dicts = tp_gather.gather_state_dicts(model_state_dict)
        all_ckpt_empty = all(len(d) == 0 for d in gathered_state_dicts)
    else:
        # VPP: gather each stage separately
        gathered_state_dicts = []
        for vpp_idx, sd in enumerate(model_state_dict):
            print_rank_0(f"Gathering state_dict for VPP stage {vpp_idx}...")
            gathered_state_dicts.append(tp_gather.gather_state_dicts(sd))
        all_ckpt_empty = all(len(inner_dict) == 0 for outer_list in gathered_state_dicts for inner_dict in outer_list)

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        print_rank_0(f"State_dicts gathered within TP group. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")

    if all_ckpt_empty:
        dist.barrier()
        return
    else:
        # Prepare mcore_dict from gathered state_dicts
        # Non-VPP: gathered_state_dicts is a list of state_dicts (one per TP rank)
        # VPP: gathered_state_dicts is a list of lists (outer: VPP stage, inner: TP rank)
        # Convert to format: {pp_rank: {tp_rank: {"model": state_dict, "checkpoint_version": 3.0}}}
        # For VPP: {pp_rank: {tp_rank: {"model0": ..., "model1": ..., "checkpoint_version": 3.0}}}
        print_rank_0("Preparing mcore_dict...")
        mem_before = 0.0
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()

        if num_vpp_stages == 1:
            # Non-VPP: gathered_state_dicts is List[state_dict]
            assert gathered_state_dicts is not None and isinstance(gathered_state_dicts, list), \
                "gathered_state_dicts should be a list for tp_rank == 0"
            assert len(gathered_state_dicts) > 0, "gathered_state_dicts should not be empty"

            # For dense models: {pp_rank: {tp_idx: {"model": state_dict, ...}}}
            # For MoE models:   {pp_rank: {ep_rank: {tp_idx: {"model": state_dict, ...}}}}
            if ep_rank is None:
                # Dense model: flat tp_shards under pp_rank
                tp_shards = {
                    tp_idx: {
                        "model": state_dict,
                        "checkpoint_version": 3.0,
                    }
                    for tp_idx, state_dict in enumerate(gathered_state_dicts)
                }
                mcore_dict = {pp_rank: tp_shards}
            else:
                # MoE model: need to organize by EP rank
                if etp_rank is not None and tp_size is not None and ep_size is not None \
                        and etp_size is not None and etp_size > 0:
                    # ETP enabled: use tp_to_ep mapping
                    _, tp_to_ep = get_etp_map(
                        tp_size,
                        ep_size,
                        etp_size
                    )
                    # Group tp_shards by their corresponding EP rank
                    ep_shards = {}
                    ep_ids = []
                    for tp_idx, state_dict in enumerate(gathered_state_dicts):
                        ep_id = ((ep_rank * etp_size) // tp_size * tp_size // etp_size) + tp_to_ep[tp_idx]
                        if ep_id not in ep_ids:
                            ep_ids.append(ep_id)
                        if ep_id not in ep_shards:
                            ep_shards[ep_id] = {}
                        ep_shards[ep_id][tp_idx] = {
                            "model": state_dict,
                            "checkpoint_version": 3.0,
                        }
                    mcore_dict = {pp_rank: ep_shards}
                else:
                    # ETP disabled: all tp_shards under current ep_rank
                    tp_shards = {
                        tp_idx: {
                            "model": state_dict,
                            "checkpoint_version": 3.0,
                        }
                        for tp_idx, state_dict in enumerate(gathered_state_dicts)
                    }
                    mcore_dict = {pp_rank: {ep_rank: tp_shards}}
        else:
            # VPP: gathered_state_dicts is List[List[state_dict]]
            assert gathered_state_dicts is not None and isinstance(gathered_state_dicts, list), \
                "gathered_state_dicts should be a list for tp_rank == 0"
            assert len(gathered_state_dicts) == num_vpp_stages, \
                f"gathered_state_dicts should have {num_vpp_stages} VPP stages"
            assert all(isinstance(g, list) and len(g) > 0 for g in gathered_state_dicts), \
                "Each VPP stage should contain a non-empty list of state_dicts"

            # Transpose: from [vpp_stage][tp_rank] to [tp_rank][vpp_stage]
            num_tp = len(gathered_state_dicts[0])
            transposed = [[gathered_state_dicts[v][t] for v in range(num_vpp_stages)] for t in range(num_tp)]

            # For dense models: {pp_rank: {tp_idx: {"model0": ..., "model1": ..., "checkpoint_version": 3.0}}}
            # For MoE models:   {pp_rank: {ep_rank: {tp_idx: {"model0": ..., ...}}}}
            if ep_rank is None:
                # Dense model: flat tp_shards under pp_rank
                tp_shards = {}
                for tp_idx, vpp_state_dicts in enumerate(transposed):
                    shard = {"checkpoint_version": 3.0}
                    for vpp_idx, state_dict in enumerate(vpp_state_dicts):
                        shard[f"model{vpp_idx}"] = state_dict
                    tp_shards[tp_idx] = shard
                mcore_dict = {pp_rank: tp_shards}
            else:
                # MoE model: need to organize by EP rank
                if etp_rank is not None and tp_size is not None and ep_size is not None \
                        and etp_size is not None and etp_size > 0:
                    # ETP enabled: use tp_to_ep mapping
                    _, tp_to_ep = get_etp_map(
                        tp_size,
                        ep_size,
                        etp_size
                    )
                    # Group tp_shards by their corresponding EP rank
                    ep_shards = {}
                    ep_ids = []
                    for tp_idx, vpp_state_dicts in enumerate(transposed):
                        ep_id = ((ep_rank * etp_size) // tp_size * tp_size // etp_size) + tp_to_ep[tp_idx]
                        if ep_id not in ep_ids:
                            ep_ids.append(ep_id)
                        if ep_id not in ep_shards:
                            ep_shards[ep_id] = {}
                        shard = {"checkpoint_version": 3.0}
                        for vpp_idx, state_dict in enumerate(vpp_state_dicts):
                            shard[f"model{vpp_idx}"] = state_dict
                        ep_shards[ep_id][tp_idx] = shard
                    mcore_dict = {pp_rank: ep_shards}
                else:
                    # ETP disabled: all tp_shards under current ep_rank
                    tp_shards = {}
                    for tp_idx, vpp_state_dicts in enumerate(transposed):
                        shard = {"checkpoint_version": 3.0}
                        for vpp_idx, state_dict in enumerate(vpp_state_dicts):
                            shard[f"model{vpp_idx}"] = state_dict
                        tp_shards[tp_idx] = shard
                    mcore_dict = {pp_rank: {ep_rank: tp_shards}}

        # Create a new parallel config for this specific PP rank only
        # IMPORTANT: Keep original pp_size but only set pp_ranks to [pp_rank]
        parallel_config.tp_ranks = list(range(len(gathered_state_dicts)))
        parallel_config.pp_ranks = [pp_rank]  # Only current pp_rank
        parallel_config.sub_file_tag = rank
        if ep_rank is not None and etp_rank is not None:
            parallel_config.ep_ranks = ep_ids
        elif ep_rank is not None and etp_rank is None:
            parallel_config.ep_ranks = [ep_rank]

        # Create HF converter
        hf_converter = HfCheckpointConverter(parallel_config, c_config, vision_patch_config=c_vision_patch_config)

        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
            print_rank_0(f"Mcore_dict prepared. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")

        # Create save directory
        os.makedirs(save_hf_path, exist_ok=True)

        # Convert from Mcore format to HF format and save
        print_rank_0("Converting to HF format and saving...")
        mem_before = 0.0
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
        try:
            hf_converter.save_hf_ckpt(mcore_dict, save_hf_path)
            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
                print_rank_0(f"HF checkpoint saved successfully. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")
        except Exception as e:
            print_rank_0(f"Error saving HF checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise

    dist.barrier()
    # Step 8: Global rank 0 consolidates all PP checkpoints to final location
    if rank == 0:
        print_rank_0("Starting checkpoint consolidation...")
        _consolidate_pp_checkpoints(
            save_hf_path,
            pp_size,
            original_hf_path=args.load  # Use the original HF checkpoint path for config files
        )

    print_rank_0("="*80)
    print_rank_0("HF checkpoint saved successfully!")
    print_rank_0("="*80)