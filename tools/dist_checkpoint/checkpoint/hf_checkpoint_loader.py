# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
HF Checkpoint Online Loading for Training
Implements online loading of HF checkpoints based on tools/dist_checkpoint modules
"""
import os
import sys
from typing import Tuple

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.distributed as dist
from megatron.core import mpu, parallel_state
from megatron.training import print_rank_0
from megatron.training.utils import unwrap_model

# Import existing dist_checkpoint modules
from tools.dist_checkpoint.core.parser import Parser
from tools.dist_checkpoint.core.topo_sharder import TopoSharder
from tools.dist_checkpoint.checkpoint.hf_checkpoint_converter import HfCheckpointConverter
from tools.dist_checkpoint.utils import time_checkpoint_operation, MemoryTracker
from tools.convert_checkpoint.utils.utils import get_etp_map
from tools.convert_checkpoint.utils.config_utils import get_yaml_config


def _is_hf_checkpoint(checkpoint_path: str) -> bool:
    """
    Detect if checkpoint is in HF format

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        bool: True if HF format, False if Mcore format
    """
    if not os.path.exists(checkpoint_path):
        return False

    # HF checkpoint features:
    # - Has config.json
    # - Has model.safetensors or pytorch_model.bin
    has_config = os.path.exists(os.path.join(checkpoint_path, "config.json"))
    has_safetensors = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))
    has_pytorch_bin = any(
        f.startswith("pytorch_model") and f.endswith(".bin")
        for f in os.listdir(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_path, f))
    )

    return has_config and (has_safetensors or has_pytorch_bin)


def _log_loaded_param_stats(unwrapped_model):
    """Print count + norm/mean of a small set of well-known parameters so we
    can tell from logs whether ckpt values actually landed in the model.
    strict=True only verifies key set + shapes, not actual values — this
    closes that gap for the rank-0 sanity check."""
    if dist.get_rank() != 0:
        return
    total = 0
    total_nonzero = 0
    samples = []
    sample_targets = (
        "embedding.word_embeddings.weight",
        "decoder.layers.0.self_attention.linear_q_down_proj.weight",
        "decoder.layers.0.mlp.router.weight",
        "output_layer.weight",
        "decoder.final_layernorm.weight",
    )
    for m in unwrapped_model:
        for name, p in m.named_parameters():
            n = p.numel()
            total += n
            if p.is_floating_point():
                total_nonzero += int((p != 0).sum().item())
            for tgt in sample_targets:
                if name.endswith(tgt) and len(samples) < 16:
                    pf = p.float()
                    samples.append((name, n, pf.norm().item(), pf.abs().mean().item()))
                    break
    print_rank_0(f"[CKPT_LOAD] total params={total:,}, nonzero floats={total_nonzero:,}")
    for name, n, norm, mean_abs in samples:
        print_rank_0(f"[CKPT_LOAD]   {name}: numel={n:,} norm={norm:.4f} mean_abs={mean_abs:.6f}")


@time_checkpoint_operation
def load_hf_checkpoint_online(
    model,
    optimizer,
    opt_param_scheduler,
    args
) -> Tuple[int, int]:
    """
    Load HF checkpoint online and shard to distributed model

    Uses tools/dist_checkpoint modules:
    1. Parser to parse config
    2. TopoSharder to initialize parallel topology
    3. HfCheckpointConverter to convert and shard
    4. Load to model

    Args:
        model: Megatron distributed model
        optimizer: Optimizer
        opt_param_scheduler: Learning rate scheduler
        args: Command line arguments

    Returns:
        (iteration, num_floating_point_operations_so_far)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank_0("="*80)
    print_rank_0("Loading HF checkpoint with online sharding")
    print_rank_0("="*80)
    print_rank_0(f"Checkpoint path: {args.load}")
    print_rank_0(f"World size: {world_size}")
    ep_size = getattr(args, 'expert_model_parallel_size', None)
    etp_size = getattr(args, 'expert_tensor_parallel_size', None)
    if ep_size is not None and ep_size > 1:
        print_rank_0(f"Parallel config: TP={args.tensor_model_parallel_size}, "
                    f"PP={args.pipeline_model_parallel_size}, EP={ep_size}, ETP={etp_size}")
    else:
        print_rank_0(f"Parallel config: TP={args.tensor_model_parallel_size}, "
                    f"PP={args.pipeline_model_parallel_size}")

    # Step 1: Parse args to get config
    print_rank_0("Parsing config from args")
    parser = Parser(args)

    # Get parallel config
    parallel_config = parser.get_parallel_config()

    print_rank_0(f"Model type: {parser.type}")

    # Step 2: Initialize TopoSharder (parallel_state already initialized by training, only get coordinates here)
    # Note: parallel_state is already set up by training initialization, TopoSharder only used to get coordinates
    topo_sharder = TopoSharder(parallel_config)
    tp_rank, pp_rank, ep_rank, etp_rank, dp_rank = topo_sharder.get_current_rank_coordinates()
    parallel_config.tp_ranks = [tp_rank]
    parallel_config.pp_ranks = [pp_rank]
    if ep_rank is not None:
        if etp_rank is not None and etp_size is not None and etp_size > 0: # with ETP
            tp_size = args.tensor_model_parallel_size
            ep_size = args.expert_model_parallel_size
            etp_size = args.expert_tensor_parallel_size
            assert ep_size >= tp_size, "With ETP, EP size must be greater than or equal to TP size!"
            _, tp_to_ep = get_etp_map(
                    tp_size,
                    ep_size,
                    etp_size
                )
            ep_id = ((ep_rank * etp_size) // tp_size * tp_size // etp_size) + tp_to_ep[tp_rank]
            parallel_config.ep_ranks = [ep_id]
        else: # without ETP
            parallel_config.ep_ranks = [ep_rank]


    # Step 3: Create HF converter
    print_rank_0("Creating HF checkpoint converter...")
    c_config = get_yaml_config(parser.config_file, parser.convert_file, for_vlm=(parser.vision_patch_convert_file is not None))
    c_vision_patch_config = get_yaml_config(
        parser.config_file, parser.vision_patch_convert_file,
        adapter_convert_file=parser.adapter_convert_file) if parser.vision_patch_convert_file is not None else None
    
    hf_converter = HfCheckpointConverter(parallel_config, c_config, vision_patch_config=c_vision_patch_config)

    # Step 4: Load and convert HF checkpoint based on model type

    print_rank_0("Processing HF checkpoint...")

    # Load HF checkpoint
    print_rank_0(f"Loading HF checkpoint from {args.load}...")
    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        print_rank_0(f"HF checkpoint loaded successfully. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")
    else:
        print_rank_0("HF checkpoint loaded successfully")

    # Convert to Mcore format
    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    mcore_dict = hf_converter.get_mcore_ckpt(args.load)
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        print_rank_0(f"Mcore conversion completed. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")

    if mcore_dict is None:
        raise RuntimeError("Failed to convert HF checkpoint to Mcore format")

    # Ensure mcore_dict is not None (for type checker)
    assert mcore_dict is not None, "mcore_dict should not be None after loading"

    # Step 6: Extract shard for current rank
    # Get state_dict corresponding to current rank's coordinates
    if pp_rank not in mcore_dict:
        raise KeyError(f"PP rank {pp_rank} not found in mcore_dict")

    if ep_rank is None:
        current_rank_state_dict = mcore_dict[pp_rank][tp_rank]
    else:
        if etp_rank is not None: # with ETP
            current_rank_state_dict = mcore_dict[pp_rank][ep_id][tp_rank]
        else: # without ETP
            current_rank_state_dict= mcore_dict[pp_rank][ep_rank][tp_rank]
            
            

    # Step 7: Load to model
    print_rank_0("Loading model state_dict...")
    unwrapped_model = unwrap_model(model)

    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

    # Handle both wrapped {'model': dict} and direct dict formats
    model_state_dict = current_rank_state_dict.get('model', current_rank_state_dict) if isinstance(current_rank_state_dict, dict) else current_rank_state_dict
    # strict=True — let PyTorch raise a RuntimeError that lists every
    # missing / unexpected key and every size-mismatch. The prior
    # strict=False (plus silent fallback) was masking real load failures
    # (q_layernorm / kv_layernorm / hc_*_scale / tid2eid / expert_bias)
    # and letting training run with randomly-initialized shards.
    if len(unwrapped_model) == 1:
        unwrapped_model[0].load_state_dict(model_state_dict, strict=True)
    else:  # vpp
        for i in range(len(unwrapped_model)):
            unwrapped_model[i].load_state_dict(
                current_rank_state_dict[f"model{i}"], strict=True
            )

    # Positive sanity check — print weight stats for a handful of well-known
    # parameters so we can tell (from logs) that ckpt values actually landed
    # in the model rather than zeros / random init. Helps diagnose loss-jump
    # symptoms where strict=True accepted shapes but values were off.
    _log_loaded_param_stats(unwrapped_model)

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        print_rank_0(f"Model state_dict loaded successfully. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")
    else:
        print_rank_0("Model state_dict loaded successfully")

    if optimizer is not None:
        optimizer.reload_model_params()

    # Step 8: Synchronize all ranks
    print_rank_0("Synchronizing all ranks...")
    dist.barrier()

    print_rank_0("="*80)
    print_rank_0("HF checkpoint loaded and sharded successfully!")
    print_rank_0("="*80)

    # Return iteration=0 (start training from HF checkpoint)
    return 0, 0

    
