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


def _scatter_state_dict_from_rank0(
    mcore_dict,
    rank_mapping,
    vpp_size,
    has_ep,
    has_etp,
    tp_to_ep,
    tp_size,
    ep_size,
    pp_size,
    tp_rank,
    pp_rank,
    ep_rank,
    etp_rank
):
    """
    Scatter state_dict from rank 0 to all ranks tensor by tensor.

    Rank 0 has complete mcore_dict, distributes each rank's slice to them.
    Supports VPP (model0, model1...) and MoE/ETP.

    Args:
        mcore_dict: Complete mcore checkpoint dict (only valid on rank 0)
        vpp_size: Number of virtual pipeline stages
        has_ep: Whether expert parallelism is enabled
        has_etp: Whether expert tensor parallelism is enabled
        tp_to_ep: Mapping from TP rank to EP rank (for ETP)
        tp_size, ep_size, pp_size: Parallel sizes
        tp_rank, pp_rank, ep_rank, etp_rank: Current rank's coordinates

    Returns:
        state_dict for current rank
    """
    rank = dist.get_rank()

    # Determine model keys for VPP
    model_keys = [f"model{i}" for i in range(vpp_size)] if vpp_size and vpp_size > 1 else ["model"]

    def _get_ep_id(ep_rank, tp_rank):
        """Calculate ep_id for ETP case."""
        if has_ep and has_etp:
            return (ep_rank // tp_size * tp_size) + tp_to_ep[tp_rank]
        return ep_rank

    def _get_slice(data, pp, ep, tp):
        """Get slice for given coordinates."""
        slice_dict = data[pp]
        if has_ep:
            slice_dict = slice_dict[ep]
        return slice_dict[tp]

    # Calculate current rank's ep_id
    ep_id = _get_ep_id(ep_rank, tp_rank) if has_ep else None

    if rank == 0:
        # Rank 0: distribute slices to all ranks using the lookup table
        # Iterate over all (pp, tp, ep) combinations from rank_mapping
        for (dst_pp, dst_tp, dst_ep), dst_global_ranks in rank_mapping.items():
            dst_ep_id = _get_ep_id(dst_ep, dst_tp) if has_ep else None
            dst_slice = _get_slice(mcore_dict, dst_pp, dst_ep_id, dst_tp)

            for dst_global_rank in dst_global_ranks:
                if dst_global_rank == 0:
                    # Skip self
                    continue

                # Send each model_key's tensors (only keys starting with 'model')
                for model_key in model_keys:
                    if not model_key.startswith('model'):
                        continue
                    model_state = dst_slice[model_key]
                    # Filter out None values and count valid tensors
                    valid_tensors = {k: v for k, v in model_state.items() if v is not None}
                    # First send number of tensors (use GPU tensor for NCCL)
                    dist.send(torch.tensor([len(valid_tensors)], dtype=torch.int32).cuda(), dst=dst_global_rank)
                    # Then send each tensor
                    for param_name, tensor in valid_tensors.items():
                        # Send name length and name
                        name_bytes = param_name.encode('utf-8')
                        dist.send(torch.tensor([len(name_bytes)], dtype=torch.int32).cuda(), dst=dst_global_rank)
                        dist.send(torch.ByteTensor(list(name_bytes)).cuda(), dst=dst_global_rank)
                        # Send shape
                        shape_tensor = torch.zeros(8, dtype=torch.int64)
                        shape_tensor[:len(tensor.shape)] = torch.tensor(tensor.shape, dtype=torch.int64)
                        dist.send(shape_tensor.cuda(), dst=dst_global_rank)
                        # Send dtype (as int)
                        dtype_map = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
                        dist.send(torch.tensor([dtype_map.get(tensor.dtype, 0)], dtype=torch.int32).cuda(), dst=dst_global_rank)
                        # Send tensor data (move to GPU for NCCL)
                        gpu_tensor = tensor.cuda()
                        dist.send(gpu_tensor, dst=dst_global_rank)

        # Rank 0 returns its own slice
        return _get_slice(mcore_dict, pp_rank, ep_id, tp_rank)

    else:
        # Other ranks: receive from rank 0
        result = {}
        for model_key in model_keys:
            result[model_key] = {}
            # Receive number of tensors (use GPU tensor for NCCL)
            num_tensors = torch.zeros(1, dtype=torch.int32).cuda()
            dist.recv(num_tensors, src=0)

            for _ in range(num_tensors[0].item()):
                # Receive name
                name_len = torch.zeros(1, dtype=torch.int32).cuda()
                dist.recv(name_len, src=0)
                name_bytes = torch.ByteTensor(int(name_len[0].item())).cuda()
                dist.recv(name_bytes, src=0)
                param_name = bytes(name_bytes.tolist()).decode('utf-8')

                # Receive shape
                shape_tensor = torch.zeros(8, dtype=torch.int64).cuda()
                dist.recv(shape_tensor, src=0)
                shape = tuple(shape_tensor[shape_tensor > 0].cpu().tolist())

                # Receive dtype
                dtype_val = torch.zeros(1, dtype=torch.int32).cuda()
                dist.recv(dtype_val, src=0)
                dtype_map_inv = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}
                dtype = dtype_map_inv.get(int(dtype_val[0].item()), torch.float32)

                # Receive tensor
                recv_tensor = torch.empty(shape, dtype=dtype).cuda()
                dist.recv(recv_tensor, src=0)
                result[model_key][param_name] = recv_tensor.cpu()

        return result


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

    # Step 1: Parse config file
    if args.yaml_file is None:
        raise ValueError(
            "--yaml-file is required when --load-format=hf\n"
            "Please provide a YAML config file with HF to Mcore mapping"
        )

    print_rank_0(f"Parsing mapping config from {args.yaml_file}")
    parser = Parser(yaml_file=args.yaml_file)

    # Get parallel config
    parallel_config = parser.get_parallel_config()

    # Get mapping config based on model type
    if parser.type == 'llm':
        mapping_cfg = parser.get_language_model_cfg()
        mapping_cfgs = [mapping_cfg]
    elif parser.type == 'vlm':
        # VLM has multiple mapping configs (no language_model key)
        mapping_cfgs = [
            parser.get_foundation_model_cfg(),
            parser.get_image_encoder_cfg(),
            parser.get_image_projector_cfg()
        ]
    else:
        raise ValueError(f"Unsupported model type: {parser.type}")

    print_rank_0(f"Model type: {parser.type}")

    # Step 2: Initialize TopoSharder (parallel_state already initialized by training, only get coordinates here)
    # Note: parallel_state is already set up by training initialization, TopoSharder only used to get coordinates
    topo_sharder = TopoSharder(parallel_config)
    tp_rank, pp_rank, ep_rank, etp_rank, dp_rank = topo_sharder.get_current_rank_coordinates()

    # Get parallel sizes for all ranks
    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    ep_size = getattr(args, 'expert_model_parallel_size', 1)
    etp_size = getattr(args, 'expert_tensor_parallel_size', 1)
    vpp_size = getattr(args, 'num_virtual_stages_per_pipeline_rank', None)

    # Compute ep_id for ETP case
    tp_to_ep = None
    ep_id = None
    if ep_rank is not None and etp_rank is not None:
        assert ep_size >= tp_size, "With ETP, EP size must be greater than or equal to TP size!"
        _, tp_to_ep = get_etp_map(tp_size, ep_size, etp_size)
        ep_id = (ep_rank // tp_size * tp_size) + tp_to_ep[tp_rank]
    elif ep_rank is not None:
        ep_id = ep_rank

    # Step 3 & 4: Only rank 0 loads and converts full checkpoint
    if rank == 0:
        print_rank_0("Creating HF checkpoint converter on rank 0 (full configuration)...")
        # Create a full parallel_config for rank 0 to generate all slices
        full_parallel_config = parser.get_parallel_config()
        full_parallel_config.tp_ranks = list(range(tp_size))
        full_parallel_config.pp_ranks = list(range(pp_size))
        if ep_rank is not None:
            if etp_rank is not None:
                # For ETP, we need to compute all ep_ids
                full_parallel_config.ep_ranks = list(range(ep_size))
            else:
                full_parallel_config.ep_ranks = list(range(ep_size))

        converter = HfCheckpointConverter(
            parallel_config=full_parallel_config,
            mapping_cfg=mapping_cfgs[0]
        )

        if parser.type == 'llm':
            print_rank_0("Processing LLM checkpoint on rank 0...")
            converter.set_mapping_cfg(mapping_cfgs[0])

            # Load HF checkpoint (only on rank 0)
            print_rank_0(f"Loading HF checkpoint from {args.load}...")
            mem_before = 0.0
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats()
            converter.load_hf(args.load, mapping_cfgs[0])
            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
                print_rank_0(f"HF checkpoint loaded successfully. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")
            else:
                print_rank_0("HF checkpoint loaded successfully")

            # Convert to Mcore format (all slices on rank 0)
            mem_before = 0.0
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats()
            mcore_dict = converter.get_mcore_ckpt()
            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
                print_rank_0(f"Mcore conversion completed (all slices). Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")

            if mcore_dict is None:
                raise RuntimeError("Failed to convert HF checkpoint to Mcore format")
        else:
            raise NotImplementedError("VLM checkpoint loading is not yet implemented")
    else:
        mcore_dict = None

    # Step 5: Build rank mapping table (all ranks participate)
    # This creates a lookup table from (pp, tp, ep) to global_rank on rank 0
    rank_mapping = topo_sharder.build_rank_mapping_table()

    # Step 6: Scatter state_dict from rank 0 to all ranks
    print_rank_0("Scattering checkpoint slices from rank 0 to all ranks...")
    current_rank_state_dict = _scatter_state_dict_from_rank0(
        mcore_dict=mcore_dict,
        rank_mapping=rank_mapping,
        vpp_size=vpp_size,
        has_ep=(ep_rank is not None),
        has_etp=(etp_rank is not None),
        tp_to_ep=tp_to_ep,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        ep_rank=ep_rank,
        etp_rank=etp_rank
    )
            
            

    # Step 7: Load to model
    print_rank_0("Loading model state_dict...")
    unwrapped_model = unwrap_model(model)

    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

    if len(unwrapped_model) == 1:
        missing_keys, unexpected_keys = unwrapped_model[0].load_state_dict(
            current_rank_state_dict['model'],
            strict=True
        )
    else: # vpp
        missing_keys = []
        unexpected_keys = []
        for i in range(len(unwrapped_model)):
            model_key = f"model{i}"
            tmp_missing_keys, tmp_unexpected_keys = unwrapped_model[i].load_state_dict(
                current_rank_state_dict[model_key],
                strict=True
            )
            if len(tmp_missing_keys) > 0:
                missing_keys.extend(tmp_missing_keys)
            if len(tmp_unexpected_keys) > 0:
                unexpected_keys.extend(tmp_unexpected_keys)

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

    
