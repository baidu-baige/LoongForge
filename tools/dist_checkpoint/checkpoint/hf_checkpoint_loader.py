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
    tp_rank, pp_rank, ep_rank, etp_rank = topo_sharder.get_current_rank_coordinates()
    parallel_config.tp_ranks = [tp_rank]
    parallel_config.pp_ranks = [pp_rank]
    if ep_rank is not None:
        parallel_config.ep_ranks = [ep_rank]
    if etp_rank is not None:
        parallel_config.etp_ranks = [etp_rank]

    # Step 3: Create HF converter
    print_rank_0("Creating HF checkpoint converter...")
    converter = HfCheckpointConverter(
        parallel_config=parallel_config,
        mapping_cfg=mapping_cfgs[0]
    )

    # Step 4: Load and convert HF checkpoint based on model type
    if parser.type == 'llm':
        # LLM: single mapping config
        print_rank_0("Processing LLM checkpoint...")

        # Set mapping config first
        converter.set_mapping_cfg(mapping_cfgs[0])

        # Load HF checkpoint
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

        # Convert to Mcore format
        mem_before = 0.0
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
        mcore_dict = converter.get_mcore_ckpt()
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
            print_rank_0(f"Mcore conversion completed. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")

        if mcore_dict is None:
            raise RuntimeError("Failed to convert HF checkpoint to Mcore format")

    elif parser.type == 'vlm':
        # VLM: multiple mapping configs (foundation, image_encoder, image_projector)
        # TODO: Implement VLM checkpoint loading
        raise NotImplementedError("VLM checkpoint loading is not yet implemented")
    else:
        raise ValueError(f"Unsupported model type: {parser.type}")

    # Ensure mcore_dict is not None (for type checker)
    assert mcore_dict is not None, "mcore_dict should not be None after loading"


    # Step 6: Extract shard for current rank
    # Get state_dict corresponding to current rank's coordinates
    if pp_rank not in mcore_dict:
        raise KeyError(f"PP rank {pp_rank} not found in mcore_dict")

    current_rank_state_dict = mcore_dict[pp_rank].get(tp_rank, {})

    if etp_rank is not None and etp_rank in current_rank_state_dict:
        current_rank_state_dict = current_rank_state_dict[etp_rank]

    # Step 7: Load to model
    print_rank_0("Loading model state_dict...")
    unwrapped_model = unwrap_model(model)

    mem_before = 0.0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

    # Load to each model shard
    for model_module in unwrapped_model:
        # Use strict=False because some parameters may not be in checkpoint
        missing_keys, unexpected_keys = model_module.load_state_dict(
            current_rank_state_dict['model'],
            strict=False
        )

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        print_rank_0(f"Model state_dict loaded successfully. Memory: Before={mem_before:.2f}GB → Peak={peak_mem_gb:.2f}GB → After={mem_after:.2f}GB, Change={mem_after-mem_before:+.2f}GB")
    else:
        print_rank_0("Model state_dict loaded successfully")

    optimizer.reload_model_params()

    # Step 8: Synchronize all ranks
    print_rank_0("Synchronizing all ranks...")
    dist.barrier()

    print_rank_0("="*80)
    print_rank_0("HF checkpoint loaded and sharded successfully!")
    print_rank_0("="*80)

    # Return iteration=0 (start training from HF checkpoint)
    return 0, 0


def test():
    """
    Test HF checkpoint online loading functionality with Qwen2.5-0.5B-Instruct

    Test scenario:
    - Initialize distributed environment
    - Create real Qwen2.5 model using model provider pattern
    - Call load_hf_checkpoint_online to load HF checkpoint
    - Verify the function executes without errors

    Run with:
        torchrun --nproc_per_node=4 tools/dist_checkpoint/checkpoint/hf_checkpoint_loader.py \
            --yaml-file configs/qwen2.5_hf_to_mcore.yaml \
            --load /mnt/cluster/models/Qwen2.5-0.5B-Instruct

    Note:
        - Parallel strategy (TP, PP, etc.) is defined in the YAML config file
        - World size should match the parallel configuration in YAML
        - Model architecture is inferred from checkpoint config
    """
    print("\n" + "="*80)
    print("HF Checkpoint Loader Test Starting...")
    print("="*80 + "\n")

    # Step 1: Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test HF checkpoint online loading')
    parser.add_argument('--yaml-file', type=str, required=True,
                       help='Path to YAML config file for HF to Mcore mapping')
    parser.add_argument('--load', type=str, required=True,
                       help='Path to HF checkpoint directory')
    test_args = parser.parse_args()

    # Step 2: Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set GPU device for each rank
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"[Test] Rank {rank}/{world_size} initialized on GPU {local_rank}")
    else:
        print(f"[Test] Rank {rank}/{world_size} initialized on CPU")

    # Step 3: Parse YAML to get parallel config
    print(f"[Test] Rank {rank} parsing YAML config...")
    from tools.dist_checkpoint.core.parser import Parser
    yaml_parser = Parser(yaml_file=test_args.yaml_file)
    parallel_config = yaml_parser.get_parallel_config()

    print(f"[Test] Rank {rank} parallel config from YAML: "
          f"TP={parallel_config.tp_size}, PP={parallel_config.pp_size}, "
          f"EP={parallel_config.ep_size}, ETP={parallel_config.etp_size}")

    # Step 4: Initialize parallel_state with config from YAML
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=parallel_config.tp_size,
            pipeline_model_parallel_size=parallel_config.pp_size
        )
        print(f"[Test] Rank {rank} initialized parallel_state with "
              f"TP={parallel_config.tp_size}, PP={parallel_config.pp_size}")

    # Step 5: Add parallel config to test_args for compatibility
    test_args.tensor_model_parallel_size = parallel_config.tp_size
    test_args.pipeline_model_parallel_size = parallel_config.pp_size

    # Step 6: Create real Qwen2.5 model using setup_model_and_optimizer pattern
    print(f"[Test] Rank {rank} creating Qwen2.5 model...")

    # Load HF config to get model architecture parameters
    import json
    config_path = os.path.join(test_args.load, "config.json")
    with open(config_path, 'r') as f:
        hf_config = json.load(f)

    print(f"[Test] Rank {rank} HF config loaded: "
          f"layers={hf_config.get('num_hidden_layers')}, "
          f"hidden={hf_config.get('hidden_size')}, "
          f"heads={hf_config.get('num_attention_heads')}")

    # Import necessary modules
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

    from aiak_training_omni.models.foundation.qwen2.qwen_model import Qwen2Model
    from aiak_training_omni.models.foundation.qwen2.qwen_config import Qwen2Config
    from megatron.core import parallel_state as ps

    # Initialize CUDA RNG tracker (required for model initialization)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(123 + rank)  # Use rank-dependent seed

    # Create Qwen2 config based on HF config
    params_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    qwen_config = Qwen2Config(
        num_layers=hf_config.get('num_hidden_layers', 24),
        hidden_size=hf_config.get('hidden_size', 896),
        num_attention_heads=hf_config.get('num_attention_heads', 14),
        ffn_hidden_size=hf_config.get('intermediate_size', 4864),
        num_query_groups=hf_config.get('num_key_value_heads', 2),

        # Vocabulary and sequence length
        padded_vocab_size=hf_config.get('vocab_size', 151936),
        max_position_embeddings=hf_config.get('max_position_embeddings', 32768),

        # Parallel config
        tensor_model_parallel_size=parallel_config.tp_size,
        pipeline_model_parallel_size=parallel_config.pp_size,

        # Required dtype parameters for pipeline parallelism
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,

        # Other required params
        use_cpu_initialization=False,
        perform_initialization=True,
        bf16=torch.cuda.is_available(),
    )

    # Create Qwen2 model directly
    print(f"[Test] Rank {rank} creating Qwen2Model...")

    # Determine if this rank needs embedding (first PP stage)
    # and output layer (last PP stage)
    pp_rank = ps.get_pipeline_model_parallel_rank()
    pp_size = ps.get_pipeline_model_parallel_world_size()

    pre_process = (pp_rank == 0)
    post_process = (pp_rank == pp_size - 1)

    model = Qwen2Model(
        config=qwen_config,
        pre_process=pre_process,
        post_process=post_process,
    )

    # 打印模型状态的键名和形状
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        print(f"模型参数总数: {len(state_dict)}")
        for key, param in state_dict.items():
            print(f"  {key}: {param.shape}")

    # Wrap in list to match expected format
    model = [model]

    if torch.cuda.is_available():
        model[0] = model[0].cuda()

    print(f"[Test] Rank {rank} model created successfully")

    # Step 7: Create mock optimizer and scheduler (not needed for checkpoint loading)
    optimizer = None
    opt_param_scheduler = None

    # Step 8: Call load_hf_checkpoint_online
    print(f"[Test] Rank {rank} calling load_hf_checkpoint_online...")

    try:
        iteration, num_floating_point_operations_so_far = load_hf_checkpoint_online(
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            args=test_args
        )

        print(f"[Test] Rank {rank} load_hf_checkpoint_online succeeded!")
        print(f"[Test] Rank {rank} returned: iteration={iteration}, "
              f"num_floating_point_operations_so_far={num_floating_point_operations_so_far}")

    except Exception as e:
        print(f"[Test] Rank {rank} load_hf_checkpoint_online failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 9: Synchronize all ranks
    dist.barrier()

    if rank == 0:
        print("\n" + "="*80)
        print("✅ HF Checkpoint Loader Test Completed Successfully!")
        print("="*80 + "\n")

    # Cleanup
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    dist.destroy_process_group()



if __name__ == "__main__":
    test()
    
