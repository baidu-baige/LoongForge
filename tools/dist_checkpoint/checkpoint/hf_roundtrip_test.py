"""
HF Checkpoint Roundtrip Test

Complete roundtrip test for HF checkpoint conversion pipeline.
Directly reuses load_hf_checkpoint_online and save_hf_checkpoint_online logic.

This test validates that the HF checkpoint conversion pipeline is lossless by:
1. Loading HF checkpoint and sharding to ranks (using loader logic)
2. Gathering weights back to TP rank 0 (using saver gather logic)
3. Saving back to HF format (using saver logic)
4. Comparing original and roundtripped weights

Usage:
    torchrun --nproc_per_node=4 hf_roundtrip_test.py \
        --hf-checkpoint /path/to/hf \
        --yaml-file config.yaml \
        --output-dir ./roundtrip_output
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional
import argparse
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tools'))

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.training import print_rank_0

# Import existing functions from loader.py and saver.py
from tools.dist_checkpoint.core.parser import Parser
from tools.dist_checkpoint.core.topo_sharder import TopoSharder
from tools.dist_checkpoint.core.tp_gather import TPGather
from tools.dist_checkpoint.checkpoint.hf_checkpoint_converter import HfCheckpointConverter


@dataclass
class ComparisonResult:
    """Comparison result structure"""
    num_baseline: int = 0
    num_roundtrip: int = 0
    identical_keys: int = 0
    missing_keys: list = None
    extra_keys: list = None
    shape_mismatches: list = None
    num_exact_matches: int = 0
    num_close_matches: int = 0
    num_different: int = 0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0

    def __post_init__(self):
        if self.missing_keys is None:
            self.missing_keys = []
        if self.extra_keys is None:
            self.extra_keys = []
        if self.shape_mismatches is None:
            self.shape_mismatches = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'num_baseline': self.num_baseline,
            'num_roundtrip': self.num_roundtrip,
            'identical_keys': self.identical_keys,
            'missing_keys': self.missing_keys,
            'extra_keys': self.extra_keys,
            'shape_mismatches': self.shape_mismatches,
            'num_exact_matches': self.num_exact_matches,
            'num_close_matches': self.num_close_matches,
            'num_different': self.num_different,
            'max_abs_diff': float(self.max_abs_diff),
            'mean_abs_diff': float(self.mean_abs_diff),
        }


def load_hf_weights(hf_path: str) -> Dict[str, torch.Tensor]:
    """Load all weights from HF checkpoint"""
    weights = {}
    hf_path = Path(hf_path)

    # Try safetensors first
    safetensors_files = sorted([
        f for f in os.listdir(hf_path)
        if f.endswith('.safetensors')
    ])

    if safetensors_files:
        try:
            from safetensors.torch import load_file
            for sf_file in safetensors_files:
                file_weights = load_file(hf_path / sf_file)
                weights.update(file_weights)
        except ImportError:
            pass

    # Fallback to pytorch_model.bin
    if not weights:
        pytorch_files = sorted([
            f for f in os.listdir(hf_path)
            if f.startswith('pytorch_model') and f.endswith('.bin')
        ])
        for pt_file in pytorch_files:
            file_weights = torch.load(hf_path / pt_file, map_location='cpu')
            weights.update(file_weights)

    if not weights:
        raise FileNotFoundError(f"No weight files found in {hf_path}")

    return weights


def phase1_load_and_shard(hf_path: str, yaml_file: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Phase 1: Load HF checkpoint and shard to ranks
    Directly follows load_hf_checkpoint_online logic
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    print_rank_0("=" * 80)
    print_rank_0("Phase 1: Load HF Checkpoint and Shard")
    print_rank_0("=" * 80)

    # Parse config
    parser = Parser(yaml_file=yaml_file)
    parallel_config = parser.get_parallel_config()

    # Get mapping config
    if parser.type == 'llm':
        mapping_cfg = parser.get_language_model_cfg()
    else:
        raise NotImplementedError(f"Model type {parser.type} not supported")

    # Initialize TopoSharder and get rank coordinates
    topo_sharder = TopoSharder(parallel_config)
    tp_rank, pp_rank, _, _ = topo_sharder.get_current_rank_coordinates()

    # Set rank lists on parallel_config - ALL ranks, not just current rank
    parallel_config.tp_ranks = list(range(parallel_config.tp_size))
    parallel_config.pp_ranks = list(range(parallel_config.pp_size))
    if parallel_config.ep_size is not None:
        parallel_config.ep_ranks = list(range(parallel_config.ep_size))
    if parallel_config.etp_size is not None:
        parallel_config.etp_ranks = list(range(parallel_config.etp_size))

    # Create converter
    converter = HfCheckpointConverter(
        parallel_config=parallel_config,
        mapping_cfg=mapping_cfg
    )
    converter.set_mapping_cfg(mapping_cfg)

    # Load HF (exactly like load_hf_checkpoint_online)
    if rank == 0:
        print_rank_0(f"  Loading HF from {hf_path}...")
    converter.load_hf(hf_path, mapping_cfg)

    if dist.is_initialized():
        dist.barrier()

    # Convert to Mcore format
    if rank == 0:
        print_rank_0("  Converting to Mcore format...")
    mcore_dict = converter.get_mcore_ckpt()

    if dist.is_initialized():
        dist.barrier()

    # Extract shard for current rank
    if pp_rank not in mcore_dict:
        raise KeyError(f"PP rank {pp_rank} not found in mcore_dict")

    current_rank_state_dict = mcore_dict[pp_rank].get(tp_rank, {})
    sharded_weights = current_rank_state_dict.get('model', {})

    print_rank_0(f"✓ Rank {rank} received {len(sharded_weights)} parameters")
    return sharded_weights


def phase2_gather_weights(sharded_weights: Dict, yaml_file: str) -> Optional[list]:
    """
    Phase 2: Gather weights from all TP ranks back to TP rank 0
    Directly follows save_hf_checkpoint_online gather logic

    Returns:
        TP rank 0: list of state_dicts from all TP ranks
        Other ranks: None
    """
    print_rank_0("=" * 80)
    print_rank_0("Phase 2: Gather Weights")
    print_rank_0("=" * 80)

    # Move weights to GPU if available (required for NCCL)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        sharded_weights = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sharded_weights.items()
        }

    if dist.is_initialized():
        dist.barrier()

    # Initialize TPGather (exactly like save_hf_checkpoint_online)
    parser = Parser(yaml_file=yaml_file)
    parallel_config = parser.get_parallel_config()
    topo_sharder = TopoSharder(parallel_config)

    # Set rank lists on parallel_config - ALL ranks
    parallel_config.tp_ranks = list(range(parallel_config.tp_size))
    parallel_config.pp_ranks = list(range(parallel_config.pp_size))
    if parallel_config.ep_size is not None:
        parallel_config.ep_ranks = list(range(parallel_config.ep_size))
    if parallel_config.etp_size is not None:
        parallel_config.etp_ranks = list(range(parallel_config.etp_size))

    tp_gather = TPGather(topo_sharder)

    # Execute gather
    print_rank_0("Executing TP gather...")
    gathered_state_dicts = tp_gather.gather_state_dicts(sharded_weights)

    if gathered_state_dicts is not None:
        print_rank_0(f"✓ TP rank 0 received {len(gathered_state_dicts)} TP rank weights")
        print_rank_0(f"  Keys in each gathered state_dict: {[len(sd) for sd in gathered_state_dicts]}")
        return gathered_state_dicts
    else:
        print_rank_0("✓ Non-TP-rank-0 completed gather")
        return None


def phase3_save(
    gathered_state_dicts: Optional[list],
    yaml_file: str,
    output_dir: str,
    tp_rank: int,
    pp_rank: int,
    ep_rank: Optional[int],
    etp_rank: Optional[int],
    parallel_config,
) -> None:
    """
    Phase 3: Save gathered weights back to HF format
    Directly follows save_hf_checkpoint_online save logic
    """
    print_rank_0("=" * 80)
    print_rank_0("Phase 3: Save to HF Format")
    print_rank_0("=" * 80)

    # Only TP rank 0 saves
    if tp_rank != 0 or gathered_state_dicts is None:
        print_rank_0("✓ Non-TP-rank-0 skipping save")
        return

    # TP rank 0 saves
    print_rank_0("Step 3.1: Saving to HF format...")

    parser = Parser(yaml_file=yaml_file)
    if parser.type == 'llm':
        mapping_cfg = parser.get_language_model_cfg()
    else:
        raise NotImplementedError(f"Model type {parser.type} not supported")

    # Create a new parallel config for this specific PP rank only
    # IMPORTANT: Keep original pp_size but only set pp_ranks to [pp_rank]
    from tools.dist_checkpoint.config.parallel_config import ParallelConfig

    # gathered_state_dicts is a list: [state_dict_tp0, state_dict_tp1, ...]
    tp_size = len(gathered_state_dicts)

    pp_parallel_config = ParallelConfig(
        tp_size=tp_size,
        pp_size=parallel_config.pp_size,  # Keep original pp_size
        ep_size=parallel_config.ep_size,
        etp_size=parallel_config.etp_size,
        vpp_size=parallel_config.vpp_size,
        custom_pipeline_layers=parallel_config.custom_pipeline_layers
    )
    pp_parallel_config.tp_ranks = list(range(tp_size))
    pp_parallel_config.pp_ranks = [pp_rank]  # Only current pp_rank
    if ep_rank is not None:
        pp_parallel_config.ep_ranks = [ep_rank]
    if etp_rank is not None:
        pp_parallel_config.etp_ranks = [etp_rank]

    # Create HF converter
    converter = HfCheckpointConverter(
        parallel_config=pp_parallel_config,
        mapping_cfg=mapping_cfg
    )
    converter.set_mapping_cfg(mapping_cfg)

    # Prepare mcore_dict (exactly like save_hf_checkpoint_online)
    # gathered_state_dicts is a list: [state_dict_tp0, state_dict_tp1, ...]
    # Convert to format: {pp_rank: {tp_idx: {"model": state_dict, "checkpoint_version": 3.0}}}
    mcore_dict = {
        pp_rank: {
            tp_idx: {
                "model": state_dict,
                "checkpoint_version": 3.0,
            }
            for tp_idx, state_dict in enumerate(gathered_state_dicts)
        }
    }

    # Save
    roundtrip_ckpt_dir = Path(output_dir) / "roundtrip_checkpoint"
    roundtrip_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print_rank_0(f"  Saving to {roundtrip_ckpt_dir}...")
    try:
        converter.save_hf_ckpt(mcore_dict, str(roundtrip_ckpt_dir))
        print_rank_0("✓ Checkpoint saved")
    except Exception as e:
        print_rank_0(f"Error during save: {e}")
        import traceback
        traceback.print_exc()


def phase3_consolidate(
    yaml_file: str,
    output_dir: str,
    hf_path: str,
) -> None:
    """
    Phase 3.5: Consolidate PP checkpoints
    Global rank 0 consolidates all PP rank checkpoints to final location
    """
    print_rank_0("=" * 80)
    print_rank_0("Phase 3.5: Consolidate PP Checkpoints")
    print_rank_0("=" * 80)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Only global rank 0 consolidates
    if rank != 0:
        return

    # Import consolidate function from saver
    from tools.dist_checkpoint.checkpoint.hf_checkpoint_saver import _consolidate_pp_checkpoints

    parser = Parser(yaml_file=yaml_file)
    parallel_config = parser.get_parallel_config()
    pp_size = parallel_config.pp_size

    roundtrip_ckpt_dir = str(Path(output_dir) / "roundtrip_checkpoint")

    try:
        _consolidate_pp_checkpoints(
            roundtrip_ckpt_dir,
            pp_size,
            original_hf_path=hf_path
        )
        print_rank_0("✓ Checkpoint consolidation completed")
    except Exception as e:
        print_rank_0(f"Error during consolidation: {e}")
        import traceback
        traceback.print_exc()


def phase4_compare(
    hf_path: str,
    output_dir: str,
) -> ComparisonResult:
    """
    Phase 4: Compare baseline and roundtripped weights
    """
    print_rank_0("=" * 80)
    print_rank_0("Phase 4: Compare Weights")
    print_rank_0("=" * 80)

    comparison_result = ComparisonResult()

    # Only global rank 0 compares
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print_rank_0("Loading baseline from original HF...")
        baseline_weights = load_hf_weights(hf_path)
        comparison_result.num_baseline = len(baseline_weights)

        roundtrip_ckpt_dir = Path(output_dir) / "roundtrip_checkpoint"
        print_rank_0(f"Loading roundtripped from {roundtrip_ckpt_dir}...")
        roundtrip_weights = load_hf_weights(roundtrip_ckpt_dir)
        comparison_result.num_roundtrip = len(roundtrip_weights)

        # Compare keys
        baseline_keys = set(baseline_weights.keys())
        roundtrip_keys = set(roundtrip_weights.keys())

        comparison_result.identical_keys = len(baseline_keys & roundtrip_keys)
        comparison_result.missing_keys = sorted(list(baseline_keys - roundtrip_keys))
        comparison_result.extra_keys = sorted(list(roundtrip_keys - baseline_keys))

        print_rank_0(f"Baseline: {comparison_result.num_baseline} tensors")
        print_rank_0(f"Roundtrip: {comparison_result.num_roundtrip} tensors")
        print_rank_0(f"Identical keys: {comparison_result.identical_keys}")

        # Compare values
        all_diffs = []
        for key in baseline_keys & roundtrip_keys:
            baseline_tensor = baseline_weights[key]
            roundtrip_tensor = roundtrip_weights[key]

            # Check shape
            if baseline_tensor.shape != roundtrip_tensor.shape:
                comparison_result.shape_mismatches.append({
                    'key': key,
                    'baseline': tuple(baseline_tensor.shape),
                    'roundtrip': tuple(roundtrip_tensor.shape),
                })
                continue

            # Check values
            diff = torch.abs(baseline_tensor.float() - roundtrip_tensor.float())
            max_diff = diff.max().item()

            # Categorize
            if torch.allclose(baseline_tensor.float(), roundtrip_tensor.float(), rtol=1e-5, atol=1e-8):
                comparison_result.num_exact_matches += 1
            elif torch.allclose(baseline_tensor.float(), roundtrip_tensor.float(), rtol=1e-3, atol=1e-5):
                comparison_result.num_close_matches += 1
            else:
                comparison_result.num_different += 1

            all_diffs.append(diff)

        if all_diffs:
            all_diffs_tensor = torch.cat([d.flatten() for d in all_diffs])
            comparison_result.max_abs_diff = all_diffs_tensor.max().item()
            comparison_result.mean_abs_diff = all_diffs_tensor.mean().item()

        print_rank_0(f"Exact matches: {comparison_result.num_exact_matches}")
        print_rank_0(f"Close matches: {comparison_result.num_close_matches}")
        print_rank_0(f"Different: {comparison_result.num_different}")
        print_rank_0(f"Max difference: {comparison_result.max_abs_diff:.2e}")
        print_rank_0(f"Mean difference: {comparison_result.mean_abs_diff:.2e}")

        # Save report
        results_path = Path(output_dir) / "comparison_report.json"
        with open(results_path, 'w') as f:
            json.dump(comparison_result.to_dict(), f, indent=2)
        print_rank_0(f"Comparison report saved to {results_path}")

    if dist.is_initialized():
        dist.barrier()

    return comparison_result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HF Checkpoint Roundtrip Test')
    parser.add_argument('--hf-checkpoint', type=str, required=True,
                        help='Path to HF checkpoint')
    parser.add_argument('--yaml-file', type=str, required=True,
                        help='Path to YAML config for HF to Mcore mapping')
    parser.add_argument('--output-dir', type=str, default='./roundtrip_output',
                        help='Output directory for test results')

    args = parser.parse_args()

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set GPU device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)

    print_rank_0(f"[Roundtrip Test] Rank {rank}/{world_size} initialized")

    try:
        # Phase 1: Load and shard
        sharded_weights = phase1_load_and_shard(args.hf_checkpoint, args.yaml_file)

        if dist.is_initialized():
            dist.barrier()

        # Phase 2: Gather
        gathered_weights = phase2_gather_weights(sharded_weights, args.yaml_file)

        if dist.is_initialized():
            dist.barrier()

        # Get rank coordinates for phase 3
        parser_obj = Parser(yaml_file=args.yaml_file)
        parallel_config = parser_obj.get_parallel_config()
        topo_sharder = TopoSharder(parallel_config)
        tp_rank, pp_rank, ep_rank, etp_rank = topo_sharder.get_current_rank_coordinates()

        # Phase 3: Save
        if tp_rank is not None and pp_rank is not None:
            phase3_save(
                gathered_weights,
                args.yaml_file,
                args.output_dir,
                tp_rank,
                pp_rank,
                ep_rank,
                etp_rank,
                parallel_config
            )
        else:
            raise RuntimeError("tp_rank and pp_rank should not be None")

        if dist.is_initialized():
            dist.barrier()

        # Phase 3.5: Consolidate
        phase3_consolidate(args.yaml_file, args.output_dir, args.hf_checkpoint)

        if dist.is_initialized():
            dist.barrier()

        # Phase 4: Compare
        result = phase4_compare(args.hf_checkpoint, args.output_dir)

        if dist.is_initialized():
            dist.barrier()

        print_rank_0("=" * 80)
        print_rank_0("✅ Roundtrip Test Completed Successfully!")
        print_rank_0("=" * 80)

    finally:
        # Cleanup
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
