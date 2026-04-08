# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
HF Checkpoint Roundtrip Test v2

Uses the same model/load/save pipeline as actual training to validate
HF <-> Mcore checkpoint conversion correctness.

Compared to v1, this version:
  - Builds a real Megatron model (guarantees consistent state_dict keys across all TP ranks)
  - Reuses load_hf_checkpoint_online / save_hf_checkpoint_online directly
  - Eliminates all custom Phase1/2/3 distributed logic and the associated deadlock risks

Flow:
  initialize_loongforge_megatron
       ↓
  get_model(llm_model_provider)
       ↓
  load_hf_checkpoint_online(model, None, None, args)   # args.load = original HF path
       ↓
  save_hf_checkpoint_online(model, args)               # args.save_hf_path = output dir
       ↓
  compare(args.load, args.save_hf_path)                # rank 0 only

Usage (see bridge_roundtrip.sh for a ready-to-run example):
    torchrun --nproc_per_node=4 hf_roundtrip_test.py \\
        --model-name qwen2.5-0.5b \\
        --tensor-model-parallel-size 2 \\
        --pipeline-model-parallel-size 2 \\
        --bf16 \\
        --seq-length 4096 \\
        --max-position-embeddings 32768 \\
        --micro-batch-size 1 \\
        --global-batch-size 1 \\
        --train-iters 0 \\
        --no-load-optim \\
        --no-load-rng \\
        --load /path/to/hf_checkpoint \\
        --save-hf-path /path/to/output \\
        --save-hf true \\
        --yaml-file /path/to/mapping.yaml \\
        --tokenizer-type HFTokenizer \\
        --hf-tokenizer-path /path/to/tokenizer
"""

import os
import sys
import json
from pathlib import Path
import logging
logging.basicConfig(level=logging.WARNING)

# Add project root and tools to Python path
_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_file_dir)))
_tools_root = os.path.dirname(_file_dir)  # tools/dist_checkpoint
for _p in [_project_root, os.path.join(_project_root, 'tools')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.distributed as dist

from megatron.core.enums import ModelType
from megatron.training.training import get_model
from megatron.training import print_rank_0

from loongforge.train.parser import parse_train_args
from loongforge.train.initialize import initialize_loongforge_megatron
from loongforge.models.foundation.llm_model_provider import llm_model_provider
from loongforge.models.omni_models.omni_model_provider import omni_model_provider
from loongforge.utils import get_model_config

from dist_checkpoint.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online
from dist_checkpoint.checkpoint.hf_checkpoint_saver import save_hf_checkpoint_online


# ---------------------------------------------------------------------------
# Index helpers  (read model.safetensors.index.json or pytorch_model.bin.index.json)
# ---------------------------------------------------------------------------

def _read_weight_map(hf_path: Path) -> dict:
    """Read weight_map {tensor_key: filename} from an HF index file.

    Priority: safetensors index > pytorch_model.bin index > single-file fallback
    (for small models without a sharded index).
    """
    for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index_file = hf_path / index_name
        if index_file.exists():
            with open(index_file) as f:
                return json.load(f)['weight_map']

    # Small models: single safetensors / pytorch_model.bin with no index file
    single_st = hf_path / 'model.safetensors'
    if single_st.exists():
        from safetensors import safe_open
        with safe_open(str(single_st), framework='pt', device='cpu') as f:
            return {k: 'model.safetensors' for k in f.keys()}

    single_bin = hf_path / 'pytorch_model.bin'
    if single_bin.exists():
        sd = torch.load(str(single_bin), map_location='cpu')
        return {k: 'pytorch_model.bin' for k in sd.keys()}

    raise FileNotFoundError(f"No weight index or weight file found in {hf_path}")


def _load_shard(hf_path: Path, filename: str) -> dict:
    """Load a single shard file and return {key: tensor}."""
    fpath = hf_path / filename
    if filename.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(str(fpath), device='cpu')
    else:
        return torch.load(str(fpath), map_location='cpu')


def _compare_shard(orig_path_str: str, rt_path_str: str,
                   base_fname: str, keys_in_shard: list,
                   rt_weight_map: dict,
                   verbose: bool = True) -> dict:
    """Compare one baseline shard against its roundtrip counterpart(s).

    Designed as a top-level function so it is picklable for thread/process pools.
    Returns per-shard statistics that the caller aggregates.

    Args:
        verbose: If True, output detailed logs for mismatched keys
    """
    from collections import defaultdict

    orig_path = Path(orig_path_str)
    rt_path   = Path(rt_path_str)

    base_shard = _load_shard(orig_path, base_fname)

    keys_by_rt_shard = defaultdict(list)
    for k in keys_in_shard:
        keys_by_rt_shard[rt_weight_map[k]].append(k)

    shape_mismatches = []
    num_exact = num_close = num_diff = 0
    diff_keys = []
    diff_details = []  # Store detailed diff info for verbose output
    local_max = 0.0
    local_sum = 0.0
    local_n   = 0

    for rt_fname, rt_keys in keys_by_rt_shard.items():
        rt_shard = _load_shard(rt_path, rt_fname)

        for key in rt_keys:
            b = base_shard[key].float()
            r = rt_shard[key].float()

            if b.shape != r.shape:
                shape_mismatches.append({
                    'key': key,
                    'baseline': tuple(b.shape),
                    'roundtrip': tuple(r.shape),
                })
                if verbose:
                    print(f"  [SHAPE MISMATCH] {key}: baseline={b.shape} vs roundtrip={r.shape}", flush=True)
                continue

            diff = torch.abs(b - r)
            local_max  = max(local_max, diff.max().item())
            local_sum += diff.sum().item()
            local_n   += diff.numel()

            # Use tolerance that captures typical bfloat16 precision differences (~1e-5 to 1e-4)
            if torch.allclose(b, r, rtol=1e-4, atol=1e-6):
                num_exact += 1
            elif torch.allclose(b, r, rtol=1e-3, atol=1e-4):
                num_close += 1
                if verbose:
                    # Calculate detailed stats for this key
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    std_diff = diff.std().item()
                    print(f"  [CLOSE] {key}: max={max_diff:.6e}, mean={mean_diff:.6e}, std={std_diff:.6e}, shape={tuple(b.shape)}", flush=True)
                    print(f"          baseline: min={b.min():.6f}, max={b.max():.6f}, mean={b.mean():.6f}", flush=True)
                    print(f"          roundtrip: min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}", flush=True)
            else:
                num_diff += 1
                diff_keys.append(key)
                if verbose:
                    # Calculate detailed stats for this key
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    std_diff = diff.std().item()
                    # Get some sample values with largest differences
                    flat_diff = diff.flatten()
                    top_vals, top_idx = torch.topk(flat_diff, min(5, flat_diff.numel()))
                    max_pos = (diff == diff.max()).nonzero(as_tuple=True)
                    detail = {
                        'key': key,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'std_diff': std_diff,
                        'shape': tuple(b.shape),
                        'baseline_stats': f"min={b.min():.6f}, max={b.max():.6f}, mean={b.mean():.6f}",
                        'roundtrip_stats': f"min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}",
                    }
                    diff_details.append(detail)
                    print(f"  [DIFF] {key}: max={max_diff:.6e}, mean={mean_diff:.6e}, std={std_diff:.6e}, shape={tuple(b.shape)}", flush=True)
                    print(f"          baseline: {detail['baseline_stats']}", flush=True)
                    print(f"          roundtrip: {detail['roundtrip_stats']}", flush=True)

            del b, r, diff

        del rt_shard

    del base_shard

    return {
        'shape_mismatches': shape_mismatches,
        'num_exact': num_exact,
        'num_close': num_close,
        'num_diff': num_diff,
        'diff_keys': diff_keys,
        'diff_details': diff_details,  # Include detailed diff info
        'local_max': local_max,
        'local_sum': local_sum,
        'local_n':   local_n,
    }


# ---------------------------------------------------------------------------
# Comparison  (parallel shard-by-shard via ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def compare_weights(original_path: str, roundtrip_path: str, output_dir: str,
                    num_workers: int = 4) -> dict:
    """Parallel stream-compare using HF index file.

    Uses ThreadPoolExecutor (safe with CUDA-initialized processes, GIL released
    by both disk I/O and torch C++ operations).  num_workers controls how many
    shards are loaded and compared concurrently.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("=" * 80, flush=True)
    print("Comparing Weights", flush=True)
    print("=" * 80, flush=True)

    orig_path = Path(original_path)
    rt_path   = Path(roundtrip_path)

    # ── Step 1: Read index (JSON parse only, near-instant) ────────────────
    print(f"  Reading baseline index  : {orig_path}", flush=True)
    base_weight_map = _read_weight_map(orig_path)

    print(f"  Reading roundtrip index : {rt_path}", flush=True)
    rt_weight_map   = _read_weight_map(rt_path)

    baseline_keys  = set(base_weight_map.keys())
    roundtrip_keys = set(rt_weight_map.keys())
    common_keys    = sorted(baseline_keys & roundtrip_keys)
    missing_keys   = sorted(baseline_keys - roundtrip_keys)
    extra_keys     = sorted(roundtrip_keys - baseline_keys)

    print(f"  Baseline  : {len(baseline_keys)} tensors", flush=True)
    print(f"  Roundtrip : {len(roundtrip_keys)} tensors", flush=True)
    print(f"  Common    : {len(common_keys)}", flush=True)
    if missing_keys:
        print(f"  Missing   : {len(missing_keys)}", flush=True)
        # Detailed logs for missing keys
        for key in missing_keys[:20]:
            print(f"    [MISSING] {key} -> {base_weight_map[key]}", flush=True)
        if len(missing_keys) > 20:
            print(f"    ... and {len(missing_keys) - 20} more", flush=True)
    if extra_keys:
        print(f"  Extra     : {len(extra_keys)}", flush=True)
        # Detailed logs for extra keys
        for key in extra_keys[:20]:
            print(f"    [EXTRA] {key} -> {rt_weight_map[key]}", flush=True)
        if len(extra_keys) > 20:
            print(f"    ... and {len(extra_keys) - 20} more", flush=True)

    # ── Step 2: Build per-shard task list ─────────────────────────────────
    keys_by_base_shard = defaultdict(list)
    for key in common_keys:
        keys_by_base_shard[base_weight_map[key]].append(key)

    shard_list   = sorted(keys_by_base_shard.items())   # [(fname, [keys]), ...]
    total_shards = len(shard_list)
    rt_map_copy  = dict(rt_weight_map)   # plain dict for thread safety

    # ── Step 3: Parallel comparison ───────────────────────────────────────
    shape_mismatches = []
    num_exact = num_close = num_diff = 0
    diff_keys   = []
    running_max = 0.0
    running_sum = 0.0
    running_n   = 0
    completed   = 0

    print(f"  Workers   : {num_workers}  |  Shards: {total_shards}", flush=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_fname = {
            executor.submit(
                _compare_shard,
                str(orig_path), str(rt_path),
                base_fname, keys_in_shard, rt_map_copy, True,  # verbose=True
            ): base_fname
            for base_fname, keys_in_shard in shard_list
        }

        for future in as_completed(future_to_fname):
            base_fname = future_to_fname[future]
            r = future.result()
            completed += 1
            print(f"  [{completed:3d}/{total_shards}] {base_fname}  "
                  f"(exact={r['num_exact']} close={r['num_close']} diff={r['num_diff']})",
                  flush=True)

            shape_mismatches.extend(r['shape_mismatches'])
            num_exact += r['num_exact']
            num_close += r['num_close']
            num_diff  += r['num_diff']
            diff_keys.extend(r['diff_keys'])

            running_max  = max(running_max, r['local_max'])
            running_sum += r['local_sum']
            running_n   += r['local_n']

    running_mean = running_sum / running_n if running_n > 0 else 0.0

    # ── Step 4: Summary ───────────────────────────────────────────────────
    print("", flush=True)
    print(f"  Shape mismatches : {len(shape_mismatches)}", flush=True)
    print(f"  Exact matches    : {num_exact}", flush=True)
    print(f"  Close matches    : {num_close}", flush=True)
    print(f"  Different        : {num_diff}", flush=True)
    print(f"  Max  |diff|      : {running_max:.2e}", flush=True)
    print(f"  Mean |diff|      : {running_mean:.2e}", flush=True)
    if diff_keys:
        print(f"  Mismatched keys  : (first 10) {diff_keys[:10]}", flush=True)

    passed = (num_diff == 0 and not missing_keys and not extra_keys and not shape_mismatches)
    if passed:
        print("✅ PASSED: All weights match!", flush=True)
    else:
        print("❌ FAILED: Weight mismatches detected!", flush=True)

    result = {
        'passed': passed,
        'num_baseline': len(baseline_keys),
        'num_roundtrip': len(roundtrip_keys),
        'num_common': len(common_keys),
        'missing_keys': missing_keys,
        'extra_keys': extra_keys,
        'shape_mismatches': shape_mismatches,
        'num_exact_matches': num_exact,
        'num_close_matches': num_close,
        'num_different': num_diff,
        'mismatched_keys': diff_keys,
        'max_abs_diff': float(running_max),
        'mean_abs_diff': float(running_mean),
    }

    report_path = Path(output_dir) / "roundtrip_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Report saved to  : {report_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Step 1: Parse args (identical to training entry) ──────────────────
    args = parse_train_args()

    # ── Step 2: Initialize Megatron (identical to training) ───────────────
    initialize_loongforge_megatron(args=args)

    print_rank_0("=" * 80)
    print_rank_0("HF Checkpoint Roundtrip Test v2")
    print_rank_0("=" * 80)
    print_rank_0(f"  Original HF : {args.load}")
    print_rank_0(f"  Output dir  : {args.save_hf_path}")

    # ── Step 3: Build model (identical to training) ───────────────────────
    # Using the real Megatron model guarantees that model.state_dict() has
    # perfectly consistent keys across all TP ranks — the root cause of the
    # gather deadlock in v1.
    print_rank_0("=" * 80)
    print_rank_0("Phase 1: Build Model")
    print_rank_0("=" * 80)
    model_config = get_model_config()
    is_vlm = True
    for name in ["image_encoder", "image_projector"]:
        if not hasattr(model_config, name):
            is_vlm = False
            break
    if is_vlm:
        model = get_model(omni_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    else:
        model = get_model(llm_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # ── Step 4: Load HF checkpoint (identical to training) ────────────────
    print_rank_0("=" * 80)
    print_rank_0("Phase 2: Load HF Checkpoint")
    print_rank_0("=" * 80)

    # Profiler for performance analysis
    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    pr.enable()
    load_hf_checkpoint_online(model, None, None, args)
    pr.disable()

    # Print profiler results (rank 0 only)
    if dist.get_rank() == 0:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(40)  # Print top 40 lines
        print(s.getvalue())

        # Print callers/callees
        s2 = io.StringIO()
        ps2 = pstats.Stats(pr, stream=s2).sort_stats('cumulative')
        ps2.print_callers(30)  # Print caller relationships
        print("\n=== Callers ===")
        print(s2.getvalue())
        # Save to file
        ps.dump_stats("profile_stats.prof")
        print("Profile saved to profile_stats.prof")

    dist.barrier()

    # ── Step 5: Save back to HF (identical to training) ───────────────────
    print_rank_0("=" * 80)
    print_rank_0("Phase 3: Save to HF Format")
    print_rank_0("=" * 80)
    save_hf_checkpoint_online(model, args)

    dist.barrier()

    # ── Step 6: Compare (rank 0 only, CPU) ────────────────────────────────
    print_rank_0("=" * 80)
    print_rank_0("Phase 4: Compare")
    print_rank_0("=" * 80)
    if dist.get_rank() == 0:
        compare_weights(args.load, args.save_hf_path, args.save_hf_path)

    dist.barrier()

    print_rank_0("=" * 80)
    print_rank_0("Roundtrip Test Finished")
    print_rank_0("=" * 80)


if __name__ == '__main__':
    main()
