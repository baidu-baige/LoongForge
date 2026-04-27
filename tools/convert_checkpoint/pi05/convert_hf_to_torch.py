# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Convert HuggingFace safetensors checkpoint to PyTorch (single-file) format.

This script converts HuggingFace model weights from .safetensors format to a single
PyTorch checkpoint file (.pt) compatible with Megatron-LM training.

Usage:
    # Basic conversion
    python convert_hf_to_torch.py \
        --input /path/to/hf_model_dir \
        --output /path/to/output/mp_rank_00/model_optim_rng.pt

    # With dtype conversion and Megatron format
    python convert_hf_to_torch.py \
        --input /path/to/hf_model_dir \
        --output /path/to/output/mp_rank_00/model_optim_rng.pt \
        --dtype fp32 \
        --prefix-add "model." \
        --megatron-format

    # Dry run to check keys without saving
    python convert_hf_to_torch.py \
        --input /path/to/hf_model_dir \
        --dry-run
"""

import argparse
import glob
import os
import sys
from collections import OrderedDict

import torch

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors library required. Install with: pip install safetensors")
    sys.exit(1)


def find_safetensor_files(input_path: str) -> list[str]:
    """Find all safetensors files in the given path, sorted by name."""
    if os.path.isfile(input_path):
        if not input_path.endswith(".safetensors"):
            print(f"WARNING: File {input_path} is not in .safetensors format")
        return [input_path]

    if os.path.isdir(input_path):
        patterns = [
            os.path.join(input_path, "*.safetensors"),
            os.path.join(input_path, "**", "*.safetensors"),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))
        # Deduplicate and sort
        files = sorted(set(files))
        if not files:
            print(f"ERROR: No .safetensors files found in {input_path}")
            sys.exit(1)
        return files

    print(f"ERROR: {input_path} does not exist")
    sys.exit(1)


def load_safetensors(files: list[str], device: str = "cpu") -> OrderedDict:
    """Load and merge state_dict from one or more safetensors files."""
    merged = OrderedDict()
    total_files = len(files)

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        print(f"  [{idx}/{total_files}] Loading {fname} ...")

        with safe_open(fpath, framework="pt", device=device) as f:
            for key in f.keys():
                if key in merged:
                    print(f"  WARNING: Duplicate key '{key}', later file will overwrite")
                merged[key] = f.get_tensor(key)

    return merged


def print_summary(state_dict: OrderedDict, show_all: bool = False):
    """Print state_dict summary."""
    total_params = len(state_dict)
    total_elements = 0
    dtype_counts: dict[str, int] = {}

    for name, tensor in state_dict.items():
        total_elements += tensor.numel()
        dt = str(tensor.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    print(f"\n{'='*60}")
    print("State Dict Summary:")
    print(f"{'='*60}")
    print(f"  Number of tensors:    {total_params}")
    print(f"  Total elements:       {total_elements:,}")
    print(f"  BF16 size estimate:   {total_elements * 2 / 1024**3:.2f} GB")
    print(f"  FP32 size estimate:   {total_elements * 4 / 1024**3:.2f} GB")
    print(f"  Data type distribution: {dtype_counts}")

    if show_all:
        print(f"\n{'='*60}")
        print("All Parameters:")
        print(f"{'='*60}")
        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            print(f"  {name:80s}  {str(t.shape):>30s}  {t.dtype}")


def build_arg_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace safetensors checkpoint to Megatron DDP format"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to safetensors file or directory containing multiple shards"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .pt file path (default: <input_dir>/model_weights.pt)"
    )
    parser.add_argument(
        "--dtype", type=str, default=None, choices=["fp32", "fp16", "bf16"],
        help="Target data type (default: preserve original)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print key list and summary only, do not save"
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print all parameter names and shapes"
    )
    parser.add_argument(
        "--prefix-remove", type=str, default=None,
        help="Remove prefix from parameter names, e.g. 'model.' changes 'model.layer1.weight' to 'layer1.weight'"
    )
    parser.add_argument(
        "--prefix-add", type=str, default=None,
        help="Add prefix to all parameter names, e.g. 'module.' changes 'layer1.weight' to 'module.layer1.weight'"
    )
    parser.add_argument(
        "--megatron-format", action="store_true",
        help="Output in Megatron checkpoint format: {'model': state_dict, 'iteration': 0, ...}, "
             "compatible with Megatron load_checkpoint"
    )
    parser.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration value for --megatron-format checkpoint (default: 0)"
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Set default output path
    if args.output is None:
        if os.path.isdir(args.input):
            args.output = os.path.join(args.input, "model_weights.pt")
        else:
            args.output = args.input.replace(".safetensors", ".pt")

    # 1. Find all safetensors files
    print(f"Scanning input path: {args.input}")
    files = find_safetensor_files(args.input)
    print(f"Found {len(files)} safetensors file(s):")
    for f in files:
        size_mb = os.path.getsize(f) / 1024**2
        print(f"  {os.path.basename(f):50s}  ({size_mb:.1f} MB)")

    # 2. Load and merge
    print("\nLoading weights ...")
    state_dict = load_safetensors(files)
    print(f"Loading complete, {len(state_dict)} parameters loaded")

    # 3. Process prefix
    if args.prefix_remove:
        prefix = args.prefix_remove
        new_sd = OrderedDict()
        removed = 0
        for key, val in state_dict.items():
            if key.startswith(prefix):
                new_sd[key[len(prefix):]] = val
                removed += 1
            else:
                new_sd[key] = val
        state_dict = new_sd
        print(f"Removed prefix '{prefix}': affected {removed} parameters")

    if args.prefix_add:
        prefix = args.prefix_add
        new_sd = OrderedDict()
        for key, val in state_dict.items():
            new_sd[prefix + key] = val
        state_dict = new_sd
        print(f"Added prefix '{prefix}': affected {len(state_dict)} parameters")

    # 4. Optional: dtype conversion
    if args.dtype:
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        target_dtype = dtype_map[args.dtype]
        print(f"Converting data type to {args.dtype} ...")
        for key in state_dict:
            if state_dict[key].is_floating_point():
                state_dict[key] = state_dict[key].to(target_dtype)

    # 5. Print summary
    print_summary(state_dict, show_all=args.show_all or args.dry_run)

    # 6. Save
    if args.dry_run:
        print("\n[dry-run] Not saving file")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        if args.megatron_format:
            # Wrap into Megatron load_checkpoint expected format:
            #   state_dict['model']      -> model weights
            #   state_dict['iteration']  -> training step
            #   state_dict['checkpoint_version'] -> version
            save_obj = {
                'model': state_dict,
                'iteration': args.iteration,
                'checkpoint_version': 3.0,
                'num_floating_point_operations_so_far': 0,
            }
            print(f"\nSaving as Megatron format (iteration={args.iteration}): {args.output} ...")
        else:
            save_obj = state_dict
            print(f"\nSaving to: {args.output} ...")

        torch.save(save_obj, args.output)
        size_gb = os.path.getsize(args.output) / 1024**3
        print(f"Save complete! File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
