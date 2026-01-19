#!/usr/bin/env python3
"""
Compare differences between two Megatron checkpoint directories
Usage: python compare_checkpoints.py <ckpt_dir1> <ckpt_dir2>
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def compare_values(v1, v2):
    """Compare two values, handle Tensor types"""
    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        if v1.shape != v2.shape:
            return False, f"shape mismatch: {v1.shape} vs {v2.shape}"
        if torch.equal(v1, v2):
            return True, "identical"
        else:
            max_diff = (v1 - v2).abs().max().item()
            return False, f"max diff: {max_diff}"
    elif type(v1) != type(v2):
        return False, f"type mismatch: {type(v1).__name__} vs {type(v2).__name__}"
    else:
        try:
            if v1 == v2:
                return True, "identical"
            else:
                return False, "different"
        except Exception:
            return False, "comparison failed"


def compare_model_dict(model1: dict, model2: dict, name: str = "model", ignore_extra_state: bool = False):
    """Compare model weight dictionaries"""
    print(f"\n[{name} Weights Comparison]")
    
    keys1 = set(model1.keys())
    keys2 = set(model2.keys())
    
    # If ignoring _extra_state, filter out from keys
    if ignore_extra_state:
        keys1 = {k for k in keys1 if '_extra_state' not in k}
        keys2 = {k for k in keys2 if '_extra_state' not in k}
        print(f"  (ignoring _extra_state fields)")
    
    if keys1 != keys2:
        print(f"  WARNING: Different {name} keys!")
        if keys1 - keys2:
            print(f"  Only in File1: {keys1 - keys2}")
        if keys2 - keys1:
            print(f"  Only in File2: {keys2 - keys1}")
    
    common_keys = keys1 & keys2
    all_same = True
    diff_weights = []
    
    for k in sorted(common_keys):
        t1 = model1[k]
        t2 = model2[k]
        
        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
            if t1.shape != t2.shape:
                all_same = False
                diff_weights.append((k, f"shape mismatch: {t1.shape} vs {t2.shape}"))
            elif not torch.equal(t1, t2):
                all_same = False
                max_diff = (t1 - t2).abs().max().item()
                diff_weights.append((k, f"max diff: {max_diff}"))
        else:
            is_same, reason = compare_values(t1, t2)
            if not is_same:
                all_same = False
                diff_weights.append((k, reason))
    
    if all_same:
        print(f"  ✓ All {len(common_keys)} {name} weights are IDENTICAL!")
    else:
        print(f"  ✗ Found {len(diff_weights)} different weights:")
        for wname, reason in diff_weights:
            print(f"    - {wname}: {reason}")
    
    return all_same


def compare_checkpoint_files(file1: str, file2: str, verbose: bool = False, ignore_extra_state: bool = False):
    """Compare two checkpoint files"""
    print(f"\n{'='*60}")
    print(f"Comparing:")
    print(f"  File1: {file1}")
    print(f"  File2: {file2}")
    print('='*60)

    # Check file size
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    print(f"\n[File Size]")
    print(f"  File1: {size1:,} bytes")
    print(f"  File2: {size2:,} bytes")
    print(f"  Diff:  {size2 - size1:,} bytes")

    # Load checkpoint
    print(f"\n[Loading checkpoints...]")
    ckpt1 = torch.load(file1, map_location='cpu', weights_only=False)
    ckpt2 = torch.load(file2, map_location='cpu', weights_only=False)

    # Compare top-level keys
    print(f"\n[Top-level Keys]")
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())
    print(f"  File1 keys: {sorted(keys1)}")
    print(f"  File2 keys: {sorted(keys2)}")
    
    if keys1 != keys2:
        print(f"  Only in File1: {keys1 - keys2}")
        print(f"  Only in File2: {keys2 - keys1}")

    # Compare differences in args
    if 'args' in ckpt1 and 'args' in ckpt2:
        print(f"\n[Args Differences]")
        args1 = ckpt1['args']
        args2 = ckpt2['args']
        
        # Convert to dictionary
        if hasattr(args1, '__dict__'):
            args1 = vars(args1)
        if hasattr(args2, '__dict__'):
            args2 = vars(args2)
        
        all_keys = set(args1.keys()) | set(args2.keys())
        diff_count = 0
        for k in sorted(all_keys):
            v1 = args1.get(k)
            v2 = args2.get(k)
            if v1 != v2:
                diff_count += 1
                print(f"  {k}:")
                print(f"    File1: {v1}")
                print(f"    File2: {v2}")
        
        if diff_count == 0:
            print("  No differences in args")
        else:
            print(f"\n  Total args differences: {diff_count}")

    # Compare model weights (support model, model0, model1, ... and other multi-model fields)
    model_keys = [k for k in (keys1 & keys2) if k.startswith('model')]
    for model_key in sorted(model_keys):
        compare_model_dict(ckpt1[model_key], ckpt2[model_key], model_key, ignore_extra_state)

    # Compare other fields
    other_keys = (keys1 & keys2) - {'args'} - set(model_keys)
    if other_keys:
        print(f"\n[Other Fields]")
        for key in sorted(other_keys):
            v1 = ckpt1[key]
            v2 = ckpt2[key]
            is_same, reason = compare_values(v1, v2)
            if is_same:
                print(f"  {key}: identical")
            else:
                print(f"  {key}: DIFFERENT ({reason})")
                if verbose:
                    print(f"    File1: {v1}")
                    print(f"    File2: {v2}")


def compare_checkpoint_dirs(dir1: str, dir2: str, verbose: bool = False, ignore_extra_state: bool = False):
    """Compare two checkpoint directories"""
    print(f"Comparing checkpoint directories:")
    print(f"  Dir1: {dir1}")
    print(f"  Dir2: {dir2}")
    
    # Find all .pt files
    pt_files1 = set()
    pt_files2 = set()
    
    for root, dirs, files in os.walk(dir1):
        for f in files:
            if f.endswith('.pt'):
                rel_path = os.path.relpath(os.path.join(root, f), dir1)
                pt_files1.add(rel_path)
    
    for root, dirs, files in os.walk(dir2):
        for f in files:
            if f.endswith('.pt'):
                rel_path = os.path.relpath(os.path.join(root, f), dir2)
                pt_files2.add(rel_path)
    
    print(f"\n[PT Files]")
    print(f"  Dir1 has {len(pt_files1)} .pt files")
    print(f"  Dir2 has {len(pt_files2)} .pt files")
    
    if pt_files1 != pt_files2:
        print(f"  Only in Dir1: {pt_files1 - pt_files2}")
        print(f"  Only in Dir2: {pt_files2 - pt_files1}")
    
    # Compare common files
    common_files = sorted(pt_files1 & pt_files2)
    print(f"\n  Comparing {len(common_files)} common files...")
    
    for rel_path in common_files:
        file1 = os.path.join(dir1, rel_path)
        file2 = os.path.join(dir2, rel_path)
        compare_checkpoint_files(file1, file2, verbose, ignore_extra_state)


def main():
    parser = argparse.ArgumentParser(description='Compare two Megatron checkpoints')
    parser.add_argument('path1', help='First checkpoint path (file or directory)')
    parser.add_argument('path2', help='Second checkpoint path (file or directory)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--ignore-extra-state', action='store_true', 
                        help='Ignore _extra_state fields (FP8 scaling factors, etc.)')
    parser.add_argument('--rank', type=str, default=None, 
                        help='Only compare specific rank, e.g., "00_000"')
    
    args = parser.parse_args()
    
    path1 = args.path1
    path2 = args.path2
    
    # If rank is specified, adjust path
    if args.rank:
        path1 = os.path.join(path1, f'mp_rank_{args.rank}', 'model_optim_rng.pt')
        path2 = os.path.join(path2, f'mp_rank_{args.rank}', 'model_optim_rng.pt')
    
    if os.path.isfile(path1) and os.path.isfile(path2):
        compare_checkpoint_files(path1, path2, args.verbose, args.ignore_extra_state)
    elif os.path.isdir(path1) and os.path.isdir(path2):
        compare_checkpoint_dirs(path1, path2, args.verbose, args.ignore_extra_state)
    else:
        print(f"Error: Both paths must be either files or directories")
        print(f"  path1: {path1} ({'file' if os.path.isfile(path1) else 'dir' if os.path.isdir(path1) else 'not found'})")
        print(f"  path2: {path2} ({'file' if os.path.isfile(path2) else 'dir' if os.path.isdir(path2) else 'not found'})")
        sys.exit(1)


if __name__ == '__main__':
    main()