# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""
HF Checkpoint Comparison Utilities

Provides tools for comparing two HF checkpoints and generating detailed reports.
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import torch


@dataclass
class ComparisonMetrics:
    """Metrics from checkpoint comparison"""
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
    largest_diffs: dict = None  # Top 10 differences by key

    def __post_init__(self):
        if self.missing_keys is None:
            self.missing_keys = []
        if self.extra_keys is None:
            self.extra_keys = []
        if self.shape_mismatches is None:
            self.shape_mismatches = []
        if self.largest_diffs is None:
            self.largest_diffs = {}


def load_hf_weights(hf_path: str) -> Dict[str, torch.Tensor]:
    """
    Load weights from HF checkpoint directory

    Args:
        hf_path: Path to HF checkpoint directory

    Returns:
        Dictionary of weight tensors
    """
    weights = {}
    hf_path = Path(hf_path)

    # Try safetensors first
    safetensors_files = sorted([
        f for f in os.listdir(hf_path)
        if f.endswith('.safetensors') and not f.startswith('model-')
    ])

    if safetensors_files:
        try:
            from safetensors.torch import load_file
            for sf_file in safetensors_files:
                file_path = hf_path / sf_file
                file_weights = load_file(file_path)
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
            file_path = hf_path / pt_file
            file_weights = torch.load(file_path, map_location='cpu')
            weights.update(file_weights)

    if not weights:
        raise FileNotFoundError(f"No weight files found in {hf_path}")

    return weights


def compare_checkpoints(
    baseline_path: str,
    roundtrip_path: str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> ComparisonMetrics:
    """
    Compare two HF checkpoints

    Args:
        baseline_path: Path to baseline HF checkpoint (or pkl file)
        roundtrip_path: Path to roundtripped HF checkpoint
        atol: Absolute tolerance for close match
        rtol: Relative tolerance for close match

    Returns:
        ComparisonMetrics object with detailed comparison results
    """
    metrics = ComparisonMetrics()

    # Load baseline
    if baseline_path.endswith('.pkl'):
        baseline_weights = torch.load(baseline_path, map_location='cpu')
    else:
        baseline_weights = load_hf_weights(baseline_path)

    metrics.num_baseline = len(baseline_weights)

    # Load roundtripped
    if roundtrip_path.endswith('.pkl'):
        roundtrip_weights = torch.load(roundtrip_path, map_location='cpu')
    else:
        roundtrip_weights = load_hf_weights(roundtrip_path)

    metrics.num_roundtrip = len(roundtrip_weights)

    # Compare keys
    baseline_keys = set(baseline_weights.keys())
    roundtrip_keys = set(roundtrip_weights.keys())

    metrics.identical_keys = len(baseline_keys & roundtrip_keys)
    metrics.missing_keys = sorted(list(baseline_keys - roundtrip_keys))
    metrics.extra_keys = sorted(list(roundtrip_keys - baseline_keys))

    # Compare values
    all_diffs = []
    largest_diffs = {}

    for key in sorted(baseline_keys & roundtrip_keys):
        baseline_tensor = baseline_weights[key]
        roundtrip_tensor = roundtrip_weights[key]

        # Check shape
        if baseline_tensor.shape != roundtrip_tensor.shape:
            metrics.shape_mismatches.append({
                'key': key,
                'baseline': tuple(baseline_tensor.shape),
                'roundtrip': tuple(roundtrip_tensor.shape),
            })
            continue

        # Check values
        diff = torch.abs(baseline_tensor.float() - roundtrip_tensor.float())
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        largest_diffs[key] = {
            'max': max_diff,
            'mean': mean_diff,
            'numel': baseline_tensor.numel(),
        }

        # Categorize
        if torch.allclose(baseline_tensor.float(), roundtrip_tensor.float(), rtol=rtol, atol=atol):
            metrics.num_exact_matches += 1
        elif torch.allclose(baseline_tensor.float(), roundtrip_tensor.float(), rtol=1e-3, atol=1e-5):
            metrics.num_close_matches += 1
        else:
            metrics.num_different += 1

        all_diffs.append(diff)

    # Compute overall statistics
    if all_diffs:
        all_diffs_tensor = torch.cat([d.flatten() for d in all_diffs])
        metrics.max_abs_diff = all_diffs_tensor.max().item()
        metrics.mean_abs_diff = all_diffs_tensor.mean().item()

    # Keep top 10 largest differences
    sorted_diffs = sorted(
        largest_diffs.items(),
        key=lambda x: x[1]['max'],
        reverse=True
    )
    metrics.largest_diffs = dict(sorted_diffs[:10])

    return metrics


def save_comparison_report(
    metrics: ComparisonMetrics,
    output_path: str,
    verbose: bool = True
) -> None:
    """
    Save comparison report to JSON

    Args:
        metrics: ComparisonMetrics object
        output_path: Path to save JSON report
        verbose: Whether to print summary to stdout
    """
    # Convert to dict for JSON serialization
    report_dict = asdict(metrics)
    report_dict['max_abs_diff'] = float(report_dict['max_abs_diff'])
    report_dict['mean_abs_diff'] = float(report_dict['mean_abs_diff'])

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)

    if verbose:
        print(f"Report saved to {output_path}")


def print_comparison_summary(metrics: ComparisonMetrics) -> None:
    """
    Print human-readable summary of comparison

    Args:
        metrics: ComparisonMetrics object
    """
    print("=" * 80)
    print("CHECKPOINT COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Baseline tensors:        {metrics.num_baseline}")
    print(f"Roundtrip tensors:       {metrics.num_roundtrip}")
    print(f"Identical keys:          {metrics.identical_keys}")
    print(f"Missing keys:            {len(metrics.missing_keys)}")
    print(f"Extra keys:              {len(metrics.extra_keys)}")
    print(f"Shape mismatches:        {len(metrics.shape_mismatches)}")
    print()
    print(f"Exact matches:           {metrics.num_exact_matches}")
    print(f"Close matches (1e-3):    {metrics.num_close_matches}")
    print(f"Different:               {metrics.num_different}")
    print()
    print(f"Max absolute difference: {metrics.max_abs_diff:.2e}")
    print(f"Mean absolute difference:{metrics.mean_abs_diff:.2e}")
    print("=" * 80)

    if metrics.missing_keys:
        print(f"\nMissing keys ({len(metrics.missing_keys)}):")
        for key in metrics.missing_keys[:10]:
            print(f"  - {key}")
        if len(metrics.missing_keys) > 10:
            print(f"  ... and {len(metrics.missing_keys) - 10} more")

    if metrics.extra_keys:
        print(f"\nExtra keys ({len(metrics.extra_keys)}):")
        for key in metrics.extra_keys[:10]:
            print(f"  - {key}")
        if len(metrics.extra_keys) > 10:
            print(f"  ... and {len(metrics.extra_keys) - 10} more")

    if metrics.shape_mismatches:
        print(f"\nShape mismatches ({len(metrics.shape_mismatches)}):")
        for item in metrics.shape_mismatches[:5]:
            print(f"  - {item['key']}: {item['baseline']} vs {item['roundtrip']}")
        if len(metrics.shape_mismatches) > 5:
            print(f"  ... and {len(metrics.shape_mismatches) - 5} more")

    if metrics.largest_diffs:
        print(f"\nLargest value differences:")
        for key, diff_info in sorted(
            metrics.largest_diffs.items(),
            key=lambda x: x[1]['max'],
            reverse=True
        )[:5]:
            print(f"  - {key}: max={diff_info['max']:.2e}, mean={diff_info['mean']:.2e}")

    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare two HF checkpoints')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline checkpoint (dir or pkl)')
    parser.add_argument('--roundtrip', type=str, required=True,
                        help='Path to roundtripped checkpoint (dir or pkl)')
    parser.add_argument('--output', type=str, default='./comparison_report.json',
                        help='Output path for comparison report')
    parser.add_argument('--atol', type=float, default=1e-8,
                        help='Absolute tolerance for exact match')
    parser.add_argument('--rtol', type=float, default=1e-5,
                        help='Relative tolerance for exact match')

    args = parser.parse_args()

    # Run comparison
    print(f"Comparing {args.baseline} vs {args.roundtrip}...")
    metrics = compare_checkpoints(
        args.baseline,
        args.roundtrip,
        atol=args.atol,
        rtol=args.rtol,
    )

    # Print summary
    print_comparison_summary(metrics)

    # Save report
    save_comparison_report(metrics, args.output)
