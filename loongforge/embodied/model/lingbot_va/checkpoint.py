# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Streaming sharded-safetensors loader for the PyTorch LingBot-VA model."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

_IGNORED_UNEXPECTED = {"patch_embedding.weight", "patch_embedding.bias"}


def _transformer_directory(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    transformer = root / "transformer"
    return transformer if transformer.is_dir() else root


def _read_index(directory: Path):
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise FileNotFoundError(
            f"Expected one sharded safetensors index in {directory}, found {len(indexes)}"
        )
    with indexes[0].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {indexes[0]}")
    return indexes[0], weight_map


def _write_report(report: Dict) -> None:
    report_dir = os.environ.get("LINGBOT_CHECKPOINT_REPORT_DIR")
    if not report_dir:
        return
    output_dir = Path(report_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lingbot_va_checkpoint_report.json"
    report["report_path"] = str(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def load_sharded_safetensors(model: torch.nn.Module, path: str) -> Dict:
    """Load checkpoint tensors shard by shard without retaining a merged state dict."""
    directory = _transformer_directory(path)
    index_path, weight_map = _read_index(directory)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    persistent_names = set(model.state_dict().keys())
    expected = {
        name: tensor
        for name, tensor in {**parameters, **buffers}.items()
        if name in persistent_names
    }
    loaded = set()
    unexpected = []
    ignored_unexpected = []
    shape_mismatch = []
    shard_weights = defaultdict(list)
    for name, shard in weight_map.items():
        shard_weights[shard].append(name)

    with torch.no_grad():
        for shard_name, names in sorted(shard_weights.items()):
            shard_path = directory / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint shard listed by index is missing: {shard_path}"
                )
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                available = set(shard.keys())
                for name in names:
                    if name not in available:
                        unexpected.append(
                            {
                                "name": name,
                                "reason": "listed in index but absent from shard",
                            }
                        )
                        continue
                    if name in _IGNORED_UNEXPECTED and name not in expected:
                        ignored_unexpected.append(name)
                        continue
                    destination = expected.get(name)
                    if destination is None:
                        unexpected.append(
                            {
                                "name": name,
                                "shape": list(shard.get_slice(name).get_shape()),
                            }
                        )
                        continue
                    source = shard.get_tensor(name)
                    if tuple(source.shape) != tuple(destination.shape):
                        shape_mismatch.append(
                            {
                                "name": name,
                                "checkpoint": list(source.shape),
                                "model": list(destination.shape),
                            }
                        )
                        del source
                        continue
                    destination.copy_(
                        source.to(device=destination.device, dtype=destination.dtype)
                    )
                    loaded.add(name)
                    del source

    missing = sorted(set(expected) - loaded)
    report = {
        "checkpoint": str(directory),
        "index": str(index_path),
        "loaded_count": len(loaded),
        "missing": missing,
        "unexpected": unexpected,
        "ignored_unexpected": sorted(ignored_unexpected),
        "shape_mismatch": shape_mismatch,
    }
    _write_report(report)
    print(
        "[lingbot-va-torch-checkpoint] "
        f"loaded={len(loaded)} missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_mismatch={len(shape_mismatch)} ignored_unexpected={len(ignored_unexpected)}"
        + (f" report={report['report_path']}" if "report_path" in report else ""),
        flush=True,
    )
    if missing or unexpected or shape_mismatch:
        details = json.dumps(
            {
                "missing": missing[:20],
                "unexpected": unexpected[:20],
                "shape_mismatch": shape_mismatch[:20],
            },
            sort_keys=True,
        )
        raise RuntimeError(f"LingBot-VA core checkpoint mismatch: {details}")
    return report
