# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Streaming sharded-safetensors loader for the PyTorch LingBot-VA model."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

_IGNORED_UNEXPECTED = {"patch_embedding.weight", "patch_embedding.bias"}


def _packed_projection_targets(name: str):
    """Map legacy Q/K/V checkpoint tensors into an optional packed parameter."""
    targets = []
    for source_suffix, target_suffix, part in (
        (".to_q.weight", ".to_qkv.weight", 0),
        (".to_k.weight", ".to_qkv.weight", 1),
        (".to_v.weight", ".to_qkv.weight", 2),
        (".to_k.bias", ".to_qkv.bias", 1),
        (".to_v.bias", ".to_qkv.bias", 2),
    ):
        if name.endswith(source_suffix):
            targets.append((name[: -len(source_suffix)] + target_suffix, part))
    for source_suffix, target_suffix, part in (
        (".to_k.weight", ".to_kv.weight", 0),
        (".to_v.weight", ".to_kv.weight", 1),
        (".to_k.bias", ".to_kv.bias", 0),
        (".to_v.bias", ".to_kv.bias", 1),
    ):
        if name.endswith(source_suffix):
            targets.append((name[: -len(source_suffix)] + target_suffix, part))
    if name.endswith(".to_q.bias"):
        targets.append((name[: -len(".to_q.bias")] + ".to_qkv.bias", 0))
    return targets


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
    loaded_packed_parts = defaultdict(set)
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
                    packed = None
                    if destination is None:
                        packed = next(
                            (
                                candidate
                                for candidate in _packed_projection_targets(name)
                                if candidate[0] in expected
                            ),
                            None,
                        )
                        if packed is None:
                            # Nothing in the model wants this tensor, so report
                            # it from the index metadata instead of reading the
                            # payload off disk.
                            unexpected.append(
                                {
                                    "name": name,
                                    "shape": list(shard.get_slice(name).get_shape()),
                                }
                            )
                            continue
                    source = shard.get_tensor(name)
                    if packed is not None:
                        packed_name, part = packed
                        destination = expected[packed_name]
                        rows = source.shape[0]
                        packed_rows = destination.shape[0]
                        groups = (
                            3
                            if packed_name.endswith("to_qkv.weight")
                            or packed_name.endswith("to_qkv.bias")
                            else 2
                        )
                        group_rows = packed_rows // groups
                        start = part * group_rows
                        stop = start + rows
                        # A part must fill its whole group slice. A shorter
                        # source would copy in, be recorded as loaded, and
                        # silently leave the rest of the slice uninitialized.
                        if (
                            tuple(source.shape[1:]) != tuple(destination.shape[1:])
                            or packed_rows % groups != 0
                            or rows != group_rows
                            or stop > packed_rows
                        ):
                            shape_mismatch.append({
                                "name": name,
                                "checkpoint": list(source.shape),
                                "model": list(destination.shape),
                            })
                            del source
                            continue
                        destination.narrow(0, start, rows).copy_(
                            source.to(device=destination.device, dtype=destination.dtype)
                        )
                        loaded_packed_parts[packed_name].add(part)
                        del source
                        continue
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

    for packed_name, parts in loaded_packed_parts.items():
        groups = 3 if ".to_qkv." in packed_name else 2
        if len(parts) == groups:
            loaded.add(packed_name)
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
    print(
        "[lingbot-va-torch-checkpoint] "
        f"loaded={len(loaded)} missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_mismatch={len(shape_mismatch)} ignored_unexpected={len(ignored_unexpected)}",
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
