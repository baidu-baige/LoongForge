# pylint: skip-file
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert a Qwen-Image-Edit Megatron FSDP (fsdp_dtensor) DCP checkpoint back to
the HuggingFace diffusers ``transformer`` safetensors format.
"""

import argparse
import json
import os
import tempfile

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

try:
    from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME
except Exception:  # pragma: no cover
    SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def parse_args():
    p = argparse.ArgumentParser(description="Convert Qwen-Image Megatron DCP -> HF.")
    p.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Megatron DCP checkpoint dir (contains latest_checkpointed_iteration.txt).",
    )
    p.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output dir for the HF diffusers transformer safetensors.",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration to load; defaults to latest_checkpointed_iteration.txt.",
    )
    return p.parse_args()


def load_dcp(load_path, iteration=None):
    """Load a DCP (fsdp_dtensor) checkpoint and return the model state dict."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    if iteration is None:
        iter_file = os.path.join(load_path, "latest_checkpointed_iteration.txt")
        with open(iter_file) as f:
            iteration = int(f.read().strip())
    dcp_dir = os.path.join(load_path, f"iter_{int(iteration):07d}")
    print(f"Loading DCP checkpoint from: {dcp_dir}")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        dcp_to_torch_save(dcp_dir, tmp_path)
        raw = torch.load(tmp_path, map_location="cpu", weights_only=False)
    finally:
        os.unlink(tmp_path)

    model_raw = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model_dict = {}
    for k, v in model_raw.items():
        k = k.removeprefix("module.")
        if k.endswith("_extra_state"):
            continue
        if not torch.is_tensor(v):
            continue
        model_dict[k] = v
    return model_dict


def mcore_key_to_hf(key):
    """Map a QwenImageModel key back to HuggingFace naming."""
    if key.startswith("decoder.layers."):
        rest = key[len("decoder.layers."):]
        idx, _, tail = rest.partition(".")
        key = f"transformer_blocks.{idx}.{tail}"
    return key


def save_huggingface_checkpoint(state_dict, save_path):
    os.makedirs(save_path, exist_ok=True)
    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    for shard_file, tensors in state_dict_split.filename_to_tensors.items():
        shard = {}
        for tensor in tensors:
            shard[tensor] = state_dict[tensor].contiguous()
        shard_path = os.path.join(save_path, shard_file)
        save_file(shard, shard_path, metadata={"format": "pt"})
        print(f"Saving HuggingFace shard to: {shard_path}")

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(save_path, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main():
    args = parse_args()
    print(f"load_path: {args.load_path}")
    print(f"save_path: {args.save_path}")
    assert os.path.abspath(args.load_path) != os.path.abspath(args.save_path)

    mcore_dict = load_dcp(args.load_path, iteration=args.iteration)
    hf_dict = {}
    for key, value in mcore_dict.items():
        hf_dict[mcore_key_to_hf(key)] = value.to(torch.bfloat16) if value.is_floating_point() else value
    print(f"converted {len(hf_dict)} tensors")

    save_huggingface_checkpoint(hf_dict, args.save_path)
    print(f"convert success! checkpoint path: {args.save_path}")


if __name__ == "__main__":
    main()
