# pylint: skip-file
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert a Qwen-Image-Edit HuggingFace DiT checkpoint to a Megatron
FSDP (fsdp_dtensor) DCP checkpoint that can be loaded via ``--load`` with
``--ckpt-format fsdp_dtensor``.
"""

import argparse
import glob
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args():
    p = argparse.ArgumentParser(description="Convert Qwen-Image HF DiT -> Megatron DCP.")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="HF diffusers transformer dir, a single .safetensors/.bin, or a glob.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output dir for the Megatron DCP checkpoint.",
    )
    p.add_argument("--iteration", type=int, default=1, help="Iteration tag for iter_XXXXXXX.")
    return p.parse_args()


def find_checkpoint_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.bin")))
    else:
        files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No checkpoint files matched: {path}")
    return files


def load_huggingface_checkpoint(path):
    """Merge all HF shards into a single state dict."""
    files = find_checkpoint_files(path)
    state_dict = {}
    for f in files:
        print(f"loading {f}")
        if f.endswith(".safetensors"):
            local = load_file(f, device="cpu")
        else:
            local = torch.load(f, map_location="cpu")
            if isinstance(local, dict) and "state_dict" in local:
                local = local["state_dict"]
        state_dict.update(local)
    return state_dict


def hf_key_to_mcore(key):
    """Map a HuggingFace key to the QwenImageModel key."""
    for prefix in ("pipe.dit.", "dit.", "model.diffusion_model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("transformer_blocks."):
        rest = key[len("transformer_blocks."):]
        idx, _, tail = rest.partition(".")
        key = f"decoder.layers.{idx}.{tail}"
    return key


def save_dcp(model_state_dict, save_path, iteration=1):
    """Save a plain state dict as a Megatron DCP (fsdp_dtensor) checkpoint.

    Adds the ``module.`` prefix to match the FSDP wrapper key format produced
    by ``generate_state_dict()`` at runtime (``module.<param_name>``), then
    converts a regular ``torch.save`` file to DCP via ``torch_save_to_dcp`` so
    no distributed process group is required.
    """
    from torch.distributed.checkpoint.format_utils import torch_save_to_dcp

    save_path = Path(save_path)
    dcp_iter_path = save_path / f"iter_{iteration:07d}"
    dcp_iter_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model_sd_with_prefix = {"module." + k: v for k, v in model_state_dict.items()}
        torch.save({"model": model_sd_with_prefix}, tmp_path)
        torch_save_to_dcp(tmp_path, str(dcp_iter_path))
    finally:
        os.unlink(tmp_path)

    with open(save_path / "latest_checkpointed_iteration.txt", "w") as f:
        f.write(str(iteration))
    print(f"DCP checkpoint saved to: {dcp_iter_path}")


def main():
    args = parse_args()
    print(f"checkpoint_path: {args.checkpoint_path}")
    print(f"save_path: {args.save_path}")
    assert os.path.abspath(args.checkpoint_path) != os.path.abspath(args.save_path)

    hf_state = load_huggingface_checkpoint(args.checkpoint_path)
    mcore_state = {}
    for key, value in hf_state.items():
        mcore_state[hf_key_to_mcore(key)] = value
    print(f"converted {len(mcore_state)} tensors")

    save_dcp(mcore_state, args.save_path, iteration=args.iteration)
    print(f"convert success! checkpoint path: {args.save_path}")


if __name__ == "__main__":
    main()
