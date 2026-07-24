# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline tool: consolidate a DCP-sharded checkpoint into a single-file
``model.safetensors`` (or ``pytorch_model.pt``).

Single-process, no distributed init required. Use this to:

* Feed a DCP checkpoint to ``load_pretrained`` (which only consumes single-file
  weights).
* Export a release artifact for downstream inference.

Usage:
    python -m loongforge.embodied.tools.dcp_to_safetensors \\
        --ckpt outputs/run/checkpoints/steps_10000 \\
        --out  outputs/run/release/steps_10000.safetensors \\
        --format safetensors

The input ``--ckpt`` path may be either a ``steps_N`` directory containing a
``dcp/`` subdirectory, or the ``dcp/`` directory itself.
"""

import argparse
import os
import sys

import torch


def _resolve_dcp_dir(ckpt: str) -> str:
    if os.path.isdir(os.path.join(ckpt, "dcp")):
        return os.path.join(ckpt, "dcp")
    if os.path.exists(os.path.join(ckpt, ".metadata")):
        return ckpt
    raise FileNotFoundError(
        f"No DCP checkpoint found at {ckpt} (expected dcp/.metadata)."
    )


def _load_full_state_dict(dcp_dir: str) -> dict:
    """Load DCP shards into a full (unsharded) tensor dict on CPU."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import (
        BytesStorageMetadata,
        TensorStorageMetadata,
    )

    reader = FileSystemReader(dcp_dir)
    metadata = reader.read_metadata()

    # Find top-level keys present in this checkpoint (e.g. "model", "optim").
    top_keys = set()
    for k in metadata.state_dict_metadata.keys():
        top_keys.add(k.split(".", 1)[0])

    if "model" not in top_keys:
        raise RuntimeError(
            f"DCP checkpoint at {dcp_dir} has no 'model' key (found {sorted(top_keys)})."
        )

    # DCP only fills entries that already exist in the destination dict, so we
    # pre-allocate tensor placeholders from metadata for every "model.*" key.
    model_sd = {}
    non_tensor = []
    for full_key, meta in metadata.state_dict_metadata.items():
        if not full_key.startswith("model."):
            continue
        sub_key = full_key[len("model."):]
        if isinstance(meta, TensorStorageMetadata):
            model_sd[sub_key] = torch.empty(meta.size, dtype=meta.properties.dtype)
        elif isinstance(meta, BytesStorageMetadata):
            non_tensor.append(sub_key)
        else:
            non_tensor.append(sub_key)

    if non_tensor:
        print(f"[dcp_to_safetensors] skipping {len(non_tensor)} non-tensor model entries")

    state = {"model": model_sd}
    dcp.load(state, storage_reader=reader)
    return state["model"]


def _save_safetensors(sd: dict, out_path: str):
    from safetensors.torch import save_file

    clean = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(clean, out_path)


def _save_pt(sd: dict, out_path: str):
    clean = {k: v.detach().cpu() for k, v in sd.items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(clean, out_path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True,
                   help="Path to a steps_N directory or its dcp/ subdirectory.")
    p.add_argument("--out", required=True,
                   help="Output single-file path (e.g. model.safetensors).")
    p.add_argument("--format", choices=["safetensors", "pt"], default="safetensors")
    args = p.parse_args(argv)

    dcp_dir = _resolve_dcp_dir(args.ckpt)
    print(f"[dcp_to_safetensors] reading shards from {dcp_dir}")
    sd = _load_full_state_dict(dcp_dir)
    print(f"[dcp_to_safetensors] consolidated {len(sd)} tensors")

    if args.format == "safetensors":
        _save_safetensors(sd, args.out)
    else:
        _save_pt(sd, args.out)
    print(f"[dcp_to_safetensors] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
