# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace to FSDP checkpoint conversion utilities."""

import argparse
import glob
import os
import tempfile

import torch
from safetensors.torch import load_file
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp


def list_safetensors_files(model_dir):
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    safetensors_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    return safetensors_files


def load_and_merge_state_dicts(safetensors_files):
    print(f"Found {len(safetensors_files)} safetensors file(s):")
    for file_path in safetensors_files:
        print(f"  - {file_path}")

    state_dict = {}
    for file_path in safetensors_files:
        print(f"Loading {file_path}...")
        shard = load_file(file_path)
        state_dict.update(shard)

    return state_dict


def build_dcp_state_dict(state_dict):
    original_keys = list(state_dict.keys())
    print(f"original keys: {original_keys}")

    new_model_dict = {}
    for key in original_keys:
        new_key = "module.model." + key
        new_model_dict[new_key] = state_dict[key]
        del state_dict[key]
        print(f"{key} -> {new_key}")

    state_dict["model"] = new_model_dict
    state_dict["args"] = None
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = 0

    return state_dict


def save_state_dict_to_pt(state_dict, model_dir, keep_pt):
    if keep_pt:
        pt_path = os.path.join(model_dir, "merged_model.pt")
    else:
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        pt_path = temp_file.name
        temp_file.close()

    print(f"Saving merged state dict to: {pt_path}")
    torch.save(state_dict, pt_path)
    return pt_path


def convert_pt_to_dcp(pt_path, dcp_output_dir, keep_pt):
    print(f"Converting to DCP format: {dcp_output_dir}...")
    torch_save_to_dcp(pt_path, dcp_output_dir)

    if not keep_pt:
        os.remove(pt_path)
        print(f"Removed temporary file: {pt_path}")

    print(f"✓ Conversion complete! DCP checkpoint saved to: {dcp_output_dir}")


def safetensors_dir_to_dcp(model_dir, dcp_output_dir, keep_pt=False):
    """
    Load all .safetensors in a directory and convert them to DCP format.

    Args:
        model_dir (str): Directory containing .safetensors files.
        dcp_output_dir (str): Output directory for DCP checkpoint.
        keep_pt (bool): Whether to keep the intermediate .pt file.
    """
    os.makedirs(dcp_output_dir, exist_ok=True)

    safetensors_files = list_safetensors_files(model_dir)
    state_dict = load_and_merge_state_dicts(safetensors_files)
    dcp_state_dict = build_dcp_state_dict(state_dict)
    pt_path = save_state_dict_to_pt(dcp_state_dict, model_dir, keep_pt)
    convert_pt_to_dcp(pt_path, dcp_output_dir, keep_pt)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert Pi05 safetensors to DCP format")
    parser.add_argument("--model-dir", required=True, help="Directory containing .safetensors files")
    parser.add_argument("--dcp-output-dir", required=True, help="Output directory for DCP checkpoint")
    parser.add_argument("--keep-pt", action="store_true", help="Keep intermediate .pt file")
    return parser


def main():
    args = build_arg_parser().parse_args()
    safetensors_dir_to_dcp(
        model_dir=args.model_dir,
        dcp_output_dir=args.dcp_output_dir,
        keep_pt=args.keep_pt,
    )


if __name__ == "__main__":
    main()


