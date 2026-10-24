# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
import os
import torch
from safetensors import safe_open
from ..logging_utils import get_logger

logger = get_logger(__name__)


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype=torch_dtype, device=device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    mmap_env = os.environ.get("VPP2_TORCH_LOAD_MMAP", "").strip().lower()
    mmap = False
    if mmap_env in {"1", "true", "yes", "on"}:
        mmap = True
    elif mmap_env in {"0", "false", "no", "off"}:
        mmap = False
    try:
        state_dict = torch.load(file_path, map_location=device, weights_only=True, mmap=mmap)
    except RuntimeError:
        if mmap_env:
            raise
        logger.warning(
            "mmap load failed for %s; retrying with default torch.load behavior.", file_path
        )
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict
