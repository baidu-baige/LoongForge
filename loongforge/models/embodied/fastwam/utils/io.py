# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from FastWAM (https://github.com/yuantianyuan01/FastWAM).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model file download, loading, and hashing utilities for FastWAM."""

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from safetensors import safe_open


@dataclass
class ModelConfig:
    """Configuration for locating or downloading a model artifact."""

    path: Union[str, list[str], None] = None
    model_id: Optional[str] = None
    origin_file_pattern: Union[str, list[str], None] = None
    download_source: Optional[str] = None
    local_model_path: Optional[str] = None
    skip_download: Optional[bool] = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None

    def check_input(self):
        """Validate that enough information is present to resolve the model."""
        if self.path is None and self.model_id is None:
            raise ValueError("ModelConfig requires either `path` or (`model_id`, `origin_file_pattern`).")

    def parse_original_file_pattern(self):
        """Return the normalized original file pattern for model lookup."""
        if self.origin_file_pattern in [None, "", "./"]:
            return "*"
        if isinstance(self.origin_file_pattern, list):
            return self.origin_file_pattern
        if self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        return self.origin_file_pattern

    def parse_download_source(self):
        """Resolve the download source from config or environment."""
        if self.download_source is not None:
            return self.download_source
        env = os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE")
        return env if env is not None else "modelscope"

    def parse_skip_download(self):
        """Resolve whether downloading should be skipped."""
        if self.skip_download is not None:
            return self.skip_download
        env = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
        if env is None:
            return False
        return env.lower() == "true"

    def reset_local_model_path(self):
        """Set the local model root from config, environment, or default path."""
        if os.environ.get("DIFFSYNTH_MODEL_BASE_PATH") is not None:
            self.local_model_path = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        elif self.local_model_path is None:
            self.local_model_path = "./checkpoints/"

    def require_downloading(self):
        """Return True when configured model files are absent and download is allowed."""
        if self.path is not None:
            return False
        origin_file_pattern = self.parse_original_file_pattern()
        local_root = os.path.join(self.local_model_path, self.model_id)
        if isinstance(origin_file_pattern, list):
            all_exist = True
            for pattern in origin_file_pattern:
                matches = glob.glob(os.path.join(local_root, pattern))
                if len(matches) == 0:
                    all_exist = False
                    break
            if all_exist:
                return False
        else:
            if len(glob.glob(os.path.join(local_root, origin_file_pattern))) > 0:
                return False
        return not self.parse_skip_download()

    def download(self):
        """Download missing model artifacts from ModelScope or HuggingFace."""
        origin_file_pattern = self.parse_original_file_pattern()
        root = os.path.join(self.local_model_path, self.model_id)
        downloaded_files = glob.glob(origin_file_pattern, root_dir=root)
        download_source = self.parse_download_source().lower()
        if download_source == "modelscope":
            from modelscope import snapshot_download

            snapshot_download(
                self.model_id,
                local_dir=root,
                allow_file_pattern=origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False,
            )
        elif download_source == "huggingface":
            from huggingface_hub import snapshot_download as hf_snapshot_download

            hf_snapshot_download(
                self.model_id,
                local_dir=root,
                allow_patterns=origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False,
            )
        else:
            raise ValueError("`download_source` should be `modelscope` or `huggingface`.")

    def download_if_necessary(self):
        """Download artifacts when needed and resolve the final local path."""
        self.check_input()
        self.reset_local_model_path()
        if self.require_downloading():
            self.download()
        if self.path is None:
            if self.origin_file_pattern in [None, "", "./"]:
                self.path = os.path.join(self.local_model_path, self.model_id)
            else:
                matches = glob.glob(os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern))
                matches.sort()
                self.path = matches
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    """Load one or more checkpoint files into a single state dict."""
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype=torch_dtype, device=device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    """Load a safetensors checkpoint and optionally cast tensor dtype."""
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    """Load a PyTorch checkpoint and unwrap common state-dict containers."""
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


def _load_keys_dict_from_safetensors(file_path):
    """Load safetensors key shapes without materializing full tensor values."""
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            keys_dict[key] = f.get_slice(key).get_shape()
    return keys_dict


def _convert_state_dict_to_keys_dict(state_dict):
    """Convert nested state-dict tensors into a nested key-shape dictionary."""
    keys_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            keys_dict[key] = list(value.shape)
        else:
            keys_dict[key] = _convert_state_dict_to_keys_dict(value)
    return keys_dict


def _load_keys_dict_from_bin(file_path):
    """Load tensor key shapes from a PyTorch checkpoint."""
    state_dict = load_state_dict_from_bin(file_path)
    return _convert_state_dict_to_keys_dict(state_dict)


def _load_keys_dict(file_path):
    """Load tensor key shapes from one or more checkpoint files."""
    if isinstance(file_path, list):
        merged = {}
        for path in file_path:
            merged.update(_load_keys_dict(path))
        return merged
    if file_path.endswith(".safetensors"):
        return _load_keys_dict_from_safetensors(file_path)
    return _load_keys_dict_from_bin(file_path)


def _convert_keys_dict_to_single_str(keys_dict, with_shape=True):
    """Serialize a key-shape dictionary into a deterministic hash input string."""
    keys = []
    for key, value in keys_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + _convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    return ",".join(keys)


def hash_model_file(path, with_shape=True):
    """Return a deterministic md5 hash for checkpoint tensor keys and shapes."""
    keys_dict = _load_keys_dict(path)
    keys_str = _convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape).encode("UTF-8")
    return hashlib.md5(keys_str).hexdigest()
