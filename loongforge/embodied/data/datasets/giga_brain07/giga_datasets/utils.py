# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_brain07 dataset support utilities (self-contained).

Consolidation of the former ``giga_datasets/utils/`` package (file/time/paths/
utils/import_utils) into a single module. Only the helpers actually used by the
surviving giga dataset reader + ``DefaultCollator`` are kept; dead helpers
(``empty_dir`` / ``list_dir`` / ``chmod_dir`` / ``get_root_dir`` /
``get_cur_time`` / ``as_list`` / ``repeat_n`` / ``split_data`` / ``wrap_call``)
were dropped. Implementations are byte-for-byte equivalent to the originals, so
runtime behaviour is unchanged.

No dependency on the vendored ``giga_train`` / ``giga_models`` / ``giga_brain_0``
packages, and no third-party helper imports: ``is_lerobot_available`` now probes
via ``importlib`` instead of ``diffusers`` while keeping the same contract
(lerobot installed and version >= 0.4.0).
"""

from __future__ import annotations

import json
import os
import pickle
import time
from importlib import import_module
from typing import Any

import numpy as np
import torch
import yaml

try:  # Prefer the C-accelerated (de)serializers when libyaml is available.
    from yaml import CDumper as _YamlDumper
except ImportError:  # pragma: no cover - pure-python fallback
    from yaml import Dumper as _YamlDumper


# ───────────────────────────── timing ─────────────────────────────
class Timer:
    """Context manager for timing code blocks and printing elapsed seconds."""

    def __init__(self, prefix: str = "Cost") -> None:
        self.prefix = prefix
        self.start_time: float | None = None

    def tic(self) -> None:
        """Start or reset the timer."""
        self.start_time = time.time()

    def toc(self) -> None:
        """Stop the timer and print the elapsed time in seconds."""
        print("{}: {}".format(self.prefix, time.time() - self.start_time))

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()


# ───────────────────────────── paths ─────────────────────────────
def get_data_dir() -> str:
    """Data directory read from env var ``GIGA_DATASETS_DIR`` with fallback."""
    return os.environ.get("GIGA_DATASETS_DIR", "./data/")


# ─────────────────────────── importing ───────────────────────────
def import_function(function_name: str, sep: str = ".") -> Any:
    """Import a function/attribute from a fully-qualified dotted name."""
    parts = function_name.split(sep)
    module_name = ".".join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])


# ───────────────────────────── file I/O ─────────────────────────────
def load_file(file_path: str, **kwargs: Any) -> Any:
    """Load a structured file based on its extension (.pkl/.pickle/.json/.yaml/.yml)."""
    if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        data = pickle.load(open(file_path, "rb"), **kwargs)
    elif file_path.endswith(".json"):
        data = json.load(open(file_path, "r"), **kwargs)
    elif file_path.endswith(".yaml") or file_path.endswith("yml"):
        kwargs.pop("Loader", None)
        data = yaml.safe_load(open(file_path, "r"), **kwargs)
    else:
        assert False
    return data


def save_file(file_path: str, data: Any, **kwargs: Any) -> None:
    """Save a Python object to disk based on file extension."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        pickle.dump(data, open(file_path, "wb"), **kwargs)
    elif file_path.endswith(".json"):
        kwargs.setdefault("indent", 4)
        json.dump(data, open(file_path, "w"), **kwargs)
    elif file_path.endswith(".yaml") or file_path.endswith("yml"):
        kwargs.setdefault("Dumper", _YamlDumper)
        yaml.dump(data, open(file_path, "w"), **kwargs)
    else:
        assert False


# ─────────────────────────── tensor stacking ───────────────────────────
def stack_data(
    data_list: list[np.ndarray | torch.Tensor],
    pad_value: int | float = 0,
    is_equal: bool = False,
) -> np.ndarray | torch.Tensor:
    """Stack arrays/tensors with optional padding when shapes differ.

    Fast path stacks directly when all shapes match; otherwise a padded canvas
    (up to 4 dims) is allocated and each sample copied in-place. Behaviour is
    identical to the original ``giga_datasets.utils.stack_data``.
    """
    equal_shape = True
    for data in data_list:
        if data.shape != data_list[0].shape:
            equal_shape = False
            break
    if is_equal:
        assert equal_shape
    if equal_shape:
        if isinstance(data_list[0], np.ndarray):
            new_data = np.stack(data_list)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.stack(data_list)
        else:
            assert False
    else:
        ndim = data_list[0].ndim
        dtype = data_list[0].dtype
        new_data_shape = [len(data_list)]
        for dim in range(ndim):
            new_data_shape.append(max(data.shape[dim] for data in data_list))
        if isinstance(data_list[0], np.ndarray):
            new_data = np.full(tuple(new_data_shape), pad_value, dtype=dtype)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.full(tuple(new_data_shape), pad_value, dtype=dtype)
        else:
            assert False
        for i, data in enumerate(data_list):
            if ndim == 1:
                new_data[i][: data.shape[0]] = data
            elif ndim == 2:
                new_data[i][: data.shape[0], : data.shape[1]] = data
            elif ndim == 3:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2]] = data
            elif ndim == 4:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2], : data.shape[3]] = data
            else:
                assert False
    return new_data


# ─────────────────────── optional-dependency probe ───────────────────────
def _probe_lerobot() -> tuple[bool, str | None]:
    """Return (available, version) for lerobot via importlib (no diffusers dep)."""
    import importlib.util

    if importlib.util.find_spec("lerobot") is None:
        return False, None
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return True, version("lerobot")
        except PackageNotFoundError:
            return True, None
    except Exception:
        return True, None


_lerobot_available, _lerobot_version = _probe_lerobot()


def is_lerobot_available() -> bool:
    """True iff ``lerobot`` is installed with version >= 0.4.0 (unchanged contract)."""
    from packaging.version import InvalidVersion, Version

    if not _lerobot_available:
        return False
    try:
        return Version(_lerobot_version) >= Version("0.4.0")
    except (InvalidVersion, TypeError):
        return False
