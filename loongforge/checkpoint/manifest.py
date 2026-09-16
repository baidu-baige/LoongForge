# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint format markers and Torch resume metadata."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from loongforge.contracts.checkpoint import TorchCheckpointMetadata


def is_hf_checkpoint(path):
    if path is None:
        return False
    return any((Path(path) / name).is_file() for name in (
        "model.safetensors.index.json", "model.safetensors",
        "pytorch_model.bin.index.json", "pytorch_model.bin",
    ))


def read_torch_metadata(path) -> TorchCheckpointMetadata:
    with (Path(path) / "resume_meta.json").open(encoding="utf-8") as file:
        return TorchCheckpointMetadata(**json.load(file))


def write_torch_metadata(path, meta: TorchCheckpointMetadata):
    # Publish the complete marker in one rename; readers never see partial JSON.
    with NamedTemporaryFile(mode="w", dir=path, encoding="utf-8", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(asdict(meta), file, indent=2)
            file.flush()
            os.replace(temporary, Path(path) / "resume_meta.json")
        finally:
            temporary.unlink(missing_ok=True)
