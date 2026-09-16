# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Catalog entry for engine selection and lazy config construction."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    config_file: Path
    model_config_path: str | None = None
    data_config_path: str | None = None

    @staticmethod
    def _load(path):
        if path is None:
            raise ValueError("This model uses Hydra configs, not Torch dataclasses")
        module, name = path.rsplit(":", 1)
        return getattr(import_module(module), name)

    @property
    def model_config_cls(self):
        return self._load(self.model_config_path)

    @property
    def data_config_cls(self):
        return self._load(self.data_config_path)
