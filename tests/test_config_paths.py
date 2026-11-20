# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from loongforge.config_paths import config_root, resolve_config


def test_known_model_configuration_resolves():
    root = config_root()
    qwen3 = resolve_config("configs/models/qwen3")

    assert root.name == "configs" or root.name == "_configs"
    assert qwen3.is_dir()
    assert list(qwen3.glob("*.yaml"))
