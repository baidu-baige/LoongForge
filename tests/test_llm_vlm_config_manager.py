# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path


LLM_VLM_ROOT = Path(__file__).parent / "llm_vlm"


def load_config_manager():
    llm_vlm_root = str(LLM_VLM_ROOT)
    if llm_vlm_root not in sys.path:
        sys.path.insert(0, llm_vlm_root)
    return importlib.import_module("tools.config_manager")


def test_llm_config_resolves_paths_from_environment(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "common.yaml").write_text(
        "\n".join(
            [
                "pfs_path: ${PFS_PATH}",
                "megatron_path: ${MEGATRON_PATH}",
                "LOONGFORGE_PATH: ${LOONGFORGE_PATH}",
                "training_log_path: ${TRAINING_LOG_PATH}",
                "save_mcore_ckpt_path_pre: $training_log_path/megatron_checkpoint",
                "dataset_cache_path: $training_log_path/dataset_cache",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (configs_dir / "model.yaml").write_text(
        "\n".join(
            [
                "model_name: test_model",
                "HF_CKPT_PATH: $pfs_path/huggingface.co/model",
                "DATA_PATH: $pfs_path/datasets/model",
                "CHECKPOINT_PATH: $save_mcore_ckpt_path_pre/test_model",
                "TENSORBOARD_PATH: $training_log_path/tensorboard/test_model.log",
                "DATA_ARGS: '--data-cache-path $dataset_cache_path'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("PFS_PATH", "/ci-llm-input")
    monkeypatch.setenv("TRAINING_LOG_PATH", "/ci-output/llm_vlm")
    monkeypatch.setenv("LOONGFORGE_PATH", "/ci-source/LoongForge/")
    monkeypatch.setenv("MEGATRON_PATH", "/ci-source/LoongForge/third_party/Loong-Megatron")

    config_manager = load_config_manager()
    config = config_manager.ConfigManager.load_config(
        object(), str(configs_dir), "model.yaml"
    )

    assert config["pfs_path"] == "/ci-llm-input"
    assert config["training_log_path"] == "/ci-output/llm_vlm"
    assert config["HF_CKPT_PATH"] == "/ci-llm-input/huggingface.co/model"
    assert config["DATA_PATH"] == "/ci-llm-input/datasets/model"
    assert (
        config["save_mcore_ckpt_path_pre"]
        == "/ci-output/llm_vlm/megatron_checkpoint"
    )
    assert config["CHECKPOINT_PATH"] == "/ci-output/llm_vlm/megatron_checkpoint/test_model"
    assert (
        config["dataset_cache_path"]
        == "/ci-output/llm_vlm/dataset_cache"
    )
    assert config["TENSORBOARD_PATH"] == "/ci-output/llm_vlm/tensorboard/test_model.log"
    assert "--data-cache-path /ci-output/llm_vlm/dataset_cache" in config["DATA_ARGS"]
