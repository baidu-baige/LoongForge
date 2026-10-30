# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Static contract for the Kimi K3 language-only training examples."""

from pathlib import Path
from runpy import run_path


ROOT = Path(__file__).resolve().parents[3]
CONSTANTS = run_path(str(ROOT / "loongforge/utils/constants.py"))


def test_kimi_k3_llm_example_uses_the_llm_training_path():
    script = (ROOT / "examples/kimi_k3/pretrain/pretrain_kimi_k3_llm.sh").read_text()

    assert CONSTANTS["LanguageModelFamilies"].KIMI_K3_LLM == "kimi_k3_llm"
    assert "MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-" in script
    assert "configs/models/kimi_k3/kimi_k3_backbone.yaml" in script
    assert "model_type=kimi_k3_llm" in script
    assert "--training-phase pretrain" in script
    assert "KimiTaskEncoder" not in script
    assert "configs/models/kimi_k3/kimi_k3.yaml" not in script


def test_kimi_k3_llm_sft_example_uses_the_llm_training_path():
    script = (ROOT / "examples/kimi_k3/finetuning/sft_kimi_k3_llm.sh").read_text()

    assert "MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-" in script
    assert "configs/models/kimi_k3/kimi_k3_backbone.yaml" in script
    assert "model_type=kimi_k3_llm" in script
    assert "--training-phase sft" in script
    assert "--chat-template kimi-k3-hf" in script
    assert "--sft-dataset openai" in script
    assert "KimiTaskEncoder" not in script
    assert "configs/models/kimi_k3/kimi_k3.yaml" not in script


if __name__ == "__main__":
    test_kimi_k3_llm_example_uses_the_llm_training_path()
    test_kimi_k3_llm_sft_example_uses_the_llm_training_path()
