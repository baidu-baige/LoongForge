# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load_config_map():
    path = ROOT / "loongforge/utils/config_map.py"
    spec = spec_from_file_location("deepseek_v41_config_map", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deepseek_v41_schedule_and_model_names():
    full = yaml.safe_load((ROOT / "configs/models/deepseek4.1/deepseek_v41_flash.yaml").read_text())
    ratios = full["csa_compress_ratios"]
    assert ratios == [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
    assert full["mtp_num_layers"] == 0
    assert full["model_type"] == "deepseek_v41"
    assert full["experimental_attention_variant"] is None
    assert full["num_layers"] == 40
    assert full["num_experts"] == 384
    assert full["num_experts_per_tok"] == 6
    assert full["engram_layer_ids"] == [1, 14]
    assert full["vision_config"]["num_hidden_layers"] == 32
    assert full["vision_config"]["hidden_size"] == 1024
    assert full["vision_config"]["downsample_ratio"] == 3


def test_deepseek_v41_examples_use_full_parameter_optimizer_defaults():
    scripts = (
        ROOT / "examples/deepseek_v4.1/pretrain/pretrain_deepseek_v41_flash_bf16.sh",
        ROOT / "examples/deepseek_v4.1/finetune/finetune_deepseek_v41_flash_bf16.sh",
    )
    for path in scripts:
        text = path.read_text()
        assert "--adam-beta2 0.95" in text
        assert "--use-precision-aware-optimizer" in text
        assert "--exp-avg-dtype bf16" in text
        assert "--exp-avg-sq-dtype bf16" in text
        assert "freeze-parameters-regex" not in text
        assert "--task-encoder deepseekv41taskencoder" in text
        assert "--deepseek-v41-multimodal" not in text
        assert text.count("--moe-router-bias-update-rate") == 1
        assert "--moe-router-bias-update-rate 0\n" in text
        assert "--moe-token-dispatcher-type alltoall" in text

    config_map = _load_config_map()
    config_path, _ = config_map.get_config_from_model_name("deepseek-v4.1-flash")
    assert config_path.endswith("configs/models/deepseek4.1")


def test_deepseek_v41_conversion_owns_only_v41_keys():
    conversion = (ROOT / "configs/models/deepseek4.1/ckpt_convert/deepseek_v41_convert.yaml").read_text()
    assert "attention.indexer.k_norm" in conversion
    assert "self_attention.indexer.k_norm" not in conversion
    assert "attention.compressor.ape" not in conversion
    assert "model.dsv41" not in conversion
    mapping = yaml.safe_load(conversion)
    for platform in ("huggingface", "mcore"):
        assert "hc_attn_alpha_pre" not in mapping["name_map"][platform]
        assert "hc_ffn_alpha_pre" not in mapping["name_map"][platform]
    hf_converter = (ROOT / "tools/mcore_checkpoint_convert/huggingface/huggingface_checkpoint.py").read_text()
    mcore_converter = (ROOT / "tools/mcore_checkpoint_convert/mcore/mcore_checkpoint.py").read_text()
    assert '("vision.", "aligner.")' in hf_converter
    assert 'for root in ("vision", "aligner")' in mcore_converter
