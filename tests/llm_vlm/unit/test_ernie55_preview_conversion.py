# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ERNIE-5.5-Flash-Preview MXFP4 keys, Preview config and shard selection."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from mcore_checkpoint_convert.ernie5_5 import normalize_ernie55_mxfp4_state_dict
from mcore_checkpoint_convert.huggingface import huggingface_checkpoint as hf_checkpoint_module
from mcore_checkpoint_convert.huggingface.huggingface_checkpoint import (
    HuggingFaceCheckpoint,
    get_hf_checkpoint_names,
)
from mcore_checkpoint_convert.huggingface.mxfp4_dequant import dequantize_mxfp4
from mcore_checkpoint_convert.utils.config_utils import get_yaml_config

BASE_CONFIG = ROOT / "configs/models/ernie5_5/ernie5_5_flash_base.yaml"
PREVIEW_CONFIG = ROOT / "configs/models/ernie5_5/ernie5_5_flash_preview.yaml"
CONVERT_CONFIG = ROOT / "configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml"
ERNIE_CONFIG = {"module": {"model_type": "ernie5_5"}}


def _preview_config():
    c_config = get_yaml_config(str(PREVIEW_CONFIG), str(CONVERT_CONFIG))
    c_config.data.module.update({
        "num_layers": 1,
        "compress_ratios": [128],
        "hidden_size": 4,
        "num_attention_heads": 2,
        "vocab_size_in_config_file": 8,
        "num_experts": 4,
    })
    return c_config


def _preview_hf_config():
    return {
        "num_hidden_layers": 1,
        "compress_ratios": [128],
        "hidden_size": 4,
        "torch_dtype": "bfloat16",
        "mqa_latent_rope_adjacent_pairing": True,
    }


def _preview_load_args():
    return SimpleNamespace(
        hf_checkpoint_device="cpu",
        max_workers=1,
        enable_full_hetero_dp=False,
        hf_dequantize_mxfp4=True,
        hf_dequantize_dtype="bfloat16",
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        mtp_num_layers=0,
    )


def _mxfp4_pair(base, rows=4, in_features=64, scale_columns=4):
    weight = torch.zeros(rows, in_features // 2, dtype=torch.uint8)
    weight = weight.repeat(1, 2).view(torch.int8)
    scale = torch.full((rows, scale_columns), 127, dtype=torch.uint8)
    return {base + ".weight": weight, base + ".weight_scale_inv": scale}


def test_mxfp4_weight_scale_inv_is_renamed_to_scale_and_idempotent():
    base = "model.layers.1.block_sparse_moe.experts.0.w1"
    # Shapes must satisfy scale width * 32 == unpacked in_features, so [4, 64]
    # packed bytes pair with [4, 4] E8M0 blocks of 32.
    weight = torch.zeros(4, 64, dtype=torch.int8)
    scale = torch.full((4, 4), 127, dtype=torch.float8_e8m0fnu)
    state_dict = {base + ".weight": weight, base + ".weight_scale_inv": scale}

    normalize_ernie55_mxfp4_state_dict(state_dict, ERNIE_CONFIG)
    assert set(state_dict) == {base + ".weight", base + ".scale"}
    assert state_dict[base + ".scale"] is scale

    normalize_ernie55_mxfp4_state_dict(state_dict, ERNIE_CONFIG)
    assert set(state_dict) == {base + ".weight", base + ".scale"}
    assert state_dict[base + ".scale"] is scale


def test_mxfp4_pair_without_dequantize_switch_reports_flag_and_key():
    # The guard fires while the scale still has its Preview name, so the message
    # can name both the missing switch and the offending pair.
    base = "model.layers.1.block_sparse_moe.experts.0.w1"
    state_dict = {
        base + ".weight": torch.zeros(4, 64, dtype=torch.int8),
        base + ".weight_scale_inv": torch.zeros(4, 4, dtype=torch.float8_e8m0fnu),
    }
    with pytest.raises(ValueError) as excinfo:
        normalize_ernie55_mxfp4_state_dict(
            state_dict, ERNIE_CONFIG, SimpleNamespace(hf_dequantize_mxfp4=False)
        )
    message = str(excinfo.value)
    assert "--hf-dequantize-mxfp4" in message
    assert base + ".weight" in message
    # The packed pair stays as loaded, so the same checkpoint can be retried with the switch.
    assert set(state_dict) == {base + ".weight", base + ".weight_scale_inv"}


def test_mxfp4_pair_with_dequantize_switch_enabled_is_renamed():
    base = "model.layers.1.block_sparse_moe.experts.0.w1"
    scale = torch.full((4, 4), 127, dtype=torch.float8_e8m0fnu)
    state_dict = {base + ".weight": torch.zeros(4, 64, dtype=torch.int8), base + ".weight_scale_inv": scale}

    normalize_ernie55_mxfp4_state_dict(
        state_dict, ERNIE_CONFIG, SimpleNamespace(hf_dequantize_mxfp4=True)
    )
    assert set(state_dict) == {base + ".weight", base + ".scale"}
    assert state_dict[base + ".scale"] is scale


def test_mxfp4_normalization_reports_shape_mismatch_with_both_shapes():
    base = "model.layers.1.block_sparse_moe.experts.0.w1"
    state_dict = {
        base + ".weight": torch.zeros(4, 8, dtype=torch.uint8),
        base + ".weight_scale_inv": torch.zeros(4, 4, dtype=torch.uint8),
    }
    with pytest.raises(ValueError, match=r"weight_scale_inv.*\(4, 4\).*\(4, 8\)"):
        normalize_ernie55_mxfp4_state_dict(state_dict, ERNIE_CONFIG)


@pytest.mark.parametrize("scale_shape", [(), (256,), (3, 4)])
def test_mxfp4_normalization_rejects_bad_scale_rank_and_leading_dims(scale_shape):
    base = "model.layers.1.block_sparse_moe.experts.0.w1"
    state_dict = {
        base + ".weight": torch.zeros(4, 64, dtype=torch.int8),
        base + ".weight_scale_inv": torch.zeros(scale_shape, dtype=torch.uint8),
    }
    with pytest.raises(ValueError) as excinfo:
        normalize_ernie55_mxfp4_state_dict(state_dict, ERNIE_CONFIG)
    message = str(excinfo.value)
    assert base in message
    assert f"scale {base}.weight_scale_inv has shape {tuple(scale_shape)}" in message
    assert "packed weight" in message and "shape (4, 64)" in message
    assert "block size 32" in message
    # Nothing was renamed, so the invalid pair is still visible to the caller.
    assert set(state_dict) == {base + ".weight", base + ".weight_scale_inv"}


def test_mxfp4_normalization_ignores_other_models_and_layouts():
    prefix = "model.layers.1.block_sparse_moe.experts.0.w1"
    other_model = {
        prefix + ".weight": torch.zeros(4, 64, dtype=torch.int8),
        prefix + ".weight_scale_inv": torch.zeros(4, 4, dtype=torch.float8_e8m0fnu),
    }
    normalize_ernie55_mxfp4_state_dict(other_model, {"module": {"model_type": "kimi_k3"}})
    assert prefix + ".weight_scale_inv" in other_model
    assert prefix + ".scale" not in other_model

    fp8 = {
        prefix + ".weight": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        prefix + ".weight_scale_inv": torch.zeros(1, dtype=torch.float32),
    }
    normalize_ernie55_mxfp4_state_dict(fp8, ERNIE_CONFIG)
    assert prefix + ".weight_scale_inv" in fp8
    assert prefix + ".scale" not in fp8

    no_scale = {prefix + ".weight": torch.zeros(4, 8, dtype=torch.int8)}
    normalize_ernie55_mxfp4_state_dict(no_scale, ERNIE_CONFIG)
    assert set(no_scale) == {prefix + ".weight"}


def test_dequantize_mxfp4_decodes_e2m1_nibbles_with_e8m0_block_scale():
    # Byte -> (low nibble, high nibble), emitted in that order: 0x1 -> 0.5,
    # 0x2 -> 1.0, 0x7 -> 6.0 (max magnitude), 0x8 -> -0.0, 0x3 -> 1.5,
    # 0xF -> -6.0, 0x5 -> 3.0, 0x4 -> 2.0.
    packed = torch.tensor([0x21, 0x87, 0xF3, 0x45] + [0x00] * 12, dtype=torch.uint8)
    weight = packed.unsqueeze(0).repeat(2, 1).view(torch.int8)
    # E8M0 raw 127 is 2^0, raw 119 is 2^-8.
    scale = torch.tensor([[127], [119]], dtype=torch.uint8)

    decoded = dequantize_mxfp4(weight, scale, dtype=torch.float32)
    assert decoded.shape == (2, 32)
    assert torch.equal(
        decoded[0],
        torch.tensor([0.5, 1.0, 6.0, -0.0, 1.5, -6.0, 3.0, 2.0] + [0.0] * 24),
    )
    assert torch.equal(
        decoded[1],
        torch.tensor(
            [0.5 / 256, 1.0 / 256, 6.0 / 256, -0.0, 1.5 / 256, -6.0 / 256, 3.0 / 256, 2.0 / 256]
            + [0.0] * 24
        ),
    )
    # torch.equal treats -0.0 as 0.0, so check the sign bit of the 0x8 nibble.
    assert torch.equal(torch.signbit(decoded[0]), torch.tensor([0, 0, 0, 1, 0, 1, 0, 0] + [0] * 24, dtype=torch.bool))


def test_preview_config_changes_only_mtp_num_layers(monkeypatch):
    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    base = OmegaConf.load(BASE_CONFIG)
    preview = OmegaConf.load(PREVIEW_CONFIG)

    assert preview.mtp_num_layers == 0
    assert preview.num_layers == 43
    assert len(preview.compress_ratios) == 43
    assert list(preview.compress_ratios) == list(base.compress_ratios)
    assert {key for key in base.keys() if base[key] != preview[key]} == {"mtp_num_layers"}

    # The shared convert yaml resolves mtp_num_layers from the module config,
    # so Preview converts without any stage ever addressing model.layers.43.
    c_config = get_yaml_config(str(PREVIEW_CONFIG), str(CONVERT_CONFIG))
    assert c_config.get_args("common")["mtp_num_layers"] == 0
    assert c_config.get_args("huggingface")["mtp_layer_id"] == 43


def test_preview_config_passes_ernie55_post_init(monkeypatch):
    pytest.importorskip("megatron.core.transformer")
    from hydra.utils import instantiate

    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    preview = OmegaConf.load(PREVIEW_CONFIG)
    # bf16 is supplied by the training CLI in real runs; ERNIE 5.5 rejects anything else.
    config = instantiate(OmegaConf.merge(preview, {"bf16": True}))
    assert config.mtp_num_layers == 0
    assert len(config.compress_ratios) == 43


def test_preview_load_renames_then_dequantizes_mxfp4_experts(tmp_path, monkeypatch):
    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    c_config = get_yaml_config(str(PREVIEW_CONFIG), str(CONVERT_CONFIG))
    c_config.data.module.update({
        "num_layers": 1,
        "compress_ratios": [128],
        "hidden_size": 4,
        "num_attention_heads": 2,
        "vocab_size_in_config_file": 8,
        "num_experts": 4,
    })
    hf_config = {
        "num_hidden_layers": 1,
        "compress_ratios": [128],
        "hidden_size": 4,
        "torch_dtype": "bfloat16",
        "mqa_latent_rope_adjacent_pairing": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(hf_config))

    base = "model.layers.0.block_sparse_moe.experts.0.w1"
    weight = torch.tensor([0x21, 0x87, 0xF3, 0x45] + [0x00] * 12, dtype=torch.uint8)
    weight = weight.repeat(4, 2).view(torch.int8)
    scale = torch.tensor([[127, 126]] * 4, dtype=torch.uint8)
    save_file({base + ".weight": weight, base + ".weight_scale_inv": scale}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        base + ".weight": "model.safetensors",
        base + ".weight_scale_inv": "model.safetensors",
    }}))

    args = SimpleNamespace(
        hf_checkpoint_device="cpu",
        max_workers=1,
        enable_full_hetero_dp=False,
        hf_dequantize_mxfp4=True,
        hf_dequantize_dtype="bfloat16",
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        mtp_num_layers=0,
    )
    checkpoint = HuggingFaceCheckpoint(c_config, args)
    checkpoint.load(str(tmp_path), True, c_config, [0], mtp_num_layers=0)
    assert set(checkpoint.state_dict) == {base + ".weight"}
    assert checkpoint.state_dict[base + ".weight"].dtype == torch.bfloat16
    assert torch.equal(
        checkpoint.state_dict[base + ".weight"],
        dequantize_mxfp4(weight, scale, dtype=torch.bfloat16),
    )


def test_preview_single_file_load_without_c_config_still_normalizes(tmp_path, monkeypatch):
    # No index file: only config.json and model.safetensors exist, so load()
    # cannot use the config argument and must fall back to the checkpoint's own.
    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    c_config = _preview_config()
    (tmp_path / "config.json").write_text(json.dumps(_preview_hf_config()))

    base = "model.layers.0.block_sparse_moe.experts.0.w1"
    pair = _mxfp4_pair(base)
    save_file(pair, tmp_path / "model.safetensors")

    checkpoint = HuggingFaceCheckpoint(c_config, _preview_load_args())
    checkpoint.load(str(tmp_path), True, None, [0], mtp_num_layers=0)
    assert set(checkpoint.state_dict) == {base + ".weight"}
    assert checkpoint.state_dict[base + ".weight"].dtype == torch.bfloat16
    assert torch.equal(
        checkpoint.state_dict[base + ".weight"],
        dequantize_mxfp4(pair[base + ".weight"], pair[base + ".weight_scale_inv"], dtype=torch.bfloat16),
    )


def test_preview_drops_shared_shard_mtp_tensors_before_mxfp4_dequant(tmp_path, monkeypatch):
    # Preview's layer-42 trunk tensors share a shard with model.mtp.0 experts.
    # Shards are read whole, so both land in the state_dict even though
    # mtp_num_layers=0 never exports the MTP ones.
    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    c_config = _preview_config()
    (tmp_path / "config.json").write_text(json.dumps(_preview_hf_config()))

    trunk_base = "model.layers.0.block_sparse_moe.experts.0.w1"
    mtp_base = "model.mtp.0.block_sparse_moe.experts.0.w1"
    trunk = _mxfp4_pair(trunk_base)
    mtp = _mxfp4_pair(mtp_base)
    shard = {
        "model.layers.0.input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "model.mtp.0.enorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        **trunk,
        **mtp,
    }
    save_file(shard, tmp_path / "shared.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.input_layernorm.weight": "shared.safetensors",
        trunk_base + ".weight": "shared.safetensors",
        trunk_base + ".weight_scale_inv": "shared.safetensors",
        mtp_base + ".weight": "shared.safetensors",
        mtp_base + ".weight_scale_inv": "shared.safetensors",
        "model.mtp.0.enorm.weight": "shared.safetensors",
    }}))

    dequant_keys = {}
    real_dequantize = hf_checkpoint_module.dequantize_mxfp4_state_dict

    def spy(state_dict, **kwargs):
        dequant_keys["seen"] = set(state_dict)
        return real_dequantize(state_dict, **kwargs)

    monkeypatch.setattr(hf_checkpoint_module, "dequantize_mxfp4_state_dict", spy)
    checkpoint = HuggingFaceCheckpoint(c_config, _preview_load_args())
    checkpoint.load(str(tmp_path), True, c_config, [0], mtp_num_layers=0)

    # Dropped before dequantization, so the discarded DSpark experts are never
    # materialized: the dequant hook saw no model.mtp.* key at all.
    assert not any(key.startswith("model.mtp.") for key in dequant_keys["seen"])
    assert trunk_base + ".weight" in dequant_keys["seen"]
    assert set(checkpoint.state_dict) == {
        "model.layers.0.input_layernorm.weight",
        trunk_base + ".weight",
    }
    assert checkpoint.state_dict[trunk_base + ".weight"].dtype == torch.bfloat16
    assert torch.equal(
        checkpoint.state_dict[trunk_base + ".weight"],
        dequantize_mxfp4(trunk[trunk_base + ".weight"], trunk[trunk_base + ".weight_scale_inv"], dtype=torch.bfloat16),
    )


def test_preview_selection_skips_dspark_mtp_and_flashindexer_shards(monkeypatch):
    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    c_config = get_yaml_config(str(PREVIEW_CONFIG), str(CONVERT_CONFIG))

    weight_map = {
        "model.embed_tokens.weight": "model-001.safetensors",
        "model.layers.43.enorm.weight": "base-mtp.safetensors",
        "model.mtp.0.enorm.weight": "dspark.safetensors",
        "model.mtp.0.e_proj.weight": "dspark.safetensors",
        "model.mtp.1.h_proj.weight": "dspark.safetensors",
        "model.mtp.2.e_proj.weight": "dspark.safetensors",
        # Preview also ships flashindexer/flashindexer.safetensors; it is absent
        # from the weight map, and a stray non-layer key must not pull it in.
        "flashindexer": "flashindexer/flashindexer.safetensors",
    }
    for layer in range(43):
        weight_map[f"model.layers.{layer}.input_layernorm.weight"] = (
            f"model-{layer // 8 + 1:03d}.safetensors"
        )
    weight_map["model.norm.weight"] = "model-006.safetensors"
    weight_map["lm_head.weight"] = "model-006.safetensors"

    selected = set(get_hf_checkpoint_names(c_config, weight_map, list(range(43)), mtp_num_layers=0))
    assert "base-mtp.safetensors" not in selected
    assert "dspark.safetensors" not in selected
    assert "flashindexer/flashindexer.safetensors" not in selected
    assert selected == {f"model-{index:03d}.safetensors" for index in range(1, 7)}
