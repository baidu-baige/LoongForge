# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ERNIE conversion through the same HF/common/MCore path as other models."""

import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from mcore_checkpoint_convert import arguments
from mcore_checkpoint_convert.huggingface.huggingface_checkpoint import (
    HuggingFaceCheckpoint,
    get_hf_checkpoint_names,
)
from mcore_checkpoint_convert.mcore.mcore_checkpoint import McoreCheckpoint
from mcore_checkpoint_convert.utils.config_utils import get_yaml_config
from mcore_checkpoint_convert.utils.utils import get_ep_map, get_layer_ids


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    monkeypatch.setenv("LOONGFORGE_PATH", str(ROOT))
    config = get_yaml_config(
        str(ROOT / "configs/models/ernie5_5/ernie5_5_flash_base.yaml"),
        str(ROOT / "configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml"),
    )
    config.data.module.update({
        "num_layers": 2,
        "compress_ratios": [128, -2],
        "hidden_size": 4,
        "ffn_hidden_size": 8,
        "num_attention_heads": 2,
        "vocab_size_in_config_file": 8,
        "num_experts": 4,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 4,
        "v_head_dim": 3,
        "d_num_attention_heads": 2,
        "d_qk_head_dim": 2,
        "d_qk_pos_emb_head_dim": 4,
        "d_kv_lora_rank": 3,
        "d_v_head_dim": 1,
    })
    hf_config = {
        "num_hidden_layers": 2,
        "compress_ratios": [128, -2],
        "hidden_size": 4,
        "vocab_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "num_attention_heads": 2,
        "head_dim": 3,
        "hybrid_mla_num_attention_heads": 2,
        "hybrid_mla_qk_nope_head_dim": 2,
        "hybrid_mla_kv_lora_rank": 3,
        "hybrid_mla_v_head_dim": 1,
        "mqa_latent_rope_adjacent_pairing": True,
        "torch_dtype": "bfloat16",
    }
    source = tmp_path / "hf"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(hf_config))
    monkeypatch.setattr(arguments, "_GLOBAL_ARGS", None)
    monkeypatch.setattr(sys, "argv", [
        "convert",
        "--load_platform=huggingface",
        "--save_platform=mcore",
        "--save_ckpt_path=" + str(tmp_path / "mcore"),
        "--expert_parallel_size=2",
        "--expert_tensor_parallel_size=1",
        "--max_workers=1",
        "--moe-grouped-gemm",
        "--safetensors",
    ])
    args = arguments.parse_args()
    return config, args, source, hf_config


def _hf_weights():
    def tensor(*shape, dtype=torch.bfloat16):
        value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        if dtype == torch.float32:
            value = value + 0.1234567
        return value.to(dtype)

    state = {
        "model.embed_tokens.weight": tensor(8, 4),
        "model.norm.weight": tensor(4),
        "lm_head.weight": tensor(8, 4),
        "model.lm_head.multimax_ranges": tensor(4),
        "model.lm_head.multimax_ts": tensor(4),
        "model.mhc_contract.hc_head_fn": tensor(4, 2, dtype=torch.float32),
        "model.mhc_contract.hc_head_base": tensor(4, dtype=torch.float32),
        "model.mhc_contract.hc_head_scale": tensor(1, dtype=torch.float32),
        "model.layers.0.self_attn.linear_q_down_proj.weight": tensor(2, 4),
        "model.layers.0.self_attn.linear_q_up_proj.weight": tensor(6, 2),
        "model.layers.0.self_attn.q_layernorm.weight": tensor(2),
        "model.layers.0.self_attn.kv_layernorm.weight": tensor(3),
        "model.layers.0.self_attn.linear_kv_proj.weight": tensor(3, 4),
        "model.layers.0.self_attn.linear_o_group_proj": tensor(2, 3),
        "model.layers.0.self_attn.core_attention.compressor.linear_wkv.weight": tensor(3, 4),
        "model.layers.0.self_attn.core_attention.compressor.linear_wgate.weight": tensor(3, 4),
        "model.layers.0.self_attn.core_attention.compressor.ape": tensor(2, 3, dtype=torch.float32),
        "model.layers.0.self_attn.core_attention.compressor.norm.weight": tensor(3),
        "model.layers.0.self_attn.core_attention.attn_sink": tensor(2, dtype=torch.float32),
        "model.layers.0.self_attn.gate_proj.weight": tensor(6, 2),
        "model.layers.0.mlp.gate_proj.weight": tensor(8, 4),
        "model.layers.0.mlp.up_proj.weight": tensor(8, 4) + 1,
        "model.layers.0.mlp.down_proj.weight": tensor(4, 8),
        "model.layers.1.self_attn.q_a_proj.weight": tensor(2, 4),
        "model.layers.1.self_attn.q_b_proj.weight": tensor(12, 2),
        "model.layers.1.self_attn.q_a_layernorm.weight": tensor(2),
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight": tensor(7, 4),
        "model.layers.1.self_attn.kv_a_layernorm.weight": tensor(3),
        "model.layers.1.self_attn.kv_b_proj.weight": tensor(6, 3),
        "model.layers.1.self_attn.core_attention.softmax_offset": tensor(2, dtype=torch.float32),
        "model.layers.1.self_attn.gate_proj.weight": tensor(2, 2),
        "model.layers.1.block_sparse_moe.gate.weight": tensor(4, 4, dtype=torch.float32),
        "model.layers.1.block_sparse_moe.gate.weight_1": tensor(4, 4, dtype=torch.float32),
        "model.layers.1.block_sparse_moe.gate.routed_scaling_factor_param": tensor(4, dtype=torch.float32),
        "model.layers.1.block_sparse_moe.e_score_correction_bias": tensor(4, dtype=torch.float32),
        "model.layers.1.block_sparse_moe.fc1_latent_proj.weight": tensor(3, 4, dtype=torch.float32),
        "model.layers.1.block_sparse_moe.fc2_latent_proj.weight": tensor(4, 3, dtype=torch.float32),
        "model.layers.1.block_sparse_moe.latent_norm.weight": tensor(3),
        "model.layers.2.enorm.weight": tensor(4),
        "model.layers.2.hnorm.weight": tensor(4),
        "model.layers.2.e_proj.weight": tensor(4, 4),
        "model.layers.2.h_proj.weight": tensor(4, 4),
        "model.layers.2.shared_head.norm.weight": tensor(4),
        "model.layers.2.hc_head_fn": tensor(4, 2, dtype=torch.float32),
        "model.layers.2.hc_head_base": tensor(4, dtype=torch.float32),
        "model.layers.2.hc_head_scale": tensor(1, dtype=torch.float32),
        "model.layers.2.block_sparse_moe.e_score_correction_bias": tensor(4, dtype=torch.float32) + 1,
    }
    for layer_id in (0, 1):
        prefix = f"model.layers.{layer_id}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            state[prefix + norm + ".weight"] = tensor(4)
        state[prefix + "self_attn.o_proj.weight"] = tensor(4, 2)
        for name in ("vha_postmix_U", "vha_postmix_V"):
            state[prefix + "self_attn." + name] = tensor(2, 1, dtype=torch.float32)
        for branch in ("self_attention", "mlp"):
            hc = prefix + branch + "_hyper_connection."
            state[hc + "mapping_proj.weight"] = tensor(4, 2, dtype=torch.float32)
            state[hc + "bias"] = tensor(2, dtype=torch.float32)
            for part in ("pre", "post", "res"):
                state[hc + "alpha_" + part] = tensor(1, dtype=torch.float32)
    for expert in range(4):
        prefix = f"model.layers.1.block_sparse_moe.experts.{expert}."
        state[prefix + "w1.weight"] = tensor(4, 3) + expert
        state[prefix + "w3.weight"] = tensor(4, 3) + expert + 16
        state[prefix + "w2.weight"] = tensor(3, 4) + expert
    for name, shape in (("w1", (4, 4)), ("w3", (4, 4)), ("w2", (4, 4))):
        state[f"model.layers.1.block_sparse_moe.shared_experts.{name}.weight"] = tensor(*shape)
    return state


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("adjacent", [False, True])
def test_hf_mcore_hf_uses_shared_converter(conversion, pp, adjacent):
    config, args, source, hf_config = conversion
    args.pipeline_model_parallel_size = pp
    hf_config["mqa_latent_rope_adjacent_pairing"] = adjacent
    config.data.args.huggingface.mqa_latent_rope_adjacent_pairing = adjacent
    (source / "config.json").write_text(json.dumps(hf_config))
    expected = _hf_weights()
    save_file(expected, source / "model.safetensors")
    _, _, experts = get_ep_map(4, 2)
    actual = {}
    for p in range(pp):
        layers = {p: get_layer_ids(config, args, p)}
        hf = HuggingFaceCheckpoint(config, args)
        hf.load(str(source), True, config, layers[p], mtp_num_layers=1)
        common = hf.convert_to_common(layers, expert_dict=experts)
        mcore = McoreCheckpoint(config, args)
        mcore.convert_from_common(common, None, layers, expert_dict=experts)
        release = str(Path(args.save_ckpt_path) / "release")
        mcore.load(release, layers, expert_dict=experts)
        assert torch.equal(
            mcore.m_dict[0]["model"]["embedding.word_embeddings.weight"],
            expected["model.embed_tokens.weight"],
        )
        common = mcore.convert_to_common(layers, expert_dict=experts)
        exported = HuggingFaceCheckpoint(config, args).convert_from_common(
            common, layers, expert_dict=experts, save_file=False,
        )
        assert not (actual.keys() & exported.keys())
        actual.update(exported)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype, name
        assert torch.equal(actual[name], expected[name]), name
    rank_dirs = list((Path(args.save_ckpt_path) / "release").glob("mp_rank_*"))
    assert len(rank_dirs) == pp * 2


def test_hf_fp8_load_materializes_partial_blocks(conversion):
    config, args, source, hf_config = conversion
    hf_config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (source / "config.json").write_text(json.dumps(hf_config))
    name = "model.layers.0.mlp.down_proj.weight"
    weight = torch.arange(12.0).reshape(3, 4).to(torch.float8_e4m3fn)
    save_file({name: weight, name + "_scale_inv": torch.tensor([[0.5]])}, source / "model.safetensors")
    hf = HuggingFaceCheckpoint(config, args)
    hf.load(str(source), True, config, [0], mtp_num_layers=1)
    assert torch.equal(hf.state_dict[name], (weight.float() * 0.5).bfloat16())
    assert name + "_scale_inv" not in hf.state_dict


def test_last_decoder_stage_loads_output_shards_with_mtp(conversion):
    config, args, _, _ = conversion
    names = {
        "model.norm.weight": "norm.safetensors",
        "lm_head.weight": "head.safetensors",
        "model.layers.2.enorm.weight": "mtp.safetensors",
    }
    files = get_hf_checkpoint_names(config, names, [1], mtp_num_layers=1, args=args)
    assert set(files) == {"norm.safetensors", "head.safetensors"}


def test_global_fp8_weight_selects_its_separate_scale_shard(conversion):
    config, args, _, _ = conversion
    names = {
        "model.embed_tokens.weight": "embedding.safetensors",
        "model.embed_tokens.weight_scale_inv": "scales.safetensors",
    }
    files = get_hf_checkpoint_names(config, names, [0], mtp_num_layers=1, args=args)
    assert set(files) == {"embedding.safetensors", "scales.safetensors"}


@pytest.mark.parametrize("tp,etp", [(2, 1), (1, 2)])
def test_unsupported_parallelism_fails_before_reading_weights(conversion, tp, etp):
    config, args, _, _ = conversion
    args.tensor_model_parallel_size = tp
    args.expert_tensor_parallel_size = etp
    with pytest.raises(ValueError, match="TP=1 and ETP=1"):
        HuggingFaceCheckpoint(config, args)


@pytest.mark.parametrize("symlink", [False, True])
def test_hf_export_preserves_source_checkpoint(conversion, tmp_path, symlink):
    from mcore_checkpoint_convert.module_convertor.model import _convert_checkpoint

    _, args, source, _ = conversion
    target = source
    if symlink:
        target = tmp_path / "output"
        target.symlink_to(source, target_is_directory=True)
    args.config_file = str(ROOT / "configs/models/ernie5_5/ernie5_5_flash_base.yaml")
    args.convert_file = str(ROOT / "configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml")
    args.save_platform = "huggingface"
    args.save_ckpt_path = str(target)
    args.hf_official_config_file = str(source / "config.json")
    (source / "model.safetensors").write_bytes(b"original weights")
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(ValueError, match="different directory"):
        _convert_checkpoint(args)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
