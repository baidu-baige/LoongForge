# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Kimi K3 operators and checkpoint normalization."""

from types import SimpleNamespace

import pytest
import torch

from loongforge.models.foundation.kimi_k3.kimi_k3_ops import (
    SITU_BETA,
    SITU_LINEAR_BETA,
    situ_and_mul,
)
from loongforge.models.foundation.kimi_k3.kimi_k3_pipeline import (
    bank_num_rows,
    pack_stage_boundary,
    unpack_stage_boundary,
)
from tools.convert_checkpoint.common.common_checkpoint import (
    K3_OUTPUT_ATTN_RES_NORM,
    LAYER_LOCAL_LAST_NAMES,
    LAYER_PREFIX,
)
from tools.convert_checkpoint.mcore.mcore_base import McoreBase
from tools.convert_checkpoint.kimi_k3.transforms import normalize_kimi_k3_state_dict
from tools.convert_checkpoint.utils.config_utils import convert_vlm_config

A_LOG = "language_model.model.layers.0.self_attn.A_log"
GATE = "language_model.model.layers.0.block_sparse_moe.gate"
EXPERT = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"


@pytest.fixture(name="config")
def _config():
    """Stand in for the converter's config object, which reads like a dict."""
    module = {"model_type": "kimi_k3", "kimi_linear_num_heads": 4, "num_experts": 8}
    return SimpleNamespace(get={"module": module}.get)


def test_situ_matches_closed_form():
    inputs = torch.randn(2, 4, 16, dtype=torch.float32)
    gate, linear = torch.chunk(inputs, 2, dim=-1)
    expected = (
        SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    ) * (SITU_LINEAR_BETA * torch.tanh(linear / SITU_LINEAR_BETA))
    torch.testing.assert_close(situ_and_mul(inputs), expected)


def test_stage_boundary_pack_unpack_and_bank_schedule():
    prefix_sum = torch.randn(5, 2, 16, dtype=torch.bfloat16)
    block_residual = torch.randn(5, 2, 3, 16, dtype=torch.bfloat16)

    packed = pack_stage_boundary(prefix_sum, block_residual)
    prefix_out, bank_out = unpack_stage_boundary(packed, hidden_size=16, num_rows=3)

    torch.testing.assert_close(prefix_out, prefix_sum, rtol=0, atol=0)
    torch.testing.assert_close(bank_out, block_residual, rtol=0, atol=0)
    assert [bank_num_rows(layer_idx, 12) for layer_idx in (1, 12, 13, 24, 25)] == [
        1,
        1,
        2,
        2,
        3,
    ]

    with pytest.raises(ValueError, match="stage-boundary payload width"):
        unpack_stage_boundary(torch.zeros(2, 1, 3 * 16), hidden_size=16, num_rows=3)


def test_final_attnres_uses_the_pipeline_stage_local_layer_index():
    converter = McoreBase.__new__(McoreBase)
    converter.name_map = {
        LAYER_PREFIX: "decoder.layers",
        K3_OUTPUT_ATTN_RES_NORM: "output_attn_res_norm",
    }
    converter.layer_prefix = "decoder.layers"
    converter.untie_embeddings_and_output_weights = True
    converter.pp = 2
    converter.add_embed_padding = False

    paths = converter.build_mcore_paths(K3_OUTPUT_ATTN_RES_NORM, m_layer_id=1)

    assert K3_OUTPUT_ATTN_RES_NORM in LAYER_LOCAL_LAST_NAMES
    assert paths.common_key == K3_OUTPUT_ATTN_RES_NORM
    assert paths.mcore_weight_path == "decoder.layers.1.output_attn_res_norm.weight"


def test_vlm_prefix_is_applied_by_the_layer_prefix_only():
    name_map = {
        "mcore": {
            LAYER_PREFIX: "decoder.layers",
            K3_OUTPUT_ATTN_RES_NORM: "output_attn_res_norm",
        },
        "huggingface": {},
    }
    config = SimpleNamespace(get={"name_map": name_map}.get)

    convert_vlm_config(config, adapter={}, vision_patch={}, for_vlm=True)

    assert name_map["mcore"][LAYER_PREFIX] == "foundation_model.decoder.layers"
    assert name_map["mcore"][K3_OUTPUT_ATTN_RES_NORM] == "output_attn_res_norm"


def test_a_log_is_truncated_to_active_heads(config):
    state = {A_LOG: torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])}
    normalize_kimi_k3_state_dict(state, config)
    torch.testing.assert_close(state[A_LOG], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_a_log_with_nonzero_padding_is_rejected(config):
    with pytest.raises(ValueError, match="A_log padding"):
        normalize_kimi_k3_state_dict({A_LOG: torch.ones(6)}, config)


def test_router_is_sliced_to_expert_count(config):
    state = {
        f"{GATE}.weight": torch.randn(10, 6),
        f"{GATE}.e_score_correction_bias": torch.randn(10),
    }
    normalize_kimi_k3_state_dict(state, config)
    assert state[f"{GATE}.weight"].shape == (8, 6)
    assert state[f"{GATE}.e_score_correction_bias"].shape == (8,)


def test_mxfp4_expert_names_are_normalized(config):
    state = {
        f"{EXPERT}.weight_packed": torch.ones(2, dtype=torch.int8),
        f"{EXPERT}.weight_scale": torch.ones(1, dtype=torch.uint8),
    }
    normalize_kimi_k3_state_dict(state, config)
    assert f"{EXPERT}.weight" in state
    assert f"{EXPERT}.scale" in state
    assert not any(key.endswith(("_packed", "weight_scale")) for key in state)
