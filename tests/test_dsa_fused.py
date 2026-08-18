# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import loongforge.train  # noqa: F401 - initialize package imports in training order
import pytest
import torch

from loongforge.models.common.experimental_attention_variant import dsa_fused


def test_fused_indexer_uses_configured_k_norm_epsilon(monkeypatch):
    recorded = []

    def init_module(self, config):
        torch.nn.Module.__init__(self)
        self.config = config

    monkeypatch.setattr(
        dsa_fused.MegatronModule,
        "__init__",
        init_module,
    )
    monkeypatch.setattr(dsa_fused, "RotaryEmbedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsa_fused, "DSAIndexerKernel", lambda: None)

    def build_module(*args, **kwargs):
        recorded.append(kwargs.get("eps"))
        return torch.nn.Identity()

    monkeypatch.setattr(dsa_fused, "build_module", build_module)

    config = SimpleNamespace(
        hidden_size=16,
        qk_pos_emb_head_dim=4,
        q_lora_rank=8,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=8,
        rope_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        init_method=lambda tensor: tensor,
        layernorm_epsilon=1e-5,
        dsa_indexer_k_norm_epsilon=1e-6,
        enable_chunkpipe=False,
    )
    submodules = dsa_fused.DSAIndexerFusedSubmodules(
        linear_wq_b=object(),
        linear_wk=object(),
        k_norm=object(),
        linear_weights_proj=object(),
    )

    dsa_fused.DSAIndexerFused(
        config=config,
        submodules=submodules,
        pg_collection=SimpleNamespace(cp=None),
    )

    assert recorded == [None, None, 1e-6, None]

    config.dsa_indexer_k_norm_epsilon = None
    dsa_fused.DSAIndexerFused(
        config=config,
        submodules=submodules,
        pg_collection=SimpleNamespace(cp=None),
    )
    assert recorded[-4:] == [None, None, 1e-5, None]


def test_fused_indexer_rejects_dense_indexer_loss():
    with pytest.raises(ValueError, match="only supports .*sparse_loss=True"):
        dsa_fused.DSAIndexerFused(
            config=SimpleNamespace(dsa_indexer_use_sparse_loss=False),
            submodules=None,
        )
