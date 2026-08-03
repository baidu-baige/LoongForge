# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.dist_checkpoint.checkpoint.hf_checkpoint_loader import (
    _load_model_state_dict,
)
from tools.dist_checkpoint.checkpoint.hf_checkpoint_converter import (
    _common_checkpoint_has_mtp_weights,
)


class _ModelWithMTP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        self.mtp = torch.nn.Linear(2, 2, bias=False)


def test_checkpoint_loader_can_initialize_only_missing_mtp_weights():
    model = _ModelWithMTP()
    initial_mtp = model.mtp.weight.detach().clone()

    missing = _load_model_state_dict(
        model,
        {"backbone.weight": torch.ones_like(model.backbone.weight)},
        allow_missing_mtp=True,
    )

    assert missing == ["mtp.weight"]
    torch.testing.assert_close(model.backbone.weight, torch.ones_like(model.backbone.weight))
    torch.testing.assert_close(model.mtp.weight, initial_mtp)


def test_checkpoint_loader_rejects_missing_non_mtp_weights():
    model = _ModelWithMTP()

    with pytest.raises(RuntimeError, match="backbone.weight"):
        _load_model_state_dict(model, {}, allow_missing_mtp=True)


def test_checkpoint_loader_rejects_unexpected_weights():
    model = _ModelWithMTP()
    state_dict = model.state_dict()
    state_dict["unexpected.weight"] = torch.ones(1)

    with pytest.raises(RuntimeError, match="unexpected.weight"):
        _load_model_state_dict(model, state_dict, allow_missing_mtp=True)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ({"layer_prefix.23.mlp.weight": torch.ones(1)}, False),
        ({"mtp_word_embeddings.weight": torch.ones(1)}, False),
        ({"layer_prefix.24.input_layernorm.weight": torch.ones(1)}, True),
        ({"layer_prefix.25.mlp.weight": torch.ones(1)}, True),
    ],
)
def test_common_checkpoint_detects_dedicated_mtp_weights(keys, expected):
    common_checkpoint = type("CommonCheckpointStub", (), {"model_dict": keys})()

    assert _common_checkpoint_has_mtp_weights(common_checkpoint, 24) is expected
