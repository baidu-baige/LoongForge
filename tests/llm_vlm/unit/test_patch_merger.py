# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PatchMerger projector."""

from types import SimpleNamespace

import torch

import loongforge.train  # noqa: F401 - initialize package imports in repository order
from loongforge.models.common import BaseMegatronModule
from loongforge.models.encoder.moon_vision_models import patch_merger_adapter


class _TupleLinear(torch.nn.Module):
    """Adapt torch Linear to the Megatron tuple-return convention."""

    def __init__(self, input_size, output_size, bias):
        """Initialize the wrapped linear layer."""
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        """Return the linear output and an empty bias result."""
        return self.linear(x), None


def _build_projector(monkeypatch, freeze=False):
    """Build a small projector with local torch modules."""
    norm_spec, linear_spec = object(), object()
    config = SimpleNamespace(
        model_spec=["unused", "unused"],
        layernorm_epsilon=1e-5,
        activation_func=torch.nn.functional.gelu,
        init_method=None,
        output_layer_init_method=None,
        add_bias_linear=True,
        freeze=freeze,
    )

    def fake_base_init(self, config):
        """Initialize the torch-only test stand-in."""
        torch.nn.Module.__init__(self)
        self.config = config

    monkeypatch.setattr(BaseMegatronModule, "__init__", fake_base_init)
    monkeypatch.setattr(
        patch_merger_adapter,
        "import_module",
        lambda *_: SimpleNamespace(
            layernorm=norm_spec,
            linear_fc1=linear_spec,
            linear_fc2=linear_spec,
        ),
    )

    def fake_build_module(spec, *args, **kwargs):
        """Build the matching torch-only test module."""
        if spec is norm_spec:
            return torch.nn.LayerNorm(kwargs["hidden_size"])
        return _TupleLinear(args[0], args[1], kwargs["bias"])

    monkeypatch.setattr(patch_merger_adapter, "build_module", fake_build_module)
    return patch_merger_adapter.PatchMergerMLP(
        config=config,
        input_size=2,
        output_size=5,
        spatial_merge_size=2,
    )


def test_projector_shape_and_trainable_parameters(monkeypatch):
    """PatchMerger projects merged patches without freezing parameters."""
    projector = _build_projector(monkeypatch)
    assert projector(torch.randn(2, 4, 2)).shape == (2, 1, 5)
    assert projector.linear_fc1.linear.in_features == 8
    assert projector.linear_fc2.linear.out_features == 5
    assert projector.linear_fc1.linear.bias is not None
    assert projector.linear_fc2.linear.bias is not None
    assert all(parameter.requires_grad for parameter in projector.parameters())


def test_projector_freeze(monkeypatch):
    """Freeze disables updates for every projector parameter."""
    projector = _build_projector(monkeypatch, freeze=True)
    assert not any(parameter.requires_grad for parameter in projector.parameters())
