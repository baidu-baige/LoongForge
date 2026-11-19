# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU compatibility checks for the built wheel and a DreamZero model path."""

from importlib import metadata
import os
from pathlib import Path
import shutil

import pytest
import torch

import loongforge
from loongforge.embodied.model.dreamzero.modules.action_encoder import MultiEmbodimentActionEncoder
from loongforge.embodied.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from loongforge.embodied.model.dreamzero.modules.wan_video_text_encoder import WanTextEncoder


def test_installed_package() -> None:
    expected_torch = os.environ.get("EXPECTED_TORCH_VERSION")
    if expected_torch is None:
        pytest.skip("This check requires an installed wheel and EXPECTED_TORCH_VERSION")

    assert torch.__version__.split("+", 1)[0] == expected_torch
    assert torch.version.cuda is None
    distribution = metadata.distribution("LoongForge")
    assert Path(loongforge.__file__).resolve() == distribution.locate_file(
        "loongforge/__init__.py"
    ).resolve()
    assert distribution.entry_points.select(group="console_scripts", name="LoongForge")
    assert shutil.which("LoongForge") is not None
    print(f"LoongForge {distribution.version} imported from {loongforge.__file__}")


def test_dreamzero_action_encoding_and_scheduler() -> None:
    torch.manual_seed(0)
    encoder = MultiEmbodimentActionEncoder(action_dim=4, hidden_size=8, num_embodiments=2)
    scheduler = FlowMatchScheduler(num_inference_steps=8, num_train_timesteps=100)
    samples = torch.zeros(2, 3, 4)
    timesteps = scheduler.timesteps[[0, 3]]

    noisy_samples = scheduler.add_noise(samples, torch.ones_like(samples), timesteps)
    expected = scheduler.sigmas[[0, 3], None, None].expand_as(samples)
    torch.testing.assert_close(noisy_samples, expected)
    output = encoder(noisy_samples, timesteps[:, None].expand(2, 3), torch.tensor([0, 1]))

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()


def test_dreamzero_wan_text_encoder_forward_and_backward() -> None:
    torch.manual_seed(0)
    encoder = WanTextEncoder(
        vocab=16, dim=8, dim_attn=8, dim_ffn=16, num_heads=2, num_layers=2, dropout=0,
    )
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    output = encoder(tokens, mask=tokens.ne(0))
    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    gradient = encoder.token_embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
