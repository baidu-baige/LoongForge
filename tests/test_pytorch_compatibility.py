# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Version-sensitive PyTorch compatibility checks.

These tests deliberately use modules whose runtime dependencies are limited to
PyTorch. They complement, rather than replace, accelerator regression tests.
"""

import os

import torch

from loongforge.embodied.model.dreamzero.modules.action_encoder import MultiEmbodimentActionEncoder
from loongforge.embodied.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler


def test_requested_torch_version() -> None:
    expected = os.environ.get("EXPECTED_TORCH_VERSION")
    if expected is not None:
        installed_version = torch.__version__.split("+", 1)[0]
        assert installed_version == expected, (
            f"expected PyTorch {expected}, got {torch.__version__}"
        )


def test_action_encoder() -> None:
    torch.manual_seed(0)
    model = MultiEmbodimentActionEncoder(
        action_dim=4,
        hidden_size=8,
        num_embodiments=2,
    )
    output = model(
        torch.randn(2, 3, 4),
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        torch.tensor([0, 1]),
    )

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()


def test_flow_match_scheduler() -> None:
    scheduler = FlowMatchScheduler(num_inference_steps=8, num_train_timesteps=100)
    samples = torch.zeros(2, 3)
    noise = torch.ones(2, 3)
    timesteps = scheduler.timesteps[[0, 3]].cpu()

    noisy_samples = scheduler.add_noise(samples, noise, timesteps)

    assert noisy_samples.shape == samples.shape
    assert torch.isfinite(noisy_samples).all()
