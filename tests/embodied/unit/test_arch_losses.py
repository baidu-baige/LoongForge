# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for decoupled supervision losses and cross-model extensibility (T40).

Covers the four Lane C v2 §1.17 contracts:
  1. Pure non-diffusion ``WorldModelSpec(diffusion=None, losses=..., loss_masks=...)``
     runs training forward + backward with zero RNG consumption, while V5 rejects
     ``diffusion=None`` whenever any stream declares ``needs_timestep=True``.
  2. ``LossSpec.target_key`` reads explicit targets from the batch and supports
     per-term ``LossSpec.loss_mask`` (taking precedence over stream-level masks),
     with ``ArchModelBase._target_keys`` / ``_mask_keys`` forwarding them.
  3. ``LossSpec.gate`` (both built-in ``BatchGate`` and a timestep-based custom
     ``LossGate``) averages only active samples, logs detached 0-D
     ``loss_<name>_active`` metrics, and preserves autograd gradients on an
     all-False micro-batch so DDP/FSDP never stalls.
  4. ``LossSpec.reducer`` receives unreduced errors alongside per-frame
     ``[B, F]`` timestep weights, ``valid``, ``active`` and ``layout`` so models
     can weight before reducing across frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from loongforge.embodied.model.arch.broadcast import BroadcastSpec
from loongforge.embodied.model.arch.builder import WorldModel
from loongforge.embodied.model.arch.diffusion.weights import Bsmntw
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.arch.heads import HeadSpec
from loongforge.embodied.model.arch.losses import (
    BatchGate,
    LossMaskSpec,
    LossSpec,
    MeanSquaredError,
    TokenMeanSquaredError,
)
from loongforge.embodied.model.arch.spec import WorldModelSpec
from loongforge.embodied.model.arch.streams.spec import StreamSpec
from tests.embodied.unit.test_arch_blocks import tiny_spec
from tests.embodied.unit.test_arch_world_model import BATCH, fake_batch, world


def _non_diffusion_spec(
    *,
    losses: tuple[LossSpec, ...] | None = None,
    loss_masks: dict[str, LossMaskSpec] | None = None,
) -> WorldModelSpec:
    """Turn the shrunk fastwam topology into a pure non-diffusion model."""
    base = tiny_spec("fastwam")
    group = base.attention_groups[0]
    streams: list[StreamSpec] = []
    for stream in group.streams:
        streams.append(replace(
            stream,
            needs_timestep=False,
            timestep_embed_width=None,
            token_timestep=None,
        ))
    heads: list[HeadSpec] = []
    for head in base.heads:
        # Non-diffusion heads take hidden states without timestep modulation.
        if head.name == "action_flow":
            heads.append(replace(head, needs_timestep_embedding=False))
        else:
            heads.append(replace(
                head,
                impl=replace(head.impl, params={"form": "linear"}),
                needs_timestep_embedding=False,
            ))
    if losses is None:
        losses = (
            LossSpec(
                name="action_reg",
                head="action_flow",
                objective=TokenMeanSquaredError(),
                target_key="action_target",
            ),
        )
    if loss_masks is None:
        loss_masks = {"action": LossMaskSpec("action_is_pad")}
    return WorldModelSpec(
        model_type="non_diffusion_toy",
        geometry=base.geometry,
        attention_groups=(replace(group, streams=tuple(streams)),),
        diffusion=None,
        heads=tuple(heads),
        losses=losses,
        conditions=base.conditions,
        visibility=base.visibility,
        loss_masks=loss_masks,
        token_labels=base.token_labels,
        attention_backend=base.attention_backend,
        attention_block_size=base.attention_block_size,
    )


def test_non_diffusion_spec_runs_forward_backward_without_rng():
    """``WorldModelSpec(diffusion=None)`` trains end-to-end with zero RNG draws
    and applies stream-level ``loss_masks``."""
    spec = _non_diffusion_spec()
    model = WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                       layout=spec.build_layout())
    pad = torch.tensor([
        [False, False, True, True],
        [False, True, True, True],
    ])
    batch = fake_batch(
        tiny_spec("fastwam"),
        action_target=torch.randn(BATCH, spec.geometry.action_horizon, spec.geometry.action_dim),
        action_is_pad=pad,
    )
    torch.manual_seed(123)
    state_before = torch.get_rng_state()
    loss, metrics = model(batch)
    state_after = torch.get_rng_state()
    assert torch.equal(state_before, state_after)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss_action_reg" in metrics
    loss.backward()
    action_grads = [
        p.grad for name, p in model.named_parameters()
        if name.startswith("heads.action_flow.")
    ]
    assert action_grads and all(g is not None for g in action_grads)


def test_diffusion_none_is_rejected_when_any_stream_needs_timestep():
    """V5: omitting ``diffusion`` is only legal when every stream has ``needs_timestep=False``."""
    base = tiny_spec("fastwam")
    with pytest.raises(ValueError, match="WorldModelSpec.diffusion is None"):
        replace(base, diffusion=None)


def test_non_diffusion_rejects_sample_and_timestep_weights():
    """``sample()`` and non-uniform ``weight_of`` require a ``DiffusionSpec``."""
    base = _non_diffusion_spec()
    with pytest.raises(ValueError, match="weight_of"):
        replace(base, losses=(
            LossSpec(name="bad", head="action_flow", target_key="action_target", weight_of=Bsmntw()),
        ))
    model = WorldModel(base, engine=JointAttentionEngine(backend="dense"),
                       layout=base.build_layout())
    with pytest.raises(NotImplementedError, match="has no DiffusionSpec"):
        model.sample(fake_batch(tiny_spec("fastwam")))


def test_target_key_and_per_term_loss_mask_override_stream_mask():
    """``LossSpec.target_key`` reads directly from batch and ``LossSpec.loss_mask``
    takes precedence over stream-level ``WorldModelSpec.loss_masks``."""
    base = tiny_spec("vpp2")
    losses = (
        LossSpec(name="video_flow", head="video_flow", enabled=False),
        LossSpec(
            name="aux_action",
            head="action_flow",
            objective=TokenMeanSquaredError(),
            target_key="teacher_action",
            loss_mask=LossMaskSpec("aux_is_pad"),
        ),
    )
    spec = replace(base, losses=losses)
    model = world(spec)
    horizon = spec.geometry.action_horizon
    dim = spec.geometry.action_dim
    teacher = torch.zeros(BATCH, horizon, dim)
    aux_pad = torch.tensor([[False, False, True, True], [False, True, True, True]])
    # Stream-level mask marks everything padded; per-term mask must win.
    stream_pad = torch.ones(BATCH, horizon, dtype=torch.bool)
    batch = fake_batch(spec, teacher_action=teacher, aux_is_pad=aux_pad, action_is_pad=stream_pad)
    runtime = model.add_noise(batch, model.draw(batch))
    pred = torch.ones(BATCH, horizon, dim)
    pred[:, 2:, :] = 100.0
    pred[1, 1:, :] = 100.0
    outputs = {
        "video_flow": torch.zeros_like(runtime.targets["video"]),
        "action_flow": pred,
    }
    loss, metrics = model.compute_losses(None, runtime, outputs, step=0, batch=batch)
    # Every unmasked element has pred=1, teacher=0 -> MSE == 1.0 exactly.
    assert torch.allclose(metrics["loss_aux_action"], torch.tensor(1.0))
    assert torch.allclose(loss, torch.tensor(1.0))

    # Missing target_key fails loudly with KeyError naming the key.
    batch_missing = {k: v for k, v in batch.items() if k != "teacher_action"}
    with pytest.raises(KeyError, match="teacher_action"):
        model.compute_losses(None, runtime, outputs, step=0, batch=batch_missing)


@dataclass(frozen=True)
class _TimestepThresholdGate:
    """Example model-owned gate: active only where ``t <= threshold`` (e.g. InternW0-Delta t <= 500)."""

    stream: str
    threshold: float

    @property
    def batch_keys(self) -> tuple[str, ...]:
        return ()

    def __call__(self, batch, *, timesteps):
        del batch
        return timesteps[self.stream] <= self.threshold


def test_batch_gate_and_timestep_gate_preserve_grad_when_all_false():
    """``LossSpec.gate`` averages active samples, logs detached ``loss_<name>_active``,
    and keeps parameter gradients alive when the whole micro-batch is gated off."""
    base = tiny_spec("vpp2")
    losses = (
        LossSpec(name="video_flow", head="video_flow", enabled=False),
        LossSpec(
            name="action_flow",
            head="action_flow",
            objective=MeanSquaredError(),
            gate=BatchGate("has_real_action"),
        ),
    )
    spec = replace(base, losses=losses)
    model = world(spec)

    # 1. Partial activation: only sample 0 is active.
    batch = fake_batch(spec, has_real_action=torch.tensor([True, False]))
    runtime = model.add_noise(batch, model.draw(batch))
    target = runtime.targets["action"]
    pred = target.clone().detach().requires_grad_(True)
    with torch.no_grad():
        pred[0].add_(2.0)   # error^2 = 4.0
        pred[1].add_(10.0)  # gated out
    outputs = {
        "video_flow": torch.zeros_like(runtime.targets["video"]),
        "action_flow": pred,
    }
    loss, metrics = model.compute_losses(None, runtime, outputs, step=0, batch=batch)
    assert torch.allclose(loss, torch.tensor(4.0))
    assert metrics["loss_action_flow_active"].ndim == 0
    assert not metrics["loss_action_flow_active"].requires_grad
    assert torch.allclose(metrics["loss_action_flow_active"], torch.tensor(0.5))

    # 2. All-False micro-batch: exact 0.0 loss, but grad_fn stays attached (DDP/FSDP safe).
    batch_none = fake_batch(spec, has_real_action=torch.tensor([False, False]))
    pred_zero = target.clone().detach().requires_grad_(True)
    outputs_zero = {
        "video_flow": torch.zeros_like(runtime.targets["video"]),
        "action_flow": pred_zero,
    }
    loss_zero, metrics_zero = model.compute_losses(
        None, runtime, outputs_zero, step=0, batch=batch_none)
    assert loss_zero.item() == 0.0
    assert loss_zero.requires_grad
    assert metrics_zero["loss_action_flow_active"].item() == 0.0
    loss_zero.backward()
    assert pred_zero.grad is not None and torch.equal(pred_zero.grad, torch.zeros_like(pred_zero))

    # 3. Timestep-based custom gate (no batch keys).
    t_spec = replace(base, losses=(
        LossSpec(name="video_flow", head="video_flow", enabled=False),
        LossSpec(
            name="action_flow",
            head="action_flow",
            objective=MeanSquaredError(),
            gate=_TimestepThresholdGate(stream="action", threshold=500.0),
        ),
    ))
    t_model = world(t_spec)
    runtime_t = replace(
        runtime,
        model_timesteps={**runtime.model_timesteps, "action": torch.tensor([200.0, 800.0])},
    )
    loss_t, metrics_t = t_model.compute_losses(None, runtime_t, outputs, step=0, batch=batch_none)
    assert torch.allclose(loss_t, torch.tensor(4.0))
    assert torch.allclose(metrics_t["loss_action_flow_active"], torch.tensor(0.5))


def test_reducer_receives_unreduced_error_and_per_frame_weights():
    """``LossSpec.reducer`` multiplies per-frame ``[B, F]`` weights before collapsing frames."""
    seen: dict[str, object] = {}

    def per_frame_reducer(error, *, weight=None, valid=None, active=None, layout=None):
        seen["error_shape"] = tuple(error.shape)
        seen["weight_shape"] = None if weight is None else tuple(weight.shape)
        seen["has_layout"] = layout is not None
        # error is [B, C, F, H, W]; collapse non-frame axes then multiply [B, F] weight.
        per_frame = error.mean(dim=(1, 3, 4))
        assert weight is not None
        return (per_frame * weight).mean()

    @dataclass(frozen=True)
    class _PerFrameWeight:
        def check(self, sampler, *, loss: str) -> None:
            del sampler, loss

        def __call__(self, t, *, sampler, num_train_timesteps: int):
            del sampler, num_train_timesteps
            return t

    base = tiny_spec("vpp2")
    losses = (
        LossSpec(
            name="video_flow",
            head="video_flow",
            objective=MeanSquaredError(),
            broadcast=BroadcastSpec(axes=("frame",)),
            weight_of=_PerFrameWeight(),
            reducer=per_frame_reducer,
        ),
        LossSpec(name="action_flow", head="action_flow", enabled=False),
    )
    spec = replace(base, losses=losses)
    model = world(spec)
    batch = fake_batch(spec)
    runtime = model.add_noise(batch, model.draw(batch))
    frames = runtime.targets["video"].shape[2]
    frame_weights = torch.arange(1, frames + 1, dtype=torch.float32).unsqueeze(0).expand(BATCH, -1)
    runtime = replace(
        runtime,
        model_timesteps={**runtime.model_timesteps, "video": frame_weights},
    )
    target = runtime.targets["video"]
    # Frame f has squared error (f + 1)^2, weight (f + 1) -> product (f + 1)^3.
    pred = target.clone()
    for f in range(frames):
        pred[:, :, f] += float(f + 1)
    outputs = {
        "video_flow": pred,
        "action_flow": torch.zeros_like(runtime.targets["action"]),
    }
    loss, _ = model.compute_losses(None, runtime, outputs, step=0, batch=batch)
    expected = torch.tensor([(f + 1) ** 3 for f in range(frames)], dtype=torch.float32).mean()
    assert seen["error_shape"] == tuple(target.shape)
    assert seen["weight_shape"] == (BATCH, frames)
    assert seen["has_layout"] is True
    assert torch.allclose(loss, expected)


def test_shell_collects_mask_gate_and_target_keys():
    """``ArchModelBase._mask_keys`` and ``_target_keys`` pick up ``loss_masks``,
    ``LossSpec.loss_mask``, ``LossSpec.gate.batch_keys`` and ``LossSpec.target_key``."""
    from loongforge.embodied.model.arch.factory.base import ArchModelBase

    base = tiny_spec("vpp2")
    spec = replace(
        base,
        losses=(
            LossSpec(name="video_flow", head="video_flow", enabled=False),
            LossSpec(
                name="action_flow",
                head="action_flow",
                target_key="teacher_action",
                loss_mask=LossMaskSpec("aux_is_pad"),
                gate=BatchGate("has_real_action"),
            ),
        ),
    )

    class _ProbeShell:
        def __init__(self, s):
            self.spec = s

    probe = _ProbeShell(spec)
    mask_keys = ArchModelBase._mask_keys(probe)
    target_keys = ArchModelBase._target_keys(probe)
    assert mask_keys == ("action_is_pad", "action_terminal_mask", "aux_is_pad", "has_real_action")
    assert target_keys == ("teacher_action",)


def test_loss_spec_and_gate_validation_errors():
    """Invalid ``target_key``, ``loss_mask`` or ``BatchGate.key`` fail at construction."""
    with pytest.raises(ValueError, match="target_key"):
        LossSpec(name="x", head="h", target_key="")
    with pytest.raises(TypeError, match="LossMaskSpec"):
        LossSpec(name="x", head="h", loss_mask="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BatchGate.key"):
        BatchGate("")
