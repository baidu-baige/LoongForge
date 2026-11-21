# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 action-stage recipe over the generic WorldModel orchestration.

The wrapper owns VAE/CLIP/text encoding and channel-concat conditioning. In
particular, zero-padded clean video latents are only a shape template for noise;
Libero does NOT replace the first noisy latent with a clean latent by default.
Runtime diffusion coordinates remain normalized. Only ``run_group`` converts
them to WAN's 1000-step embedding domain; no stream-side adapter is required.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .builder import DiffusionRuntime, WorldModel
from .diffusion.timesteps import UniformShift
from .layout import TokenLayout
from .spec.world import WorldModelSpec
from .streams.conditions import ConditionBus
from .streams.engine import AttentionGroup, EngineOutput, JointAttentionEngine, StreamEngine

__all__ = ["VPP2TrainingWorldModel", "VPP2DiffusionRuntime",
           "VPP2ActionScheduler", "masked_action_loss"]


class VPP2ActionScheduler:
    """Master's continuous shifted-uniform scheduler, in the WAN input domain.

    Keep sampling in fp32, cast AFTER multiplying by 1000, and compute weight
    statistics on the descending, endpoint-excluding fp64 grid. These details
    matter for master parity, especially when actions are bf16.
    """

    num_train_timesteps = 1000

    def __init__(self, shift: float = 5.0) -> None:
        if not 0 < shift < float("inf"):
            raise ValueError("action shift must be finite and positive")
        self.shift = float(shift)
        self.eps = 1e-10
        steps = self.num_train_timesteps
        u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t = self._phi(u) * float(steps)
        y = torch.exp(-2.0 * ((t - steps / 2.0) / steps) ** 2)
        self._y_min = float(y.min().item())
        self._weight_norm_const = float((y - self._y_min).mean().item())

    def _phi(self, u: Tensor) -> Tensor:
        return self.shift * u / (1.0 + (self.shift - 1.0) * u)

    def sample_training_t(self, batch_size: int, device: torch.device,
                          dtype: torch.dtype) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        return (self._phi(u) * float(self.num_train_timesteps)).to(dtype=dtype)

    def training_weight(self, timestep: Tensor) -> Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - steps / 2.0) / steps) ** 2)
        weight = (y - self._y_min) / (self._weight_norm_const + self.eps)
        return weight.reshape(()) if weight.numel() == 1 else weight

    def add_noise(self, sample: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            device=sample.device, dtype=sample.dtype)
        if sigma.ndim:
            sigma = sigma.view(-1, *([1] * (sample.ndim - 1)))
        return (1 - sigma) * sample + sigma * noise

    @staticmethod
    def training_target(sample: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        del timestep
        return noise - sample


def masked_action_loss(prediction: Tensor, target: Tensor,
                       training_weight: Tensor,
                       action_is_pad: Tensor | None = None) -> Tensor:
    """fp32 dimension MSE, per-sample valid-token mean, then weighted batch mean."""
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("action prediction and target must match [B,T,D]")
    if not all(prediction.shape):
        raise ValueError("action prediction dimensions must be nonempty")
    if training_weight.numel() not in (1, prediction.shape[0]):
        raise ValueError("training_weight must be scalar or have one value per sample")
    token_loss = F.mse_loss(prediction.float(), target.float(),
                           reduction="none").mean(dim=2)
    if action_is_pad is None:
        per_sample = token_loss.mean(dim=1)
    else:
        if action_is_pad.shape != prediction.shape[:2]:
            raise ValueError("action_is_pad must match action [B,T]")
        valid = (~action_is_pad.to(device=token_loss.device, dtype=torch.bool)).to(
            dtype=token_loss.dtype)
        per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    weight = training_weight.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weight.reshape(-1)).mean()


@dataclass
class VPP2DiffusionRuntime(DiffusionRuntime):
    """Raw padding is distinct from padding after terminal inclusion."""

    action_is_pad: Tensor | None = None
    action_loss_is_pad: Tensor | None = None
    # Preserve the master dtype/rounding independently of normalized runtime t.
    model_timesteps: dict[str, Tensor] = field(default_factory=dict)


class VPP2TrainingWorldModel(WorldModel):
    """Action-only VPP2; forward is inherited without replacing the architecture.

    Batch: stream ``batch_key`` tensors video [B,C,F,H,W], action [B,T,D],
    plus each condition's ``source_key``. Optional masks: action_is_pad [B,T],
    action_terminal_mask [B,T] (subset of padding), image_is_pad [B,N].
    ``fuse_vae_embedding_in_latents`` defaults to False, as in master Libero.
    Explicit True uses the wrapper-materialized video's first clean latent.

    The engine must implement frozen-stream execution. A shallow engine copy
    adds video to frozen_streams without changing an engine shared by callers.
    Shared condition encoders remain trainable through the action branch.
    """

    def __init__(self, spec: WorldModelSpec, *, engine: StreamEngine,
                 layout: TokenLayout, training_stage: str = "action") -> None:
        if training_stage != "action":
            raise NotImplementedError("VPP2TrainingWorldModel supports only training_stage='action'")
        if {stream.name for stream in spec.streams} != {"video", "action"}:
            raise ValueError("VPP2 action training requires video and action streams")
        if len(spec.attention_groups) != 1 or set(spec.attention_groups[0].stream_names) != {"video", "action"}:
            raise ValueError("VPP2 action training requires one video/action attention group")
        sampler = spec.diffusion.sampler_for("action")
        if not isinstance(sampler.distribution, UniformShift) or not sampler.broadcast.is_per_sample:
            raise ValueError("VPP2 action training requires per-sample UniformShift")
        if not isinstance(engine, JointAttentionEngine):
            raise TypeError("VPP2 action training requires JointAttentionEngine frozen_streams support")
        if "action" in engine.frozen_streams:
            raise ValueError("the action stream must not be frozen")
        recipe_engine = copy(engine)
        recipe_engine.frozen_streams = engine.frozen_streams | frozenset({"video"})
        super().__init__(spec, engine=recipe_engine, layout=layout)
        self.training_stage = training_stage
        self.action_scheduler = VPP2ActionScheduler(sampler.distribution.shift)
        self.blocks["video"].requires_grad_(False)
        for head in spec.heads:
            if head.source_stream == "video":
                self.heads[head.name].requires_grad_(False)
        for condition in spec.conditions:
            if condition.name in self.condition_encoders and set(condition.targets) == {"video"}:
                self.condition_encoders[condition.name].requires_grad_(False)
        action_losses = [loss for loss in spec.diffusion.losses if loss.head == "action_flow"]
        if len(action_losses) != 1:
            raise ValueError("VPP2 action training requires exactly one action_flow loss")
        self._action_loss_spec = action_losses[0]

    def _latents(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        streams = {stream.name: stream for stream in self.spec.streams}
        video = self._raw_tensor(batch, streams["video"])
        action = self._raw_tensor(batch, streams["action"])
        if video.ndim != 5 or action.ndim != 3 or video.shape[0] != action.shape[0]:
            raise ValueError("video [B,C,F,H,W] and action [B,T,D] must share batch size")
        if not video.is_floating_point() or not action.is_floating_point():
            raise ValueError("video and action must be floating point")
        if not all(video.shape) or not all(action.shape):
            raise ValueError("video and action dimensions must be nonempty")
        return video, action

    def sample_timesteps(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        video, action = self._latents(batch)
        t = self.action_scheduler.sample_training_t(action.shape[0], action.device, action.dtype)
        # fp64 normalized coordinates preserve the already-cast master t exactly
        # through the reversible input-scale boundary, including bf16 rounding.
        return {"video": torch.ones(video.shape[0], device=video.device, dtype=torch.float64),
                "action": t.to(torch.float64) / 1000.0}

    def build_conditions(self, batch: Mapping[str, Any]) -> ConditionBus:
        return super().build_conditions(batch)

    def add_noise(self, batch: Mapping[str, Any],
                  timesteps: Mapping[str, Tensor]) -> VPP2DiffusionRuntime:
        video, action = self._latents(batch)
        for name, tensor in (("video", video), ("action", action)):
            t = timesteps[name]
            if t.shape != (tensor.shape[0],) or not bool(torch.isfinite(t).all()):
                raise ValueError(f"{name} timestep must be finite [B]")
            if bool(((t < 0) | (t > 1)).any()):
                raise ValueError(f"{name} timestep must lie in [0,1]")
        if not bool((timesteps["video"] == 1).all()):
            raise ValueError("action-stage video timestep must be pure noise (1)")
        raw_pad = batch.get("action_is_pad")
        if raw_pad is not None:
            if not isinstance(raw_pad, Tensor) or raw_pad.shape != action.shape[:2]:
                raise ValueError("action_is_pad must match action [B,T]")
            raw_pad = raw_pad.to(device=action.device, dtype=torch.bool).clone()
        loss_pad = raw_pad
        terminal = batch.get("action_terminal_mask")
        if terminal is not None:
            if raw_pad is None or not isinstance(terminal, Tensor) or terminal.shape != raw_pad.shape:
                raise ValueError("action_terminal_mask must match action_is_pad [B,T]")
            terminal = terminal.to(device=action.device, dtype=torch.bool)
            if bool((terminal & ~raw_pad).any()):
                raise ValueError("action_terminal_mask must be a subset of action_is_pad")
            loss_pad = raw_pad & ~terminal
        image_pad = batch.get("image_is_pad")
        if image_pad is not None and (
                not isinstance(image_pad, Tensor) or image_pad.ndim != 2
                or image_pad.shape[0] != video.shape[0]):
            raise ValueError("image_is_pad must be [B,N]")
        model_t = {"video": (timesteps["video"] * 1000).to(video),
                   "action": (timesteps["action"] * 1000).to(action)}
        noise_video, noise_action = torch.randn_like(video), torch.randn_like(action)
        noised_video = noise_video
        if batch.get("fuse_vae_embedding_in_latents", False):
            noised_video = noise_video.clone()
            noised_video[:, :, :1] = video[:, :, :1].detach()
        return VPP2DiffusionRuntime(
            timesteps=dict(timesteps),
            noised={"video": noised_video,
                    "action": self.action_scheduler.add_noise(action, noise_action, model_t["action"])},
            noise={"video": noise_video, "action": noise_action},
            targets={"action": self.action_scheduler.training_target(action, noise_action, model_t["action"])},
            attention_key_valid=batch.get("attention_key_valid"),
            attention_labels=batch.get("attention_labels"),
            action_is_pad=raw_pad, action_loss_is_pad=loss_pad, model_timesteps=model_t)

    def stream_inputs(self, runtime: DiffusionRuntime) -> dict[str, Tensor]:
        return super().stream_inputs(runtime)

    def run_group(self, group: AttentionGroup, inputs: Mapping[str, Tensor],
                  conditions: ConditionBus, runtime: DiffusionRuntime) -> EngineOutput:
        if not isinstance(runtime, VPP2DiffusionRuntime):
            raise TypeError("VPP2 run_group requires VPP2DiffusionRuntime")
        index = self.spec.attention_groups.index(group)
        plan = self.plans[index]
        labels = runtime.attention_labels if plan.visibility is not None else None
        plan = plan.bind(labels=labels, key_valid=runtime.attention_key_valid)
        return self.engine.run(
            group, {name: inputs[name] for name in group.stream_names},
            {name: self.blocks[name] for name in group.stream_names},
            plan, self.layout, conditions=conditions,
            timesteps={name: runtime.model_timesteps[name] for name in group.stream_names},
            captures={name: layers for name, layers in self.captures.items()
                      if name in group.stream_names})

    def compute_losses(self, output: EngineOutput, runtime: DiffusionRuntime,
                       outputs: Mapping[str, Tensor], *,
                       step: int) -> tuple[Tensor, dict[str, Any]]:
        del output
        if not isinstance(runtime, VPP2DiffusionRuntime):
            raise TypeError("VPP2 compute_losses requires VPP2DiffusionRuntime")
        loss_action = masked_action_loss(
            outputs["action_flow"], runtime.targets["action"],
            self.action_scheduler.training_weight(runtime.model_timesteps["action"]),
            runtime.action_loss_is_pad) * self._action_loss_spec.weight_at(step)
        return loss_action, {"loss_action": loss_action.detach(),
                             "loss_video": loss_action.detach().new_zeros(())}
