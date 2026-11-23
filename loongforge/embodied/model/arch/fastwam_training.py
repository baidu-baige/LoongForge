# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM's joint training recipe, executed by the actual arch engine.

The wrapper supplies clean VAE latents, encoded conditions and pixel-frame
``image_is_pad`` [B, 1+4k]. No pixel encoding or legacy model forward lives here.
Video engine timesteps are [B,L_video] in WAN's 1000-step domain; action
timesteps are [B]. Runtime/loss coordinates remain normalized [B].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .builder import DiffusionRuntime, WorldModel
from .diffusion.timesteps import UniformShift
from .heads.flow import PatchFlowHead
from .layout import TokenLayout
from .spec.world import WorldModelSpec
from .streams.blocks import ActionStream, WanVideoStream
from .streams.blocks.wan_dit import (
    QKV, RMSNorm, SelfAttentionIO, SlottedCrossAttention, WanDiTBlock,
)
from .streams.conditions import ConditionBus
from .streams.engine import AttentionGroup, EngineOutput, StreamEngine
from .streams.state import StreamState
from .vpp2_training import VPP2ActionScheduler, masked_action_loss

__all__ = ["FastWAMTrainingWorldModel", "FastWAMDiffusionRuntime",
           "FastWAMVideoStream", "FastWAMActionStream", "FastWAMVideoHead",
           "video_loss_per_sample"]


@dataclass
class FastWAMConditionBus(ConditionBus):
    """Invocation-local validity masks, since generic ConditionBus has none."""

    slot_masks: dict[str, dict[str, Tensor]] = field(default_factory=dict)


class _MaskedContext(dict):
    def __init__(self, values, masks):
        super().__init__(values)
        self.masks = masks


def _rotary_pairs(x, cos, sin, *, conjugate=False):
    if x.is_cuda:
        from flash_attn.ops.triton.rotary import apply_rotary

        return apply_rotary(
            x, cos, sin, interleaved=True, conjugate=conjugate)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    if conjugate:
        s = -s
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * c - odd * s, even * s + odd * c), -1).flatten(-2)


class _FastWAMRotary(torch.autograd.Function):
    """Master's input-dtype rotation and conjugate-rotation backward."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rotary_pairs(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        return _rotary_pairs(
            grad_output.contiguous(), cos, sin, conjugate=True), None, None


class _FastWAMSelfAttentionIO(SelfAttentionIO):
    @classmethod
    def from_module(cls, module: SelfAttentionIO):
        result = cls.__new__(cls)
        torch.nn.Module.__init__(result)
        result.num_heads, result.head_dim = module.num_heads, module.head_dim
        for name in ("q", "k", "v", "o", "norm_q", "norm_k"):
            setattr(result, name, getattr(module, name))
        result.train(module.training)
        return result

    def qkv(self, x: Tensor, *, freqs: Tensor | None) -> QKV:
        q = self._heads(self.norm_q(self.q(x)))
        k = self._heads(self.norm_k(self.k(x)))
        v = self._heads(self.v(x))
        if freqs is not None:
            if freqs.shape[0] < x.shape[1]:
                raise ValueError("FastWAM RoPE table does not cover the sequence")
            freqs = freqs[:x.shape[1]]
            # Master first materializes FP32 tables, then casts to Q/K dtype.
            cos = freqs[:, 0].real.float().contiguous().to(q)
            sin = freqs[:, 0].imag.float().contiguous().to(q)
            q = _FastWAMRotary.apply(q, cos, sin)
            k = _FastWAMRotary.apply(k, cos, sin)
        return QKV(q=q, k=k, v=v)


class _MaskedCrossAttention(SlottedCrossAttention):
    """Same weights and SDPA math, with actual per-sample context validity."""

    @classmethod
    def from_module(cls, module: SlottedCrossAttention):
        # Reuse already-built modules without new parameters or RNG draws.
        result = cls.__new__(cls)
        torch.nn.Module.__init__(result)
        result.slots, result.num_heads, result.head_dim = (
            module.slots, module.num_heads, module.head_dim)
        for name in ("q", "o", "norm_q", "kv"):
            setattr(result, name, getattr(module, name))
        return result

    def forward(self, x: Tensor, context: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(context, _MaskedContext) or not context.masks:
            return super().forward(x, context)
        if set(context) != set(self.slots):
            raise KeyError("FastWAM context slots must match cross-attention slots")
        shape = (self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).unflatten(-1, shape).transpose(1, 2)
        total = None
        for slot in self.slots:
            kv, ctx = self.kv[slot], context[slot]
            k = kv.norm_k(kv.k(ctx)).unflatten(-1, shape).transpose(1, 2)
            v = kv.v(ctx).unflatten(-1, shape).transpose(1, 2)
            mask = context.masks.get(slot)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None if mask is None else mask[:, None, None, :])
            total = out if total is None else total + out
        return self.o(total.transpose(1, 2).flatten(2))


class _FastWAMBlock(WanDiTBlock):
    @classmethod
    def from_module(cls, block: WanDiTBlock):
        result = cls.__new__(cls)
        torch.nn.Module.__init__(result)
        for name, module in block.named_children():
            setattr(result, name, module)
        result.modulation = block.modulation
        result.train(block.training)
        return result

    @staticmethod
    def _norm_modulate(norm, x, shift, scale):
        from loongforge.embodied.model.fastwam.utils.norm_modulate import (
            triton_norm_modulate,
        )

        # Master tokenizes into contiguous hidden states. Preserve its fused
        # kernel dispatch even when arch residuals retain a transposed view.
        x = x.contiguous()
        return triton_norm_modulate(
            x, scale.expand_as(x), shift.expand_as(x), float(norm.eps))

    def attention_io(self, x, *, shift_msa, scale_msa, freqs):
        if not x.is_cuda:
            return super().attention_io(
                x, shift_msa=shift_msa, scale_msa=scale_msa, freqs=freqs)
        return self.self_attn.qkv(
            self._norm_modulate(self.norm1, x, shift_msa, scale_msa),
            freqs=freqs)

    def post_attention(self, x, attn_out, *, gate_msa, shift_mlp,
                       scale_mlp, gate_mlp, context):
        if not x.is_cuda:
            return super().post_attention(
                x, attn_out, gate_msa=gate_msa, shift_mlp=shift_mlp,
                scale_mlp=scale_mlp, gate_mlp=gate_mlp, context=context)
        x = x + gate_msa * self.self_attn.project(attn_out)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norm3(x), context)
        elif context:
            raise ValueError("FastWAM block without cross-attention received context")
        return x + gate_mlp * self.ffn(
            self._norm_modulate(self.norm2, x, shift_mlp, scale_mlp))


class _ContextMaskStream:
    """Local stream specialization; no edits or behavior changes to VPP2."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for index, block in enumerate(self.blocks):
            block = _FastWAMBlock.from_module(block)
            self.blocks[index] = block
            # Reuse the same projections/parameters, with FastWAM-only RoPE.
            block.self_attn = _FastWAMSelfAttentionIO.from_module(block.self_attn)
            # FastWAM uses ATen RMSNorm, including its backward, rather than
            # the generic arch implementation's explicit FP32 reduction.
            for module in tuple(block.modules()):
                for name, child in tuple(module.named_children()):
                    if name in ("norm_q", "norm_k") and isinstance(child, RMSNorm):
                        norm = FastWAMRMSNorm(child.weight.numel(), eps=child.eps)
                        norm.weight = child.weight
                        norm.train(child.training)
                        setattr(module, name, norm)
            if block.cross_attn is not None:
                block.cross_attn = _MaskedCrossAttention.from_module(block.cross_attn)

    def _embed_context(self, bus: ConditionBus | None):
        values = super()._embed_context(bus)
        if isinstance(bus, FastWAMConditionBus):
            return _MaskedContext(values, bus.slot_masks.get(self.spec.name, {}))
        return values

    def prepare(self, x: Tensor, *, timestep: Tensor | None = None,
                conditions: ConditionBus | None = None) -> StreamState:
        state = super().prepare(x, timestep=timestep, conditions=conditions)
        if isinstance(conditions, FastWAMConditionBus):
            # Engine state copies discard dictionary subclass attributes.
            state.extras["cross_kv_masks"] = conditions.slot_masks.get(self.spec.name, {})
        return state

    def post_attn(self, state: StreamState, attn_out: Tensor, *,
                  local_layer_id: int) -> StreamState:
        masks = state.extras.get("cross_kv_masks")
        if masks:
            extras = dict(state.extras)
            extras["cross_kv"] = _MaskedContext(extras["cross_kv"], masks)
            state = StreamState(stream=state.stream, hidden=state.hidden,
                                modulation=state.modulation, extras=extras)
        return super().post_attn(state, attn_out, local_layer_id=local_layer_id)


class FastWAMRMSNorm(RMSNorm):
    """Master's matching/mixed-dtype ATen normalization with unchanged weights."""

    def forward(self, x: Tensor) -> Tensor:
        if self.weight.dtype == x.dtype:
            return F.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.eps)
        return F.rms_norm(x, self.weight.shape, eps=self.eps) * self.weight


class FastWAMActionStream(_ContextMaskStream, ActionStream):
    """Sample-level action timestep with masked text/proprio cross attention."""


class FastWAMVideoStream(_ContextMaskStream, WanVideoStream):
    """Token-separated WAN time embeddings without changing shared streams."""

    def prepare(self, x: Tensor, *, timestep: Tensor | None = None,
                conditions: ConditionBus | None = None) -> StreamState:
        if timestep is None or timestep.ndim == 1:
            return super().prepare(x, timestep=timestep, conditions=conditions)
        if timestep.shape != (x.shape[0], self.geometry.padded_seq_len):
            raise ValueError("video token timesteps must be [B,L_video]")
        state = super().prepare(x, timestep=timestep, conditions=conditions)
        batch, length = timestep.shape
        state.extras["timestep_embedding"] = state.extras["timestep_embedding"].reshape(
            batch, length, self.spec.residual_width)
        state.modulation = state.modulation.reshape(
            batch, length, self.spec.modulation_slots.width, self.spec.residual_width)
        return state

    def _modulation(self, block, state: StreamState) -> dict[str, Tensor]:
        if state.modulation is None or state.modulation.ndim != 4:
            return super()._modulation(block, state)
        combined = state.modulation + block.modulation.unsqueeze(1)
        return self.spec.modulation_slots.unpack(combined, slot_axis=2)


class FastWAMVideoHead(PatchFlowHead):
    """The output head consumes the same token-separated embeddings as WAN."""

    def __init__(self, *, form: str = "patch_flow", **kwargs) -> None:
        if form != "patch_flow":
            raise ValueError("FastWAM video head supports only form 'patch_flow'")
        super().__init__(**kwargs)

    def forward(self, hidden: Tensor, *, timestep_embedding: Tensor | None = None,
                conditions=None) -> Tensor:
        if timestep_embedding is None or timestep_embedding.ndim != 3:
            return super().forward(hidden, timestep_embedding=timestep_embedding,
                                   conditions=conditions)
        self._check_hidden(hidden)
        if timestep_embedding.shape != hidden.shape:
            raise ValueError("video head time embeddings must match [B,L,W]")
        slots = self.spec.modulation_slots.unpack(
            self.modulation.unsqueeze(1) + timestep_embedding.unsqueeze(2), slot_axis=2)
        tokens = self.head(self.norm(hidden) * (1 + slots["scale_out"]) + slots["shift_out"])
        return self._unpatchify(tokens)


class _SampledTimesteps(dict):
    """Invocation-local noise, drawn alongside t to preserve master's RNG order."""

    def __init__(self, timesteps: dict[str, Tensor], noise: dict[str, Tensor]):
        super().__init__(timesteps)
        self.noise = noise


@dataclass
class FastWAMDiffusionRuntime(DiffusionRuntime):
    model_timesteps: dict[str, Tensor] = field(default_factory=dict)
    action_is_pad: Tensor | None = None
    image_is_pad: Tensor | None = None
    fuse_first_frame: bool = True


def video_loss_per_sample(prediction: Tensor, target: Tensor,
                          image_is_pad: Tensor | None = None, *,
                          include_initial_video_step: bool = False,
                          temporal_factor: int = 4) -> Tensor:
    """Master's fp32 frame MSE and all-padded temporal pooling, valid mean."""
    if prediction.ndim != 5 or prediction.shape != target.shape:
        raise ValueError("video prediction and target must match [B,C,F,H,W]")
    if not include_initial_video_step:
        prediction, target = prediction[:, :, 1:], target[:, :, 1:]
    token_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").mean((1, 3, 4))
    if token_loss.shape[1] == 0:
        return token_loss.sum(dim=1)
    if image_is_pad is None:
        return token_loss.mean(dim=1)
    if temporal_factor <= 0 or image_is_pad.ndim != 2 or image_is_pad.shape[0] != prediction.shape[0]:
        raise ValueError("image_is_pad must be [B,1+temporal_factor*k]")
    if image_is_pad.shape[1] < 1 or (image_is_pad.shape[1] - 1) % temporal_factor:
        raise ValueError("image_is_pad cannot align with temporal latent steps")
    pad = image_is_pad.to(device=token_loss.device, dtype=torch.bool)
    tail = pad[:, 1:].reshape(pad.shape[0], -1, temporal_factor).all(dim=2)
    latent_pad = torch.cat((pad[:, :1], tail), dim=1) if include_initial_video_step else tail
    if latent_pad.shape != token_loss.shape:
        raise ValueError("video-loss mask steps disagree with latent steps")
    valid = (~latent_pad).to(token_loss.dtype)
    return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


class FastWAMTrainingWorldModel(WorldModel):
    """Joint trainable video/action model; inherits WorldModel.forward unchanged.

    Optional batch ``fuse_vae_embedding_in_latents`` overrides the constructor
    default. ``first_frame_latents`` [B,C,1,H,W], when supplied, is the wrapper's
    separately encoded first frame (otherwise the first clean video latent).
    Padding affects loss only, not attention. Terminal-padding exceptions are a
    VPP2 policy and deliberately do not apply here.
    """

    def __init__(self, spec: WorldModelSpec, *, engine: StreamEngine,
                 layout: TokenLayout, fuse_vae_embedding_in_latents: bool = True) -> None:
        if spec.model_type != "fastwam" or {s.name for s in spec.streams} != {"video", "action"}:
            raise ValueError("FastWAM training requires the fastwam video/action provider")
        if len(spec.attention_groups) != 1:
            raise ValueError("FastWAM requires one joint attention group")
        if getattr(engine, "frozen_streams", ()):
            raise ValueError("FastWAM joint training must not freeze either stream")
        schedulers = {}
        for name in ("video", "action"):
            sampler = spec.diffusion.sampler_for(name)
            if not isinstance(sampler.distribution, UniformShift) or not sampler.broadcast.is_per_sample:
                raise ValueError("FastWAM requires per-sample UniformShift schedulers")
            # Shared continuous scheduler math is identical to master's
            # WanContinuousFlowMatchScheduler; no legacy model imports needed.
            schedulers[name] = VPP2ActionScheduler(sampler.distribution.shift)
        losses = {loss.head: loss for loss in spec.diffusion.losses}
        if len(spec.diffusion.losses) != 2 or set(losses) != {"video_flow", "action_flow"}:
            raise ValueError("FastWAM requires exactly video_flow and action_flow losses")
        if any(spec.head(name).weight != 1 for name in losses):
            raise ValueError("FastWAM weights belong on LossSpec, not HeadSpec")
        if spec.geometry.patch_size[0] != 1:
            raise ValueError("first-frame separation requires temporal patch size 1")
        super().__init__(spec, engine=engine, layout=layout)
        if not isinstance(self.blocks["video"], FastWAMVideoStream) or not isinstance(
                self.heads["video_flow"], FastWAMVideoHead):
            raise TypeError("fastwam provider must build token-separated video stream/head")
        self.video_scheduler, self.action_scheduler = schedulers["video"], schedulers["action"]
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self._loss_specs = losses

    def _latents(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        video = self._raw_tensor(batch, self.spec.stream("video"))
        action = self._raw_tensor(batch, self.spec.stream("action"))
        if not isinstance(video, Tensor) or not isinstance(action, Tensor):
            raise ValueError("video/action must be tensors")
        geo = self.spec.geometry
        if video.ndim != 5 or tuple(video.shape[1:]) != (geo.latent_channels, *geo.latent_grid):
            raise ValueError("video must be [B,C,F,H,W] matching geometry")
        if action.ndim != 3 or tuple(action.shape[1:]) != (geo.action_horizon, geo.action_dim):
            raise ValueError("action must be [B,T,D] matching geometry")
        if video.shape[0] == 0 or video.shape[0] != action.shape[0] or video.device != action.device:
            raise ValueError("video/action must share nonempty batch and device")
        if not video.is_floating_point() or not action.is_floating_point():
            raise ValueError("video/action must be floating point")
        return video, action

    def sample_timesteps(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        video, action = self._latents(batch)
        noise_video = torch.randn_like(video)
        tv = self.video_scheduler.sample_training_t(video.shape[0], video.device, video.dtype)
        noise_action = torch.randn_like(action)
        ta = self.action_scheduler.sample_training_t(action.shape[0], action.device, action.dtype)
        return _SampledTimesteps(
            {"video": tv.double() / 1000, "action": ta.double() / 1000},
            {"video": noise_video, "action": noise_action})

    def build_conditions(self, batch: Mapping[str, Any]) -> ConditionBus:
        """Carry master's text mask to both slots, appending valid proprio.

        ``context_mask`` describes only raw text [B,L], not the appended token.
        A missing mask means all text is valid. All-True masks are retained:
        there is no numerical need to discard them for this SDPA path.
        """
        base = super().build_conditions(batch)
        bus = FastWAMConditionBus(conditions=base.conditions)
        text = bus.require("text").value
        mask = batch.get("context_mask")
        if mask is None:
            mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != text.shape[:2]:
            raise ValueError("context_mask must be bool [B,L] matching text context")
        mask = mask.to(device=text.device)
        for name in ("video", "action"):
            parts = [mask]
            for condition in self.spec.appended_to("text", name):
                token = bus.require(condition).value
                parts.append(torch.ones(token.shape[:2], device=mask.device, dtype=torch.bool))
            bus.slot_masks[name] = {"text": torch.cat(parts, dim=1)}
        return bus

    def add_noise(self, batch: Mapping[str, Any],
                  timesteps: Mapping[str, Tensor]) -> FastWAMDiffusionRuntime:
        video, action = self._latents(batch)
        noise = (timesteps.noise if isinstance(timesteps, _SampledTimesteps)
                 else {"video": torch.randn_like(video), "action": torch.randn_like(action)})
        model_t, noised, targets = {}, {}, {}
        for name, clean, scheduler in (("video", video, self.video_scheduler),
                                       ("action", action, self.action_scheduler)):
            t = timesteps[name]
            if t.shape != (clean.shape[0],) or not bool(torch.isfinite(t).all()) or bool(((t < 0) | (t > 1)).any()):
                raise ValueError(f"{name} timestep must be finite normalized [B] in [0,1]")
            model_t[name] = (t * 1000).to(clean)
            noised[name] = scheduler.add_noise(clean, noise[name], model_t[name])
            targets[name] = scheduler.training_target(clean, noise[name], model_t[name])
        fuse = bool(batch.get("fuse_vae_embedding_in_latents", self.fuse_vae_embedding_in_latents))
        if fuse:
            first = batch.get("first_frame_latents", video[:, :, :1])
            if not isinstance(first, Tensor) or first.shape != video[:, :, :1].shape:
                raise ValueError("first_frame_latents must be [B,C,1,H,W]")
            noised["video"][:, :, :1] = first.detach().to(video)
        action_pad, image_pad = batch.get("action_is_pad"), batch.get("image_is_pad")
        for name, pad, expected in (("action_is_pad", action_pad, action.shape[:2]),
                                    ("image_is_pad", image_pad, (video.shape[0], self.spec.geometry.num_frames))):
            if pad is not None and (not isinstance(pad, Tensor) or tuple(pad.shape) != tuple(expected)):
                raise ValueError(f"{name} must match {tuple(expected)}")
        return FastWAMDiffusionRuntime(
            timesteps=dict(timesteps), noise=noise, noised=noised, targets=targets,
            model_timesteps=model_t, action_is_pad=action_pad, image_is_pad=image_pad,
            fuse_first_frame=fuse, attention_key_valid=batch.get("attention_key_valid"),
            attention_labels=batch.get("attention_labels"))

    def run_group(self, group: AttentionGroup, inputs: Mapping[str, Tensor],
                  conditions: ConditionBus, runtime: DiffusionRuntime) -> EngineOutput:
        if not isinstance(runtime, FastWAMDiffusionRuntime):
            raise TypeError("FastWAM run_group requires FastWAMDiffusionRuntime")
        index = self.spec.attention_groups.index(group)
        plan = self.plans[index]
        plan = plan.bind(labels=runtime.attention_labels if plan.visibility is not None else None,
                         key_valid=runtime.attention_key_valid)
        tv = runtime.model_timesteps["video"]
        token_t = tv[:, None].expand(-1, self.spec.geometry.padded_seq_len).clone()
        if runtime.fuse_first_frame:
            token_t[:, :self.spec.geometry.tokens_per_frame] = 0
        return self.engine.run(
            group, {name: inputs[name] for name in group.stream_names},
            {name: self.blocks[name] for name in group.stream_names},
            plan, self.layout, conditions=conditions,
            timesteps={"video": token_t, "action": runtime.model_timesteps["action"]},
            captures=self.captures)

    def compute_losses(self, output: EngineOutput, runtime: DiffusionRuntime,
                       outputs: Mapping[str, Tensor], *, step: int) -> tuple[Tensor, dict[str, Any]]:
        del output
        if not isinstance(runtime, FastWAMDiffusionRuntime):
            raise TypeError("FastWAM compute_losses requires FastWAMDiffusionRuntime")
        video = video_loss_per_sample(
            outputs["video_flow"], runtime.targets["video"], runtime.image_is_pad,
            include_initial_video_step=not runtime.fuse_first_frame,
            temporal_factor=self.spec.geometry.vae_temporal_stride)
        loss_video = (video * self.video_scheduler.training_weight(
            runtime.model_timesteps["video"]).to(video)).mean()
        loss_action = masked_action_loss(
            outputs["action_flow"], runtime.targets["action"],
            self.action_scheduler.training_weight(runtime.model_timesteps["action"]),
            runtime.action_is_pad)
        loss_video = loss_video * self._loss_specs["video_flow"].weight_at(step)
        loss_action = loss_action * self._loss_specs["action_flow"].weight_at(step)
        total = loss_video + loss_action
        return total, {"loss_video": loss_video.detach(), "loss_action": loss_action.detach(),
                       "loss": total.detach()}
