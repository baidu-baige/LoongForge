# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicated-compute optimizer-state-shard parameter and optimizer-state ownership.

Compute replicas stay complete on every rank; the fp32 masters and the optimizer
state that follows them are distributed. This module orchestrates the pieces:

    ParameterRegistry      metadata, masters, resolved wire dtypes
    OwnershipPlanner       greedy whole-tensor owners
    GradientReducer        reduce-to-owner, overlapped or serial
    OptimizerAdapter       parameter identity and update timing
    ParameterSynchronizer  owner-to-replica publication
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from loongforge.embodied.train.trainers.optimizer_state_shard.gradient_reducer import (
    GradientReducer,
)
from loongforge.embodied.train.trainers.optimizer_state_shard.optimizer_adapter import (
    OptimizerAdapter,
)
from loongforge.embodied.train.trainers.optimizer_state_shard.parameter_sync import (
    ParameterSynchronizer,
)
from loongforge.embodied.train.trainers.optimizer_state_shard.registry import (
    ParameterRegistry,
)

# Overlap scheduling constants, deliberately not exposed as configuration:
# deeper parameter-sync queues (depth 4 and 6) benchmarked at 0.94x on 8 GPUs
# at GBS80, so the depth below is the retained optimum. The parameter in-flight
# cap bounds peak memory rather than throughput.
_PARAM_SYNC_DEPTH = 2
_PARAM_INFLIGHT_BYTES = 2048 * 1024 * 1024
# Default bucket sizes, used when the model config does not pin one. Overlapped
# sync wants finer buckets so collectives start early; the serial path prefers
# fewer, larger ones.
_BUCKET_MB_OVERLAP = 256
_BUCKET_MB_SERIAL = 1024


class OptimizerStateShardManager:
    """Own FP32 master/state shards while retaining complete compute replicas."""

    def __init__(
        self,
        module,
        group=None,
        rank=None,
        world_size=None,
        parameter_policy=None,
        grad_reduce_dtype: str | None = None,
        param_sync_precision: str | None = None,
        param_sync_fp8_block: int = 256,
        param_sync_fp8_reprime_interval: int = 0,
        param_sync_fp8_include=None,
        param_sync_bf16_with_fp8: bool = False,
        grad_overlap: bool = True,
        param_overlap: bool = True,
        bucket_mb: int | None = None,
        grad_inflight_mb: int = 3072,
    ):
        """Register the parameters, then wire the gradient and publish paths."""
        self.module = module
        self.group = group
        self.rank = dist.get_rank(group) if rank is None else rank
        self.world_size = dist.get_world_size(group) if world_size is None else world_size
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.parameter_policy = parameter_policy

        # Collective precision is configured from the model YAML
        # (``model.grad_reduce_dtype`` / ``model.param_sync_precision``); override
        # per run on the command line, e.g. ``model.param_sync_precision=bf16``.
        # Gradient reduction and parameter publication are downcast per parameter,
        # never per collective.
        self.registry = ParameterRegistry(
            module,
            self.rank,
            self.world_size,
            parameter_policy,
            grad_reduce_dtype=grad_reduce_dtype,
            param_sync_precision=param_sync_precision,
            param_sync_fp8_block=param_sync_fp8_block,
            param_sync_fp8_reprime_interval=param_sync_fp8_reprime_interval,
            param_sync_fp8_include=param_sync_fp8_include,
            param_sync_bf16_with_fp8=param_sync_bf16_with_fp8,
        )
        self._bucket_mb = int(
            bucket_mb
            if bucket_mb
            else (_BUCKET_MB_OVERLAP if grad_overlap else _BUCKET_MB_SERIAL)
        )
        self.adapter = OptimizerAdapter(self.registry)
        self.gradient_reducer = GradientReducer(
            self.registry,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            bucket_mb=self._bucket_mb,
            inflight_bytes=int(grad_inflight_mb) * 1024 * 1024,
            overlap=grad_overlap,
        )
        self.parameter_synchronizer = ParameterSynchronizer(
            self.registry,
            self.adapter,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            bucket_mb=self._bucket_mb,
            overlap=param_overlap,
            sync_depth=_PARAM_SYNC_DEPTH,
            inflight_bytes=_PARAM_INFLIGHT_BYTES,
            compensation=self.registry.param_sync_compensation,
            quantization=self.registry.param_sync_quantization,
            fp8_block=self.registry.param_sync_fp8_block,
            fp8_reprime_interval=self.registry.param_sync_fp8_reprime_interval,
        )

    # -- registry passthroughs ------------------------------------------------

    @property
    def compute(self):
        """Return the ``{name: compute parameter}`` mapping."""
        return self.registry.compute

    @property
    def master(self):
        """Return the ``{name: fp32 master}`` mapping owned by this rank."""
        return self.registry.master

    @property
    def ownership(self):
        """Return the ownership records, in registration order."""
        return self.registry.ownership

    @property
    def specs(self):
        """Return the ``{name: ownership record}`` mapping."""
        return self.registry.specs

    @property
    def grad_reduce_mode(self) -> str:
        """Resolved gradient-reduction precision mode."""
        return self.registry.grad_reduce_mode

    @property
    def param_sync_mode(self) -> str:
        """Resolved parameter-publication precision mode."""
        return self.registry.param_sync_mode

    @property
    def param_sync_compensation(self) -> str:
        """Resolved parameter-publication compensation mode."""
        return self.registry.param_sync_compensation

    @property
    def param_sync_precision(self) -> str:
        """Resolved parameter-publication precision (the single public enum)."""
        return self.registry.param_sync_precision

    @property
    def bucket_mb(self) -> int:
        """Resolved collective bucket size in MiB."""
        return self._bucket_mb

    def precision_summary(self):
        """Return the per-class resolved-precision rows for startup logging."""
        return self.registry.precision_summary()

    def optimizer_view(self):
        """Return a module-like view over the fp32 master parameters."""
        return self.adapter.optimizer_view()

    def named_master_parameters(self):
        """Return the ``(name, master parameter)`` pairs owned by this rank."""
        return self.registry.named_master_parameters()

    # -- gradient lifecycle ---------------------------------------------------

    def begin_gradient_sync(self):
        """Arm the step so per-parameter hooks can launch collectives when overlapped."""
        self.gradient_reducer.begin()

    def finish_gradient_sync(self):
        """Drain the in-flight gradient collectives, or reduce serially if off."""
        self.gradient_reducer.finish()

    def reduce_gradients_serially(self):
        """Average gradients onto owners in one blocking pass, without overlap."""
        self.gradient_reducer.reduce_serial()

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> float:
        """Clip owned master gradients by global norm and return that norm."""
        # Keep the fp64 per-element accumulation. Measured and rejected:
        # ``torch._foreach_norm`` is 1.019x end-to-end but accumulates in fp32
        # opmath, moving loss up to 0.963% and grad norm up to 3.65% over 20 steps.
        # The clipping coefficient feeds every parameter, so the accumulation
        # dtype has to be the widest one in the step, not the fastest.
        local_sq = torch.zeros((), dtype=torch.float64, device=self.registry.device)
        for parameter in self.master.values():
            if parameter.grad is not None:
                local_sq.add_(parameter.grad.detach().double().square().sum())
        if self.world_size > 1:
            dist.all_reduce(local_sq, group=self.group)
        norm = local_sq.sqrt()
        if max_norm > 0:
            coefficient = max_norm / (norm + 1e-6)
            if coefficient < 1:
                for parameter in self.master.values():
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
        return float(norm.item())

    # -- parameter publication ------------------------------------------------

    def set_in_step_updated_names(self, names):
        """Record which masters the inner optimizer updates during ``step()``."""
        self.adapter.set_in_step_names(names)
        self.parameter_synchronizer.invalidate()

    def on_master_updated(self, parameter):
        """Mark a master parameter as updated and launch any ready publishes."""
        self.parameter_synchronizer.note_updated(self.adapter.name_for(parameter))

    def begin_parameter_sync(self):
        """Reset the publish plan so optimizer updates can start collectives."""
        self.parameter_synchronizer.begin()

    def finish_parameter_sync(self):
        """Drain the in-flight publishes, or publish serially if overlap is off."""
        self.parameter_synchronizer.finish()

    def start_serial_parameter_sync(self):
        """Launch bounded owner-to-replica parameter synchronization."""
        self.parameter_synchronizer.start_serial()

    def finish_serial_parameter_sync(self):
        """Wait for every queued parameter broadcast to land."""
        self.parameter_synchronizer.finish_serial()

    def clear_pending_work(self):
        """Abandon half-issued collectives after a failure, on both paths."""
        self.gradient_reducer.clear_pending()
        self.parameter_synchronizer.clear_pending()

    # -- state ----------------------------------------------------------------

    def state_dict(self):
        """Return ownership metadata plus the fp32 master weights on CPU."""
        return {
            "world_size": self.world_size,
            "ownership": [item.__dict__ for item in self.ownership],
            "master": {
                name: value.detach().cpu().clone() for name, value in self.master.items()
            },
            "param_sync_compensation": self.param_sync_compensation,
            "param_sync_quantization": self.registry.param_sync_quantization,
            "param_sync_residuals": self.parameter_synchronizer.compensation_state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        """Restore the fp32 masters, rejecting any ownership-schema mismatch."""
        if state["world_size"] != self.world_size:
            raise RuntimeError(
                f"optimizer-state-shard checkpoint world_size={state['world_size']} does not match "
                f"{self.world_size}"
            )
        expected = [item.__dict__ for item in self.ownership]
        # Back-compat: an earlier ownership schema carried an ``expert_counts``
        # field (always empty in practice). Drop it before comparing so those
        # checkpoints still resume against the current schema.
        saved_ownership = [
            {key: value for key, value in item.items() if key != "expert_counts"}
            for item in state["ownership"]
        ]
        if saved_ownership != expected:
            raise RuntimeError("optimizer-state-shard parameter ownership schema does not match checkpoint")
        if set(state["master"]) != set(self.master):
            raise RuntimeError("optimizer-state-shard FP32 master keys do not match checkpoint")
        for name, value in state["master"].items():
            target = self.master[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid FP32 master tensor for {name}")
            target.copy_(value.to(target.device))
        self.parameter_synchronizer.load_compensation_state_dict(
            state.get("param_sync_residuals")
        )

    @torch.no_grad()
    def load_master_tensors(self, tensors):
        """Copy fp32 masters by name, ignoring the checkpoint's ownership layout.

        Used when resuming into a different world size: ownership is recomputed
        from the new layout, so only the tensor values carry over.
        """
        missing = sorted(set(self.master) - set(tensors))
        if missing:
            raise RuntimeError(
                f"optimizer-state-shard checkpoint is missing {len(missing)} FP32 masters this rank "
                f"now owns, e.g. {missing[:3]}"
            )
        for name, target in self.master.items():
            value = tensors[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid FP32 master tensor for {name}")
            target.copy_(value.to(target.device))


__all__ = ["OptimizerStateShardManager"]
