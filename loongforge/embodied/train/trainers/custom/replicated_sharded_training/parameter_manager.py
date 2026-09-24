# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parameter and optimizer-state ownership for replicated-sharded training.

Compute replicas stay complete on every rank; the fp32 masters and the optimizer
state that follows them are distributed. This module orchestrates the pieces:

    ParameterRegistry      metadata, masters, resolved wire dtypes
    GradientReducer        reduce-to-owner
    ParameterSynchronizer  owner-to-replica publication
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from loongforge.embodied.train.trainers.custom.replicated_sharded_training.gradient_reducer import (
    GradientReducer,
)
from loongforge.embodied.train.trainers.custom.replicated_sharded_training.parameter_sync import (
    ParameterSynchronizer,
)
from loongforge.embodied.train.trainers.custom.replicated_sharded_training.registry import (
    MasterParameterView,
    ParameterRegistry,
)

GRAD_REDUCE_MODES = ("fp32", "bf16", "mixed", "compute")
PARAM_SYNC_PRECISIONS = (
    "fp32", "bf16", "bf16_ef", "bf16_ef_delta", "fp8_e4m3_delta",
)
# Overlap scheduling constants, deliberately not exposed as configuration:
# deeper parameter-sync queues (depth 4 and 6) benchmarked at 0.94x on 8 GPUs at
# GBS80, so the depth below is the retained optimum. The parameter in-flight cap
# bounds peak memory rather than throughput.
PARAM_SYNC_DEPTH = 2
PARAM_INFLIGHT_BYTES = 2048 * 1024 * 1024
# Default bucket sizes, used when the model config does not pin one. Overlapped
# sync wants finer buckets so collectives start early; the serial path prefers
# fewer, larger ones.
BUCKET_MB_OVERLAP = 256
BUCKET_MB_SERIAL = 1024


@dataclass(frozen=True)
class CollectiveConfig:
    """Resolved collective plan: precision per axis plus overlap scheduling."""

    gradient_reduce_dtype: str = "fp32"
    parameter_sync_precision: str = "fp32"
    grad_overlap: bool = True
    param_overlap: bool = True
    bucket_mb: int | None = None
    grad_inflight_mb: int = 3072
    fp8_block_size: int = 256
    fp8_reprime_interval: int = 0
    fp8_include: tuple[str, ...] = ()
    fp8_bf16_shadow: bool | None = None

    @classmethod
    def from_model_config(cls, cfg) -> "CollectiveConfig":
        """Read the model-config field names, which stay the public YAML."""

        def get(key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)
        include = get("param_sync_fp8_include", ()) or ()
        return cls(
            gradient_reduce_dtype=get("grad_reduce_dtype", "fp32"),
            parameter_sync_precision=get("param_sync_precision", "fp32"),
            grad_overlap=bool(get("grad_overlap", True)),
            param_overlap=bool(get("param_overlap", True)),
            bucket_mb=get("comm_bucket_mb", None),
            grad_inflight_mb=int(get("grad_inflight_mb", 3072)),
            fp8_block_size=int(get("param_sync_fp8_block", 256)),
            fp8_reprime_interval=int(get("param_sync_fp8_reprime_interval", 0)),
            fp8_include=tuple(include),
            fp8_bf16_shadow=get("param_sync_bf16_with_fp8", None),
        )

    def validate(self) -> "CollectiveConfig":
        """Reject unsupported plans at startup, and resolve the bucket size.

        Both precision axes are free: every (grad_reduce, param_sync) pair of valid
        modes is implemented, and the FP8 fields are inert by design outside
        ``fp8_e4m3_delta`` (a policy may declare them while a run publishes bf16),
        so the axes are validated independently rather than as a matrix.
        """
        if self.gradient_reduce_dtype not in GRAD_REDUCE_MODES:
            raise ValueError(
                f"unsupported gradient_reduce_dtype={self.gradient_reduce_dtype!r}; "
                f"expected one of {GRAD_REDUCE_MODES}"
            )
        if self.parameter_sync_precision not in PARAM_SYNC_PRECISIONS:
            raise ValueError(
                f"unsupported parameter_sync_precision={self.parameter_sync_precision!r}; "
                f"expected one of {PARAM_SYNC_PRECISIONS}"
            )
        if self.grad_inflight_mb <= 0:
            raise ValueError("grad_inflight_mb must be positive")
        if self.bucket_mb is not None and int(self.bucket_mb) < 0:
            raise ValueError("bucket_mb must be non-negative (0 selects the default)")
        return replace(self, bucket_mb=self.resolved_bucket_mb)

    @property
    def resolved_bucket_mb(self) -> int:
        """Bucket size actually used, applying the overlap-dependent default."""
        if self.bucket_mb:
            return int(self.bucket_mb)
        return BUCKET_MB_OVERLAP if self.grad_overlap else BUCKET_MB_SERIAL

    def resolved(self) -> dict:
        """Return the startup-loggable resolved plan."""
        return {
            "gradient_reduce_dtype": self.gradient_reduce_dtype,
            "parameter_sync_precision": self.parameter_sync_precision,
            "grad_overlap": self.grad_overlap,
            "param_overlap": self.param_overlap,
            "bucket_mb": self.resolved_bucket_mb,
            "grad_inflight_mb": self.grad_inflight_mb,
            "fp8_block_size": self.fp8_block_size,
            "fp8_reprime_interval": self.fp8_reprime_interval,
            "fp8_include": list(self.fp8_include),
            "fp8_bf16_shadow": bool(self.fp8_bf16_shadow),
        }


class ReplicatedShardedManager:
    """Own FP32 master/state shards while retaining complete compute replicas."""

    def __init__(
        self,
        module,
        group=None,
        rank=None,
        world_size=None,
        parameter_policy=None,
        collective_config: CollectiveConfig | None = None,
    ):
        """Register the parameters, then wire the gradient and publish paths.

        ``collective_config`` is the single source of the precision/fp8 plan *and*
        the overlap/bucket/in-flight scheduling knobs. When omitted, the
        CollectiveConfig defaults apply (fp32 collectives, overlap on).
        """
        self.module = module
        self.group = group
        self.rank = dist.get_rank(group) if rank is None else rank
        self.world_size = dist.get_world_size(group) if world_size is None else world_size
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.parameter_policy = parameter_policy
        if collective_config is None:
            collective_config = CollectiveConfig()

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
            grad_reduce_dtype=collective_config.gradient_reduce_dtype,
            param_sync_precision=collective_config.parameter_sync_precision,
            param_sync_fp8_block=collective_config.fp8_block_size,
            param_sync_fp8_reprime_interval=collective_config.fp8_reprime_interval,
            param_sync_fp8_include=collective_config.fp8_include,
            param_sync_bf16_with_fp8=collective_config.fp8_bf16_shadow,
        )
        self.refresh()
        self._in_step_names: set[str] | None = None
        # The resolved plan takes precision from the registry's reconciled values
        # and the scheduling knobs from the supplied collective config, so the
        # strategies and the startup log agree with what actually runs.
        config = CollectiveConfig(
            gradient_reduce_dtype=self.registry.grad_reduce_mode,
            parameter_sync_precision=self.registry.param_sync_precision,
            grad_overlap=collective_config.grad_overlap,
            param_overlap=collective_config.param_overlap,
            bucket_mb=collective_config.bucket_mb,
            grad_inflight_mb=collective_config.grad_inflight_mb,
            fp8_block_size=self.registry.param_sync_fp8_block,
            fp8_reprime_interval=self.registry.param_sync_fp8_reprime_interval,
            fp8_include=tuple(self.registry.param_sync_fp8_include),
            fp8_bf16_shadow=self.registry.param_sync_bf16_with_fp8,
        ).validate()
        self._collective_config = config
        self._bucket_mb = config.resolved_bucket_mb
        self.gradient_reducer = GradientReducer(
            self.registry,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            bucket_mb=config.resolved_bucket_mb,
            inflight_bytes=int(config.grad_inflight_mb) * 1024 * 1024,
            overlap=config.grad_overlap,
        )
        self.parameter_synchronizer = ParameterSynchronizer(
            self.registry,
            self,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            bucket_mb=config.resolved_bucket_mb,
            overlap=config.param_overlap,
            sync_depth=PARAM_SYNC_DEPTH,
            inflight_bytes=PARAM_INFLIGHT_BYTES,
            compensation=self.registry.param_sync_compensation,
            quantization=self.registry.param_sync_quantization,
            fp8_block=self.registry.param_sync_fp8_block,
            fp8_reprime_interval=self.registry.param_sync_fp8_reprime_interval,
        )

    @property
    def gradient(self):
        return self.gradient_reducer

    @property
    def parameter(self):
        return self.parameter_synchronizer

    @property
    def config(self):
        return self._collective_config

    def begin(self) -> None:
        """Parameter-publish window open. The gradient path has its own begin."""
        self.parameter_synchronizer.begin()

    def finish(self) -> None:
        self.parameter_synchronizer.finish()

    def start_serial(self) -> None:
        self.parameter_synchronizer.start_serial()

    def finish_serial(self) -> None:
        self.parameter_synchronizer.finish_serial()

    def note_updated(self, name) -> None:
        self.parameter_synchronizer.note_updated(name)

    def invalidate(self) -> None:
        self.parameter_synchronizer.invalidate()

    def clear_pending(self) -> None:
        self.parameter_synchronizer.clear_pending()

    def compensation_state_dict(self):
        return self.parameter_synchronizer.compensation_state_dict()

    def load_compensation_state_dict(self, state) -> None:
        self.parameter_synchronizer.load_compensation_state_dict(state)

    def refresh(self):
        """Rebuild the identity index after the masters are (re)created."""
        self._name_by_id = {
            id(parameter): name for name, parameter in self.registry.master.items()
        }

    def name_for(self, parameter):
        """Return the registry name of ``parameter``, or ``None`` if unmanaged."""
        return self._name_by_id.get(id(parameter))

    def set_in_step_names(self, names):
        """Record which masters the optimizer updates during ``step()``.

        Masters an optimizer finishes early can start publishing before
        ``step()`` returns; the rest must wait for it. Without this information
        every master is treated as publishable early, which is correct but
        serializes the launch queue behind the slowest one.
        """
        self._in_step_names = None if names is None else set(names)

    @property
    def in_step_names(self):
        """Return the in-step-update name set, or ``None`` if unknown."""
        return self._in_step_names

    def updates_in_step(self, record) -> bool:
        """True when ``record``'s master is expected to be updated in-step."""
        if self._in_step_names is None:
            return True
        return record.name in self._in_step_names

    def collective_summary(self):
        """Return the resolved collective plan for startup logging."""
        return self._collective_config.resolved()

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

    def parameter_manifest(self):
        """Return the validated parameter capability/checkpoint manifest."""
        return self.registry.manifest()

    def optimizer_view(self):
        """Return a module-like view over the fp32 master parameters."""
        return MasterParameterView(self.registry.master.items())

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
        self.set_in_step_names(names)
        self.parameter_synchronizer.invalidate()

    def on_master_updated(self, parameter):
        """Mark a master parameter as updated and launch any ready publishes."""
        self.parameter_synchronizer.note_updated(self.name_for(parameter))

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
            "manifest": self.parameter_manifest(),
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
        self.registry.validate_manifest(state.get("manifest"), allow_reshard=False)
        if state["world_size"] != self.world_size:
            raise RuntimeError(
                f"replicated-sharded checkpoint world_size={state['world_size']} does not match "
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
            raise RuntimeError("replicated-sharded parameter ownership schema does not match checkpoint")
        if set(state["master"]) != set(self.master):
            raise RuntimeError("replicated-sharded FP32 master keys do not match checkpoint")
        for name, value in state["master"].items():
            target = self.master[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid FP32 master tensor for {name}")
        for name, value in state["master"].items():
            target = self.master[name]
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
                f"replicated-sharded checkpoint is missing {len(missing)} FP32 masters this rank "
                f"now owns, e.g. {missing[:3]}"
            )
        for name, target in self.master.items():
            value = tensors[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid FP32 master tensor for {name}")
        for name, target in self.master.items():
            value = tensors[name]
            target.copy_(value.to(target.device))

    def validate_checkpoint_manifest(self, manifest, *, allow_reshard=False):
        """Validate shared checkpoint metadata before rank-local state loading."""
        self.registry.validate_manifest(
            manifest, allow_reshard=allow_reshard,
            saved_world_size=manifest.get("world_size") if manifest else None,
        )


__all__ = ["CollectiveConfig", "ReplicatedShardedManager"]
