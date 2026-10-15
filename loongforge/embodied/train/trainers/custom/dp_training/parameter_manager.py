# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The custom_dp manager: the trainer-facing seam over the owner-distributed state.

Compute replicas stay complete on every rank; the fp32 masters and the optimizer
state that follows them are distributed. This module assembles and owns the
pieces:

    ParameterPlan          metadata, masters, resolved wire dtypes
    GradientReducer        reduce-to-owner
    ParameterSynchronizer  owner-to-replica publication
    checkpoint IO          rank-local master and optimizer-state save/load

and is the single object a trainer talks to. The lifecycle, once per step:

    setup
    -> begin_gradient_sync        (arm, before the last accumulation micro-step)
    -> forward / backward
    -> finish_gradient_sync       (drain the reduce-to-owner collectives)
    -> clip_grad_norm_
    -> begin_parameter_sync / optimizer step / finish_parameter_sync
    -> clear_pending_work

``build_parameter_manager()`` is the assembly entry point, so a new model reuses the
wiring instead of copying it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from loongforge.embodied.distributed.custom_dp_utils.precision_policy import (
    GRAD_REDUCE_MODES,
    PARAM_SYNC_PRECISIONS,
)
from loongforge.embodied.distributed.custom_dp_utils.custom_dp_config import (
    DEFAULT_GRAD_REDUCE_DTYPE,
    DEFAULT_PARAM_SYNC_PRECISION,
    CustomDpConfig,
)
from loongforge.embodied.train.trainers.custom.dp_training.checkpoint_io import (
    RankLocalCheckpointIO,
)
from loongforge.embodied.train.trainers.custom.dp_training.gradient_reducer import (
    GradientReducer,
)
from loongforge.embodied.train.trainers.custom.dp_training.parameter_sync import (
    ParameterSynchronizer,
)
from loongforge.embodied.train.trainers.custom.dp_training.parameter_plan import (
    MasterParameterView,
    ParameterPlan,
)

# Overlap scheduling constants, deliberately not exposed as configuration:
# deeper parameter-sync queues measured slower on 8 GPUs at GBS80, so the depth
# below is the retained optimum. The parameter in-flight cap bounds peak memory
# rather than throughput.
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

    gradient_reduce_dtype: str = DEFAULT_GRAD_REDUCE_DTYPE
    parameter_sync_precision: str = DEFAULT_PARAM_SYNC_PRECISION
    grad_overlap: bool = True
    param_overlap: bool = True
    bucket_mb: int | None = None
    grad_inflight_mb: int = 3072
    fp8_reprime_interval: int = 0
    fp8_include: tuple[str, ...] = ()
    fp8_bf16_shadow: bool | None = None

    @classmethod
    def from_model_config(cls, cfg) -> "CollectiveConfig":
        """Read the plan from the model config, whose field names are the YAML.

        Every field is required: the model config inherits them from
        ``CustomDpConfig``, so a missing one is a wiring bug that should
        surface at startup rather than be papered over with a local default.
        """
        return cls(
            gradient_reduce_dtype=cfg.grad_reduce_dtype,
            parameter_sync_precision=cfg.param_sync_precision,
            grad_overlap=bool(cfg.grad_overlap),
            param_overlap=bool(cfg.param_overlap),
            bucket_mb=cfg.comm_bucket_mb,
            grad_inflight_mb=int(cfg.grad_inflight_mb),
            fp8_reprime_interval=int(cfg.param_sync_fp8_reprime_interval),
            fp8_include=tuple(cfg.param_sync_fp8_include or ()),
            fp8_bf16_shadow=cfg.param_sync_bf16_with_fp8,
        )

    @classmethod
    def from_training_args(cls, args) -> "CollectiveConfig":
        """Read the plan from the ``--custom-dp-*`` command-line knobs."""
        return cls(
            gradient_reduce_dtype=args.custom_dp_grad_reduce_dtype,
            parameter_sync_precision=args.custom_dp_param_sync_precision,
            grad_overlap=bool(args.custom_dp_grad_overlap),
            param_overlap=bool(args.custom_dp_param_overlap),
            bucket_mb=args.custom_dp_comm_bucket_mb,
            grad_inflight_mb=int(args.custom_dp_grad_inflight_mb),
            fp8_reprime_interval=int(args.custom_dp_param_sync_fp8_reprime_interval),
            fp8_include=tuple(args.custom_dp_param_sync_fp8_include or ()),
            fp8_bf16_shadow=args.custom_dp_param_sync_bf16_with_fp8,
        )

    @classmethod
    def from_sources(cls, model_cfg, training_args) -> "CollectiveConfig":
        """Pick the plan's source: the model config if it carries the knobs.

        A model config that inherits ``CustomDpConfig`` keeps its YAML in charge (and
        the strict every-field-required check); any other model is tuned through
        the ``--custom-dp-*`` arguments.
        """
        if isinstance(model_cfg, CustomDpConfig):
            return cls.from_model_config(model_cfg)
        return cls.from_training_args(training_args)

    def validate(self) -> "CollectiveConfig":
        """Reject unsupported plans at startup, and resolve the bucket size.

        Both precision axes are free: every (grad_reduce, param_sync) pair of valid
        modes is implemented, and the FP8 fields are inert by design outside
        ``fp8_e4m3_delta`` (a policy may declare them while a run publishes bf16),
        so the axes are validated independently rather than as a matrix.
        """
        if str(self.gradient_reduce_dtype).lower() not in GRAD_REDUCE_MODES:
            raise ValueError(
                f"unsupported gradient_reduce_dtype={self.gradient_reduce_dtype!r}; "
                f"expected one of {GRAD_REDUCE_MODES}"
            )
        if str(self.parameter_sync_precision).lower() not in PARAM_SYNC_PRECISIONS:
            raise ValueError(
                f"unsupported parameter_sync_precision={self.parameter_sync_precision!r}; "
                f"expected one of {PARAM_SYNC_PRECISIONS}"
            )
        if self.grad_inflight_mb <= 0:
            raise ValueError("grad_inflight_mb must be positive")
        if self.bucket_mb is not None and int(self.bucket_mb) < 0:
            raise ValueError("bucket_mb must be non-negative (0 selects the default)")
        return replace(
            self,
            gradient_reduce_dtype=str(self.gradient_reduce_dtype).lower(),
            parameter_sync_precision=str(self.parameter_sync_precision).lower(),
            bucket_mb=self.resolved_bucket_mb,
        )

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
            "fp8_reprime_interval": self.fp8_reprime_interval,
            "fp8_include": list(self.fp8_include),
            "fp8_bf16_shadow": bool(self.fp8_bf16_shadow),
        }


class ParameterManager:
    """Own the fp32 master/state shards while retaining complete compute replicas.

    This is the whole optimizer stack for a custom_dp run: it builds the
    parameter plan, the gradient reducer, the parameter synchronizer and the rank-local
    checkpoint IO, and exposes the lifecycle a trainer drives. Every collective and
    every master-weight mutation runs here, so a model never has to reach past it.
    """

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
        CollectiveConfig defaults apply (bf16 gradient reduce, fp32 parameter
        sync, overlap on).
        """
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.module = module
        self.group = group
        self.rank = dist.get_rank(group) if rank is None else rank
        self.world_size = dist.get_world_size(group) if world_size is None else world_size
        self.parameter_policy = parameter_policy
        # One plan object, validated once and then handed down verbatim: the
        # plan, the two collective paths and the startup log all read it, so
        # they cannot disagree about what runs.
        self._collective_config = (collective_config or CollectiveConfig()).validate()
        config = self._collective_config
        self._bucket_mb = config.resolved_bucket_mb

        # Collective precision is configured from the model YAML
        # (``model.grad_reduce_dtype`` / ``model.param_sync_precision``); override
        # per run on the command line, e.g. ``model.param_sync_precision=bf16``.
        # Gradient reduction and parameter publication are downcast per parameter,
        # never per collective.
        self.plan = ParameterPlan(
            module,
            self.rank,
            self.world_size,
            parameter_policy,
            grad_reduce_dtype=config.gradient_reduce_dtype,
            param_sync_precision=config.parameter_sync_precision,
            param_sync_fp8_reprime_interval=config.fp8_reprime_interval,
            param_sync_fp8_include=config.fp8_include,
            param_sync_bf16_with_fp8=config.fp8_bf16_shadow,
        )
        self.refresh()
        self._in_step_names: set[str] | None = None
        self.gradient_reducer = GradientReducer(
            self.plan,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            bucket_mb=config.resolved_bucket_mb,
            inflight_bytes=int(config.grad_inflight_mb) * 1024 * 1024,
            overlap=config.grad_overlap,
        )
        self.parameter_synchronizer = ParameterSynchronizer(
            self.plan,
            self,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            bucket_mb=config.resolved_bucket_mb,
            overlap=config.param_overlap,
            sync_depth=PARAM_SYNC_DEPTH,
            inflight_bytes=PARAM_INFLIGHT_BYTES,
            compensation=self.plan.param_sync_compensation,
            quantization=self.plan.param_sync_quantization,
            fp8_reprime_interval=self.plan.param_sync_fp8_reprime_interval,
        )
        # Bound by ``attach_optimizer`` once the optimizer over the masters exists.
        self.optimizer = None
        self.checkpoint_io = None

    # -- optimizer attachment -------------------------------------------------

    def attach_optimizer(self, optimizer):
        """Bind the optimizer over the fp32 masters and its rank-local checkpoint IO.

        The shared checkpoint backend owns the file layout and the aux files; the
        object installed here owns the masters and the per-parameter optimizer
        state, which are rank-local.
        """
        self.optimizer = optimizer
        self.checkpoint_io = RankLocalCheckpointIO(self, optimizer)
        optimizer.zero_checkpoint_io = self.checkpoint_io
        return optimizer

    def optimizer_view(self):
        """Return a module-like view over the fp32 master parameters."""
        return MasterParameterView(self.plan.master.items())

    # -- plan passthroughs ------------------------------------------------

    def refresh(self):
        """Rebuild the identity index after the masters are (re)created."""
        self._name_by_id = {
            id(parameter): name for name, parameter in self.plan.master.items()
        }

    def name_for(self, parameter):
        """Return the plan name of ``parameter``, or ``None`` if unmanaged."""
        return self._name_by_id.get(id(parameter))

    @property
    def master(self):
        """Return the ``{name: fp32 master}`` mapping owned by this rank."""
        return self.plan.master

    @property
    def parameter_plan(self):
        """Return the plan, which answers ``optimizer_kind`` per parameter."""
        return self.plan

    @property
    def grad_reduce_mode(self) -> str:
        """Resolved gradient-reduction precision mode."""
        return self.plan.grad_reduce_mode

    @property
    def param_sync_precision(self) -> str:
        """Resolved parameter-publication precision (the single public enum)."""
        return self.plan.param_sync_precision

    def precision_summary(self):
        """Return the per-class resolved-precision rows for startup logging."""
        return self.plan.precision_summary()

    def parameter_manifest(self):
        """Return the validated parameter capability/checkpoint manifest."""
        return self.plan.manifest()

    def collective_summary(self):
        """Return the resolved collective plan for startup logging."""
        return self._collective_config.resolved()

    # -- parameter-publish scheduling -----------------------------------------

    def set_in_step_updated_names(self, names):
        """Record which masters the inner optimizer updates during ``step()``.

        Masters an optimizer finishes early can start publishing before ``step()``
        returns; the rest must wait for it. Without this information every master
        is treated as publishable early, which is correct but serializes the
        launch queue behind the slowest one.
        """
        self._in_step_names = None if names is None else set(names)
        self.parameter_synchronizer.invalidate()

    @property
    def in_step_names(self):
        """Return the in-step-update name set, or ``None`` if unknown."""
        return self._in_step_names

    def updates_in_step(self, record) -> bool:
        """True when ``record``'s master is expected to be updated in-step."""
        if self._in_step_names is None:
            return True
        return record.name in self._in_step_names

    def on_master_updated(self, parameter):
        """Mark a master parameter as updated and launch any ready publishes."""
        self.parameter_synchronizer.note_updated(self.name_for(parameter))

    # -- gradient lifecycle ---------------------------------------------------

    def begin_gradient_sync(self):
        """Arm the step so per-parameter hooks can launch collectives when overlapped."""
        self.gradient_reducer.begin()

    def finish_gradient_sync(self):
        """Drain the in-flight gradient collectives, or reduce serially if off."""
        self.gradient_reducer.finish()

    @torch.no_grad()
    def scrub_nan_gradients(self) -> None:
        """Replace non-finite master gradients in place."""
        for parameter in self.master.values():
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> float:
        """Clip owned master gradients by global norm and return that norm."""
        # Keep the fp64 per-element accumulation. Measured and rejected:
        # ``torch._foreach_norm`` is marginally faster but accumulates in fp32
        # opmath, which perturbs the loss and the grad norm over 20 steps. The
        # clipping coefficient feeds every parameter, so the accumulation dtype has
        # to be the widest one in the step, not the fastest.
        local_sq = torch.zeros((), dtype=torch.float64, device=self.plan.device)
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

    def begin_parameter_sync(self):
        """Reset the publish plan so optimizer updates can start collectives."""
        self.parameter_synchronizer.begin()

    def finish_parameter_sync(self):
        """Drain the in-flight publishes, or publish serially if overlap is off."""
        self.parameter_synchronizer.finish()

    def publish_serially(self):
        """Launch and drain owner-to-replica parameter synchronization, unbounded."""
        self.parameter_synchronizer.start_serial()
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
            "ownership": [item.__dict__ for item in self.plan.ownership],
            "manifest": self.parameter_manifest(),
            "master": {
                name: value.detach().cpu().clone() for name, value in self.master.items()
            },
            "param_sync_compensation": self.plan.param_sync_compensation,
            "param_sync_quantization": self.plan.param_sync_quantization,
            "param_sync_residuals": self.parameter_synchronizer.compensation_state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        """Restore the fp32 masters, rejecting any ownership-schema mismatch."""
        self.plan.validate_manifest(state.get("manifest"), allow_reshard=False)
        if state["world_size"] != self.world_size:
            raise RuntimeError(
                f"custom_dp checkpoint world_size={state['world_size']} does not match "
                f"{self.world_size}"
            )
        if state["ownership"] != [item.__dict__ for item in self.plan.ownership]:
            raise RuntimeError("custom_dp parameter ownership schema does not match checkpoint")
        if set(state["master"]) != set(self.master):
            raise RuntimeError("custom_dp FP32 master keys do not match checkpoint")
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
                f"custom_dp checkpoint is missing {len(missing)} FP32 masters this rank "
                f"now owns, e.g. {missing[:3]}"
            )
        for name, target in self.master.items():
            value = tensors[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid FP32 master tensor for {name}")
        for name, target in self.master.items():
            value = tensors[name]
            target.copy_(value.to(target.device))


def build_parameter_manager(
    module,
    parameter_policy,
    *,
    rank,
    world_size,
    group=None,
    collective_config=None,
) -> ParameterManager:
    """Assemble the whole optimizer stack from a policy and the collective config.

    Covers the parameter plan (ownership included), the fp32 masters, the gradient
    reducer, the parameter synchronizer and the rank-local checkpoint IO.
    ``attach_optimizer`` binds the optimizer later, because the optimizer is built
    over the masters this call produces.
    """
    return ParameterManager(
        module,
        group=group,
        rank=rank,
        world_size=world_size,
        parameter_policy=parameter_policy,
        collective_config=collective_config,
    )


__all__ = [
    "CollectiveConfig",
    "ParameterManager",
    "build_parameter_manager",
]
