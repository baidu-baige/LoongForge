# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``WorldModel``: the implementation that makes the specs collaborate.

Signatures and step order follow
``docs/design/mot_spec_model_matrix.zh-CN.md`` §4.5 (four ``__init__`` steps,
seven training steps, three inference steps, six extension points).

This class does **assembly and orchestration only**, and contains no model
algorithm:
  - shape facts live in ``WorldModelSpec`` (already past V1-V9 at construction);
  - how a layer computes lives in ``StreamModule`` (Lane B);
  - noising and loss live in ``DiffusionSpec`` (Lane C);
  - visibility lives in ``Visibility`` (Lane A).
Every ``NotImplementedError`` in this file therefore points at one of those
lanes, not at a missing algorithm.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .diffusion.spec import DiffusionSpec
from .losses import LossSpec
from .heads.spec import HeadSpec
from .layout import LabelView, TokenLayout
from .spec.core import build
from .spec.world import WorldModelSpec
from .streams.conditions import ConditionBus, ConditionSpec
from .streams.engine import AttentionGroup, EngineOutput, StreamEngine
from .streams.spec import StreamSpec
from .visibility.combine import Visibility, from_predicate
from .visibility.mask import AttentionPlan

__all__ = ["DiffusionDraw", "DiffusionRuntime", "STEP_HOOK_EVENTS", "StepHook",
           "WorldModel", "join_stream_visibility",
           "materialize_conditions", "query_in_stream"]


def materialize_conditions(specs: Sequence[ConditionSpec],
                           batch: Mapping[str, Any], *,
                           training: bool,
                           encoders: Mapping[str, nn.Module] | None = None
                           ) -> ConditionBus:
    """Materialize each ``ConditionSpec`` into a ``Condition`` on one bus.

    An entry with ``enabled=False`` is **skipped entirely** rather than
    materialized as a zero tensor: a disabled condition must not occupy a name
    on the bus, otherwise V7's reconciliation of "names on the bus" against
    "each stream's cross_kv_slots" would count it.

    ``encoders`` is ``WorldModel.condition_encoders``, keyed by condition name.
    A condition with an ``encoder`` goes through
    ``value = encoders[name](batch[source_key])``; one without takes
    ``value = batch[source_key]`` directly. Encoders are **not built here**:
    they hold parameters, so they must sit in ``WorldModel``'s submodule tree
    to be visible to the optimizer and the checkpoint.

    One path has no landing spot yet and raises on being hit rather than
    quietly producing an empty value: ``dropout_prob > 0`` while ``training``.
    CFG dropout can either zero the condition or swap in a learnable null
    vector, and the two implementations have different weight shapes, so the
    choice has to be made first.

    Args:
        specs: Condition declarations to materialize.
        batch: Raw batch; ``source_key`` is looked up here.
        training: Whether the model is in training mode (gates CFG dropout).
        encoders: Condition name to encoder module, usually
            ``WorldModel.condition_encoders``.

    Returns:
        A ``ConditionBus`` holding one entry per enabled condition.

    Raises:
        NotImplementedError: If CFG dropout would apply.
        ValueError: If a condition has no ``source_key``.
        KeyError: If ``batch`` or ``encoders`` lacks a required entry.
    """
    bus = ConditionBus()
    for spec in specs:
        if not spec.enabled:
            continue
        if training and spec.dropout_prob > 0.0:
            raise NotImplementedError(
                f"condition {spec.name!r} has dropout_prob="
                f"{spec.dropout_prob}; the semantics of CFG condition "
                "dropout (zero it out vs swap in a learnable null vector) "
                "are still undecided"
            )
        if spec.source_key is None:
            raise ValueError(
                f"condition {spec.name!r} has no source_key, so value has "
                "nowhere to come from. With an encoder, source_key is the "
                "source of the **encoder's input** and is not optional "
                "(see the ConditionSpec.source_key comment)"
            )
        if spec.source_key not in batch:
            raise KeyError(
                f"condition {spec.name!r} reads its raw input from "
                f"batch[{spec.source_key!r}], but batch only has "
                f"{sorted(batch)}"
            )
        value = batch[spec.source_key]
        if spec.encoder is not None:
            available = encoders or {}
            if spec.name not in available:
                raise KeyError(
                    f"condition {spec.name!r} declares an encoder, but "
                    f"encoders does not hold it; present: "
                    f"{sorted(available)}. Encoders are built into "
                    "condition_encoders by step 1 of WorldModel.__init__"
                )
            value = available[spec.name](value)
        bus.add(spec.materialize(value))
    return bus


def query_in_stream(name: str) -> Visibility:
    """Select queries that fall on stream ``name``; a ``restrict`` where-clause.

    Deliberately not one of the nine atomic rules in ``visibility/rules.py``:
    those nine describe a relation between two tokens, while this one looks at
    the query side only. It is a **scope selector**, used to confine a rule to
    the rows belonging to its own stream.

    Args:
        name: Stream name the query must belong to.

    Returns:
        A ``Visibility`` that is True on rows of that stream, False elsewhere.
    """

    @from_predicate(f"query_in_stream({name!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        names = q.layout.stream_names
        if name not in names:
            raise KeyError(
                f"query_in_stream: layout has no stream {name!r}; existing "
                f"streams {list(names)}"
            )
        if not q.has("stream_id"):  # A single-stream layout has no stream label
            return torch.ones((q.layout.total, k.layout.total),
                              dtype=torch.bool)
        return (q.get("stream_id") == names.index(name)).expand(
            q.layout.total, k.layout.total)

    return rule


def join_stream_visibility(declared: Mapping[str, Visibility]
                           ) -> Visibility | None:
    """Combine "stream name to that stream's query-side rule" into one mask.

    Each rule is first ``restrict``-ed to the query rows of its own stream,
    then all of them are ANDed together. ``restrict`` keeps everything visible
    outside the region it matches, so after the AND each stream's rows are
    governed only by its own rule, and rows of streams with no declared rule
    stay fully visible.

    A plain ``&`` would be wrong: it would also mask out the query rows of the
    other streams according to this rule. The symptom is cross-stream edges
    disappearing silently, with unchanged shapes and a loss that still goes
    down.

    Args:
        declared: Stream name to the visibility rule for that stream's queries.

    Returns:
        The joint visibility, or None when nothing was declared.
    """
    if not declared:
        return None
    joint: Visibility | None = None
    for name in sorted(declared):
        scoped = declared[name].restrict(where=query_in_stream(name))
        joint = scoped if joint is None else joint & scoped
    return joint


@dataclass
class DiffusionRuntime:
    """Diffusion intermediates of one forward pass, collected into one object.

    Why it exists: the current ``training_loss`` passes values between its
    seven steps through a dozen local variables, three of which (``t``,
    ``t_expanded``, ``timestep_input``) have different shapes but similar
    meanings, so a call site easily picks the wrong one. Inside a class,
    picking the wrong one becomes an AttributeError.

    Attributes:
        timesteps: Stream name to [B, *axes], the noising coefficient sigma
            in [0, 1] (dtype per ``RngPolicy.sigma_dtype``).
        noised: Stream name to x_t.
        noise: Stream name to epsilon, needed to compute the target.
        targets: Stream name to the regression target.
        model_timesteps: Stream name to the training timestep
            ``sigma * num_train_timesteps`` after the ``RngPolicy`` cast. The
            engine hands it to the time embeddings and ``LossSpec.weight_of``
            reads it (Lane C v2 ``NoisedStream.t``).
        loss_valid: Stream name to a bool [B, T] of positions that count in
            that stream's flow losses (True counts), over the units left after
            the clean prefix frames. ``add_noise`` fills it from
            ``WorldModelSpec.loss_masks``; a stream without an entry counts
            everything.
    """

    timesteps: dict[str, Tensor]
    noised: dict[str, Tensor]
    noise: dict[str, Tensor]
    targets: dict[str, Tensor]
    model_timesteps: dict[str, Tensor] = field(default_factory=dict)
    loss_valid: dict[str, Tensor] = field(default_factory=dict)

    def require(self, stream: str) -> tuple[Tensor, Tensor, Tensor]:
        """Return (x_t, target, timestep) for one stream.

        Raises ``KeyError`` listing the streams that do exist when the
        requested one is absent.
        """
        try:
            return (self.noised[stream], self.targets[stream],
                    self.timesteps[stream])
        except KeyError:
            raise KeyError(
                f"DiffusionRuntime has no stream {stream!r}; present: "
                f"{sorted(self.noised)}"
            ) from None


@dataclass(frozen=True)
class DiffusionDraw:
    """Every random number of one training step, drawn before any noising.

    Produced by ``WorldModel.draw`` and consumed by ``WorldModel.add_noise``,
    which draws nothing itself. Keeping the draws in one object is what makes
    their order a declaration (``TimestepSampler.rng``) instead of a side
    effect of which method happens to call ``torch.rand`` first.

    Attributes:
        noise: Stream name to epsilon, shaped like the stream's raw tensor.
        sigma: Stream name to [B, *axes] noising coefficients in [0, 1].
        model_timesteps: Stream name to the training timesteps
            ``sigma * num_train_timesteps`` (see ``RngPolicy.quantize``).
    """

    noise: dict[str, Tensor]
    sigma: dict[str, Tensor]
    model_timesteps: dict[str, Tensor]


# What ``WorldModel.register_step_hook`` reports, in the order of one forward.
STEP_HOOK_EVENTS = ("draw", "runtime", "outputs", "losses")
StepHook = Callable[["WorldModel", str, Any], None]


class WorldModel(nn.Module):
    """The world model. ``spec`` in, ``(loss, metrics)`` out (interface I4).

    The four ``__init__`` steps (model_matrix §4.5.1, order is fixed):
      1. build each stream's block per spec (``build(stream.impl, ...)``) plus
         the encoders of conditions that declare one (``condition_encoders``);
      2. materialize structural visibility into ``AttentionPlan`` (once at
         construction, not once per step);
      3. build the output heads, and derive the engine's ``captures`` requests
         from the head table;
      4. resolve each stream's "joint layer index to local layer index" map.

    Why step 2 happens at construction: the structural channel is a
    batch-shared constant, so recomputing it every step would push the
    construction of an [L, L] tensor into the training loop.
    """

    def __init__(self, spec: WorldModelSpec, *,
                 engine: StreamEngine,
                 layout: TokenLayout) -> None:
        super().__init__()
        self.spec = spec
        self.engine = engine
        self.layout = layout
        # OrderedDict, not dict: RemovableHandle keeps a weak reference to it.
        self._step_hooks: OrderedDict[int, StepHook] = OrderedDict()

        # 1. Build the stream blocks. All parameters live here; the engine
        #    holds none.
        #    ``world=spec`` is injected because a condition's raw width
        #    (``ConditionSpec.width``) and the channel count of a channel
        #    concat are **cross-layer** shape facts, and only
        #    ``WorldModelSpec`` sees the stream names and the condition table
        #    at once. Without it, every stream would walk the condition table
        #    itself and the "slot name to width" map would have N
        #    implementations.
        self.blocks = nn.ModuleDict({
            stream.name: build(stream.impl, spec=stream,
                               geometry=spec.geometry, world=spec)
            for stream in spec.streams
        })

        # 1'. Build the condition encoders. This is the **third parameter
        #    namespace**, a sibling of blocks and equally part of L2: one
        #    condition can be injected into several streams (VPP2's proprio
        #    feeds both), yet legacy keeps only one set of encoder weights --
        #    in the checkpoint it is a top-level payload key
        #    (``vpp2.py:1800-1801``), not part of the ``mot`` subtree.
        #    Hanging it under one stream would force N copies, one per target
        #    stream (that is a different model), or force an arbitrary host
        #    stream (and the whole chain breaks when that stream is disabled).
        self.condition_encoders = nn.ModuleDict({
            cond.name: build(cond.encoder, spec=cond,
                             geometry=spec.geometry, world=spec)
            for cond in spec.conditions if cond.encoder is not None
        })

        # 2. Materialize structural visibility. One plan per attention group
        #    (not per stream): joint attention shares one [total, total] mask
        #    across the whole group.
        self.plans: dict[int, AttentionPlan] = {
            index: AttentionPlan.from_visibility(
                self._group_visibility(group), layout)
            for index, group in enumerate(spec.attention_groups)
        }

        # 3. Build the heads and derive the capture requests. The requests come
        #    from the head table so they are not written out in two places.
        self.heads = nn.ModuleDict({
            head.name: build(head.impl, spec=head, geometry=spec.geometry,
                             world=spec)
            for head in spec.heads
        })
        self.captures: dict[str, tuple[int, ...]] = spec.captures()

        # 4. Layer index maps. Resolved once at construction, looked up during
        #    forward.
        self.layer_index_maps: dict[str, dict[int, int]] = {}
        for group in spec.attention_groups:
            self.layer_index_maps.update(group.layer_index_maps())

    def _group_visibility(self, group: AttentionGroup) -> Visibility | None:
        """Joint visibility of one group. None when nobody declared any.

        None means fully bidirectional, which is the normal case for a
        monolithic model.

        The keys of ``spec.visibility`` are **stream names** and the values are
        that stream's query-side rule; see ``join_stream_visibility`` for how
        they combine. Only declarations of this group's streams are taken --
        streams of other groups are not in this group's ``[total, total]``.
        """
        return join_stream_visibility({
            name: self.spec.visibility[name]
            for name in group.stream_names
            if name in self.spec.visibility})

    # ---- Training path. Seven steps, order is fixed (model_matrix §4.5.2) ----

    def forward(self, batch: Mapping[str, Any]
                ) -> tuple[Tensor, dict[str, Any]]:
        """Interface **I4**: ``(loss, metrics)``. This signature does not move.

        The seven steps:
          1. draw every random number of the step -> ``DiffusionDraw`` (per
             stream, in the order ``TimestepSampler.rng`` declares);
          2. fill the ``ConditionBus`` (materialize each ``ConditionSpec``);
          3. add noise and compute regression targets -> ``DiffusionRuntime``
             (deterministic: consumes the draw);
          4. pick the raw tensor each stream feeds the engine (tokenization
             happens inside ``prepare``);
          5. run the engine group by group (``StreamEngine.run``);
          6. run the output heads (``HiddenStateRef.resolve`` fetches hidden
             states);
          7. compute each loss per ``LossSpec`` and sum them weighted.

        Step 3 must come after step 1 and before step 5: the noise needs the
        draw, and the engine consumes x_t. Step 6 must come after step 5: the heads
        consume the engine's output. Those two are the non-reorderable
        constraints from interfaces §7.

        ``global_step`` is read from the batch (defaulting to 0) instead of
        being counted inside this class: weight ramping (``WeightRamp``) needs
        the **optimizer step**, which only the trainer knows. A local counter
        would diverge under gradient accumulation and on resume from a
        checkpoint. Training a model whose losses ramp needs the step, so a
        training batch without it raises there (``_global_step``).

        Args:
            batch: Raw batch; stream inputs, condition sources, and
                ``global_step`` are all read from it.

        Steps 1, 3, 6 and 7 are reported to ``register_step_hook`` observers.

        Returns:
            A tuple of the total loss and a metrics dict.
        """
        step = self._global_step(batch)
        draw = self.draw(batch)                                     # 1
        self._notify_step("draw", draw)
        conditions = self.build_conditions(batch)                   # 2
        runtime = self.add_noise(batch, draw)                       # 3
        self._notify_step("runtime", runtime)
        inputs = self.stream_inputs(runtime)                        # 4
        merged = EngineOutput(states={}, captured={})               # 5
        for group in self.spec.attention_groups:
            partial = self.run_group(group, inputs, conditions, runtime)
            merged.states.update(partial.states)
            merged.captured.update(partial.captured)
        outputs = self._run_heads(merged)                           # 6
        self._notify_step("outputs", MappingProxyType(outputs))
        loss, metrics = self.compute_losses(
            merged, runtime, outputs, step=step, batch=batch)       # 7
        self._notify_step("losses", (loss, MappingProxyType(metrics)))
        return loss, metrics

    def register_step_hook(self, hook: StepHook) -> RemovableHandle:
        """Observe the L3/L4 intermediates of every training step. Read-only.

        ``hook(model, event, value)`` runs once per ``forward`` for each event
        in ``STEP_HOOK_EVENTS``: ``"draw"`` after step 1 (``DiffusionDraw``),
        ``"runtime"`` after step 3 (``DiffusionRuntime``), ``"outputs"`` after
        step 6 (head name -> prediction) and ``"losses"`` after step 7
        (``(loss, metrics)``). The calls sit in ``forward``, outside activation
        checkpointing, so recomputation in backward never repeats them -- unlike
        a forward hook on a module inside a checkpointed layer.

        The hook observes and must not mutate: its return value is ignored and
        the two mappings are read-only views. An experiment that has to replace
        an intermediate (replaying noise, say) overrides ``draw`` or
        ``add_noise`` in a test subclass, where the substitution is visible,
        instead of hiding it in a hook. Hooks are not saved in the state dict.

        Args:
            hook: Called as ``hook(model, event, value)``.

        Returns:
            A handle whose ``remove()`` unregisters the hook.
        """
        handle = RemovableHandle(self._step_hooks)
        self._step_hooks[handle.id] = hook
        return handle

    def _global_step(self, batch: Mapping[str, Any]) -> int:
        """The batch's ``global_step``; 0 when it has none.

        A training forward of a model whose losses ramp with the step raises
        instead (Lane C v2 §1.9): its ramps would stay at their step-0 values
        and nothing would report it. Evaluation keeps the 0.
        """
        if "global_step" in batch:
            return int(batch["global_step"])
        ramped = [loss.name for loss in self.spec.losses
                  if loss.enabled and loss.ramp is not None and not loss.ramp.is_constant]
        if self.training and ramped:
            raise ValueError(
                f"losses {ramped} ramp with the training step, but the batch has no "
                "global_step: pass the trainer's iteration to ArchWorldModelBase.forward "
                "or put global_step in the batch")
        return 0

    def _notify_step(self, event: str, value: Any) -> None:
        for hook in tuple(self._step_hooks.values()):
            hook(self, event, value)

    def _raw_tensor(self, batch: Mapping[str, Any],
                    stream: StreamSpec) -> Tensor:
        """Fetch one stream's **raw tensor**.

        The key is ``StreamSpec.batch_key`` (empty falls back to the stream
        name). This is the only place that reads a stream's raw input from the
        batch. In the current code it is a series of hard-coded
        ``batch["latents"]`` lookups inside ``training_loss``, which means
        changing dataset means editing model code.
        """
        key = stream.batch_key
        if key not in batch:
            raise KeyError(
                f"stream {stream.name!r} reads its raw tensor from "
                f"batch[{key!r}] (StreamSpec.source_key), but batch only has "
                f"{sorted(batch)}"
            )
        return batch[key]

    # ---- The six extension points (model_matrix §4.5.4). Subclasses override
    # ---- these, never forward. Step 1 is ``draw`` (it replaced
    # ---- ``sample_timesteps``, see ``__init_subclass__``) ----

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "sample_timesteps" in vars(cls):
            # forward no longer calls it, so an override would be dead code
            # while the step drew its random numbers in a different order.
            raise TypeError(
                f"{cls.__name__} overrides WorldModel.sample_timesteps, which was "
                "replaced by draw(): noise and timesteps are drawn together in the "
                "order TimestepSampler.rng declares, and add_noise(batch, draw) "
                "draws nothing. Declare the order and t dtype on the provider "
                "(RngPolicy, RectifiedFlow.num_train_timesteps) instead")

    def draw(self, batch: Mapping[str, Any]) -> DiffusionDraw:
        """Extension point 1, step 1. Every random draw of the training step.

        Streams are drawn in the key order of ``DiffusionSpec.timesteps``
        (V5 makes that exactly the streams with ``needs_timestep=True``);
        within a stream ``TimestepSampler.rng.order`` decides whether epsilon
        or the timestep comes first. ``add_noise`` draws nothing, so this is
        the one place the step's random numbers come from, and their order is
        a provider declaration that a test can pin.

        Epsilon has the raw tensor's shape and dtype. The timestep is drawn in
        float32 and turned into ``(sigma, t)`` by ``RngPolicy.quantize``:
        under ``like_sample`` both follow the raw tensor's dtype, so
        ``add_noise``'s ``(1-sigma)*x_0 + sigma*eps`` does not silently
        promote a bf16 x_t to fp32. Such a promotion turns into an implicit
        cast at the engine entry: memory doubles while the numbers still look
        fine.

        Axis lengths come from the **layout labels**
        (``labels_of(axis).max() + 1``) rather than being recomputed from the
        geometry: the labels are I2's only source of axis facts, and computing
        it twice would diverge on padding.
        """
        if self.spec.diffusion is None:
            return DiffusionDraw(noise={}, sigma={}, model_timesteps={})
        steps = self.spec.diffusion.schedule.num_train_timesteps
        noise: dict[str, Tensor] = {}
        sigma: dict[str, Tensor] = {}
        model_timesteps: dict[str, Tensor] = {}
        for name, timestep_sampler in self.spec.diffusion.timesteps.items():
            raw = self._raw_tensor(batch, self.spec.stream(name))
            dtype = raw.dtype if raw.is_floating_point() else torch.float32
            axis_sizes = tuple(int(self.layout.labels_of(axis).max()) + 1
                               for axis in timestep_sampler.broadcast.axes)
            for kind in timestep_sampler.rng.draws:
                if kind == "noise":
                    noise[name] = torch.randn_like(raw)
                    continue
                drawn = timestep_sampler.sample(
                    int(raw.shape[0]), device=raw.device, dtype=torch.float32,
                    axis_sizes=axis_sizes)
                sigma[name], model_timesteps[name] = timestep_sampler.rng.quantize(
                    drawn, num_train_timesteps=steps, sample_dtype=dtype)
        return DiffusionDraw(noise=noise, sigma=sigma, model_timesteps=model_timesteps)

    def build_conditions(self, batch: Mapping[str, Any]) -> ConditionBus:
        """Extension point 2, step 2. Materializes each ``ConditionSpec``."""
        return materialize_conditions(self.spec.conditions, batch,
                                      training=self.training,
                                      encoders=self.condition_encoders)

    def add_noise(self, batch: Mapping[str, Any],
                  draw: DiffusionDraw) -> DiffusionRuntime:
        """Extension point 3, step 3. Defaults to ``DiffusionSpec.schedule``.

        Deterministic: epsilon and sigma come from ``draw``, so an override can
        reshape or mask them but never changes the step's random numbers.

        ``sigma`` is reshaped to ``[B, 1, ...]`` here and deliberately does **not**
        go through ``BroadcastSpec.expand``: expand produces the token-shaped
        ``[B, total, 1]``, whereas noising happens on the **raw tensor**
        (video ``[B, C, F, H, W]``, action ``[B, T, D]``), and the axes do not
        line up. Token-shaped broadcasting only becomes meaningful after L2's
        ``prepare``. A per-frame timestep therefore raises explicitly here
        instead of being broadcast into a shape that happens to multiply.

        Two declarations shape the result beyond the schedule:
          - a stream whose ``StreamSpec.token_timestep`` pins a clean prefix
            gets those latent frames back as x_0 after noising (fastwam
            ``mot/fastwam.py:529``); L2 gives them timestep 0 from the same
            declaration. Epsilon is still drawn for every frame, so the random
            numbers consumed do not change;
          - ``WorldModelSpec.loss_masks`` becomes ``loss_valid``: the units
            that are not padding (or are kept), minus the clean prefix.
        """
        if self.spec.diffusion is None:
            noised = {spec.name: self._raw_tensor(batch, spec) for spec in self.spec.streams}
            loss_valid: dict[str, Tensor] = {}
            for name, clean in noised.items():
                valid = self._loss_valid(batch, name, clean, prefix=0)
                if valid is not None:
                    loss_valid[name] = valid
            return DiffusionRuntime(timesteps={}, noised=noised, noise={},
                                    targets={}, model_timesteps={},
                                    loss_valid=loss_valid)
        schedule = self.spec.diffusion.schedule
        noised = {}
        targets: dict[str, Tensor] = {}
        loss_valid = {}
        for name, timestep_sampler in self.spec.diffusion.timesteps.items():
            if not timestep_sampler.broadcast.is_per_sample:
                raise NotImplementedError(
                    f"stream {name!r} samples its timestep per axis "
                    f"{timestep_sampler.broadcast.axes}, while the default add_noise "
                    "only spreads [B] into [B, 1, ...]. Per-axis noising "
                    "first needs the map `label axis -> which dim of the raw "
                    "tensor` (it is per-modality); until then, override this "
                    "method"
                )
            sigma = draw.sigma[name]
            if sigma.ndim != 1:
                raise ValueError(
                    f"stream {name!r} per-sample timestep should be "
                    f"[B], got {tuple(sigma.shape)}"
                )
            clean = self._raw_tensor(batch, self.spec.stream(name))
            if sigma.shape[0] != clean.shape[0]:
                raise ValueError(
                    f"stream {name!r} timestep batch is {sigma.shape[0]}, "
                    f"while the raw tensor batch is {clean.shape[0]}"
                )
            shaped = sigma.to(dtype=clean.dtype).view(-1, *([1] * (clean.ndim - 1)))
            eps = draw.noise[name]
            x_t = schedule.add_noise(clean, eps, shaped)
            prefix = self._clean_prefix(name, clean)
            if prefix:
                x_t = torch.cat([clean[:, :, :prefix], x_t[:, :, prefix:]], dim=2)
            noised[name] = x_t
            targets[name] = schedule.target(clean, eps, shaped)
            valid = self._loss_valid(batch, name, clean, prefix)
            if valid is not None:
                loss_valid[name] = valid
        for spec in self.spec.streams:
            if spec.name not in noised and spec.name in self.spec.loss_masks:
                clean = self._raw_tensor(batch, spec)
                valid = self._loss_valid(batch, spec.name, clean, prefix=0)
                if valid is not None:
                    loss_valid[spec.name] = valid
        return DiffusionRuntime(timesteps=dict(draw.sigma), noised=noised,
                                noise=dict(draw.noise), targets=targets,
                                model_timesteps=dict(draw.model_timesteps),
                                loss_valid=loss_valid)

    def _clean_prefix(self, stream: str, raw: Tensor) -> int:
        """Leading latent frames of ``stream`` that are clean conditioning.

        Read from ``StreamSpec.token_timestep``, the declaration L2 uses to pin
        their timestep to 0, so the count has one source. Frames are axis 2 of
        the raw video ``[B, C, F, H, W]``.

        Args:
            stream: Stream name.
            raw: A raw-shaped tensor of that stream (x_0 or its target).

        Returns:
            The clean prefix length; 0 when the stream declares none.

        Raises:
            ValueError: The raw tensor is not ``[B, C, F, H, W]`` with
                ``F = geometry.latent_grid[0]``, or the prefix leaves no
                noised frame.
        """
        scope = self.spec.stream(stream).token_timestep
        if scope is None or not scope.clean_prefix:
            return 0
        frames = self.spec.geometry.latent_grid[0]
        if raw.ndim != 5 or raw.shape[2] != frames or scope.clean_prefix >= frames:
            raise ValueError(
                f"stream {stream!r} keeps its first {scope.clean_prefix} latent "
                "frames clean (StreamSpec.token_timestep), which needs a raw "
                f"video [B, C, F, H, W] with F = geometry.latent_grid[0] = {frames} "
                f"> {scope.clean_prefix}; got {tuple(raw.shape)}")
        return scope.clean_prefix

    def _loss_valid(self, batch: Mapping[str, Any], stream: str, raw: Tensor,
                    prefix: int) -> Tensor | None:
        """``WorldModelSpec.loss_masks`` of ``stream`` as a ``loss_valid`` mask.

        Args:
            batch: The training batch.
            stream: Stream name.
            raw: The stream's raw tensor.
            prefix: Its clean prefix length. Those columns are dropped: the
                losses slice the prefix frames off before reducing.

        Returns:
            bool ``[B, units - prefix]``, True where the unit counts; None
            when the stream declares no padding or the batch carries none.
        """
        loss_mask = self.spec.loss_masks.get(stream)
        if loss_mask is None:
            return None
        padded = loss_mask.padded(batch, raw, stream=stream,
                                temporal_stride=self.spec.geometry.vae_temporal_stride)
        if padded is None:
            return None
        return ~padded[:, prefix:]

    def stream_inputs(self, runtime: DiffusionRuntime) -> dict[str, Tensor]:
        """Extension point 4, step 4. Picks the **raw tensor** each stream feeds.

        The default is ``runtime.noised`` itself: the x_t produced by noising
        already is the stream's raw input (video ``[B, C, F, H, W]``, action
        ``[B, T, D]``), and tokenization happens in
        ``StreamModule.prepare`` (L2 detailed design §1.1 assigns patchify
        there). So this method is a **pure forward**, and it exists for one
        reason: some models have streams whose input is not the noised
        quantity (EWAM's vl stream reads image-text features from the batch and
        is not in ``runtime.noised`` at all), and overriding this one method is
        then enough.

        Patchify does not belong here: it would move geometry-dependent
        algorithm into the assembly layer, while ``prepare`` already has to
        read ``geometry.patch_size``.
        """
        missing = [spec.name for spec in self.spec.streams
                   if spec.name not in runtime.noised]
        if missing:
            raise KeyError(
                f"WorldModel.stream_inputs: streams {missing} have no raw "
                "tensor. They are not in runtime.noised and this method was "
                "not overridden. A pure condition stream "
                "(needs_timestep=False) must override this method and read "
                "its own raw input from batch"
            )
        return dict(runtime.noised)

    def run_group(self, group: AttentionGroup,
                  inputs: Mapping[str, Tensor],
                  conditions: ConditionBus,
                  runtime: DiffusionRuntime) -> EngineOutput:
        """Extension point 5, step 5. Looks up the group's plan and captures.

        Delegates to ``self.engine.run``.

        Only this group's streams are passed in ``captures``: streams of other
        groups are not participants of this run, and passing them would be
        rejected by the engine as unknown stream names (that check is correct,
        do not work around it).
        """
        index = self.spec.attention_groups.index(group)
        captures = {name: layers for name, layers in self.captures.items()
                    if name in group.stream_names}
        return self.engine.run(
            group, {name: inputs[name] for name in group.stream_names},
            {name: self.blocks[name] for name in group.stream_names},
            self.plans[index], self.layout, conditions=conditions,
            timesteps={name: runtime.model_timesteps.get(name)
                       for name in group.stream_names},
            captures=captures)

    def _run_heads(self, output: EngineOutput) -> dict[str, Tensor]:
        """Step 6. **Deliberately not an extension point**: §4.5.4's six do not
        include it, because ``HiddenStateRef`` already declares which layer of
        which stream a head consumes, so there is no decision left to override.

        The timestep a head consumes is the **embedded**
        ``[B, residual_width]`` (from the source stream's
        ``StreamState.extras["t_embed"]``), not the scalar in [0, 1]: a head's
        modulation shares one set of ``ModulationSlots`` with the in-stream
        modulation, and those slots take the embedding as input. Passing the
        scalar would pass the shape check by broadcasting and get every number
        wrong.

        ``conditions`` is always None: ``HeadSpec`` currently has no condition
        declaration field, and forwarding the source stream's conditions to the
        head would let the head consume a condition nobody declared for it
        (which is exactly the pair of things ``HiddenStateRef`` keeps apart).
        A head that needs conditions (action heads often need proprio) first
        needs a declaration added to ``HeadSpec``.
        """
        out: dict[str, Tensor] = {}
        for head in self.spec.heads:
            hidden = head.source.resolve(output)
            timestep_embedding: Tensor | None = None
            if head.needs_timestep_embedding:
                extras = output.require(head.source_stream).extras
                if "t_embed" not in extras:
                    raise KeyError(
                        f"head {head.name!r} declares needs_timestep_embedding=True, "
                        f"but source stream {head.source_stream!r} has no "
                        f"'t_embed' in StreamState.extras; present: "
                        f"{sorted(extras)}. Head modulation consumes the "
                        "embedded [B, residual_width], not the scalar in "
                        "[0, 1]"
                    )
                timestep_embedding = extras["t_embed"]
            out[head.name] = self.heads[head.name](hidden, timestep_embedding=timestep_embedding,
                                                   conditions=None)
        return out

    def _loss_target(self, loss: LossSpec, head: HeadSpec,
                     runtime: DiffusionRuntime,
                     batch: Mapping[str, Any] | None = None) -> tuple[Tensor, bool]:
        """Resolve ``(target, from_stream_flow)`` for ``loss`` in fixed order:
        1. ``batch[loss.target_key]`` when ``target_key`` is declared;
        2. ``runtime.targets[loss.head]`` for subclass-injected head targets;
        3. ``runtime.targets[head.source_stream]`` for flow targets."""
        if loss.target_key is not None:
            if batch is None or loss.target_key not in batch:
                present = sorted(batch) if batch is not None else []
                raise KeyError(
                    f"LossSpec({loss.name!r}) reads its target from "
                    f"batch[{loss.target_key!r}], but batch only has {present}")
            return batch[loss.target_key], False
        if loss.head in runtime.targets:
            return runtime.targets[loss.head], False
        if head.source_stream in runtime.targets:
            return runtime.targets[head.source_stream], True
        raise KeyError(
            f"LossSpec({loss.name!r}) has no regression target: "
            f"runtime.targets holds neither the head name {loss.head!r} nor "
            f"the source stream {head.source_stream!r}; present: "
            f"{sorted(runtime.targets)}. Declare target_key on LossSpec or "
            "write runtime.targets in add_noise"
        )

    def _gate_mask(self, loss: LossSpec, batch: Mapping[str, Any] | None,
                   runtime: DiffusionRuntime, prediction: Tensor) -> Tensor | None:
        """Evaluate ``loss.gate`` into a bool ``[B]`` mask on ``prediction.device``."""
        if loss.gate is None:
            return None
        active = loss.gate(batch or {}, timesteps=runtime.model_timesteps)
        if active is None:
            return torch.ones(prediction.shape[0], device=prediction.device, dtype=torch.bool)
        if active.ndim != 1 or active.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"LossSpec({loss.name!r}).gate returned shape {tuple(active.shape)}, "
                f"expected [B={prediction.shape[0]}]")
        return active.to(device=prediction.device, dtype=torch.bool)

    def _loss_value(self, loss: LossSpec, runtime: DiffusionRuntime,
                    outputs: Mapping[str, Tensor], *,
                    batch: Mapping[str, Any] | None = None,
                    active: Tensor | None = None) -> Tensor:
        """One head loss term before ``LossSpec.weight_at``."""
        head = self.spec.head(str(loss.head))
        stream = head.source_stream
        prediction = outputs[head.name]
        target, is_stream_target = self._loss_target(loss, head, runtime, batch)
        target = target.to(device=prediction.device)
        valid = None
        if loss.loss_mask is not None:
            if batch is not None:
                padded = loss.loss_mask.padded(
                    batch, target, stream=loss.name,
                    temporal_stride=self.spec.geometry.vae_temporal_stride)
                if padded is not None:
                    valid = ~padded
        elif is_stream_target:
            prefix = self._clean_prefix(stream, target)
            if prefix:
                prediction, target = prediction[:, :, prefix:], target[:, :, prefix:]
            valid = runtime.loss_valid.get(stream)
        elif loss.target_key is not None and stream not in (
                self.spec.diffusion.timesteps if self.spec.diffusion is not None else ()):
            valid = runtime.loss_valid.get(stream)
        weight = None
        if self.spec.diffusion is not None and stream in self.spec.diffusion.timesteps:
            weight = loss.weight_of(
                runtime.model_timesteps[stream], sampler=self.spec.diffusion.sampler_for(stream),
                num_train_timesteps=self.spec.diffusion.schedule.num_train_timesteps)
        if loss.reducer is not None:
            unreduced_fn = getattr(loss.objective, "unreduced", None)
            error = (unreduced_fn(prediction, target) if callable(unreduced_fn)
                     else loss.objective(prediction, target, valid=valid))
            reduced = loss.reducer(error, weight=weight, valid=valid,
                                   active=active, layout=self.layout)
            if active is not None:
                reduced = reduced + 0.0 * prediction.reshape(-1)[:1].sum().float()
            return reduced
        per_sample = loss.objective(prediction, target, valid=valid)
        if weight is not None:
            per_sample = per_sample * weight.to(
                device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
        if active is None:
            return per_sample.mean()
        counted = active.to(dtype=per_sample.dtype)
        gated = (per_sample * counted).sum() / counted.sum().clamp(min=1.0)
        return gated + 0.0 * prediction.reshape(-1)[:1].sum().float()

    def compute_losses(self, output: EngineOutput,
                       runtime: DiffusionRuntime,
                       outputs: Mapping[str, Tensor], *,
                       step: int,
                       batch: Mapping[str, Any] | None = None) -> tuple[Tensor, dict[str, Any]]:
        """Extension point 6, step 7. Weighted sum over ``WorldModelSpec.losses``."""
        del output              # Heads already ran in step 6; only outputs used
        terms: list[Tensor] = []
        metrics: dict[str, Any] = {}
        for loss in self.spec.losses:
            if loss.stream is not None:
                raise NotImplementedError(
                    f"LossSpec({loss.name!r}) is computed on stream "
                    f"{loss.stream!r}. The semantics of a stream-side loss "
                    "(computed on the residual itself, bypassing the head) "
                    "are still undecided: the residual is token shaped while "
                    "the target is raw shaped, and an inverse patchify is "
                    "missing in between"
                )
            if not loss.broadcast.is_per_sample and loss.reducer is None:
                raise NotImplementedError(
                    f"LossSpec({loss.name!r}).broadcast="
                    f"{loss.broadcast.axes} asks for per-axis weighting "
                    "without a Reducer; supply LossSpec.reducer to weight "
                    "per-frame or per-token errors before reduction"
                )
            head = self.spec.head(str(loss.head))
            active = self._gate_mask(loss, batch, runtime, outputs[head.name])
            value = self._loss_value(loss, runtime, outputs, batch=batch, active=active)
            weight = loss.weight_at(step)
            terms.append(weight * value)
            metrics[f"loss_{loss.name}"] = value.detach()
            metrics[f"{loss.name}/weight"] = weight
            if active is not None:
                metrics[f"loss_{loss.name}_active"] = active.float().mean().detach()
        total = torch.stack(terms).sum()
        metrics["loss"] = total.detach()
        return total, metrics

    # ---- Inference path. Three steps (model_matrix §4.5.3) ----

    @torch.no_grad()
    def sample(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """Target three-step inference orchestration."""
        diffusion: DiffusionSpec | None = self.spec.diffusion
        if diffusion is None:
            raise NotImplementedError(
                f"model {self.spec.model_type!r} has no DiffusionSpec "
                "(a non-diffusion model); the single-step inference path "
                "belongs to the assembly layer and is not implemented yet"
            )
        if not diffusion.is_iterative:
            raise NotImplementedError(
                f"model {self.spec.model_type!r} has a DiffusionSpec with no "
                "sampling_gate/solver (a single-step feed-forward "
                "model); the "
                "single-step inference path belongs to the assembly layer "
                "and is not implemented yet"
            )
        raise NotImplementedError(
            "WorldModel.sample three-step orchestration is not implemented "
            "yet (assembly layer); criteria in "
            "mot_spec_model_matrix.zh-CN.md §4.5.3"
        )
