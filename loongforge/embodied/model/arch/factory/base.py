# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""What **every** arch model shares, and nothing a single family shares.

Three things live here, shared by construction rather than by convention:

* ``remap_state_dict`` and the ``dual_stream_wan`` rule tables. All three
  family members save through ``{"mot": <mot>.state_dict()}``, so the source
  namespace is ``mixtures.<stream>.`` for every one of them and only the extra
  branches (VPP2's CLIP, InternW0-Delta's future-delta head) differ.
* ``ArchModelConfig``, which holds the six spec sections ``build_world_model``
  reads plus the runtime fields that are not part of the spec at all. It
  carries **no weight path**: where the frozen towers come from depends on
  which towers a family has, so that field belongs to the family config.
* ``ArchModelBase``, the abstract shell. It owns what is true of any arch
  model -- spec construction, the engine-capability check, dtype resolution
  and the remapped warm start -- and leaves the batch path abstract, because
  "pixels in" is a statement about one family and not about the architecture.

The per-family layer lives in ``factory/families/``. A family base binds the
engine it is built on (``engine_cls``) and implements ``encode_batch`` /
``forward``; a model package under ``factory/<model_name>/`` then adds only
its registry key, its ``config_cls`` and its ``transfer`` table.

Three families are planned, one of which exists:

* ``families/dual_stream_wan.py`` -- video + action streams over joint
  attention, with a frozen Wan VAE and CLIP encoder in the shell. Holds
  ``vpp2`` and ``internw0_delta``.
* **VLM-as-a-stream** (unbuilt). The VLM is a stream *inside* the core, so its
  weights live in ``core.blocks.vlm.*``, it is trained with the rest, and the
  shell's ``encode_batch`` produces image and text tokens rather than latents.
  ``pi05`` is this shape (spec_config §5.4).
* **VLM-as-a-tower** (unbuilt). The VLM is *outside* the core and frozen; the
  core holds the action stream alone and the VLM's output arrives through a
  condition slot (``pretrained: {vlm: ...}`` plus a ``CROSS_KV`` condition,
  spec_config §5.6: ``groot_n1_6``, ``xvla``). Mechanically it is this file's
  ``source_encoder`` path, not a new concept.

The two VLA families are described here rather than stubbed as empty modules:
the boundary that matters is whether the VLM's parameters are inside the core,
because that single answer decides the state-dict namespace, whether the
optimizer sees those parameters, and which of ``encode_batch`` / a condition
``source_encoder`` carries the modality. An empty file would not record it.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from ..builder import WorldModel
from ..spec.world import WorldModelSpec, build_world_model_spec

logger = logging.getLogger(__name__)

__all__ = ["ArchModelBase", "ArchModelConfig", "CATCH_ALL",
           "DUAL_STREAM_WAN_RENAME_RULES", "HEAD_RULES",
           "PROPRIO_PAYLOAD_KEY", "RenameRules", "TEXT_RULES",
           "attach_proprio", "remap_state_dict", "resolve_dtype"]


RenameRules = Tuple[Tuple[str, str], ...]


def remap_state_dict(state_dict: Mapping[str, Any], rules: RenameRules, *,
                     drop: Iterable[str] = ()) -> Dict[str, Any]:
    """Rename keys by an **ordered** rule table; the first match wins.

    Why an ordered regex table instead of one dict of all keys: the legacy
    VPP2 checkpoint holds 2312 tensors (1298 video + 1014 action, with
    ``num_layers=40``), so a key-by-key table would be longer than the model
    and would be entirely wrong for a different layer count. The rule table's
    size is independent of the layer count.

    Three situations deliberately raise instead of being skipped:
      1. a key matches no rule -- skipping it shows up later as a strict-load
         "missing" error, and by then you cannot tell "the rules did not cover
         it" from "the checkpoint never had it";
      2. two source keys map to the same target key -- the later write would
         silently overwrite the earlier one;
      3. ``drop`` names a prefix that does not exist -- that drop entry is
         stale.

    ``drop`` is a **declared** discard (weights the legacy model has and the
    new architecture does not, for example ``video_kv_projections``, which is
    only built when ``use_video_tokens=True``). It is kept separate from "no
    rule matched" on purpose: the first is a decision, the second is an
    oversight.

    Args:
        state_dict: Source mapping of legacy key to tensor.
        rules: Ordered ``(pattern, replacement)`` pairs applied with
            ``re.subn(..., count=1)``.
        drop: Key prefixes to discard, each of which must match something.

    Returns:
        A new dict keyed by the new architecture's names.

    Raises:
        ValueError: On a mapping collision or a stale ``drop`` prefix.
        KeyError: If any key matches no rule.
    """
    dropped: set[str] = set()
    out: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    unmatched: list[str] = []
    prefixes = tuple(drop)
    for key, value in state_dict.items():
        hit = next((prefix for prefix in prefixes
                    if key.startswith(prefix)), None)
        if hit is not None:
            dropped.add(hit)
            continue
        for pattern, replacement in rules:
            new_key, count = re.subn(pattern, replacement, key, count=1)
            if count:
                if new_key in out:
                    raise ValueError(
                        f"remap collision: {key!r} and {sources[new_key]!r} "
                        f"both map to {new_key!r}. The later write would "
                        f"silently overwrite the earlier one"
                    )
                out[new_key] = value
                sources[new_key] = key
                break
        else:
            unmatched.append(key)
    if unmatched:
        raise KeyError(
            f"{len(unmatched)} keys match no rule, first 5: "
            f"{unmatched[:5]}. Skipping them turns into a strict-load "
            "missing error, and by then you cannot tell a rule gap from a "
            "key the ckpt never had"
        )
    stale = sorted(set(prefixes) - dropped)
    if stale:
        raise ValueError(
            f"drop prefixes {stale} matched no key in this ckpt; "
            "a stale drop silently discards real weights on the next ckpt"
        )
    return out

# ---- the dual_stream_wan rename family ----

# The source namespace is ``mixtures.<stream>.`` for all three family members:
# saving goes through ``{"mot": <mot>.state_dict()}`` (``vpp2.py:1799``,
# ``fastwam/mot/fastwam.py:1169``,
# ``InternW0-Delta wam/training/checkpoint.py:11``), and every ``MoT`` has
# exactly one submodule, ``self.mixtures = nn.ModuleDict`` (``mot.py:28``,
# ``fastwam/mot/model.py:74``, ``mixture_of_transformers.py:90``), keyed
# ``"video"`` / ``"action"``.
# Inside ``VPP2``, ``video_expert`` / ``mot`` / ``dit`` are **three
# registrations of the same objects** (``vpp2.py:104-107``), so one tensor has
# three keys in ``VPP2.state_dict()`` -- we match the prefix that is on disk,
# not the in-memory aliases.
#
# One fact runs through the whole table: weights live in exactly three
# namespaces (``mot_spec_dataflow.svg``) -- L2's ``blocks.<stream>.*`` and
# ``condition_encoders.<condition>.*``, plus L4's ``heads.<head>.*``. So the
# legacy habit of growing the head inside the expert (``WanVideoDiT.head``) is
# a **cross-layer move** here, not a prefix rename: it is the only class of
# rule in this table that changes the tree shape, which is why it comes first.
HEAD_RULES: RenameRules = (
    # Cross-layer move: legacy grows the output head inside the expert
    # (``WanVideoDiT.head`` / ``ActionDiT.head``), while in the new
    # architecture the head is a standalone L4 object.
    # Must precede the catch-all, or it would be treated as an in-stream
    # parameter.
    # The video head's inner Linear is also named head, so the source key
    # literally reads ``head.head.weight``.
    (r"^mixtures\.video\.head\.head\.", "heads.video_flow.head."),
    (r"^mixtures\.video\.head\.modulation$", "heads.video_flow.modulation"),
    # The action head is a bare ``nn.Linear`` (``action_dit.py:150``,
    # ``fastwam/action/dit.py:132``, ``InternW0-Delta action_dit.py:128``), so
    # it has no modulation and no inner head. The ``ActionHead`` sitting next
    # to it in all three repos is dead code.
    (r"^mixtures\.action\.head\.", "heads.action_flow.head."),
)

TEXT_RULES: RenameRules = (
    # Condition projectors are filed by **slot name**: legacy gave each
    # condition its own name (``img_emb`` / ``text_embedding``), while the new
    # architecture unifies them under ``context_embedding.<slot>``, where the
    # slot is ``ConditionSpec.name``.
    (r"^mixtures\.(video|action)\.text_embedding\.",
     r"blocks.\1.context_embedding.text."),
    # The cross-attention KV side is grouped by slot name too.
    (r"^mixtures\.(video|action)\.blocks\.(\d+)\.cross_attn\.(k|v|norm_k)\.",
     r"blocks.\1.blocks.\2.cross_attn.kv.text.\3."),
)

CATCH_ALL: RenameRules = (
    # Everything else only swaps the top-level prefix: the submodule names
    # ``self_attn.{q,k,v,o,norm_q,norm_k}`` / ``norm3`` / ``ffn.{0,2}`` /
    # ``modulation`` / ``time_embedding.{0,2}`` / ``time_projection.1`` /
    # ``patch_embedding`` / ``action_encoder`` are **deliberately** kept
    # byte-identical to legacy in the L2 implementation, so this single rule
    # covers the vast majority of the tensors.
    (r"^mixtures\.(video|action)\.", r"blocks.\1."),
)

DUAL_STREAM_WAN_RENAME_RULES: RenameRules = (
    HEAD_RULES + TEXT_RULES + CATCH_ALL)
"""The rules every member of the ``dual_stream_wan`` family shares.

This is the **whole** table for fastwam: its video expert asserts
``has_image_input is False`` and ``require_clip_embedding is False``
(``fastwam/wan/dit.py:567-568``), so there is no CLIP branch to rename, and
its cross-attention KV is the plain ``k`` / ``v`` / ``norm_k``. The
``img_emb`` / ``k_img`` names exist in that repo only inside the Diffusers
importer (``fastwam/utils/state_dict.py:89-118``) and never reach a saved
checkpoint. VPP2 and InternW0-Delta each prepend their own rules.
"""

PROPRIO_PAYLOAD_KEY = "proprio_encoder"
"""The proprio encoder's key in legacy ckpts, **a sibling of** ``"mot"``.

Written at ``vpp2.py:1800-1801``, read at ``:1841-1842``; the same sibling
appears in fastwam (``mot/fastwam.py:1174``) and in InternW0-Delta
(``wam/training/checkpoint.py:15-16``). It is not inside the ``mot`` subtree
because the legacy ``nn.Linear`` hangs off the top-level model rather than off
any single expert -- its output is concatenated into the context shared by
both streams, so assigning it to either stream would be wrong.

The new architecture states this plainly as the third parameter namespace,
``condition_encoders.<condition>.*``, so this top-level key lands exactly on
``condition_encoders.proprio.{weight,bias}`` (``TokenEncoder`` subclasses
``nn.Linear`` instead of wrapping one precisely so those two keys match
literally).
"""


def attach_proprio(out: Dict[str, Any], payload: Mapping[str, Any]) -> None:
    """Move the sibling ``proprio_encoder`` into ``condition_encoders.proprio``.

    ``TokenEncoder`` subclasses ``nn.Linear``, so ``weight`` and ``bias`` match
    literally. InternW0-Delta's encoder carries a third parameter,
    ``type_embedding``, which its forward simply **adds to the output**
    (``proprio_encoder.py:16-17``: ``super().forward(input) + type_embedding``)
    with nothing non-linear in between, so it is a second bias and is folded
    into the first. Folding is exact, and it is preferable to giving the new
    architecture a parameter whose only job is to be added to another one.

    Args:
        out: Remapped state dict, modified in place.
        payload: The full checkpoint; a bare ``mot`` subtree carries no
            proprio encoder and leaves ``out`` untouched.
    """
    encoder = payload.get(PROPRIO_PAYLOAD_KEY)
    if encoder is None:
        return
    extra = encoder.get("type_embedding")
    for name, value in encoder.items():
        if name == "type_embedding":
            continue
        if name == "bias" and extra is not None:
            value = value + extra
        out[f"condition_encoders.proprio.{name}"] = value

# ---- the shared ``model:`` section ----
#
# Two rules the spec sections exist to enforce:
#
# * No YAML line names a model or a topology. ``--model-name`` selects the
#   schema row, the schema row selects the subclass, and the subclass default
#   fills ``model_type``. Stream count, visibility, condition table and loss
#   terms are Python code in the provider.
# * Every number has one source. ``action_horizon`` and the text width are
#   read back off the built spec rather than restated here, because the data
#   pipeline needs them and a second copy in YAML would drift from the first.


@dataclass(frozen=True)
class ArchModelConfig:
    """Shared ``model:`` section for every arch provider.

    Attributes:
        model_type: Provider registry key. Set by each subclass's default; a
            YAML file never writes it.
        geometry: Overrides for ``GeometrySpec`` fields. Reaches the provider as
            a keyword argument, because visibility and the flow heads are
            derived from the geometry.
        streams: ``{stream: {scalar: value}}``, flattened to
            ``<stream>_<scalar>`` provider arguments (widths, layer counts).
        diffusion: ``{stream: {knob: value}}``, passed to the provider as
            ``diffusion=`` and resolved over its defaults before it builds the
            stream's objects (``resolve_diffusion``), e.g. the shift of
            ``UniformShift``. A stream with no knob (vpp2's constant video
            timestep) takes no entry.
        heads: ``{head: {weight|warmup_steps|enabled}}``, applied to the loss
            term that consumes the head after the provider has run.
        attention_backend: Backend name, or ``None`` to keep the provider's
            choice. Which names are accepted is answered by the family's
            engine (``ArchModelBase.validate_spec``), not by a list here.
        attention_block_size: Block size for ``flex``, which is the only backend
            that takes one.
        tokenizer_max_len: Length of the cached text embeddings, i.e. the token
            count of the ``text`` cross-attention slot.
        dtype: Parameter dtype for the built model.
    """

    model_type: str = ""

    geometry: Dict[str, Any] = field(default_factory=dict)
    streams: Dict[str, Any] = field(default_factory=dict)
    diffusion: Dict[str, Any] = field(default_factory=dict)
    heads: Dict[str, Any] = field(default_factory=dict)
    attention_backend: Optional[str] = None
    attention_block_size: Optional[int] = None

    tokenizer_max_len: int = 128
    dtype: str = "bfloat16"

    def provider_kwargs(self) -> Dict[str, Any]:
        """Collect the per-model provider arguments.

        Every field a *model* config declares is a provider argument, so the
        set is computed as "fields of this class that no ancestor declares".
        Listing them by hand in each subclass would be a second place to
        forget.

        The excluded set is the union over **every dataclass ancestor**, not
        the fields of this class alone: a family config sits between the two
        (``Vpp2Config`` -> ``DualStreamWanConfig`` -> ``ArchModelConfig``) and
        its fields -- the frozen-tower weight path, for one -- are shell
        business that no provider signature accepts.

        ``None`` means "no opinion" rather than "pass None": the provider's own
        default is the single source for an unset knob, and forwarding ``None``
        would overwrite it with a value no signature accepts.

        Returns:
            Keyword arguments for the provider, excluding every inherited field.
        """
        inherited = frozenset(
            f.name for ancestor in type(self).__mro__[1:]
            if is_dataclass(ancestor) for f in fields(ancestor))
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in inherited and getattr(self, f.name) is not None}

    def build_spec(self) -> WorldModelSpec:
        """Build the ``WorldModelSpec`` this config describes.

        Not cached: the config is frozen, the spec is a tree of frozen
        dataclasses, and building it is pure dataclass construction. Callers who
        need the spec more than once hold on to the result.
        """
        # Registration is an import side effect, and the provider package
        # cannot be imported at module scope here: the providers import
        # ``spec.world``, which this module also imports. Importing inside the
        # one function that needs a populated registry keeps that cycle out
        # while guaranteeing the lookup never sees an empty table.
        from ..spec import providers as _providers  # noqa: F401

        return build_world_model_spec(self)

    @property
    def action_horizon(self) -> int:
        """Action chunk length, read back off the spec for the data pipeline.

        The dataset needs this number to slice action windows. Declaring it as a
        field here as well as in ``geometry:`` would let the two disagree, and
        the symptom would be a shape error far from either declaration.
        """
        return self.build_spec().geometry.action_horizon

    @property
    def video_dit_config(self) -> Dict[str, Any]:
        """Compatibility view for the collator's ``video_dit_config["text_dim"]``.

        The text slot's width lives in the condition table, which is the only
        place that decides how wide the cached text embeddings must be. This
        property exposes it under the key the existing FastWAM collator reads so
        that the collator can be reused unchanged.
        """
        return {"text_dim": self.build_spec().condition("text").width}

# ---- dtype resolution ----

_DTYPES: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32, "full": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config dtype string to a ``torch.dtype``.

    Args:
        name: Value of ``ArchModelConfig.dtype``.

    Returns:
        The matching torch dtype.

    Raises:
        ValueError: The alias is not one of the recognised spellings.
    """
    try:
        return _DTYPES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unsupported arch dtype {name!r}; expected one of "
            f"{sorted(_DTYPES)}") from None


class ArchModelBase(nn.Module, ABC):
    """Abstract trainer-facing wrapper around one ``WorldModel``.

    ``WorldModel`` is pure assembly: it takes a batch of *latents* and returns
    ``(loss, metrics)``. The trainer hands the model whatever its data pipeline
    produces -- pixels for a video family, image and text tokens for a VLM
    family. Translating one into the other is the shell's job, and it is the
    part this class leaves **abstract**: ``encode_batch`` and ``forward`` are
    the two methods a family must write, and there is no ``if is_vla`` branch
    anywhere below.

    What is concrete here is what holds for any arch model: the config type
    check, spec construction, the engine-capability check, assembling the core
    on the family's engine, dtype resolution, and the remapped warm start.

    Attributes:
        engine_cls: The ``StreamEngine`` implementation this family's topology
            is built on, bound by the family base. It is a class attribute and
            not a config field because the engine *is* the topology: joint
            attention over equal-depth streams is a different model from
            sequential attention, not a different setting of one. The backend
            **kernel** stays a YAML knob (``attention_backend``), because that
            one does not change what is computed.
        config_cls: The ``ArchModelConfig`` subclass this model is built from.
            ``from_pretrained`` rejects anything else, so a YAML row bound to
            the wrong schema fails at construction instead of at the first
            missing provider argument.
        transfer: The checkpoint rename table for this model, from the same
            package's ``transfer.py``. ``None`` means no table exists yet, and
            ``load_pretrained`` then raises rather than loading nothing --
            which would look like "trained from scratch with a warm start".
        config: The typed config the model was built from.
        core: The assembled ``WorldModel``. Every trainable parameter lives
            here or in nothing at all; any tower a family adds is frozen.
    """

    engine_cls: Optional[type] = None
    config_cls: type = ArchModelConfig
    transfer: Optional[Callable[..., Dict[str, Any]]] = None

    def __init__(self, config: ArchModelConfig, core: WorldModel) -> None:
        """Wire a built core into one module.

        Args:
            config: Typed config instance for this model.
            core: Assembled ``WorldModel``.
        """
        super().__init__()
        self.config = config
        self.core = core
        self._dtype = resolve_dtype(config.dtype)

    @property
    def spec(self) -> WorldModelSpec:
        """The built spec. Single source for geometry, conditions and losses."""
        return self.core.spec

    @classmethod
    def validate_spec(cls, spec: WorldModelSpec) -> None:
        """Check the built spec against the engine this family actually runs.

        This is the one place that answers "can this backend run here", and it
        answers it by asking the engine. The spec layer deliberately does not
        keep a backend whitelist: a second list that no engine consults drifts
        from the implementations in both directions -- a name in the list with
        no kernel passes validation and raises in the engine constructor, and
        a kernel an engine does implement gets rejected before it is asked.

        Called by ``from_pretrained`` right after ``build_spec()``, i.e. before
        a single parameter is allocated.

        Args:
            spec: The spec just built from the config.

        Raises:
            NotImplementedError: This class binds no ``engine_cls``, which
                means a family base was skipped.
            ValueError: The spec names a backend this family's engine does not
                implement.
        """
        if cls.engine_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} binds no engine_cls; a model class must "
                "inherit from a family base in factory/families/, which is "
                "what decides the attention topology")
        supported = tuple(getattr(cls.engine_cls, "SUPPORTED_BACKENDS", ()))
        if spec.attention_backend not in supported:
            raise ValueError(
                f"model {spec.model_type!r}: attention_backend="
                f"{spec.attention_backend!r} is not implemented by "
                f"{cls.engine_cls.__name__}, the engine this family runs on; "
                f"it supports {list(supported)}")

    @classmethod
    def build_engine(cls, spec: WorldModelSpec) -> Any:
        """Instantiate the family's engine for one built spec.

        The backend comes from the spec rather than from an argument because
        ``WorldModelSpec`` already validated the backend/block-size pair;
        re-deciding it here would allow a run whose engine and whose masks
        disagree. Both fields are universal spec fields, so the default works
        for any engine that names its constructor arguments the same way; a
        family whose engine takes something else overrides this.

        Args:
            spec: The validated spec.

        Returns:
            A ``StreamEngine``.
        """
        return cls.engine_cls(backend=spec.attention_backend,
                              block_size=spec.attention_block_size)

    @classmethod
    def build_towers(cls, cfg: Any, spec: WorldModelSpec) -> Dict[str, Any]:
        """Build the frozen towers this family's ``__init__`` takes.

        Returns the keyword arguments ``from_pretrained`` passes to ``cls``
        beyond ``(cfg, core)``. The default is empty: a family with no frozen
        tower -- the VLM-as-a-stream shape, where the VLM is a core stream --
        needs nothing here.

        Towers are built here rather than inside ``__init__`` so that tests
        and any caller without CUDA can construct the model with stubs:
        ``WanVideoVAE.__init__`` allocates its normalisation statistics
        straight onto ``cuda``.

        Args:
            cfg: The typed config; it carries the weight path.
            spec: The validated spec; it declares which towers exist, through
                ``StreamSpec.source_encoder`` and
                ``ConditionSpec.source_encoder``.

        Returns:
            Keyword arguments for ``cls(cfg, core, **kwargs)``.
        """
        return {}

    @classmethod
    def from_pretrained(cls, cfg: Any) -> "ArchModelBase":
        """Build the model from a typed config (the ``build_model`` entry point).

        Five steps, in order: reject a config of the wrong schema, build the
        spec, check it against this family's engine, assemble the core, then
        build the frozen towers the spec declares and cast to the configured
        dtype.

        Args:
            cfg: An instance of this class's ``config_cls``.

        Returns:
            A ready-to-train model.

        Raises:
            TypeError: ``cfg`` is not an instance of ``cls.config_cls``.
        """
        if not isinstance(cfg, cls.config_cls):
            raise TypeError(
                f"{cls.__name__}.from_pretrained expects a "
                f"{cls.config_cls.__name__} instance; got "
                f"{type(cfg).__name__}")
        spec = cfg.build_spec()
        cls.validate_spec(spec)
        core = WorldModel(spec, engine=cls.build_engine(spec),
                          layout=spec.build_layout())
        model = cls(cfg, core, **cls.build_towers(cfg, spec))
        return model.to(dtype=resolve_dtype(cfg.dtype))

    @abstractmethod
    def encode_batch(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """Map one collated sample onto the keys the spec declares.

        Abstract because this is exactly where the families differ: the
        dual-stream Wan family VAE-encodes pixels, a VLM-as-a-stream family
        tokenises images and text instead. Two keys come out of any
        implementation: one tensor per stream, named by
        ``StreamSpec.batch_key``, and one per enabled ``ConditionSpec``, named
        by its ``source_key``.

        Args:
            sample: The collated batch, in whatever space the family's data
                pipeline produces.

        Returns:
            The batch ``WorldModel.forward`` consumes.
        """

    @abstractmethod
    def forward(self, batch: Any, iteration: Optional[int] = None
                ) -> Tuple[Tensor, Dict[str, Any]]:
        """Run one training step. Interface I4: ``(loss, metrics)``.

        Abstract rather than ``self.core(self.encode_batch(...))``: a family
        with a frozen tower inside the step, or with an auxiliary loss the core
        does not own, computes more than that, and a default would be silently
        correct for one family only. Every implementation names ``iteration``
        and passes the batch through ``_sample_at_step``.

        Args:
            batch: The trainer's batch object or a plain dict.
            iteration: Optimizer steps completed before this one, from the
                trainer (``_select_forward_kwargs`` passes it only to a forward
                that names it). It becomes the core's ``global_step``, which
                ``LossSpec.weight_at`` reads for ``WeightRamp``; without it
                every ramped loss stays at its start value.

        Returns:
            The weighted total loss and the per-term metrics dict.

        Raises:
            ValueError: The sample already carries a different
                ``global_step``; two step counters would disagree silently.
        """

    @staticmethod
    def _sample_at_step(batch: Any, iteration: Optional[int]) -> Dict[str, Any]:
        """The batch as a sample dict, with the trainer's iteration as its step.

        Args:
            batch: The trainer's batch object (``to_sample()``) or a plain dict.
            iteration: Optimizer steps completed before this one, or ``None``.

        Returns:
            The sample; ``global_step`` is set when ``iteration`` is given and
            left as the batch has it otherwise.

        Raises:
            ValueError: The sample already carries a different ``global_step``.
        """
        sample = batch.to_sample() if hasattr(batch, "to_sample") else dict(batch)
        if iteration is not None:
            given = sample.get("global_step")
            if given is not None and int(given) != int(iteration):
                raise ValueError(
                    f"batch global_step={int(given)} disagrees with the trainer "
                    f"iteration={int(iteration)}")
            sample = {**sample, "global_step": int(iteration)}
        return sample

    def _mask_keys(self) -> tuple[str, ...]:
        """Batch keys of the boolean masks the spec reads, in declaration order."""
        keys: list[str] = []
        for mask_spec in self.spec.loss_masks.values():
            keys.extend(mask_spec.batch_keys)
        for loss in self.spec.losses:
            if loss.loss_mask is not None:
                keys.extend(loss.loss_mask.batch_keys)
            if loss.gate is not None:
                keys.extend(loss.gate.batch_keys)
        return tuple(dict.fromkeys(keys))

    def _target_keys(self) -> tuple[str, ...]:
        """Batch keys of explicit ``LossSpec.target_key`` targets."""
        return tuple(dict.fromkeys(
            loss.target_key for loss in self.spec.losses if loss.target_key is not None))

    @staticmethod
    def _require(sample: Mapping[str, Any], key: str) -> Tensor:
        """Fetch a required tensor, naming the available keys when it is absent.

        Args:
            sample: Collated batch.
            key: Key the caller needs.

        Returns:
            The tensor at ``key``.

        Raises:
            KeyError: ``key`` is missing or ``None``.
        """
        value = sample.get(key)
        if value is None:
            raise KeyError(
                f"arch batch needs {key!r}, but the sample only has "
                f"{sorted(k for k, v in sample.items() if v is not None)}")
        return value

    def load_pretrained(self, path: str, device: Any = None) -> Optional[int]:
        """Load upstream weights into the core (the trainer's warm-start hook).

        The checkpoint is remapped before loading: upstream keeps the output
        heads inside the experts and names the condition projectors after the
        modality they happened to carry, while this architecture keeps heads in
        ``heads.*`` and projectors under their slot name. The rename table is
        the subclass's ``transfer`` attribute, defined in
        ``factory/<model_name>/transfer.py``.

        ``strict=True``: a warm start that silently leaves half the backbone
        randomly initialised trains, converges slowly and looks like a
        hyper-parameter problem.

        Args:
            path: Checkpoint file written by the upstream implementation.
            device: Optional device to move the model to afterwards.

        Returns:
            The optimizer step recorded in the checkpoint, if it has one.

        Raises:
            NotImplementedError: This subclass declares no rename table.
        """
        if self.transfer is None:
            raise NotImplementedError(
                f"no checkpoint rename table for model_type="
                f"{self.config.model_type!r}; write one in this model's "
                f"factory/<model_name>/transfer.py and point the subclass's "
                f"'transfer' attribute at it")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.core.load_state_dict(self.transfer(payload), strict=True)
        logger.info("[arch] core weights loaded from %s", path)
        if device is not None:
            self.to(device)
        step = payload.get("step") if isinstance(payload, dict) else None
        return int(step) if isinstance(step, int) else None
