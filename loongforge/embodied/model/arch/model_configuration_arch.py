# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Typed YAML configs for the ``arch`` world models.

One base class plus one tiny subclass per model. The base holds everything that
is shared: the six spec sections ``build_world_model`` reads, and the runtime
fields that are not part of the spec at all (dtype, encoder repo id, text
context length).

A subclass adds **only provider keyword arguments** — the handful of numbers the
six sections cannot express, such as how many video frames the action stream may
attend to. That split is what makes ``provider_kwargs`` work by reflection: any
field declared below the base is, by construction, a provider argument.

Two rules the sections exist to enforce:

* No YAML line names a model or a topology. ``--model-name`` selects the schema
  row, the schema row selects the subclass, and the subclass default fills
  ``model_type``. Stream count, visibility, condition table and loss terms are
  Python code in the provider.
* Every number has one source. ``action_horizon`` and the text width are read
  back off the built spec rather than restated here, because the data pipeline
  needs them and a second copy in YAML would drift from the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from .spec.world import WorldModelSpec, build_world_model


@dataclass(frozen=True)
class WorldModelConfig:
    """Shared ``model:`` section for every arch provider.

    Attributes:
        model_type: Provider registry key. Set by each subclass's default; a
            YAML file never writes it.
        geometry: Overrides for ``GeometrySpec`` fields. Reaches the provider as
            a keyword argument, because visibility and the flow heads are
            derived from the geometry.
        streams: ``{stream: {scalar: value}}``, flattened to
            ``<stream>_<scalar>`` provider arguments (widths, layer counts).
        diffusion: ``{stream: {knob: value}}``, same flattening. Today the only
            knob is ``shift``; a stream whose distribution has no scalar (vpp2's
            constant video timestep) simply has no entry.
        heads: ``{head: {weight|warmup_steps|enabled}}``, applied to the loss
            term that consumes the head after the provider has run.
        attention_backend: ``dense`` / ``flex`` / ``sdpa`` / ``flash``, or
            ``None`` to keep the provider's choice.
        attention_block_size: Block size for ``flex``, which is the only backend
            that takes one.
        model_id: Repo or directory holding the VAE and CLIP image encoder. The
            arch streams are built from the spec, so this is the *encoder* stack
            only; fine-tuned DiT weights arrive via ``--pretrained-checkpoint``.
        tokenizer_max_len: Length of the cached text embeddings, i.e. the token
            count of the ``text`` cross-attention slot.
        dtype: Parameter dtype for the built model.
        compile_vae_encode: Wrap the VAE encode call in ``torch.compile``.
    """

    model_type: str = ""

    geometry: Dict[str, Any] = field(default_factory=dict)
    streams: Dict[str, Any] = field(default_factory=dict)
    diffusion: Dict[str, Any] = field(default_factory=dict)
    heads: Dict[str, Any] = field(default_factory=dict)
    attention_backend: Optional[str] = None
    attention_block_size: Optional[int] = None

    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P"
    tokenizer_max_len: int = 128
    dtype: str = "bfloat16"
    compile_vae_encode: bool = False

    def provider_kwargs(self) -> Dict[str, Any]:
        """Collect the per-model provider arguments.

        Every field a subclass declares is a provider argument, so the set is
        computed as "fields of this class that the base does not have". Listing
        them by hand in each subclass would be a second place to forget.

        ``None`` means "no opinion" rather than "pass None": the provider's own
        default is the single source for an unset knob, and forwarding ``None``
        would overwrite it with a value no signature accepts.

        Returns:
            Keyword arguments for the provider, excluding the shared sections.
        """
        base = frozenset(f.name for f in fields(WorldModelConfig))
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in base and getattr(self, f.name) is not None}

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
        from .spec import providers as _providers  # noqa: F401

        return build_world_model(self)

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


@dataclass(frozen=True)
class Vpp2Config(WorldModelConfig):
    """``--model-name arch_vpp2``.

    Attributes:
        visible_video_frames: How many leading video frames the action stream
            may attend to. Shapes a visibility rule, hence a provider argument
            and not a section key.
        inference_steps: Denoising steps for the action stream at eval time.
        proprio_dim: Width of the raw proprioception vector, i.e. the input
            width of the token encoder whose output is appended to the text
            slot. Unrelated to any stream width.
    """

    model_type: str = "arch_vpp2"
    visible_video_frames: Optional[int] = None
    inference_steps: Optional[int] = None
    proprio_dim: Optional[int] = None


@dataclass(frozen=True)
class InternW0DeltaConfig(WorldModelConfig):
    """``--model-name internw0_delta``.

    Attributes:
        condition_frames: Clean history frames prepended to the noised video
            sequence. Enters the frame count the provider hands to the geometry,
            so it cannot be a plain geometry override.
        inference_steps: Denoising steps for the action stream at eval time.
        proprio_dim: Width of the raw proprioception vector.
    """

    model_type: str = "internw0_delta"
    condition_frames: Optional[int] = None
    inference_steps: Optional[int] = None
    proprio_dim: Optional[int] = None
