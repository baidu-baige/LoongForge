# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The gate: ``--model-name`` → YAML → typed config → arch model.

Everything the other ``test_arch_*`` files check lives inside the ``arch``
package. This file checks the five seams that connect it to the trainer, each of
which fails silently rather than loudly when it breaks:

1. ``MODEL_SCHEMA`` rows resolve to YAML files that exist and merge cleanly into
   the typed config classes.
2. ``geometry.num_frames`` equals ``data.num_video_frames``. The VAE sees the
   frames the dataset sampled, so these two numbers are one fact written twice;
   a disagreement only surfaces as a latent-grid shape error deep in a forward.
3. The model registry maps both arch names onto the one shell, and does *not*
   steal ``fastwam`` from ``FastWAMPolicy``.
4. The shell really converts a pixel batch into ``(loss, metrics)`` — with stub
   encoders, because the real VAE allocates its normalisation buffers on CUDA.
5. Each ``diffusion:`` key is one the provider lists for ``resolve_diffusion``
   before it builds the object; any other key raises when the spec is built.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from loongforge.embodied.model.arch.builder import WorldModel
from loongforge.embodied.model.arch.factory import (ArchModelConfig,
                                                    DualStreamWanBase,
                                                    DualStreamWanConfig,
                                                    InternW0DeltaArchModel,
                                                    Vpp2ArchModel, Vpp2Config)
from loongforge.embodied.model.arch.diffusion.loss import WeightRamp
from loongforge.embodied.model.arch.diffusion.timesteps import UniformShift
from loongforge.embodied.model.arch.spec.core import Spec
from loongforge.embodied.model.arch.spec.overrides import resolve_diffusion
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.arch.towers import build_wan_clip, build_wan_vae
from loongforge.embodied.model.registry import MODEL_REGISTRY
from loongforge.embodied.train.config_map import get_config_path, get_model_schema
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer

from test_arch_blocks import BATCH, CONTEXT_LEN, tiny_spec

ARCH_MODELS = ("vpp2", "internw0_delta")


def load_config(model_name: str):
    """Reproduce the parser's merge for one model name.

    Mirrors ``train/parser.py``: a structured default built from the schema's
    config class, with the YAML ``model:`` section merged on top.

    Args:
        model_name: A ``MODEL_SCHEMA`` key.

    Returns:
        The materialised model config instance.
    """
    schema = get_model_schema(model_name)
    yaml_cfg = OmegaConf.load(get_config_path(model_name))
    merged = OmegaConf.merge(OmegaConf.structured(schema.model_config_cls),
                             yaml_cfg.get("model", {}))
    return OmegaConf.to_object(merged)


# ---- 1. schema rows and the YAML merge ----


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_yaml_merges_into_the_typed_config(model_name):
    """The YAML is a set of overrides on the dataclass, not a free-form dict.

    ``OmegaConf.to_object`` on a structured merge rejects unknown keys and wrong
    types, so this one call covers every field name in the file.
    """
    cfg = load_config(model_name)
    assert cfg.model_type == model_name
    assert cfg.build_spec().model_type == model_name


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_model_identity_is_not_written_in_the_yaml(model_name):
    """Rule R5: no YAML line names the model, a class path or a topology.

    ``model_type`` comes from the config subclass default, which the
    ``--model-name`` argument selected through ``MODEL_SCHEMA``. A file that also
    wrote it would let the two disagree, and the loser would be silent.
    """
    section = OmegaConf.load(get_config_path(model_name)).model
    assert "model_type" not in section
    assert "_target_" not in section
    # Trainable DiT weights arrive via ``--pretrained-checkpoint``; ``model_id``
    # is the encoder stack and is a different fact.
    assert "pretrained" not in section


# ---- 2. the frame count the dataset and the VAE have to agree on ----


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_geometry_frame_count_matches_the_dataloader(model_name):
    """``geometry.num_frames`` is the pixel frame count handed to the VAE.

    The dataset samples ``num_video_frames`` frames out of
    ``(T - 1) * action_video_freq_ratio + 1`` raw ones, so the tensor that
    reaches ``vae.encode`` has exactly ``data.num_video_frames`` frames. If the
    geometry says otherwise, the latent grid check in ``encode_video`` fires on
    the first batch of a real run — long after the config was written.
    """
    yaml_cfg = OmegaConf.load(get_config_path(model_name))
    assert yaml_cfg.model.geometry.num_frames == yaml_cfg.data.num_video_frames
    assert (yaml_cfg.model.geometry.action_video_freq_ratio
            == yaml_cfg.data.action_video_freq_ratio)


# ---- 3. the registries ----


ARCH_SUBCLASSES = {"vpp2": Vpp2ArchModel,
                   "internw0_delta": InternW0DeltaArchModel}


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_registry_key_resolves_to_its_subclass(model_name):
    """Each arch model owns its registry key, and the key picks that class.

    The two subclasses differ only in ``config_cls`` and ``transfer``, which is
    exactly what a wrong row here would silently swap: the model would build
    and train, then load a checkpoint through the other model's rename table.
    """
    assert MODEL_REGISTRY[model_name] is ARCH_SUBCLASSES[model_name]


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_every_arch_subclass_has_a_transfer(model_name):
    """A registered arch model can warm-start, i.e. ``transfer`` is not None.

    ``load_pretrained`` raises ``NotImplementedError`` when the attribute is
    missing, and the failure arrives at warm-start time on a real checkpoint —
    long after the class was registered and the config was written.
    """
    cls = ARCH_SUBCLASSES[model_name]
    assert cls.transfer is not None
    assert cls.config_cls is not DualStreamWanBase.config_cls


def test_gate_does_not_take_over_fastwam():
    """``fastwam`` keeps its own policy class.

    ``@register_model`` decorators stack, so claiming one name too many is a
    one-character mistake that would silently reroute an existing model.
    """
    from loongforge.embodied.model.fastwam.modeling_fastwam import FastWAMPolicy

    assert MODEL_REGISTRY["fastwam"] is FastWAMPolicy


def test_arch_models_take_their_data_packages_preprocessor():
    """Preprocessor lookup is by ``model_type``; this pins whose each model gets.

    internw0_delta consumes FastWAM's pixel-space batch and reads the same four
    config attributes, so a separate collator would be a copy of FastWAM's.
    vpp2 runs master's data package since #371, collator included.
    """
    import loongforge.embodied.data.datasets.fastwam.transforms  # noqa: F401
    import loongforge.embodied.data.datasets.vpp2.transforms  # noqa: F401
    from loongforge.embodied.data.datasets.fastwam.transforms.fastwam_collator \
        import FastWAMPreprocessor
    from loongforge.embodied.data.datasets.transforms.collator import (
        _PREPROCESSOR_REGISTRY)
    from loongforge.embodied.data.datasets.vpp2.transforms.vpp2_collator \
        import VPP2Preprocessor

    assert _PREPROCESSOR_REGISTRY["internw0_delta"] is FastWAMPreprocessor
    assert _PREPROCESSOR_REGISTRY["vpp2"] is VPP2Preprocessor


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_config_exposes_what_the_collator_reads(model_name):
    """The four attributes ``FastWAMPreprocessor`` reads off ``model_cfg``.

    Two of them are properties read back off the built spec rather than fields:
    restating the action horizon or the text width in YAML would let the data
    pipeline and the model disagree by one edit.
    """
    cfg = load_config(model_name)
    assert cfg.action_horizon == cfg.build_spec().geometry.action_horizon
    assert cfg.video_dit_config["text_dim"] == cfg.build_spec().condition("text").width
    assert cfg.tokenizer_max_len > 0
    assert cfg.model_id


# ---- 4. the shell, end to end on stub encoders ----


class StubVAE(torch.nn.Module):
    """Stand-in for the Wan VAE: pixels in, a latent grid of the right shape out.

    The real encoder allocates its mean/std buffers on CUDA at construction
    time, which is why the arch shell takes both encoders by injection.
    """

    def __init__(self, spec: WorldModelSpec) -> None:
        super().__init__()
        self.channels = spec.geometry.latent_channels
        self.grid = spec.geometry.latent_grid

    def encode(self, video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """Return ``[B, C, F, H, W]`` derived from the pixel batch size only."""
        return torch.randn(video.shape[0], self.channels, *self.grid)


class StubImageEncoder(torch.nn.Module):
    """Stand-in for the CLIP image encoder: one frame in, a token grid out."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def encode_image(self, frames: torch.Tensor) -> torch.Tensor:
        """Return ``[B, CONTEXT_LEN, width]`` for a ``[B, 1, 3, H, W]`` input."""
        assert frames.ndim == 5 and frames.shape[1] == 1
        return torch.randn(frames.shape[0], CONTEXT_LEN, self.width)


def make_shell(spec: WorldModelSpec) -> Vpp2ArchModel:
    """A shell around ``spec``, wired to stub encoders."""
    from loongforge.embodied.model.arch.factory import Vpp2Config

    core = WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=spec.build_layout())
    return Vpp2ArchModel(
        Vpp2Config(dtype="float32"), core,
        vae=StubVAE(spec),
        image_encoder=StubImageEncoder(spec.condition("clip_image").width))


@pytest.fixture
def shell() -> Vpp2ArchModel:
    """A vpp2 shell at test width, wired to stub encoders.

    The spec comes from the real provider (so the condition table, visibility
    and loss terms are the production ones) with the widths and geometry shrunk
    — the shell's job is tensor plumbing, which is width-independent.
    """
    return make_shell(tiny_spec("vpp2"))


def pixel_batch(shell: Vpp2ArchModel) -> dict:
    """A FastWAM-style sample dict: pixels, actions, cached text, proprio.

    This is exactly what ``FastWAMPreparedBatch.to_sample()`` produces, which is
    the point — the shell's input contract is the collator's output contract.
    """
    geo = shell.spec.geometry
    text = shell.spec.condition("text")
    return {
        "video": torch.rand(BATCH, 3, geo.num_frames, geo.frame_height,
                            geo.frame_width * geo.num_views) * 2 - 1,
        "action": torch.randn(BATCH, geo.action_horizon, geo.action_dim),
        "context": torch.randn(BATCH, CONTEXT_LEN, text.width),
        "context_mask": torch.ones(BATCH, CONTEXT_LEN, dtype=torch.bool),
        "proprio": torch.randn(BATCH, geo.action_horizon,
                               shell.spec.condition("proprio").source_width),
    }


def test_shell_turns_a_pixel_batch_into_a_scalar_loss(shell):
    """The trainer-facing contract: a pixel batch in, ``(loss, metrics)`` out."""
    loss, metrics = shell(pixel_batch(shell))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for term in shell.spec.diffusion.losses:
        assert metrics[f"loss_{term.name}"].ndim == 0


def test_shell_fills_one_batch_key_per_stream_and_condition(shell):
    """``encode_batch`` is driven by the spec, not by a per-model key list.

    Stream keys come from ``StreamSpec.batch_key`` and condition keys from
    ``ConditionSpec.source_key``, so adding a condition to a provider cannot
    leave the shell behind.
    """
    encoded = shell.encode_batch(pixel_batch(shell))
    for stream in shell.spec.streams:
        assert stream.batch_key in encoded
    for cond in shell.spec.conditions:
        if cond.enabled:
            assert cond.source_key in encoded


def test_shell_passes_the_declared_masks_through(shell):
    """Mask keys come off the spec (``DiffusionSpec.batch_padding``), arrive as
    bool, and stay absent when the sample has none."""
    horizon = shell.spec.geometry.action_horizon
    sample = pixel_batch(shell)
    assert "action_is_pad" not in shell.encode_batch(sample)
    sample["action_is_pad"] = torch.zeros(BATCH, horizon, dtype=torch.uint8)
    encoded = shell.encode_batch(sample)
    assert encoded["action_is_pad"].dtype is torch.bool
    assert "action_terminal_mask" not in encoded


def test_shell_checks_the_latent_grid_against_the_geometry(shell):
    """A VAE that disagrees with the geometry fails here, naming both grids.

    Without this check the mismatch reaches the patch embedding as an
    off-by-one-frame shape error, which reads like a bug in the stream.
    """
    shell.vae.grid = (shell.vae.grid[0] + 1, *shell.vae.grid[1:])
    with pytest.raises(ValueError, match="latent grid"):
        shell(pixel_batch(shell))


def test_i2v_block_is_four_mask_channels_plus_clean_leading_frames(shell):
    """The 20-channel i2v condition: ``mask(4)`` then the clean condition frames.

    The mask is 1 on the conditioned leading frames and the latent half is zero
    everywhere else, which is how "this frame is given" is told apart from "this
    frame happens to be dark".
    """
    cond = shell.spec.condition("cond_latent")
    latents = torch.randn(BATCH, shell.spec.geometry.latent_channels,
                          *shell.spec.geometry.latent_grid)
    block = shell._i2v_channels(cond, latents)
    assert block.shape[1] == cond.width == 4 + latents.shape[1]
    assert torch.all(block[:, :4, :1] == 1.0)
    assert torch.all(block[:, :4, 1:] == 0.0)
    assert torch.equal(block[:, 4:, :1], latents[:, :, :1])
    assert torch.all(block[:, 4:, 1:] == 0.0)


def test_shell_rejects_a_config_of_the_wrong_type():
    """``from_pretrained`` takes the subclass's own ``config_cls``, and says so.

    The gate hands over whatever ``MODEL_SCHEMA`` built, so a wrong schema row
    shows up here; the alternative is an ``AttributeError`` on ``build_spec``.
    """
    with pytest.raises(TypeError, match="Vpp2Config"):
        Vpp2ArchModel.from_pretrained(object())


# ---- 5. the family extension points ----


def test_validate_spec_rejects_a_backend_the_engine_does_not_implement():
    """Backend capability is answered by the engine the family actually runs.

    ``flash`` used to sit in a whitelist next to the spec that no engine
    consulted, so a config naming it passed spec validation and then raised in
    the engine constructor. The check now names the engine and its list.
    """
    spec = dataclasses.replace(tiny_spec("vpp2"), attention_backend="flash")
    with pytest.raises(ValueError, match="JointAttentionEngine"):
        Vpp2ArchModel.validate_spec(spec)


def test_validate_spec_needs_a_family_base_to_bind_an_engine():
    """A model class that skipped the family layer has no engine to ask.

    The abstract base deliberately binds none: which engine runs the streams
    *is* the topology, so a model inheriting it directly has not said what it
    is yet.
    """
    class Orphan(DualStreamWanBase):
        engine_cls = None

    with pytest.raises(NotImplementedError, match="engine_cls"):
        Orphan.validate_spec(tiny_spec("vpp2"))


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_every_provider_declares_the_frozen_towers_it_needs(model_name):
    """A tower is declared next to the key it fills, not inferred by name.

    ``build_towers`` reads these two fields, so a provider that forgets one
    would build a shell with ``vae=None`` and fail at the first batch.
    """
    spec = tiny_spec(model_name)
    video = next(s for s in spec.streams if s.name == "video")
    assert video.source_encoder is not None
    assert video.source_encoder.impl is build_wan_vae
    clip = next((c for c in spec.conditions if c.name == "clip_image"), None)
    if clip is not None:
        assert clip.source_encoder.impl is build_wan_clip


def test_the_shared_config_carries_no_weight_path():
    """Where the frozen towers come from is a family question, not a shared one.

    A VLM-as-a-stream family has no frozen tower at all, so a weight path on
    the shared config would be a field it must leave filled in with a Wan repo
    id it never reads.
    """
    shared = {f.name for f in dataclasses.fields(ArchModelConfig)}
    assert "model_id" not in shared
    assert "compile_vae_encode" not in shared
    assert "model_id" in {f.name for f in
                          dataclasses.fields(DualStreamWanConfig)}


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_provider_kwargs_stops_at_the_family_fields(model_name):
    """Only a *model* config's own fields are provider arguments.

    The exclusion set is the union over every dataclass ancestor rather than
    the shared base alone: with a family config in between, hardcoding the
    base would start forwarding ``model_id`` to a provider whose signature has
    no such parameter.
    """
    cfg = load_config(model_name)
    kwargs = cfg.provider_kwargs()
    assert "model_id" not in kwargs
    assert "compile_vae_encode" not in kwargs
    own = {f.name for f in dataclasses.fields(type(cfg))} - {
        f.name for f in dataclasses.fields(DualStreamWanConfig)}
    assert set(kwargs) <= own


def replace_video_stream(spec: WorldModelSpec, **changes) -> WorldModelSpec:
    """Rewrite the ``video`` stream in place, inside its attention group.

    ``WorldModelSpec.streams`` is a property derived from ``attention_groups``, so a
    stream cannot be swapped by ``dataclasses.replace(spec, streams=...)``.

    Args:
        spec: A built spec.
        changes: Field overrides applied to the ``video`` ``StreamSpec``.

    Returns:
        The same spec with its video stream replaced.
    """
    groups = tuple(
        dataclasses.replace(group, streams=tuple(
            dataclasses.replace(s, **changes) if s.name == "video" else s
            for s in group.streams))
        for group in spec.attention_groups)
    return dataclasses.replace(spec, attention_groups=groups)


def stub_towers(spec: WorldModelSpec, seen: dict) -> WorldModelSpec:
    """Swap both ``source_encoder`` specs for stubs that record their path.

    The real builders import the dreamzero VAE and allocate on CUDA, and the
    ``Spec`` holds a direct function reference, so the substitution has to
    happen in the spec rather than by patching the module.

    Args:
        spec: A built spec.
        seen: Dict the stubs write their received path into.

    Returns:
        The same spec with both towers stubbed.
    """
    def fake_vae(*, spec, path):
        seen["vae"] = path
        return StubVAE(spec)

    def fake_clip(*, path):
        seen["clip"] = path
        return StubImageEncoder(1280)

    conditions = tuple(dataclasses.replace(c, source_encoder=Spec(fake_clip))
                       if c.name == "clip_image" else c
                       for c in spec.conditions)
    return dataclasses.replace(
        replace_video_stream(spec, source_encoder=Spec(fake_vae)),
        conditions=conditions)


def test_build_towers_injects_the_weight_path_the_family_config_carries():
    """The spec says *which* tower; the family config says *where from*.

    Keeping the path out of the spec is what lets the spec layer stay
    allocation free, so this is the seam where the two meet.
    """
    seen: dict = {}
    spec = stub_towers(tiny_spec("vpp2"), seen)
    towers = Vpp2ArchModel.build_towers(Vpp2Config(model_id="/weights"), spec)
    assert seen == {"vae": "/weights", "clip": "/weights"}
    assert isinstance(towers["vae"], StubVAE)
    assert isinstance(towers["image_encoder"], StubImageEncoder)


def test_build_towers_refuses_a_video_stream_with_no_tower():
    """A family that reads pixels cannot reach latents without a VAE.

    Failing in ``build_towers`` names the missing declaration; the alternative
    is ``vae=None`` and an ``AttributeError`` on the first batch.
    """
    spec = replace_video_stream(tiny_spec("vpp2"), source_encoder=None)
    with pytest.raises(ValueError, match="source_encoder"):
        Vpp2ArchModel.build_towers(Vpp2Config(), spec)


def test_trainer_iteration_drives_the_loss_weight_ramp():
    """The trainer's ``iteration`` is the core's ``global_step``.

    ``_select_forward_kwargs`` passes ``iteration`` only to a forward that
    names it, and nothing else writes ``global_step`` into the batch. Without
    it every ``WeightRamp`` would stay at its start value: internw0_delta's
    ``track_align`` (0.0 -> 0.1) and any YAML ``warmup_steps`` would never
    train, so a training forward without it raises.
    """
    spec = tiny_spec("vpp2")
    ramp = WeightRamp(start_step=0, end_step=10, start_value=0.0, end_value=1.0)
    spec = replace(spec, diffusion=replace(spec.diffusion, losses=tuple(
        replace(loss, ramp=ramp) if loss.head == "action_flow" else loss
        for loss in spec.diffusion.losses)))
    shell = make_shell(spec)
    kwargs = FinetuneTrainer._select_forward_kwargs(SimpleNamespace(model=shell), iteration=5)
    assert kwargs == {"iteration": 5}
    _, metrics = shell(pixel_batch(shell), **kwargs)
    assert metrics["action_flow/weight"] == 0.5
    _, metrics = shell(pixel_batch(shell), iteration=10)
    assert metrics["action_flow/weight"] == 1.0
    with pytest.raises(ValueError, match="no global_step"):
        shell(pixel_batch(shell))


def test_batch_global_step_must_agree_with_the_trainer_iteration(shell):
    """Two step counters that disagree raise instead of one winning silently."""
    batch = {**pixel_batch(shell), "global_step": 3}
    shell(batch, iteration=3)
    with pytest.raises(ValueError, match="disagrees"):
        shell(batch, iteration=4)


# ---- 5. the diffusion: section lands on fields the provider declared ----


def load_config_with(model_name: str, *dotlist: str, drop: tuple[str, ...] = ()):
    """``load_config`` plus command-line overrides, as ``train/parser.py`` merges them.

    Args:
        model_name: A ``MODEL_SCHEMA`` key.
        *dotlist: ``key=value`` overrides under ``model:``, as on the command line.
        drop: ``model:`` sections emptied before merging, to build without them.

    Returns:
        The materialised model config instance.
    """
    schema = get_model_schema(model_name)
    section = OmegaConf.load(get_config_path(model_name)).get("model", {})
    for name in drop:
        section[name] = {}
    merged = OmegaConf.merge(OmegaConf.structured(schema.model_config_cls), section,
                             OmegaConf.from_dotlist(list(dotlist)))
    return OmegaConf.to_object(merged)


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_shipped_diffusion_section_restates_the_provider_values(model_name):
    """Writing the section after the provider changes nothing by itself.

    Built with and without its ``diffusion:`` section, each model gets equal
    samplers: the shipped YAML only restates the provider's numbers.
    """
    with_section = load_config_with(model_name).build_spec().diffusion
    without = load_config_with(model_name, drop=("diffusion",)).build_spec().diffusion
    assert dict(with_section.timesteps) == dict(without.timesteps)


def test_yaml_shift_reaches_the_sampler_and_the_bsmntw_weight():
    """``diffusion.action.shift`` moves the sampler, and bsmntw follows it.

    Bsmntw normalizes on the shifted training grid and reads the shift off the
    sampler when it is called, so no constant computed from the old shift is
    left behind. An int in YAML lands as a float.
    """
    base = load_config("vpp2").build_spec().diffusion
    moved = load_config_with("vpp2", "diffusion.action.shift=3").build_spec().diffusion
    sampler = moved.sampler_for("action")
    assert sampler.distribution == UniformShift(shift=3.0)
    assert type(sampler.distribution.shift) is float
    (loss,) = [loss for loss in moved.losses if loss.head == "action_flow"]
    t = torch.tensor([500.0])
    before = loss.weight_of(t, sampler=base.sampler_for("action"), num_train_timesteps=1000)
    after = loss.weight_of(t, sampler=sampler, num_train_timesteps=1000)
    assert not torch.equal(before, after)


@pytest.mark.parametrize("override, error, message", [
    ("diffusion.video.shift=3.0", ValueError, "diffusion.video.shift: stream 'video' has no diffusion keys"),
    ("diffusion.action.num_layers=2", ValueError,
     "diffusion.action.num_layers is not a diffusion key of stream 'action'"),
    ("diffusion.action.weight=0.5", ValueError,
     "diffusion.action.weight is not a diffusion key of stream 'action'"),
    ("diffusion.proprio.shift=1.0", ValueError, "diffusion.proprio.shift: stream 'proprio' has no diffusion keys"),
    ("diffusion.action.shift=true", TypeError, "diffusion.action.shift must be float, got bool"),
    ("diffusion.action.shift=0", ValueError,
     "UniformShift.shift must be positive, got 0.0"),
    ("streams.action.shift=3.0", ValueError, "does not consume ['action_shift']"),
])
def test_a_diffusion_key_that_lands_nowhere_fails_at_build(override, error, message):
    """Wrong section, wrong stream, wrong type or out of range: each one raises.

    The two sections used to be flattened into one keyword table, so
    ``streams.action.shift`` silently moved the action timesteps and a stream
    size written under ``diffusion:`` silently changed the stream.
    """
    with pytest.raises(error, match=re.escape(message)):
        load_config_with("vpp2", override).build_spec()


@dataclass(frozen=True)
class _CappedShift:
    """Stand-in for a distribution a model writes in its own directory."""

    shift: float = 3.0
    t_max: float = 0.95

    def sample(self, shape, *, device, dtype):
        u = torch.rand(shape, device=device, dtype=dtype)
        return (self.shift * u / (1.0 + (self.shift - 1.0) * u)).clamp(max=self.t_max)


def test_a_model_owned_distribution_gets_its_keys_without_common_code():
    """A new distribution needs no marker: its provider lists the keys for ``resolve_diffusion``.

    YAML overrides the provider's value before the one construction, and the
    provider's table is left as it was.
    """
    defaults = {"video": {"shift": 2.0, "t_max": 0.95}}
    values = resolve_diffusion("toy", defaults, {"video": {"t_max": 0.9}})
    assert _CappedShift(**values["video"]) == _CappedShift(shift=2.0, t_max=0.9)
    assert defaults == {"video": {"shift": 2.0, "t_max": 0.95}}
    assert resolve_diffusion("toy", defaults, None) == defaults


def test_a_key_type_follows_its_default():
    """The default's type is the key's type: an int key refuses a float, a float key takes an int."""
    defaults = {"action": {"num_steps": 10, "shift": 5.0}}
    with pytest.raises(TypeError, match=re.escape("diffusion.action.num_steps must be int, got float 4.0")):
        resolve_diffusion("toy", defaults, {"action": {"num_steps": 4.0}})
    assert resolve_diffusion("toy", defaults, {"action": {"shift": 3}})["action"]["shift"] == 3.0
    with pytest.raises(TypeError, match=re.escape("default for diffusion.action.other: a diffusion key is float")):
        resolve_diffusion("toy", {"action": {"other": None}}, None)


def test_diffusion_blocks_reject_unknown_empty_streams_and_non_mappings():
    """Stream-block shape and empty-block rules match ``streams:`` and ``heads:``.

    An empty block on a valid stream keeps its defaults, whereas an empty block
    on an unknown or key-free stream still raises so ``diffusion: {actoin: {}}``
    cannot pass silently. ``None`` and lists are rejected as non-mappings, and
    unknown or legal key lists are sorted deterministically.
    """
    defaults = {"video": {"t_max": 0.95, "shift": 2.0}, "action": {"shift": 5.0}}
    assert resolve_diffusion("toy", defaults, {"action": {}}) == defaults
    assert load_config_with("vpp2", "diffusion.action={}").build_spec().diffusion.sampler_for(
        "action").distribution == UniformShift(shift=5.0)
    with pytest.raises(ValueError, match=re.escape("diffusion.actoin: stream 'actoin' has no diffusion keys")):
        resolve_diffusion("toy", defaults, {"actoin": {}})
    with pytest.raises(ValueError, match=re.escape("diffusion.video: stream 'video' has no diffusion keys")):
        load_config_with("vpp2", "diffusion.video={}").build_spec()
    for bad in (None, [1.0], True):
        with pytest.raises(TypeError, match=re.escape(f"diffusion.action must be a mapping, got {type(bad).__name__}")):
            resolve_diffusion("toy", defaults, {"action": bad})
    with pytest.raises(TypeError, match=re.escape("diffusion must be a mapping, got list")):
        resolve_diffusion("toy", defaults, ["action"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=re.escape(
            "diffusion.video got unknown keys ['alpha', 'zeta'] of stream 'video'; "
            "legal diffusion keys are ['shift', 't_max']")):
        resolve_diffusion("toy", defaults, {"video": {"zeta": 1.0, "alpha": 2.0}})
