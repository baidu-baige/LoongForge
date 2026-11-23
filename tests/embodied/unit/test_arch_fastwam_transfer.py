# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tensor parity with MASTER's WanVideoDiT/ActionDiT registration tree.

The source fixture builds MASTER-shaped modules independently of the rename
table, including FastWAMPolicy's shared expert registrations. It does not import
MASTER's GPU-only TransformerEngine/flash-attention dependencies or run kernels.
Initializer files are serialized in memory: tests create no checkpoint artifacts.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import loongforge.embodied.model.arch.transfer as transfer
from test_arch_blocks import tiny_spec
from test_arch_world_model import world


def _attention(width, attn_width, *, cross=False):
    attention = nn.Module()
    for name in ("q", "k", "v"):
        setattr(attention, name, nn.Linear(width, attn_width))
    attention.o = nn.Linear(attn_width, width)
    attention.norm_q = nn.Module()
    attention.norm_q.weight = nn.Parameter(torch.ones(attn_width))
    attention.norm_k = nn.Module()
    attention.norm_k.weight = nn.Parameter(torch.ones(attn_width))
    return attention


def _expert(core, name):
    """MASTER constructor tensor names/shapes, not an inverse rename table."""
    spec = core.blocks[name].spec
    width = spec.residual_width
    text_width = core.spec.condition("text").width
    expert = nn.Module()
    expert.text_embedding = nn.Sequential(
        nn.Linear(text_width, width), nn.GELU(), nn.Linear(width, width))
    expert.time_embedding = nn.Sequential(
        nn.Linear(spec.timestep_embed_width, width), nn.SiLU(),
        nn.Linear(width, width))
    expert.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
    expert.blocks = nn.ModuleList()
    for _ in range(spec.num_layers):
        block = nn.Module()
        attn_width = spec.num_heads * spec.head_dim
        block.self_attn = _attention(width, attn_width)
        block.cross_attn = _attention(width, attn_width, cross=True)
        block.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        block.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        block.norm3 = nn.LayerNorm(width)
        block.ffn = nn.Sequential(
            nn.Linear(width, spec.mlp_width), nn.GELU(),
            nn.Linear(spec.mlp_width, width))
        block.modulation = nn.Parameter(torch.randn(1, 6, width))
        expert.blocks.append(block)
    if name == "video":
        geometry = core.spec.geometry
        expert.patch_embedding = nn.Conv3d(
            geometry.latent_channels + core.spec.concat_channels(name), width,
            geometry.patch_size, stride=geometry.patch_size)
        expert.head = nn.Module()
        expert.head.modulation = nn.Parameter(torch.randn(1, 2, width))
        expert.head.head = nn.Linear(width, core.heads["video_flow"].head.out_features)
    else:
        expert.action_encoder = nn.Linear(core.spec.geometry.action_dim, width)
        expert.head = nn.Linear(width, core.spec.geometry.action_dim)
    return expert


@pytest.fixture
def pair():
    core = world(tiny_spec("fastwam"))
    master = nn.Module()
    master.video_expert = _expert(core, "video")
    master.action_expert = _expert(core, "action")
    master.mot = nn.Module()
    master.mot.mixtures = nn.ModuleDict({
        "video": master.video_expert, "action": master.action_expert})
    master.dit = master.mot
    master.proprio_encoder = nn.Linear(
        core.spec.condition("proprio").source_width,
        core.spec.condition("text").width)
    # Make every source value distinguishable from initial target values.
    with torch.no_grad():
        for i, parameter in enumerate(master.parameters()):
            parameter.fill_(0.125 + i / 1000)
    return core, master


def _saved(master):
    return {"mot": master.mot.state_dict(),
            "proprio_encoder": master.proprio_encoder.state_dict(),
            "step": 10, "optimizer": {}}


def _initializers(core, master):
    stream = core.blocks["action"]
    spec = stream.spec
    return master.video_expert.state_dict(), {
        "meta": {
            "hidden_dim": spec.residual_width, "ffn_dim": spec.mlp_width,
            "num_layers": spec.num_layers, "num_heads": spec.num_heads,
            "attn_head_dim": spec.head_dim,
            "text_dim": core.spec.condition("text").width,
            "freq_dim": spec.timestep_embed_width,
            "eps": stream.blocks[0].norm1.eps,
        },
        "backbone_state_dict": {
            k: v for k, v in master.action_expert.state_dict().items()
            if not k.startswith(("action_encoder.", "head."))},
        "policy": {"skip_prefixes": ["action_encoder.", "head."],
                   "projection_initialization": "random"},
    }


def _files(monkeypatch, video, action, *, envelope=False):
    """Exercise actual torch checkpoint reader with in-memory serialized files."""
    blobs = {}
    for name, payload in (("video.pt", {"dit": video} if envelope else video),
                          ("action.pt", action)):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        blobs[name] = buffer.getvalue()
    load = torch.load

    def read(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return load(io.BytesIO(blobs[Path(path).name]), **kwargs)

    monkeypatch.setattr(Path, "is_file", lambda path: path.name in blobs)
    monkeypatch.setattr(torch, "load", read)
    return "/video.pt", "/action.pt"


def _assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize("form", ["saved", "bare", "mot", "dit", "core", "policy", "envelope"])
def test_complete_tensor_map_and_strict_state_parity(pair, form):
    core, master = pair
    payload = _saved(master)
    expected = transfer.fastwam_to_arch(payload)
    if form == "bare":
        payload = master.mot.state_dict()
        expected = {k: v for k, v in expected.items()
                    if not k.startswith("condition_encoders.proprio.")}
    elif form in ("mot", "dit"):
        payload = {f"{form}.{k}": v for k, v in master.mot.state_dict().items()}
        payload.update({f"proprio_encoder.{k}": v
                        for k, v in master.proprio_encoder.state_dict().items()})
    elif form == "core":
        payload = master.state_dict()
    elif form in ("policy", "envelope"):
        wrapper = nn.Module()
        wrapper.core = master
        wrapper.dit = master.dit
        payload = wrapper.state_dict()
        if form == "envelope":
            payload = {"state_dict": payload}
    mapped = transfer.fastwam_to_arch(payload)
    _assert_same(mapped, expected)
    if form == "bare":
        with pytest.raises(RuntimeError, match="Missing key"):
            core.load_state_dict(mapped, strict=True)
    else:
        assert mapped.keys() == core.state_dict().keys()
        report = core.load_state_dict(mapped, strict=True)
        assert not report.missing_keys and not report.unexpected_keys
        _assert_same(core.state_dict(), mapped)
        # Read representative cross-layer/slot/attention-width moves directly.
        assert torch.equal(mapped["heads.video_flow.head.weight"],
                           master.video_expert.head.head.weight)
        assert torch.equal(mapped["blocks.action.blocks.0.cross_attn.kv.text.k.weight"],
                           master.action_expert.blocks[0].cross_attn.k.weight)
        assert mapped["blocks.video.blocks.0.self_attn.q.weight"].shape == (16, 32)


@pytest.mark.parametrize("change", ["value", "shape", "dtype"])
def test_shared_alias_conflicts_fail_loud(pair, change):
    _, master = pair
    payload = dict(master.state_dict())
    key = "dit.mixtures.video.blocks.0.modulation"
    value = payload[key]
    payload[key] = {"value": value + 1, "shape": value[..., :1],
                    "dtype": value.double()}[change]
    with pytest.raises(ValueError, match="alias collision"):
        transfer.fastwam_to_arch(payload)


def test_equal_alias_keeps_first_tensor():
    value = torch.ones(2)
    mapped = transfer.fastwam_to_arch({
        "video_expert.head.modulation": value,
        "core.mot.mixtures.video.head.modulation": value.clone()})
    assert mapped["heads.video_flow.modulation"] is value
    # Generic remapping retains its original collision behavior (VPP2 unchanged).
    with pytest.raises(ValueError, match="remap collision"):
        transfer.remap_state_dict(
            {"a.w": value, "b.w": value}, ((r"^[ab]\.", "z."),))


@pytest.mark.parametrize("key", [
    "mixtures.third.weight", "mixtures.video.something_new.weight",
    "mixtures.action.blocks.0.ffn.9.weight", "core.vae.encoder.weight",
    "mixtures.video.blocks.0.self_attn.norm_q._extra_state"])
def test_unknown_keys_are_not_silently_dropped(key):
    with pytest.raises(KeyError, match="match no rule"):
        transfer.fastwam_to_arch({key: torch.ones(1)})


def test_explicit_auxiliary_drop_and_stale_drop():
    payload = {"core.vae.encoder.weight": torch.ones(1),
               "core.video_expert.head.modulation": torch.ones(1)}
    assert set(transfer.fastwam_to_arch(payload, drop=("core.vae.",))) == {
        "heads.video_flow.modulation"}
    with pytest.raises(ValueError, match="matched no key"):
        transfer.fastwam_to_arch(payload, drop=("core.text_encoder.",))


def test_full_transfer_shape_mismatch_is_strict(pair):
    core, master = pair
    payload = _saved(master)
    payload["mot"]["mixtures.video.patch_embedding.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="size mismatch"):
        core.load_state_dict(transfer.fastwam_to_arch(payload), strict=True)


@pytest.mark.parametrize("envelope", [False, True])
def test_split_initializer_values_and_random_exclusions(pair, monkeypatch, envelope):
    core, master = pair
    before = {k: v.clone() for k, v in core.state_dict().items()}
    identities = {k: id(v) for k, v in core.named_parameters()}
    video, action = _initializers(core, master)
    paths = _files(monkeypatch, video, action, envelope=envelope)
    report = transfer.load_fastwam_initializers(core, *paths)
    assert report["video_tensors"] == len(video)
    assert report["action_backbone_tensors"] == len(action["backbone_state_dict"])
    assert report["validation_only"] is False
    random = {
        "blocks.action.action_encoder.weight", "blocks.action.action_encoder.bias",
        "heads.action_flow.head.weight", "heads.action_flow.head.bias",
        "condition_encoders.proprio.weight", "condition_encoders.proprio.bias"}
    assert set(report["random_keys"]) == random
    wanted = transfer.fastwam_to_arch(_saved(master))
    for key, value in core.state_dict().items():
        assert torch.equal(value, before[key] if key in random else wanted[key]), key
    assert identities == {k: id(v) for k, v in core.named_parameters()}


@pytest.mark.parametrize("failure", [
    "video_shape", "action_shape", "missing", "unknown", "excluded_input",
    "excluded_output", "meta", "nonfinite", "policy", "payload", "nontensor"])
def test_initializer_errors_precede_any_mutation(pair, monkeypatch, failure):
    core, master = pair
    before = {k: v.clone() for k, v in core.state_dict().items()}
    video, action = _initializers(core, master)
    backbone = action["backbone_state_dict"]
    error, match = RuntimeError, "Strict FastWAM"
    if failure == "video_shape":
        video["patch_embedding.weight"] = torch.zeros(1)
    elif failure == "action_shape":
        backbone["blocks.0.self_attn.q.weight"] = torch.zeros(1)
    elif failure == "missing":
        backbone.pop("blocks.0.modulation")
    elif failure == "unknown":
        backbone["mystery.weight"] = torch.ones(1)
        error, match = KeyError, "match no rule"
    elif failure.startswith("excluded"):
        key = "action_encoder.weight" if failure == "excluded_input" else "head.weight"
        backbone[key] = master.action_expert.state_dict()[key]
    elif failure in ("meta", "nonfinite"):
        action["meta"]["eps"] = 0.1 if failure == "meta" else float("nan")
        error, match = ValueError, "meta.eps"
    elif failure == "policy":
        action["policy"]["skip_prefixes"].append("video_kv_projections.")
        error, match = ValueError, "policy"
    elif failure == "payload":
        action["mystery"] = torch.ones(1)
        error, match = KeyError, "payload keys"
    else:
        backbone["blocks.0.modulation"] = "not a tensor"
        error, match = TypeError, "named tensor"
    paths = _files(monkeypatch, video, action)
    with pytest.raises(error, match=match):
        transfer.load_fastwam_initializers(core, *paths)
    _assert_same(core.state_dict(), before)


def test_meta_validation_does_not_materialize(pair, monkeypatch):
    core, master = pair
    video, action = _initializers(core, master)
    paths = _files(monkeypatch, video, action)
    core.to(device="meta")
    report = transfer.load_fastwam_initializers(core, *paths)
    assert report["validation_only"]
    assert all(v.is_meta for v in core.state_dict().values())


def test_mixed_meta_state_rejected(pair):
    core, _ = pair
    core.heads["action_flow"].to(device="meta")
    with pytest.raises(ValueError, match="entirely meta or materialized"):
        transfer.load_fastwam_initializers(core, "/video.pt", "/action.pt")


@pytest.mark.parametrize("paths", [(None, None), ("/video.pt", None), (None, "/action.pt")])
def test_explicit_both_paths_required(pair, paths):
    core, _ = pair
    with pytest.raises(ValueError, match="explicit video and action"):
        transfer.load_fastwam_initializers(core, *paths)


def test_remote_model_ids_are_explicitly_unsupported(pair, monkeypatch):
    core, _ = pair
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    with pytest.raises(FileNotFoundError, match="Remote model_id is not resolved"):
        transfer.load_fastwam_initializers(core, "Wan-AI/Wan2.2-TI2V-5B", "/action.pt")


@pytest.mark.parametrize("missing", ["video", "action"])
def test_initializer_rejects_missing_checkpoint_file(pair, monkeypatch, missing):
    core, master = pair
    before = {k: v.clone() for k, v in core.state_dict().items()}
    paths = list(_files(monkeypatch, *_initializers(core, master)))
    paths[0 if missing == "video" else 1] = f"/missing-{missing}.pt"
    monkeypatch.setattr(Path, "is_dir", lambda _: False)
    field = "dit" if missing == "video" else "backbone_state_dict"
    with pytest.raises(FileNotFoundError, match=(
            f"FastWAM {field} requires an explicit checkpoint FILE: "
            f"/missing-{missing}\\.pt")):
        transfer.load_fastwam_initializers(core, *paths)
    _assert_same(core.state_dict(), before)


@pytest.mark.parametrize("index_file", [False, True])
def test_initializer_rejects_missing_wan_index(pair, monkeypatch, index_file):
    core, _ = pair
    before = {k: v.clone() for k, v in core.state_dict().items()}
    index = Path("/wan/diffusion_pytorch_model.safetensors.index.json")
    monkeypatch.setattr(Path, "is_dir", lambda path: path == index.parent)
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    with pytest.raises(FileNotFoundError, match="Wan shard index not found"):
        transfer.load_fastwam_initializers(
            core, index if index_file else index.parent, "/action.pt")
    _assert_same(core.state_dict(), before)


def test_optional_policy_is_not_required(pair, monkeypatch):
    core, master = pair
    video, action = _initializers(core, master)
    action.pop("policy")
    paths = _files(monkeypatch, video, action)
    assert not transfer.load_fastwam_initializers(core, *paths)["validation_only"]


def _shards(monkeypatch, video, *, failure=None):
    """Real safetensors bytes and HF index, with no filesystem artifacts."""
    import safetensors.torch as safe

    names = ("diffusion_pytorch_model-00001-of-00002.safetensors",
             "diffusion_pytorch_model-00002-of-00002.safetensors")
    tensors = [{}, {}]
    weight_map = {}
    for i, (key, value) in enumerate(video.items()):
        tensors[i % 2][key] = value
        weight_map[key] = names[i % 2]
    key = next(iter(tensors[0]))
    if failure == "duplicate":
        tensors[1][key] = tensors[0][key]
    elif failure == "placement":
        weight_map[key] = names[1]
    elif failure == "missing_key":
        tensors[0].pop(key)
    elif failure == "shape":
        tensors[0][key] = torch.zeros(1)
    elif failure == "unknown":
        tensors[0]["unknown.weight"] = torch.zeros(1)
        weight_map["unknown.weight"] = names[0]
    elif failure == "unsafe_path":
        weight_map[key] = "../outside.safetensors"
    elif failure == "invalid_index":
        weight_map = {}
    blobs = {name: safe.save(state) for name, state in zip(names, tensors)}
    if failure == "missing_shard":
        blobs.pop(names[0])
    index_path = Path("/wan/diffusion_pytorch_model.safetensors.index.json")
    original_is_file = Path.is_file
    original_open = Path.open
    monkeypatch.setattr(Path, "is_dir", lambda path: str(path) == "/wan")
    monkeypatch.setattr(Path, "is_file", lambda path:
                        path == index_path or
                        (path.parent == Path("/wan") and path.name in blobs) or
                        original_is_file(path))

    def open_index(path, *args, **kwargs):
        if path == index_path:
            return io.StringIO(json.dumps({"weight_map": weight_map}))
        return original_open(path, *args, **kwargs)

    opened = set()

    def load_shard(path, *, device):
        assert device == "cpu"
        name = Path(path).name
        assert name not in opened, "Each indexed shard must be loaded only once"
        opened.add(name)
        return safe.load(blobs[name])

    monkeypatch.setattr(Path, "open", open_index)
    monkeypatch.setattr(safe, "load_file", load_shard)
    return index_path


@pytest.mark.parametrize("index_file", [False, True])
def test_local_indexed_wan_shards_load_complete_video(pair, monkeypatch, index_file):
    core, master = pair
    video, action = _initializers(core, master)
    before = {k: v.clone() for k, v in core.state_dict().items()}
    _, action_path = _files(monkeypatch, video, action)
    index = _shards(monkeypatch, video)
    report = transfer.load_fastwam_initializers(
        core, index if index_file else index.parent, action_path)
    assert report["video_tensors"] == len(video)
    expected = transfer.fastwam_to_arch(_saved(master))
    for key, value in core.state_dict().items():
        assert torch.equal(value, before[key] if key in report["random_keys"]
                           else expected[key]), key


@pytest.mark.parametrize("failure,error,match", [
    ("duplicate", ValueError, "Duplicate FastWAM Wan shard keys"),
    ("placement", ValueError, "Wan shard/index mismatch"),
    ("missing_key", ValueError, "Wan shard/index missing keys"),
    ("missing_shard", FileNotFoundError, "index references missing shard"),
    ("shape", RuntimeError, "shape_mismatches"),
    ("unknown", KeyError, "match no rule"),
    ("unsafe_path", ValueError, "Invalid FastWAM Wan index entry"),
    ("invalid_index", ValueError, "index needs a nonempty weight_map"),
])
def test_invalid_wan_shards_fail_before_mutation(pair, monkeypatch, failure, error, match):
    core, master = pair
    video, action = _initializers(core, master)
    before = {k: v.clone() for k, v in core.state_dict().items()}
    _, action_path = _files(monkeypatch, video, action)
    index = _shards(monkeypatch, video, failure=failure)
    with pytest.raises(error, match=match):
        transfer.load_fastwam_initializers(core, index.parent, action_path)
    _assert_same(core.state_dict(), before)


def test_optional_proprio_absent_from_full_and_split_load(monkeypatch):
    spec = tiny_spec("fastwam")
    spec = replace(spec, conditions=tuple(c for c in spec.conditions
                                          if c.name != "proprio"))
    core = world(spec)
    master = nn.Module()
    master.video_expert = _expert(core, "video")
    master.action_expert = _expert(core, "action")
    master.mot = nn.Module()
    master.mot.mixtures = nn.ModuleDict({
        "video": master.video_expert, "action": master.action_expert})
    mapped = transfer.fastwam_to_arch(master.state_dict())
    assert mapped.keys() == core.state_dict().keys()
    core.load_state_dict(mapped, strict=True)
    paths = _files(monkeypatch, *_initializers(core, master))
    report = transfer.load_fastwam_initializers(core, *paths)
    assert len(report["random_keys"]) == 4
    assert not any(k.startswith("condition_encoders.proprio.")
                   for k in report["random_keys"])
