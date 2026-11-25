# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Arch fastwam L3 versus the in-tree legacy ``FastWAM.training_loss`` (Lane C v2 T39).

The oracle is the legacy method itself, compiled verbatim from
``model/fastwam/mot/fastwam.py`` together with ``_compute_video_loss_per_sample``,
on the legacy scheduler (``action/schedulers.py``) built from the legacy
config's numbers. Only the network between noising and the loss is replaced:
``build_inputs`` returns the case's latents and masks, both experts' ``pre_dit``
record what L3 hands them, and ``post_dit`` returns one injected prediction per
stream. These tests therefore pin L3 (draw order, t quantization, the clean
first frame, targets, padding, bsmntw weights and both reductions), not L0-L2,
and need no external checkout.
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import fields
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from torch.nn import functional as F

from loongforge.embodied.model.arch.builder import WorldModel
from loongforge.embodied.model.arch.diffusion.loss import Bsmntw
from loongforge.embodied.model.arch.diffusion.timesteps import TimestepSampler, UniformShift
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.fastwam.action.schedulers import WanContinuousFlowMatchScheduler
from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig

from test_arch_blocks import tiny_spec

LEGACY = Path(__file__).resolve().parents[3] / "loongforge/embodied/model/fastwam/mot/fastwam.py"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
PADDINGS = ("no-pad", "action-pad", "video-pad", "both-pad", "all-pad")
CASES = list(itertools.product(DTYPES, PADDINGS))


def _legacy_config() -> dict[str, dict[str, Any]]:
    """The legacy defaults: both schedulers and the loss weights (``:108-118``)."""
    return {field.name: field.default_factory() for field in fields(FastWAMModelConfig)
            if field.name in ("video_scheduler", "action_scheduler", "loss")}


def _methods(*names: str) -> dict:
    """Compile the named ``FastWAM`` methods verbatim."""
    tree = ast.parse(LEGACY.read_text(), filename=str(LEGACY))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastWAM")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names)
    scope = {"torch": torch, "F": F, "Optional": Optional, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LEGACY), "exec"), scope)
    return {name: scope[name] for name in names}


def _spec():
    """The real fastwam provider, shrunk: four latent frames, the first clean."""
    return tiny_spec("fastwam")


def _case_inputs(spec, case, device):
    """Clean latents, actions, both predictions and the case's masks, from a fixed seed."""
    dtype, padding = DTYPES[case[0]], case[1]
    geo = spec.geometry
    assert (geo.num_frames, geo.latent_grid[0]) == (13, 4)
    torch.manual_seed(13)
    latents = torch.randn(2, geo.latent_channels, *geo.latent_grid).to(device, dtype)
    action = torch.randn(2, geo.action_horizon, geo.action_dim).to(device, dtype)
    pred_video = torch.randn_like(latents)
    pred_action = torch.randn_like(action)
    masks = {}
    if padding in ("action-pad", "both-pad"):
        masks["action_is_pad"] = torch.tensor([[False, False, True, True], [False] * 4])
    if padding in ("video-pad", "both-pad"):
        # 13 pixel frames; the last 6 pad latent frame 3 only (frames 9-12)
        masks["image_is_pad"] = torch.tensor([[False] * 7 + [True] * 6, [False] * 13])
    if padding == "all-pad":
        masks["action_is_pad"] = torch.ones(2, geo.action_horizon, dtype=torch.bool)
        masks["image_is_pad"] = torch.tensor([[False] + [True] * 12] * 2)
    masks = {key: value.to(device) for key, value in masks.items()}
    return latents, action, pred_video, pred_action, masks


def run_legacy(case, device) -> dict:
    """Legacy ``training_loss`` on one case: what it hands the network, and its losses."""
    spec = _spec()
    latents, action, pred_video, pred_action, masks = _case_inputs(spec, case, device)
    config, seen = _legacy_config(), {}
    shell = SimpleNamespace(
        device=device, vae=SimpleNamespace(temporal_downsample_factor=4),
        loss_lambda_video=config["loss"]["lambda_video"],
        loss_lambda_action=config["loss"]["lambda_action"])
    for stream in ("video", "action"):
        numbers = config[f"{stream}_scheduler"]
        scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=numbers["num_train_timesteps"], shift=numbers["train_shift"])
        target = scheduler.training_target
        scheduler.training_target = (lambda *args, _s=stream, _t=target:
                                     seen.setdefault(f"{_s} target", _t(*args)))
        setattr(shell, f"train_{stream}_scheduler", scheduler)
    for name, function in _methods("training_loss", "_compute_video_loss_per_sample").items():
        setattr(shell, name, MethodType(function, shell))
    shell.build_inputs = lambda sample, tiled=False: dict(
        input_latents=latents, context=None, context_mask=None, action=action,
        action_is_pad=masks.get("action_is_pad"), image_is_pad=masks.get("image_is_pad"),
        first_frame_latents=latents[:, :, 0:1], fuse_vae_embedding_in_latents=True)

    def pre_dit(stream, tokens_key):
        def seam(**kwargs):
            seen[stream] = kwargs
            return {"tokens": kwargs[tokens_key], "meta": {"tokens_per_frame": 1},
                    "freqs": None, "context": None, "context_mask": None, "t_mod": None}
        return seam

    shell.video_expert = SimpleNamespace(pre_dit=pre_dit("video", "x"),
                                         post_dit=lambda tokens, pre: pred_video)
    shell.action_expert = SimpleNamespace(pre_dit=pre_dit("action", "action_tokens"),
                                          post_dit=lambda tokens, pre: pred_action)
    shell._build_mot_attention_mask = lambda **kwargs: None
    shell.mot = lambda **kwargs: {"video": None, "action": None}
    torch.manual_seed(2026)
    loss, logs = shell.training_loss({})
    return {"video x_t": seen["video"]["x"], "video t": seen["video"]["timestep"],
            "action x_t": seen["action"]["action_tokens"], "action t": seen["action"]["timestep"],
            "video target": seen["video target"], "action target": seen["action target"],
            "total loss": loss,
            "loss_video": torch.tensor(logs["loss_video"], dtype=torch.float64),
            "loss_action": torch.tensor(logs["loss_action"], dtype=torch.float64),
            "rng next": torch.rand(4, device=device)}


def run_arch(case, device) -> dict:
    """The generic path on the same inputs: draw, add_noise and compute_losses."""
    spec = _spec()
    latents, action, pred_video, pred_action, masks = _case_inputs(spec, case, device)
    model = WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                       layout=spec.build_layout())
    batch = dict(video=latents, action=action, **masks)
    torch.manual_seed(2026)
    runtime = model.add_noise(batch, model.draw(batch))
    rng_next = torch.rand(4, device=device)
    loss, metrics = model.compute_losses(
        None, runtime, {"video_flow": pred_video, "action_flow": pred_action}, step=0)
    return {"video x_t": runtime.noised["video"], "video t": runtime.model_timesteps["video"],
            "action x_t": runtime.noised["action"], "action t": runtime.model_timesteps["action"],
            "video target": runtime.targets["video"], "action target": runtime.targets["action"],
            "total loss": loss,
            "loss_video": torch.tensor(metrics["loss_video_flow"].item(), dtype=torch.float64),
            "loss_action": torch.tensor(metrics["loss_action_flow"].item(), dtype=torch.float64),
            "rng next": rng_next}


def _exact(label, actual, expected):
    """Zero tolerance; the failure message carries the size of the miss."""
    assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), (
        label, actual.dtype, expected.dtype, tuple(actual.shape), tuple(expected.shape))
    if not torch.equal(actual, expected):
        delta = (actual.double() - expected.double()).abs().max().item()
        pytest.fail(f"{label}: not exact, max_abs={delta:.3g}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: "-".join(case))
@pytest.mark.parametrize("device", DEVICES)
def test_fastwam_l3_matches_the_legacy_training_loss(device, case):
    """Same seed, same inputs: every draw, x_t, target and both losses are bit-equal."""
    device = torch.device(device)
    expected = run_legacy(case, device)
    actual = run_arch(case, device)
    assert actual.keys() == expected.keys()
    for label, value in expected.items():
        _exact(label, actual[label], value)


def test_the_cases_exercise_the_masks():
    """Guard against a vacuous harness: the padded cases move both terms."""
    cpu = torch.device("cpu")
    plain = run_legacy(("fp32", "no-pad"), cpu)
    both = run_legacy(("fp32", "both-pad"), cpu)
    assert not torch.equal(plain["loss_video"], both["loss_video"])
    assert not torch.equal(plain["loss_action"], both["loss_action"])
    assert run_legacy(("bf16", "all-pad"), cpu)["loss_action"].item() == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stream", ["video", "action"])
def test_bsmntw_matches_the_legacy_training_weight(stream, dtype):
    numbers = _legacy_config()[f"{stream}_scheduler"]
    legacy = WanContinuousFlowMatchScheduler(
        num_train_timesteps=numbers["num_train_timesteps"], shift=numbers["train_shift"])
    sampler = TimestepSampler(UniformShift(shift=numbers["train_shift"]))
    torch.manual_seed(7)
    t = torch.cat([torch.tensor([0.0, 1.0, 499.0, 500.0, 999.0, 1000.0]),
                   torch.rand(64) * 1000]).to(DTYPES[dtype])
    _exact("bsmntw", Bsmntw()(t, sampler=sampler, num_train_timesteps=1000),
           legacy.training_weight(t))
