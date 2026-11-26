# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The generic arch ``WorldModel`` on the vpp2 provider versus actual master
``model/vpp2`` code, action stage (L3/L4; Lane C v2 T39).

The oracle runs master's own source unmodified: the scheduler module, the
``LiberoActionModel`` ``build_inputs`` / ``training_loss`` /
``_training_loss_action_only`` bodies with the three ``VPP2`` helpers they
call, and the video ``Head`` / ``post_dit`` / ``unpatchify`` plus the action
``post_dit``. Only the DiT forward between noising and the loss is replaced by
one injected prediction, so these tests pin L3 (draw order, t quantization,
noising, targets, padding, timestep weights, reduction) and L4 (heads), not
L0-L2.

There is no VPP2 recipe here: master's action stage trains the action term
only, which the generic path expresses as the ``video_flow`` term switched off
(YAML ``heads.video_flow.enabled: false``); the padding comes from the
provider's ``DiffusionSpec.batch_padding``. Freezing the video stream is not
an L3 matter and is not compared.

The live parity tests need the sibling ``LoongForge_master_vpp2`` checkout and
skip without it, so CI does not depend on an external tree. The last test
checks that the L3 steps do not sync the host.
"""

import ast
import importlib.util
import itertools
import math
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Optional

import pytest
import torch
from torch.nn import functional as F

from loongforge.embodied.model.arch.builder import WorldModel
from loongforge.embodied.model.arch.diffusion.loss import Bsmntw
from loongforge.embodied.model.arch.diffusion.timesteps import TimestepSampler, UniformShift
from loongforge.embodied.model.arch.spec.core import get_provider
import loongforge.embodied.model.arch.spec.providers  # noqa: F401 -- register providers
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.arch.factory import vpp2_to_arch


REPO = Path(__file__).resolve().parents[3]
MASTER = REPO.parent / "LoongForge_master_vpp2"
LEGACY = MASTER / "loongforge/embodied/model/vpp2/modules"
BATCH, HORIZON, ACTION_DIM, TEXT_LEN, TEXT_DIM = 2, 4, 6, 3, 8
# Joint attention needs equal head counts, head widths and depths (V2/V3).
SCALARS = dict(
    video_residual_width=32, video_num_heads=2, video_head_dim=8,
    video_num_layers=2, video_ffn_width=32,
    action_residual_width=16, action_num_heads=2, action_head_dim=8,
    action_num_layers=2, action_ffn_width=16,
)
# CUDA draws use a different generator than CPU ones, so order is checked on both.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
SHIFTS = {"libero": 5.0, "robodojo": 3.0}
# (dtype, action shift, padding, horizon). "all-pad" must give a zero loss,
# not NaN (Lane C v2 T21); horizon 1 keeps the length-1 action axis (T22).
CASES = [*itertools.product(DTYPES, SHIFTS, ("no-pad", "terminal-pad"), (HORIZON,)),
         ("bf16", "libero", "all-pad", HORIZON), ("bf16", "libero", "no-pad", 1)]


def case_id(case: tuple) -> str:
    dtype, shift, padding, horizon = case
    return f"{dtype}-{shift}-{padding}-h{horizon}"


def _tiny_spec(action_shift: float, horizon: int = HORIZON):
    """The real VPP2 provider, shrunk, in master's action stage.

    The shift goes in the way YAML's ``diffusion.action.shift`` does, and the
    video term is off as ``heads.video_flow.enabled: false`` turns it off. The
    frequency ratio must divide the horizon (V2); L3 does not read it.
    """
    geo = get_provider("vpp2")(action_width=ACTION_DIM, **SCALARS).geometry
    # Shrunk through the provider's geometry override, so the visibility and
    # token labels it derives from the geometry are those of the shrunk one.
    spec = get_provider("vpp2")(action_width=ACTION_DIM, **SCALARS, geometry=dict(
        frame_height=4 * geo.vae_spatial_stride, frame_width=4 * geo.vae_spatial_stride,
        num_frames=1 + geo.vae_temporal_stride, action_horizon=horizon,
        action_video_freq_ratio=2 if horizon % 2 == 0 else 1, action_dim=ACTION_DIM),
        diffusion={"action": {"shift": action_shift}})
    losses = tuple(replace(loss, enabled=False) if loss.head == "video_flow" else loss
                   for loss in spec.diffusion.losses)
    return replace(spec, diffusion=replace(spec.diffusion, losses=losses))


def _tiny_model(action_shift: float, horizon: int = HORIZON) -> WorldModel:
    """The generic WorldModel on the shrunk provider, assembled as production does."""
    spec = _tiny_spec(action_shift, horizon)
    return WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=spec.build_layout())


def _exec_module(name: str, path: Path, patch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    patch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _methods(path: Path, cls_name: str, *names: str) -> dict:
    """Compile the named methods of one master class verbatim."""
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), f"{path.name}: {cls_name} lacks {names}"
    scope = {"torch": torch, "F": F, "Optional": Optional, "Dict": Dict, "Any": Any,
             "nullcontext": nullcontext}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[name] for name in names}


def load_master(patch: pytest.MonkeyPatch, legacy: Path = LEGACY) -> SimpleNamespace:
    """Master's scheduler, heads and training methods, executed from ``legacy``."""
    scheduler = _exec_module("_vpp2_master_scheduler", legacy / "scheduler.py", patch)
    dit = _exec_module("_vpp2_master_wan_video_dit", legacy / "wan_video_dit.py", patch)
    clip = _exec_module("_vpp2_master_wan_video_clip", legacy / "wan_video_clip.py", patch)
    return SimpleNamespace(
        Scheduler=scheduler.WanContinuousFlowMatchScheduler,
        Head=dit.Head, WanVideoDiT=dit.WanVideoDiT,
        ClipEncoder=clip.WanImageEncoderCLIP, ClipVisionTransformer=clip.VisionTransformer,
        training={**_methods(legacy / "libero_action_model.py", "LiberoActionModel",
                             "build_inputs", "training_loss", "_training_loss_action_only"),
                  **_methods(legacy / "vpp2.py", "VPP2", "_pack_action_tokens",
                             "_pack_action_is_pad", "_build_action_video_conditioning_mask")},
        action_post_dit=_methods(legacy / "action_dit.py", "ActionDiT", "post_dit")["post_dit"])


@pytest.fixture(scope="module")
def master():
    if not LEGACY.is_dir():
        pytest.skip(f"live master parity needs the master checkout at {MASTER}")
    with pytest.MonkeyPatch.context() as patch:
        yield load_master(patch)


def _master_shell(master, *, device, dtype, action_shift, first_frame, prediction, seen):
    """Master's L3 methods bound to exactly the attributes they read.

    The network seams record what L3 hands over and return ``prediction``.
    """
    shell = SimpleNamespace(
        device=device, torch_dtype=dtype, action_only_first_frame_fast_path=True,
        proprio_encoder=None, action_chunk_size=1, raw_action_dim=ACTION_DIM,
        action_visible_video_frames=1, action_video_timestep_cutoff=None,
        vae=SimpleNamespace(temporal_downsample_factor=4), loss_lambda_action=1.0,
        train_video_scheduler=master.Scheduler(shift=5.0, train_sampling_strategy="pure_noise"),
        train_action_scheduler=master.Scheduler(shift=action_shift),
        config=SimpleNamespace(perf_action_visible_kv_sdpa=False,
                               perf_action_graph_visible_kv=False),
        video_expert=SimpleNamespace(patch_size=(1, 2, 2)))
    for name, function in master.training.items():
        setattr(shell, name, MethodType(function, shell))
    loss_only = shell._training_loss_action_only

    def record(key, result):
        def seam(*args, **kwargs):
            seen[key] = kwargs
            return result(*args, **kwargs)
        return seam

    shell._encode_video_latents = lambda video, tiled=False: first_frame.clone()
    shell._build_frozen_video_cache = record("video", lambda **_: (None, None, 0))
    shell._training_loss_action_only = record("loss", loss_only)
    shell.action_expert = SimpleNamespace(
        pre_dit=record("action", lambda **kw: {
            "tokens": kw["action_tokens"], "freqs": None, "t_mod": None,
            "context": None, "context_mask": None}),
        post_dit=lambda tokens, pre_state: prediction)
    shell.mot = SimpleNamespace(perf_action_execution="eager",
                                forward_action_with_video_cache=lambda **kw: kw["action_tokens"])
    return shell


def _exact(label, actual, expected):
    """Zero tolerance; the failure message carries the size of the miss."""
    assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), (
        label, actual.dtype, expected.dtype, tuple(actual.shape), tuple(expected.shape))
    if not torch.equal(actual, expected):
        delta = (actual.double() - expected.double()).abs().max().item()
        pytest.fail(f"{label}: not exact, max_abs={delta:.3g}")


def _compare(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys(), (sorted(actual), sorted(expected))
    for label, value in expected.items():
        _exact(label, actual[label], value)


def _case_inputs(model, case, device):
    """One case's inputs from a fixed seed: master sample, arch batch, first frame, prediction."""
    dtype, _, padding, horizon = case
    dtype = DTYPES[dtype]
    geo = model.spec.geometry
    torch.manual_seed(11)
    first_frame = torch.randn(BATCH, geo.latent_channels, 1, *geo.latent_grid[1:]).to(device, dtype)
    action = torch.randn(BATCH, horizon, ACTION_DIM).to(device, dtype)
    prediction = torch.randn(BATCH, horizon, ACTION_DIM).to(device, dtype)
    masks = {}
    if padding == "terminal-pad":
        assert horizon == 4, "the terminal-pad masks are written for horizon 4"
        masks = dict(
            action_is_pad=torch.tensor([[False, False, True, True], [False, True, True, True]]),
            action_terminal_mask=torch.tensor([[False, False, True, False], [False] * 4]),
            image_is_pad=torch.tensor([[False] * 3 + [True] * 2, [False] * 5]))
    elif padding == "all-pad":
        masks = dict(action_is_pad=torch.ones(BATCH, horizon, dtype=torch.bool))
    masks = {name: value.to(device) for name, value in masks.items()}
    sample = dict(video=torch.zeros(BATCH, 3, geo.num_frames, geo.frame_height, geo.frame_width),
                  context=torch.randn(BATCH, TEXT_LEN, TEXT_DIM),
                  context_mask=torch.ones(BATCH, TEXT_LEN, dtype=torch.bool),
                  clip_fea=torch.randn(BATCH, 2, TEXT_DIM), action=action, **masks)
    video = torch.zeros(BATCH, geo.latent_channels, *geo.latent_grid, device=device, dtype=dtype)
    video[:, :, :1] = first_frame
    return sample, dict(video=video, action=action, **masks), first_frame, prediction


def run_master(master, case, device) -> dict:
    """Master's action-stage L3 on one case; what it hands its network seams, and its loss."""
    dtype, shift, _, horizon = case
    sample, _, first_frame, prediction = _case_inputs(_tiny_model(SHIFTS[shift], horizon),
                                                      case, device)
    seen = {}
    shell = _master_shell(master, device=device, dtype=DTYPES[dtype], action_shift=SHIFTS[shift],
                          first_frame=first_frame, prediction=prediction, seen=seen)
    torch.manual_seed(2026)
    loss, logs = shell.training_loss(sample)
    out = {"video noise": seen["video"]["latents"], "video t": seen["video"]["timestep_video"],
           "noisy action": seen["action"]["action_tokens"], "action t": seen["action"]["timestep"],
           "action target": seen["loss"]["target_action"], "total loss": loss,
           "loss_action": torch.tensor(float(logs["loss_action"]), dtype=torch.float64),
           "loss_video": torch.tensor(float(logs["loss_video"]), dtype=torch.float64),
           # The next draw pins how far the step moved the generator.
           "rng next": torch.rand(4, device=device)}
    if seen["loss"]["action_is_pad"] is not None:
        out["loss padding"] = seen["loss"]["action_is_pad"]
    return out


def _arch_outputs(runtime, prediction) -> dict:
    """The injected action prediction; the switched-off video term gets zeros."""
    return {"action_flow": prediction, "video_flow": torch.zeros_like(runtime.targets["video"])}


def run_arch(case, device) -> dict:
    """The generic path on the same inputs: draw, add_noise and compute_losses.

    Master logs the action stage's video term as 0; the generic path logs the
    switched-off term unweighted with weight 0, so the comparable number is
    their product.
    """
    _, shift, _, horizon = case
    model = _tiny_model(SHIFTS[shift], horizon)
    _, batch, _, prediction = _case_inputs(model, case, device)
    torch.manual_seed(2026)
    runtime = model.add_noise(batch, model.draw(batch))
    rng_next = torch.rand(4, device=device)
    loss, metrics = model.compute_losses(None, runtime, _arch_outputs(runtime, prediction), step=0)
    video = metrics["video_flow/weight"] * metrics["loss_video_flow"].item()
    out = {"video noise": runtime.noised["video"], "video t": runtime.model_timesteps["video"],
           "noisy action": runtime.noised["action"], "action t": runtime.model_timesteps["action"],
           "action target": runtime.targets["action"], "total loss": loss,
           "loss_action": torch.tensor(metrics["loss_action_flow"].item(), dtype=torch.float64),
           "loss_video": torch.tensor(video, dtype=torch.float64),
           "rng next": rng_next}
    if "action" in runtime.loss_valid:
        out["loss padding"] = ~runtime.loss_valid["action"]
    return out


@pytest.mark.parametrize("case", CASES, ids=case_id)
@pytest.mark.parametrize("device", DEVICES)
def test_actual_master_action_stage_l3_parity(master, device, case):
    """Same seed, same inputs: every draw, target, mask and the loss are bit-equal."""
    device = torch.device(device)
    _compare(run_arch(case, device), run_master(master, case, device))


BSMNTW_SHIFTS = (5.0, 3.0, 1.0)


def bsmntw_inputs(dtype: torch.dtype) -> torch.Tensor:
    """Training timesteps: the grid ends and the 500 boundary, then 64 random ones."""
    torch.manual_seed(7)
    return torch.cat([torch.tensor([0.0, 1.0, 499.0, 500.0, 999.0, 1000.0]),
                      torch.rand(64) * 1000]).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shift", BSMNTW_SHIFTS)
def test_bsmntw_matches_master_training_weight(master, shift, dtype):
    """``Bsmntw`` on the training t equals master's scheduler weight, bit for bit."""
    sampler = TimestepSampler(UniformShift(shift=shift))
    t = bsmntw_inputs(DTYPES[dtype])
    expected = master.Scheduler(shift=shift).training_weight(t)
    _exact("bsmntw", Bsmntw()(t, sampler=sampler, num_train_timesteps=1000), expected)
    _exact("bsmntw scalar", Bsmntw()(t[3:4], sampler=sampler, num_train_timesteps=1000),
           master.Scheduler(shift=shift).training_weight(t[3:4]))


def _transfer(head: torch.nn.Module, legacy: torch.nn.Module, prefix: str, name: str) -> None:
    """Load ``legacy`` weights into ``head`` through the production rename table."""
    mapped = vpp2_to_arch({f"{prefix}.{k}": v for k, v in legacy.state_dict().items()})
    head.load_state_dict({k.removeprefix(f"heads.{name}."): v for k, v in mapped.items()},
                         strict=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_video_head_matches_master_head_and_unpatchify(master, dtype):
    model = _tiny_model(5.0)
    geo, width = model.spec.geometry, model.spec.stream("video").residual_width
    legacy = master.Head(width, geo.latent_channels, geo.patch_size, eps=1e-6)
    torch.manual_seed(5)
    with torch.no_grad():
        for parameter in legacy.parameters():
            parameter.copy_(torch.randn_like(parameter))
    head = model.heads["video_flow"]
    _transfer(head, legacy, "mixtures.video.head", "video_flow")
    head.to(dtype)
    legacy.to(dtype)
    hidden = torch.randn(BATCH, math.prod(geo.patch_grid), width).to(dtype)
    embedding = torch.randn(BATCH, width).to(dtype)
    shell = SimpleNamespace(head=legacy, patch_size=tuple(geo.patch_size))
    shell.unpatchify = MethodType(master.WanVideoDiT.unpatchify, shell)
    expected = master.WanVideoDiT.post_dit(
        shell, hidden, {"meta": {"grid_size": geo.patch_grid}, "t": embedding})
    _exact("video head", head(hidden, timestep_embedding=embedding), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_action_head_matches_master_post_dit(master, dtype):
    model = _tiny_model(5.0)
    width = model.spec.stream("action").residual_width
    legacy = torch.nn.Linear(width, ACTION_DIM)
    head = model.heads["action_flow"]
    _transfer(head, legacy, "mixtures.action.head", "action_flow")
    head.to(dtype)
    legacy.to(dtype)
    torch.manual_seed(6)
    hidden = torch.randn(BATCH, HORIZON, width).to(dtype)
    expected = master.action_post_dit(SimpleNamespace(head=legacy), hidden, {})
    _exact("action head", head(hidden), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host-sync detection needs CUDA")
@pytest.mark.parametrize("padding", ["no-pad", "terminal-pad"])
def test_l3_steps_do_not_sync_the_host(padding):
    """draw, add_noise and compute_losses stay on the device, padding masks
    included (Lane C v2 §1.13, §1.16, T23).

    One step runs first, so one-time setup is not counted.
    """
    device = torch.device("cuda")
    model = _tiny_model(SHIFTS["libero"]).to(device)
    _, batch, _, prediction = _case_inputs(model, ("bf16", "libero", padding, HORIZON), device)

    def step():
        runtime = model.add_noise(batch, model.draw(batch))
        return model.compute_losses(None, runtime, _arch_outputs(runtime, prediction), step=0)

    step()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
