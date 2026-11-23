# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only real-engine training and production-master math parity."""

import ast
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from loongforge.embodied.model.arch.builder import WorldModel
from loongforge.embodied.model.arch.diffusion.loss import WeightRamp
from loongforge.embodied.model.arch.fastwam_training import (
    FastWAMTrainingWorldModel, video_loss_per_sample,
)
from loongforge.embodied.model.arch.layout import TokenLayout
from loongforge.embodied.model.arch.spec.core import get_provider
import loongforge.embodied.model.arch.spec.providers  # noqa: F401 -- register providers
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.arch.vpp2_training import VPP2ActionScheduler


MASTER = Path(__file__).resolve().parents[4] / "LoongForge_master_vpp2"


def test_fastwam_normmod_preserves_values_with_contiguous_kernel_input(monkeypatch):
    """Exercise the real dispatch seam without importing CUDA dependencies."""
    from loongforge.embodied.model.arch.fastwam_training import _FastWAMBlock

    observed = {}
    name = "loongforge.embodied.model.fastwam.utils.norm_modulate"
    module = ModuleType(name)

    def record_kernel(x, scale, shift, eps):
        observed.update(x=x, scale=scale, shift=shift, eps=eps)
        return x * (1 + scale) + shift

    module.triton_norm_modulate = record_kernel
    monkeypatch.setitem(sys.modules, name, module)
    source = torch.arange(24., requires_grad=True)
    x = source.reshape(2, 4, 3).transpose(1, 2)
    original_stride = x.stride()
    shift = torch.ones(2, 1, 4, requires_grad=True)
    scale = torch.full((2, 1, 4), 0.5, requires_grad=True)
    output = _FastWAMBlock._norm_modulate(SimpleNamespace(eps=1e-6), x, shift, scale)
    assert observed["x"].is_contiguous()
    assert torch.equal(observed["x"], x)
    assert x.stride() == original_stride and not x.is_contiguous()
    assert torch.equal(observed["shift"], shift.expand_as(x))
    assert torch.equal(observed["scale"], scale.expand_as(x))
    assert observed["eps"] == 1e-6
    assert torch.equal(output, x * 1.5 + shift)
    output.sum().backward()
    assert torch.equal(source.grad, torch.full_like(source, 1.5))
    assert torch.equal(shift.grad, torch.full_like(shift, 3.))
    assert torch.equal(scale.grad, x.detach().sum(dim=1, keepdim=True))


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def production(name):
    """Execute exact production AST definitions, avoiding CUDA-heavy imports."""
    root = MASTER / "loongforge/embodied/model/fastwam"
    path = root / ("action/schedulers.py" if name == "WanContinuousFlowMatchScheduler"
                   else "mot/fastwam.py")
    if not path.exists():
        pytest.skip(f"production master source not available: {path}")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == ("WanContinuousFlowMatchScheduler"
                                 if name == "WanContinuousFlowMatchScheduler" else "FastWAM"))
    node = cls if name == cls.name else next(n for n in cls.body
                                            if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    scope = {"torch": torch, "F": F, "Optional": __import__("typing").Optional}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def tiny_spec():
    spec = get_provider("fastwam")(
        proprio_dim=3, action_width=2,
        geometry=dict(frame_height=32, frame_width=32, num_views=1,
                      num_frames=9, latent_channels=2, action_horizon=4,
                      action_video_freq_ratio=2),
        video_residual_width=16, video_num_heads=2, video_head_dim=8,
        video_num_layers=2, video_ffn_width=24,
        action_residual_width=8, action_num_heads=2, action_head_dim=8,
        action_num_layers=2, action_ffn_width=16)
    return replace(spec, conditions=tuple(replace(c, width=8) for c in spec.conditions))


def model_and_batch(**kwargs):
    spec = tiny_spec()
    model = FastWAMTrainingWorldModel(
        spec, engine=JointAttentionEngine(backend="dense"),
        layout=TokenLayout.from_geometry(spec.geometry, spec.streams), **kwargs)
    batch = dict(video=torch.randn(2, 2, *spec.geometry.latent_grid),
                 action=torch.randn(2, 4, 2), context=torch.randn(2, 3, 8),
                 proprio=torch.randn(2, 3),
                 action_is_pad=torch.tensor([[False, True, False, True], [True]*4]),
                 image_is_pad=torch.tensor([[False, True, True, False, True, True, True, True, True],
                                            [True]*9]))
    return model, batch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shift", [1., 5.])
def test_scheduler_exact_master_math(dtype, shift):
    master = production("WanContinuousFlowMatchScheduler")(shift=shift)
    arch = VPP2ActionScheduler(shift=shift)
    torch.manual_seed(32)
    expected = master.sample_training_t(3, torch.device("cpu"), dtype)
    torch.manual_seed(32)
    actual = arch.sample_training_t(3, torch.device("cpu"), dtype)
    assert torch.equal(actual, expected)
    assert torch.equal(master.training_weight(expected), arch.training_weight(actual))
    x, noise = torch.randn(3, 4, 2).to(dtype), torch.randn(3, 4, 2).to(dtype)
    assert torch.equal(master.add_noise(x, noise, expected), arch.add_noise(x, noise, actual))
    assert torch.equal(master.training_target(x, noise, expected), arch.training_target(x, noise, actual))


@pytest.mark.parametrize("fuse", [True, False])
def test_rng_order_noise_pinning_and_normalized_runtime(fuse):
    model, batch = model_and_batch(fuse_vae_embedding_in_latents=fuse)
    scheduler = production("WanContinuousFlowMatchScheduler")
    for dtype in (torch.float32, torch.bfloat16):
        sample = {k: v.to(dtype) if v.is_floating_point() else v for k, v in batch.items()}
        torch.manual_seed(51)
        nv = torch.randn_like(sample["video"])
        tv = scheduler(shift=5).sample_training_t(2, torch.device("cpu"), dtype)
        na = torch.randn_like(sample["action"])
        ta = scheduler(shift=1).sample_training_t(2, torch.device("cpu"), dtype)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(51)
        times = model.sample_timesteps(sample)
        runtime = model.add_noise(sample, times)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(runtime.noise["video"], nv)
        assert torch.equal(runtime.noise["action"], na)
        assert torch.equal(runtime.model_timesteps["video"], tv)
        assert torch.equal(runtime.model_timesteps["action"], ta)
        expected = scheduler(shift=5).add_noise(sample["video"], nv, tv)
        if fuse:
            expected[:, :, :1] = sample["video"][:, :, :1]
        assert torch.equal(runtime.noised["video"], expected)
        assert torch.equal(runtime.noised["action"], scheduler(shift=1).add_noise(sample["action"], na, ta))


@pytest.mark.parametrize("include_first", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_video_loss_production_pooling_and_gradients(include_first, masked):
    pred = torch.randn(2, 2, 3, 2, 2, requires_grad=True)
    target = torch.randn_like(pred)
    pad = torch.tensor([[False, True, True, False, True, True, True, True, True],
                        [True]*9]) if masked else None
    master = production("_compute_video_loss_per_sample")
    sliced_pred = pred if include_first else pred[:, :, 1:]
    sliced_target = target if include_first else target[:, :, 1:]
    expected = master(SimpleNamespace(vae=SimpleNamespace(temporal_downsample_factor=4)),
                      sliced_pred, sliced_target, pad, include_first)
    actual = video_loss_per_sample(pred, target, pad, include_initial_video_step=include_first)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    if not include_first:
        assert torch.count_nonzero(pred.grad[:, :, 0]) == 0
    if masked:
        assert actual[1] == 0
        assert torch.count_nonzero(pred.grad[:, :, 2]) == 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_engine_forward_backward_and_production_training_loss(monkeypatch, checkpoint):
    model, batch = model_and_batch()
    model.engine.checkpoint_layers = checkpoint
    outputs, states, runtime_holder = {}, {}, {}
    for name, head in model.heads.items():
        head.register_forward_hook(lambda module, args, result, name=name: outputs.update({name: result}))
    original_run = model.engine.run
    def run(*args, **kwargs):
        assert kwargs["timesteps"]["video"].shape == (2, model.spec.geometry.padded_seq_len)
        assert torch.count_nonzero(kwargs["timesteps"]["video"][:, :model.spec.geometry.tokens_per_frame]) == 0
        result = original_run(*args, **kwargs)
        states.update(result.states)
        return result
    monkeypatch.setattr(model.engine, "run", run)
    original_noise = model.add_noise
    def add_noise(*args):
        result = original_noise(*args)
        runtime_holder["runtime"] = result
        return result
    monkeypatch.setattr(model, "add_noise", add_noise)
    assert FastWAMTrainingWorldModel.forward is WorldModel.forward
    torch.manual_seed(77)
    loss, metrics = model(batch)
    runtime = runtime_holder["runtime"]
    # Replay production training_loss against the real arch predictions. Only
    # encoder/expert execution is stubbed; production draws/noising/loss run.
    scheduler = production("WanContinuousFlowMatchScheduler")
    master_video_loss = production("_compute_video_loss_per_sample")
    video_expert = SimpleNamespace(
        pre_dit=lambda **kw: dict(tokens=states["video"].hidden, freqs=None,
                                 context=kw["context"], context_mask=None, t_mod=None,
                                 meta=dict(tokens_per_frame=model.spec.geometry.tokens_per_frame)),
        post_dit=lambda *args: outputs["video_flow"])
    action_expert = SimpleNamespace(
        pre_dit=lambda **kw: dict(tokens=states["action"].hidden, freqs=None,
                                 context=kw["context"], context_mask=None, t_mod=None),
        post_dit=lambda *args: outputs["action_flow"])
    master = SimpleNamespace(
        build_inputs=lambda *args, **kw: dict(input_latents=batch["video"], action=batch["action"],
            context=batch["context"], context_mask=None, action_is_pad=batch["action_is_pad"],
            image_is_pad=batch["image_is_pad"], first_frame_latents=batch["video"][:, :, :1],
            fuse_vae_embedding_in_latents=True),
        train_video_scheduler=scheduler(shift=5), train_action_scheduler=scheduler(shift=1),
        device=torch.device("cpu"), video_expert=video_expert, action_expert=action_expert,
        mot=lambda **kw: kw["embeds_all"], _build_mot_attention_mask=lambda **kw: None,
        vae=SimpleNamespace(temporal_downsample_factor=4), loss_lambda_video=1., loss_lambda_action=1.)
    master._compute_video_loss_per_sample = lambda **kw: master_video_loss(master, **kw)
    torch.manual_seed(77)
    expected, expected_metrics = production("training_loss")(master, batch)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    for key in ("loss_video", "loss_action"):
        assert metrics[key].item() == expected_metrics[key]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.blocks["video"].parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.blocks["action"].parameters())
    # Time embedding follows token axis all the way through the output head.
    assert states["video"].extras["timestep_embedding"].shape == states["video"].hidden.shape
    assert runtime.timesteps["video"].shape == (2,)


def test_provider_visibility_and_defaults():
    model, _ = model_and_batch()
    default = get_provider("fastwam")()
    assert default.model_type == "fastwam"
    assert default.geometry.vae_spatial_stride == 16
    assert default.geometry.latent_grid == (3, 14, 28)
    assert default.diffusion.sampler_for("action").distribution.shift == 1
    mask = model.plans[0].structural
    n = model.spec.geometry.video_seq_len
    first = model.spec.geometry.tokens_per_frame
    assert mask[:first, :first].all()
    assert not mask[:first, first:n].any()
    assert mask[first:n, :n].all()
    assert not mask[:n, n:].any()
    assert mask[n:, :first].all()
    assert not mask[n:, first:n].any()
    assert mask[n:, n:].all()


def test_batch_validation_and_fuse_override():
    model, batch = model_and_batch()
    for extra, message in ((dict(action_is_pad=torch.zeros(2, 1)), "action_is_pad"),
                           (dict(image_is_pad=torch.zeros(2, 3)), "image_is_pad"),
                           (dict(first_frame_latents=torch.zeros(2, 2, 2, 2, 2)), "first_frame_latents")):
        sample = dict(batch, **extra)
        with pytest.raises(ValueError, match=message):
            model.add_noise(sample, model.sample_timesteps(sample))
    sample = dict(batch, fuse_vae_embedding_in_latents=False)
    runtime = model.add_noise(sample, model.sample_timesteps(sample))
    assert not runtime.fuse_first_frame
    assert not torch.equal(runtime.noised["video"][:, :, :1], batch["video"][:, :, :1])


def test_spec_loss_weights_and_all_padding_zero():
    model, batch = model_and_batch()
    specs = model._loss_specs
    model._loss_specs = {name: replace(loss, weight=3. if name == "video_flow" else 2.)
                         for name, loss in specs.items()}
    runtime = model.add_noise(batch, {"video": torch.tensor([.5, .5]),
                                    "action": torch.tensor([.25, .25])})
    outputs = {"video_flow": runtime.targets["video"] + 1,
               "action_flow": runtime.targets["action"] + 1}
    total, metrics = model.compute_losses(None, runtime, outputs, step=0)
    torch.testing.assert_close(metrics["loss_video"], model.video_scheduler.training_weight(
        runtime.model_timesteps["video"])[0] * 1.5)
    torch.testing.assert_close(metrics["loss_action"], model.action_scheduler.training_weight(
        runtime.model_timesteps["action"])[0])
    assert total == metrics["loss_video"] + metrics["loss_action"]


def test_first_frame_override_and_padding_prediction_gradients():
    model, batch = model_and_batch()
    batch["first_frame_latents"] = torch.full_like(batch["video"][:, :, :1], 7.)
    runtime = model.add_noise(batch, model.sample_timesteps(batch))
    assert torch.equal(runtime.noised["video"][:, :, :1], batch["first_frame_latents"])
    pred_video = torch.randn_like(batch["video"], requires_grad=True)
    pred_action = torch.randn_like(batch["action"], requires_grad=True)
    loss, _ = model.compute_losses(
        None, runtime, {"video_flow": pred_video, "action_flow": pred_action}, step=0)
    loss.backward()
    assert torch.count_nonzero(pred_video.grad[:, :, 0]) == 0
    assert torch.count_nonzero(pred_video.grad[1]) == 0
    assert torch.count_nonzero(pred_action.grad[batch["action_is_pad"]]) == 0
    assert torch.count_nonzero(pred_action.grad[~batch["action_is_pad"]]) > 0


def test_weight_ramp_uses_optimizer_step():
    model, batch = model_and_batch()
    model._loss_specs = {name: replace(loss, ramp=WeightRamp(
        start_step=0, end_step=10, start_value=0., end_value=1.))
                         for name, loss in model._loss_specs.items()}
    runtime = model.add_noise(batch, model.sample_timesteps(batch))
    outputs = {"video_flow": runtime.targets["video"] + 1,
               "action_flow": runtime.targets["action"] + 1}
    half, _ = model.compute_losses(None, runtime, outputs, step=5)
    full, _ = model.compute_losses(None, runtime, outputs, step=10)
    torch.testing.assert_close(half * 2, full)


def test_unfused_real_forward_backward():
    model, batch = model_and_batch(fuse_vae_embedding_in_latents=False)
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())


def test_vpp2_training_adapter_remains_sample_level_and_frozen():
    from loongforge.embodied.model.arch.vpp2_training import VPP2TrainingWorldModel
    from test_arch_blocks import tiny_spec as existing_tiny_spec
    from test_arch_world_model import fake_batch

    spec = existing_tiny_spec("vpp2")
    engine = JointAttentionEngine(backend="dense")
    model = VPP2TrainingWorldModel(
        spec, engine=engine, layout=TokenLayout.from_geometry(spec.geometry, spec.streams))
    batch = fake_batch(spec)
    loss, metrics = model(batch)
    loss.backward()
    assert metrics["loss_video"] == 0
    assert not engine.frozen_streams  # adapter's copy must not change the caller
    assert all(not p.requires_grad and p.grad is None for p in model.blocks["video"].parameters())
    assert all(p.grad is not None for p in model.blocks["action"].parameters())
    assert model.blocks["video"].__class__.__name__ == "WanVideoStream"


@pytest.mark.parametrize("input_dtype,weight_dtype", [
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
])
def test_fastwam_rmsnorm_exact_master_forward_and_backward(input_dtype, weight_dtype):
    from loongforge.embodied.model.arch.fastwam_training import FastWAMRMSNorm

    path = MASTER / "loongforge/embodied/model/fastwam/wan/dit.py"
    if not path.exists():
        pytest.skip(f"production master source not available: {path}")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WanRMSNorm")
    scope = {"torch": torch, "nn": torch.nn, "F": F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), "exec"), scope)
    master = scope["WanRMSNorm"](16, eps=1e-6).to(dtype=weight_dtype)
    arch = FastWAMRMSNorm(16, eps=1e-6).to(dtype=weight_dtype)
    with torch.no_grad():
        master.weight.copy_(torch.randn_like(master.weight))
        arch.weight.copy_(master.weight)
    x = torch.randn(2, 3, 16).to(input_dtype).requires_grad_()
    y = x.detach().clone().requires_grad_()
    expected, actual = master(x), arch(y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
    torch.testing.assert_close(arch.weight.grad, master.weight.grad, rtol=0, atol=0)


def test_norm_specialization_keeps_parameter_identity_and_rng():
    from loongforge.embodied.model.arch.fastwam_training import FastWAMRMSNorm
    from loongforge.embodied.model.arch.streams.blocks.wan_dit import RMSNorm

    model, _ = model_and_batch()
    for stream in ("video", "action"):
        norms = [module for module in model.blocks[stream].modules()
                 if isinstance(module, RMSNorm)]
        assert norms and all(isinstance(norm, FastWAMRMSNorm) for norm in norms)
        assert all(norm.eps == 1e-6 for norm in norms)
    spec = tiny_spec()
    common = dict(engine=JointAttentionEngine(backend="dense"),
                  layout=TokenLayout.from_geometry(spec.geometry, spec.streams))
    torch.manual_seed(53)
    specialized = FastWAMTrainingWorldModel(spec, **common)
    state = torch.get_rng_state()
    # Generic factories must consume the same constructor RNG as the specialized
    # ones; replacing norms cannot initialize any new random weights.
    from loongforge.embodied.model.arch.spec.core import Spec
    from loongforge.embodied.model.arch.streams.blocks import ActionStream, WanVideoStream
    from loongforge.embodied.model.arch.heads.flow import build_flow_head
    streams = tuple(replace(s, impl=Spec(WanVideoStream if s.name == "video" else ActionStream))
                    for s in spec.streams)
    heads = tuple(replace(h, impl=Spec(build_flow_head, {"form": "patch_flow"}))
                  if h.name == "video_flow" else h for h in spec.heads)
    group = replace(spec.attention_groups[0], streams=streams)
    generic_spec = replace(spec, attention_groups=(group,), heads=heads)
    torch.manual_seed(53)
    generic = WorldModel(generic_spec, **common)
    assert torch.equal(state, torch.get_rng_state())
    assert specialized.state_dict().keys() == generic.state_dict().keys()
    for name, value in specialized.state_dict().items():
        assert torch.equal(value, generic.state_dict()[name]), name
    for stream in ("video", "action"):
        original_norms = dict(generic.blocks[stream].named_modules())
        for name, norm in specialized.blocks[stream].named_modules():
            if isinstance(norm, FastWAMRMSNorm):
                assert norm.eps == original_norms[name].eps


@pytest.mark.parametrize("checkpoint", [False, True])
def test_context_masks_and_gradients_are_invocation_local(checkpoint):
    model, batch = model_and_batch()
    model.engine.checkpoint_layers = checkpoint
    batch["context_mask"] = torch.tensor([[True, False, True], [False, False, True]])
    batch["context"].requires_grad_()
    bus = model.build_conditions(batch)
    expected = torch.cat((batch["context_mask"], torch.ones(2, 1, dtype=torch.bool)), dim=1)
    for stream in ("video", "action"):
        assert torch.equal(bus.slot_masks[stream]["text"], expected)
    changed = dict(batch, context=batch["context"].detach().clone())
    changed["context"][~batch["context_mask"]] = 1000
    torch.manual_seed(54)
    original_loss, _ = model(batch)
    torch.manual_seed(54)
    changed_loss, _ = model(changed)
    torch.testing.assert_close(changed_loss, original_loss, rtol=0, atol=0)
    original_loss.backward()
    assert torch.count_nonzero(batch["context"].grad[~batch["context_mask"]]) == 0
    assert torch.count_nonzero(batch["context"].grad[batch["context_mask"]]) > 0
    other = model.build_conditions(dict(batch, context_mask=torch.ones_like(batch["context_mask"])))
    assert other is not bus
    assert other.slot_masks["video"]["text"].all()
    assert torch.equal(bus.slot_masks["video"]["text"], expected)
