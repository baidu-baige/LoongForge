# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Actual master FastWAM versus native arch, with tiny CPU-only networks.

The oracle executes complete master modules, not a reconstructed loss or arch
predictions fed back into legacy code. Only unavailable CUDA kernels/imports
and the VAE encoder are replaced. No transformer/condition/loss is stubbed.
Run with ``pytest -s`` for exactness and maximum-error evidence, even on failure.
The sibling master checkout is intentionally required (not silently skipped).
"""

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from loongforge.embodied.model.arch.fastwam_training import FastWAMTrainingWorldModel
from loongforge.embodied.model.arch.layout import TokenLayout
from loongforge.embodied.model.arch.spec.core import get_provider
import loongforge.embodied.model.arch.spec.providers  # noqa: F401
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.arch.transfer import fastwam_to_arch


REPO = Path(__file__).resolve().parents[3]
MASTER = REPO.parent / "LoongForge_master_vpp2"
LEGACY = Path("loongforge/embodied/model/fastwam")
PREFIX = "_fastwam_actual_master_parity"
SOURCE_MODULES = (
    "utils.gradient", "utils.state_dict", "wan.dit", "action.dit",
    "action.schedulers", "mot.model", "mot.fastwam",
)


def _kernel_rotary(x, cos, sin, *, interleaved, conjugate=False):
    """FlashAttention's interleaved pair rotation, including its backward."""
    assert interleaved and x.device.type == "cpu"
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    if conjugate:
        s = -s
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)


def _norm_modulate(x, scale, shift, eps):
    assert x.device.type == "cpu"
    return F.layer_norm(x, (x.shape[-1],), None, None, eps) * (1 + scale) + shift


class _UnavailableTE(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise AssertionError("CPU parity must use the actual WAN RMSNorm, not TE")


@pytest.fixture
def actual_master(monkeypatch):
    """Import whole master modules under an isolated, reversible namespace.

    AST edits affect imports ONLY: redirect internal dependencies to master,
    replace FlashAttention rotary/TE/Triton bindings, and omit the unused
    pretrained loader. All class and function definitions remain untouched.
    """
    assert MASTER.is_dir(), f"Actual master checkout is required: {MASTER}"
    for package in ("", ".utils", ".wan", ".action", ".mot"):
        module = ModuleType(PREFIX + package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for suffix in SOURCE_MODULES:
        path = MASTER / LEGACY / (suffix.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        body = []
        for node in tree.body:
            if isinstance(node, ast.Import) and any(
                alias.name == "transformer_engine.pytorch" for alias in node.names
            ):
                continue
            if isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if name in (
                    "flash_attn.ops.triton.rotary",
                    "loongforge.embodied.model.fastwam.utils.norm_modulate",
                    "loongforge.embodied.model.fastwam.wan.loader",
                ):
                    continue
                if name.startswith("loongforge.embodied.model.fastwam"):
                    node.module = name.replace("loongforge.embodied.model.fastwam", PREFIX, 1)
            body.append(node)
        tree.body = body
        module = ModuleType(f"{PREFIX}.{suffix}")
        module.__file__ = str(path)
        module.__dict__.update(
            te=SimpleNamespace(RMSNorm=_UnavailableTE),
            apply_rotary=_kernel_rotary, triton_norm_modulate=_norm_modulate,
        )
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return SimpleNamespace(
        WanVideoDiT=sys.modules[f"{PREFIX}.wan.dit"].WanVideoDiT,
        ActionDiT=sys.modules[f"{PREFIX}.action.dit"].ActionDiT,
        MoT=sys.modules[f"{PREFIX}.mot.model"].MoT,
        FastWAM=sys.modules[f"{PREFIX}.mot.fastwam"].FastWAM,
    )


@pytest.fixture(autouse=True)
def cpu_math():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def test_training_oracle_sources_match_master():
    """Verify exact paths with git diff; whitelist only unused inference methods."""
    for suffix in (*SOURCE_MODULES, "utils.norm_modulate"):
        relative = LEGACY / (suffix.replace(".", "/") + ".py")
        master, local = MASTER / relative, REPO / relative
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", str(master), str(local)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode in (0, 1), result.stderr
        if suffix != "mot.fastwam":
            assert result.returncode == 0, result.stdout
        else:
            # Master adds batched image/action inference. No training definition
            # (including input preparation/proprio) may differ.
            def definitions(path):
                tree = ast.parse(path.read_text())
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                           and n.name == "FastWAM")
                return {n.name: ast.dump(n, include_attributes=False) for n in cls.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            a, b = definitions(master), definitions(local)
            differences = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
            assert differences <= {"_encode_input_image_latents_tensor", "infer_action", "_action_noise"}, (
                differences, result.stdout
            )
        print(f"master-source verified: {relative}")


class _LatentVAE:
    temporal_downsample_factor = 4

    def __init__(self, latents):
        self.latents = latents
        self.calls = 0

    def encode(self, *args, **kwargs):
        self.calls += 1
        return self.latents.clone()

    def to(self, *args, **kwargs):
        """Honor actual FastWAM.to without constructing or registering a VAE."""
        self.latents = self.latents.to(*args, **kwargs)
        return self


def _pair(actual, checkpoint):
    torch.manual_seed(103)
    spec = get_provider("fastwam")(
        proprio_dim=3, action_width=2,
        geometry=dict(frame_height=32, frame_width=64, num_views=1,
                      num_frames=9, latent_channels=2, action_horizon=8,
                      action_video_freq_ratio=1),
        video_residual_width=16, video_num_heads=2, video_head_dim=8,
        video_num_layers=2, video_ffn_width=24,
        action_residual_width=8, action_num_heads=2, action_head_dim=8,
        action_num_layers=2, action_ffn_width=16,
    )
    spec = replace(spec, conditions=tuple(replace(c, width=8) for c in spec.conditions))
    common = dict(text_dim=8, freq_dim=256, eps=1e-6, num_heads=2,
                  attn_head_dim=8, num_layers=2, rmsnorm_impl="wan",
                  use_gradient_checkpointing=checkpoint)
    video = actual.WanVideoDiT(
        hidden_dim=16, in_dim=2, out_dim=2, ffn_dim=24,
        patch_size=spec.geometry.patch_size, has_image_input=False,
        seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal", **common,
    )
    action = actual.ActionDiT(hidden_dim=8, action_dim=2, ffn_dim=16, **common)
    mot = actual.MoT({"video": video, "action": action},
                     mot_checkpoint_mixed_attn=checkpoint)
    latents = torch.randn(2, 2, *spec.geometry.latent_grid)
    vae = _LatentVAE(latents)
    master = actual.FastWAM(video, action, mot, vae, text_dim=8, proprio_dim=3,
                            action_train_shift=1, device="cpu")
    arch = FastWAMTrainingWorldModel(
        spec, engine=JointAttentionEngine(backend="dense", checkpoint_layers=checkpoint),
        layout=TokenLayout.from_geometry(spec.geometry, spec.streams),
    )
    transferred = fastwam_to_arch(master.state_dict())
    assert set(transferred) == set(arch.state_dict()), "Full trainable transfer must be strict"
    arch.load_state_dict(transferred, strict=True)
    assert all(p.requires_grad for p in master.parameters())
    assert all(p.requires_grad for p in arch.parameters())
    assert sum(p.numel() for p in master.parameters()) == sum(p.numel() for p in arch.parameters())
    for name, value in arch.state_dict().items():
        assert torch.equal(value, transferred[name]), f"initial weights: {name}"
        assert value.data_ptr() != transferred[name].data_ptr(), f"shared storage: {name}"
    print(f"full-transfer: exact=True tensors={len(transferred)} "
          f"trainable_numel={sum(p.numel() for p in arch.parameters())}")
    sample = dict(
        video=torch.zeros(2, 3, 9, 32, 64), action=torch.randn(2, 8, 2),
        context=torch.randn(2, 4, 8), proprio=torch.randn(2, 2, 3),
        action_is_pad=torch.tensor([[False, True, False, True, False, False, True, True],
                                    [True] * 8]),
        image_is_pad=torch.tensor([[False, True, True, False, True, True, True, True, True],
                                   [True] * 9]),
    )
    arch_sample = dict(sample, video=latents.clone(), proprio=sample["proprio"][:, 0])
    return master, arch, sample, arch_sample


def _compare(label, actual, expected, failures):
    """Keep zero tolerances; collect evidence so backward/optimizer still run."""
    assert actual.shape == expected.shape, f"{label}: shape mismatch"
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    exact = torch.equal(actual, expected)
    delta = (actual.detach().double() - expected.detach().double()).abs()
    print(f"{label}: exact={exact} max_abs={delta.max().item():.12g}")
    if not exact:
        failures.append(f"{label}: max_abs={delta.max().item():.12g}")


def _mapped_parameters(master, *, gradients=False):
    state = {}
    for name, parameter in master.named_parameters():
        if gradients:
            assert parameter.grad is not None, f"master unused trainable weight: {name}"
            state[name] = parameter.grad
        else:
            state[name] = parameter.detach()
    return fastwam_to_arch(state)


@pytest.mark.parametrize("masked_text", [False, True], ids=["all-text-valid", "mixed-text-mask"])
@pytest.mark.parametrize("checkpoint", [False, True], ids=["eager", "checkpoint"])
def test_actual_master_full_training_parity(actual_master, monkeypatch, masked_text, checkpoint):
    master, arch, sample, arch_sample = _pair(actual_master, checkpoint)
    mask = (torch.tensor([[True, False, True, False], [False, True, False, True]])
            if masked_text else torch.ones(2, 4, dtype=torch.bool))
    sample["context_mask"] = arch_sample["context_mask"] = mask
    failures, predictions, stochastic = [], {}, {}
    # Observe real outputs; never substitute either side's predictions.
    for stream, expert in (("video", master.video_expert), ("action", master.action_expert)):
        original = expert.post_dit
        def observe(*args, original=original, stream=stream, **kwargs):
            result = original(*args, **kwargs)
            predictions[f"master_{stream}"] = result
            return result
        monkeypatch.setattr(expert, "post_dit", observe)
    for stream, head in (("video", arch.heads["video_flow"]), ("action", arch.heads["action_flow"])):
        head.register_forward_hook(
            lambda module, args, result, stream=stream: predictions.update({f"arch_{stream}": result})
        )
    for stream, scheduler in (("video", master.train_video_scheduler),
                              ("action", master.train_action_scheduler)):
        original = scheduler.add_noise
        def noise(clean, eps, timestep, original=original, stream=stream):
            stochastic[stream] = (eps.clone(), timestep.clone())
            return original(clean, eps, timestep)
        monkeypatch.setattr(scheduler, "add_noise", noise)
    original = arch.add_noise
    def runtime(*args, **kwargs):
        result = original(*args, **kwargs)
        for stream in ("video", "action"):
            assert torch.equal(result.noise[stream], stochastic[stream][0]), f"{stream} noise"
            assert torch.equal(result.model_timesteps[stream], stochastic[stream][1]), f"{stream} t"
        return result
    monkeypatch.setattr(arch, "add_noise", runtime)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01,
                                   foreach=False) for model in (master, arch)]
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(701 + step)
        expected_loss, expected_metrics = master.training_loss(sample)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(701 + step)
        loss, metrics = arch(arch_sample)
        assert torch.equal(torch.get_rng_state(), expected_rng), "training RNG consumption differs"
        for stream in ("video", "action"):
            _compare(f"step{step}/{stream}-prediction", predictions[f"arch_{stream}"],
                     predictions[f"master_{stream}"], failures)
        _compare(f"step{step}/total-loss", loss, expected_loss, failures)
        for name in ("loss_video", "loss_action"):
            _compare(f"step{step}/{name}", metrics[name],
                     torch.tensor(expected_metrics[name], dtype=metrics[name].dtype), failures)
        expected_loss.backward()
        loss.backward()
        gradients = _mapped_parameters(master, gradients=True)
        assert set(gradients) == set(dict(arch.named_parameters()))
        for name, parameter in arch.named_parameters():
            assert parameter.grad is not None, f"arch unused trainable weight: {name}"
            _compare(f"step{step}/gradient/{name}", parameter.grad, gradients[name], failures)
        for optimizer in optimizers:
            optimizer.step()
        updated = _mapped_parameters(master)
        for name, parameter in arch.named_parameters():
            _compare(f"step{step}/optimizer-weight/{name}", parameter, updated[name], failures)
        for moment in ("exp_avg", "exp_avg_sq", "step"):
            master_state = fastwam_to_arch({
                name: optimizers[0].state[parameter][moment]
                for name, parameter in master.named_parameters()
            })
            for name, parameter in arch.named_parameters():
                _compare(f"step{step}/optimizer-{moment}/{name}",
                         optimizers[1].state[parameter][moment], master_state[name], failures)
    assert master.vae.calls == 2, "Actual build_inputs must encode once per training invocation"
    assert not failures, "NOT EXACT actual-master parity:\n" + "\n".join(failures)
