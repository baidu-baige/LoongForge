# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU (gloo) tests for ``TensorWiseOptimizer``.

Ranks are started with ``spawn``, not ``fork``: importing the checkpoint backend
can initialize CUDA in the parent (through the optional ``dmuon`` package), and a
forked child then fails as soon as anything touches CUDA.
"""

from __future__ import annotations

import os
import re
import socket
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from loongforge.embodied.distributed import checkpoint
from loongforge.embodied.optimizer.tensor_wise_optimizer import validation
from loongforge.embodied.optimizer.tensor_wise_optimizer.precision_policy import (
    DefaultParameterPolicy,
    ParameterClassPolicy,
)
from loongforge.embodied.optimizer.tensor_wise_optimizer.tensor_wise_optimizer import (
    CollectiveConfig,
    TensorWiseOptimizer,
)
from loongforge.embodied.optimizer.tensor_wise_optimizer.optimizer_builder import (
    build_muon_optimizer,
)

WORLD_SIZE = 2
GRAD_ACCUM = 2
LR = 1e-2


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(worker, *args, world_size=WORLD_SIZE):
    """Run ``worker(rank, *args)`` on ``world_size`` gloo ranks."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    mp.start_processes(
        _bootstrap,
        args=(worker, args, world_size),
        nprocs=world_size,
        start_method="spawn",
    )


def _bootstrap(rank, worker, args, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4))


def _batch(step, micro, rank):
    generator = torch.Generator().manual_seed(1000 * step + 10 * micro + rank)
    return torch.randn(3, 8, generator=generator), torch.randn(3, 4, generator=generator)


def _mse_loss(model, step, micro, rank):
    x, y = _batch(step, micro, rank)
    dtype = next(model.parameters()).dtype
    return nn.functional.mse_loss(model(x.to(dtype)).float(), y)


def _adamw_factory(dp):
    return torch.optim.AdamW(
        [p for _, p in dp.optimizer_view().named_parameters()], lr=LR, foreach=False
    )


def _muon_factory(dp):
    muon, adamw = dp.parameter_plan.split_masters_by_optimizer()
    return build_muon_optimizer(muon, lr=LR, adamw_parameters=adamw)


def _build(model, factory, policy=None, **config):
    config.setdefault("gradient_reduce_dtype", "fp32")
    return TensorWiseOptimizer(
        model,
        policy or DefaultParameterPolicy(),
        factory,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        collective_config=CollectiveConfig(**config),
    )


def _train(model, optimizer, steps, start=0, clip=1.0, loss_fn=_mse_loss):
    norms = []
    for step in range(start, start + steps):
        optimizer.zero_grad()
        for micro in range(GRAD_ACCUM):
            if micro == GRAD_ACCUM - 1:
                optimizer.prepare_for_backward()
            loss = loss_fn(model, step, micro, dist.get_rank()) / GRAD_ACCUM
            loss.backward()
        optimizer.scrub_nan_gradients()
        norms.append(optimizer.clip_grad_norm_(clip))
        optimizer.step()
    return norms


def _replicas(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _assert_replicas_agree(model):
    for name, parameter in model.named_parameters():
        gathered = [torch.empty_like(parameter) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameter.detach().contiguous())
        assert all(torch.equal(gathered[0], other) for other in gathered[1:]), name


# -- model structures beyond the plain MLP --------------------------------------


def _frozen_model():
    """A frozen first layer, like a frozen vision tower or teacher."""
    model = _model()
    model[0].requires_grad_(False)
    return model


class _Routed(nn.Module):
    """Three experts; the loss picks one per micro-batch and never picks expert 2."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.experts = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
        self.out = nn.Linear(16, 4)

    def forward(self, x, expert):
        return self.out(self.experts[expert](torch.tanh(self.inp(x))))


def _routed_loss(model, step, micro, rank):
    # Steps 0-1 send the ranks to different experts, so each of experts 0/1 has
    # a gradient on some ranks only; step 2 uses expert 0 everywhere, so expert 1
    # has none on any rank.
    expert = (rank + step) % 2 if step < 2 else 0
    x, y = _batch(step, micro, rank)
    return nn.functional.mse_loss(model(x, expert), y)


class _Tied(nn.Module):
    """Two layers sharing one weight, as tied embeddings and LM heads do."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.mid = nn.Linear(8, 8)
        self.mid.weight = self.inp.weight
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(torch.tanh(self.mid(torch.tanh(self.inp(x)))))


class _Transformer(nn.Module):
    """One self-attention block (packed ``in_proj_weight``) over 4 tokens."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 4 * 16)
        self.block = nn.TransformerEncoderLayer(
            16, nhead=2, dim_feedforward=32, dropout=0.0, batch_first=True
        )
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        return self.out(self.block(self.inp(x).view(x.shape[0], 4, 16)).mean(dim=1))


class _FusedExperts(nn.Module):
    """A fused ``[experts, in, out]`` weight stack, as MoE layers store them."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.experts = nn.Parameter(torch.randn(3, 16, 16) / 4)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        hidden = torch.einsum("bi,eio->bo", torch.tanh(self.inp(x)), self.experts) / 3
        return self.out(hidden)


def _qwen2_model():
    """A two-layer random-init HF Qwen2 with tied embeddings."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(config)


def _qwen2_loss(model, step, micro, rank):
    generator = torch.Generator().manual_seed(1000 * step + 10 * micro + rank)
    ids = torch.randint(0, 32, (2, 6), generator=generator)
    return model(input_ids=ids, labels=ids, use_cache=False).loss


def _seeded(build):
    def make():
        torch.manual_seed(0)
        return build()

    return make


# name -> (model factory, loss); workers receive the name, so nothing is pickled.
_CASES = {
    "mlp": (_model, _mse_loss),
    "frozen": (_frozen_model, _mse_loss),
    "routed": (_seeded(_Routed), _routed_loss),
    "tied": (_seeded(_Tied), _mse_loss),
    "transformer": (_seeded(_Transformer), _mse_loss),
    "fused_experts": (_seeded(_FusedExperts), _mse_loss),
    "qwen2": (_seeded(_qwen2_model), _qwen2_loss),
}


def _reference_train(case, steps, build_optimizer):
    """One process stepping on the mean gradient of every rank's micro-batches."""
    make, loss_fn = _CASES[case]
    world_size = dist.get_world_size()
    reference = make()
    optimizer = build_optimizer(reference)
    norms = []
    for step in range(steps):
        optimizer.zero_grad()
        for micro in range(GRAD_ACCUM):
            for source_rank in range(world_size):
                loss = loss_fn(reference, step, micro, source_rank)
                (loss / (GRAD_ACCUM * world_size)).backward()
        grads = [p.grad for p in reference.parameters() if p.grad is not None]
        norms.append(float(torch.norm(torch.stack([g.norm() for g in grads]))))
        optimizer.step()
    return reference, norms


def _reference_adamw(module):
    return torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad], lr=LR, foreach=False
    )


def _key_bias_slice(name):
    """Elements of an attention key bias, or ``None`` for any other tensor.

    The softmax ignores a shift shared by every key, so the key bias gets a
    (near-)zero gradient and AdamW turns reduction-order roundoff into lr-sized
    steps. Those elements are covered by ``_assert_replicas_agree`` only.
    """
    if name.endswith("in_proj_bias"):
        return slice(16, 32)  # packed [q | k | v] with d_model 16
    if name.endswith("k_proj.bias"):
        return slice(None)
    return None


def _assert_matches(model, reference):
    for (name, got), want in zip(model.named_parameters(), reference.parameters()):
        got, want = got.detach().clone(), want.detach().clone()
        skip = _key_bias_slice(name)
        if skip is not None:
            got[skip] = 0
            want[skip] = 0
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, msg=name)


def _check_frozen(model, optimizer, initial):
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"0.weight", "0.bias"}
    parameters = dict(model.named_parameters())
    for name in frozen:
        assert torch.equal(parameters[name], initial[name]), name
    assert not frozen & set(optimizer.master)
    owners = {p["name"]: p["owner_rank"] for p in optimizer.parameter_manifest()["parameters"]}
    assert all(owners[name] is None for name in frozen)


def _check_routed(model, optimizer, initial):
    parameters = dict(model.named_parameters())
    for name in ("experts.2.weight", "experts.2.bias"):
        assert torch.equal(parameters[name], initial[name]), name


def _check_tied(model, optimizer, initial):
    assert model.mid.weight is model.inp.weight
    names = [p["name"] for p in optimizer.parameter_manifest()["parameters"]]
    assert "inp.weight" in names and "mid.weight" not in names


_CHECKS = {"frozen": _check_frozen, "routed": _check_routed, "tied": _check_tied}


# -- data-parallel equivalence ------------------------------------------------


def _worker_matches_reference(rank, case, overlap):
    make, loss_fn = _CASES[case]
    model = make()
    initial = _replicas(model)
    optimizer = _build(model, _adamw_factory, grad_overlap=overlap, param_overlap=overlap)
    norms = _train(model, optimizer, steps=3, clip=0.0, loss_fn=loss_fn)

    reference, ref_norms = _reference_train(case, 3, _reference_adamw)
    assert norms == pytest.approx(ref_norms, rel=1e-5)
    _assert_matches(model, reference)
    _assert_replicas_agree(model)
    if case in _CHECKS:
        _CHECKS[case](model, optimizer, initial)


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("case", ["mlp", "frozen", "routed", "tied", "transformer"])
def test_matches_single_process_reference(case, overlap):
    _run(_worker_matches_reference, case, overlap)


@pytest.mark.parametrize("case", ["mlp", "routed"])
def test_matches_reference_on_three_ranks(case):
    # Three ranks over six or ten tensors: owners carry unequal loads.
    _run(_worker_matches_reference, case, True, world_size=3)


def test_matches_reference_on_hf_qwen2():
    pytest.importorskip("transformers")
    _run(_worker_matches_reference, "qwen2", True)


# -- Muon route: 3-D expert stacks, bf16 compute, LR scheduler -----------------


def _split_by_ndim(module):
    muon, adamw = [], []
    for name, parameter in module.named_parameters():
        (muon if parameter.ndim in (2, 3) else adamw).append((name, parameter))
    return muon, adamw


def _reference_muon(module):
    muon, adamw = _split_by_ndim(module)
    return build_muon_optimizer(muon, lr=LR, adamw_parameters=adamw)


def _worker_muon_3d(rank):
    model = _CASES["fused_experts"][0]()
    # fp32 compute and collectives, so the reduction order is the only
    # difference from the reference; matrices and the 3-D stack go to Muon.
    policy = ParameterClassPolicy(compute_fp32_markers=("inp", "experts", "out"))
    optimizer = _build(model, _muon_factory, policy)
    assert "experts" in optimizer.parameter_plan.muon_parameter_names()
    norms = _train(model, optimizer, steps=3, clip=0.0)

    reference, ref_norms = _reference_train("fused_experts", 3, _reference_muon)
    assert norms == pytest.approx(ref_norms, rel=1e-5)
    _assert_matches(model, reference)
    _assert_replicas_agree(model)


def test_muon_route_handles_3d_expert_stacks():
    _run(_worker_muon_3d)


def _worker_muon_bf16(rank):
    model = _model()
    # No markers: bf16 compute replicas, matrices on Muon, 1-D tensors on AdamW.
    optimizer = _build(
        model, _muon_factory, ParameterClassPolicy(),
        gradient_reduce_dtype="bf16", parameter_sync_precision="bf16",
    )
    assert optimizer.in_step_names == set(optimizer.parameter_plan.muon_parameter_names())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
    before = _replicas(model)
    for step in range(3):
        _train(model, optimizer, steps=1, start=step)
        scheduler.step()
    assert all(group["lr"] == LR * 0.125 for group in optimizer.optim.param_groups)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    for name, master in optimizer.master.items():
        replica = dict(model.named_parameters())[name]
        assert torch.equal(master.to(replica.dtype), replica.detach()), name
    _assert_replicas_agree(model)


def test_muon_route_publishes_bf16_replicas():
    _run(_worker_muon_bf16)


# -- rank-local checkpoint round trip ------------------------------------------


def _checkpoint_ctx(rank):
    return SimpleNamespace(
        rank=rank, world_size=dist.get_world_size(), is_main=rank == 0, barrier=dist.barrier
    )


def _worker_checkpoint(rank, path):
    ctx = _checkpoint_ctx(rank)
    model = _model()
    optimizer = _build(model, _muon_factory, ParameterClassPolicy())
    _train(model, optimizer, steps=2)
    checkpoint._save_zero_optimizer_state(path, optimizer, ctx)
    _train(model, optimizer, steps=2, start=2)
    uninterrupted = _replicas(model)

    resumed_model = _model()
    resumed = _build(resumed_model, _muon_factory, ParameterClassPolicy())
    checkpoint._load_zero_optimizer_state(path, resumed, ctx)
    # Replicas come from the published masters, not from the module state.
    _train(resumed_model, resumed, steps=2, start=2)
    for name, parameter in resumed_model.named_parameters():
        assert torch.equal(parameter.detach(), uninterrupted[name]), name


def test_checkpoint_round_trip_is_bitwise():
    with tempfile.TemporaryDirectory() as path:
        _run(_worker_checkpoint, path)


def _worker_rejects_other_precision(rank, path):
    ctx = _checkpoint_ctx(rank)
    kwargs = dict(gradient_reduce_dtype="bf16")
    model = _model()
    saved = _build(model, _muon_factory, ParameterClassPolicy(), parameter_sync_precision="bf16", **kwargs)
    _train(model, saved, steps=1)
    checkpoint._save_zero_optimizer_state(path, saved, ctx)

    other = _build(_model(), _muon_factory, ParameterClassPolicy(), parameter_sync_precision="fp32", **kwargs)
    with pytest.raises(RuntimeError, match="wire configuration mismatch"):
        checkpoint._load_zero_optimizer_state(path, other, ctx)


def test_checkpoint_rejects_a_different_precision_plan():
    with tempfile.TemporaryDirectory() as path:
        _run(_worker_rejects_other_precision, path)


# -- startup validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(gradient_reduce_dtype="fp16"), "gradient_reduce_dtype"),
        (dict(parameter_sync_precision="int8"), "parameter_sync_precision"),
        (dict(grad_inflight_mb=0), "grad_inflight_mb"),
        (dict(bucket_mb=-1), "bucket_mb"),
    ],
)
def test_collective_config_rejects_invalid_plans(overrides, match):
    with pytest.raises(ValueError, match=match):
        CollectiveConfig(**overrides).validate()


def test_policy_rejects_a_bare_string_marker():
    with pytest.raises(ValueError, match="sequence of non-empty strings"):
        ParameterClassPolicy(compute_fp32_markers="inp")


def _training_args(**overrides):
    args = dict(
        trainer_type="LingbotVlaV2Trainer",
        optimizer="AdamW",
        distributed_strategy="ddp",
        zero_optimizer=False,
        fp8=False,
        save_interval=0,
        save_training_state=False,
        use_lora=False,
        save_format="dcp",
    )
    args.update({name: None for name in validation._DDP_ONLY_ARGS})
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(optimizer="muon"),
        dict(save_interval=10, save_training_state=True),
        # Other trainers are not checked beyond the Muon rule.
        dict(trainer_type="FinetuneTrainer", distributed_strategy="fsdp", fp8=True),
    ],
)
def test_validation_accepts_supported_combinations(overrides):
    validation.validate_tensor_wise_args(_training_args(**overrides))


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(trainer_type="FinetuneTrainer", optimizer="muon"), "requires a tensor_wise trainer"),
        (dict(distributed_strategy="fsdp"), "requires --distributed-strategy=ddp"),
        (dict(zero_optimizer=True), "incompatible with --zero-optimizer"),
        (dict(ddp_bucket_cap_mb=25), "remove: --ddp-bucket-cap-mb"),
        (dict(fp8=True), "--fp8 is not supported"),
        (dict(save_interval=10, save_training_state=True, save_format="hf"), "--save-format=dcp"),
        (dict(save_interval=10, save_training_state=True, use_lora=True), "--save-format=dcp"),
    ],
)
def test_validation_rejects_unsupported_combinations(overrides, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        validation.validate_tensor_wise_args(_training_args(**overrides))


# -- fp8 codec ------------------------------------------------------------------


@pytest.mark.parametrize("numel", [1, 127, 128, 129, 1000])
def test_block_rows_pad_to_whole_scale_blocks(numel):
    from loongforge.embodied.optimizer.tensor_wise_optimizer import parameter_sync as sync

    rows = sync._as_block_rows(torch.arange(numel, dtype=torch.float32) + 1)
    assert rows.shape == (sync._fp8_scale_count(numel), sync.FP8_BLOCK)
    assert rows.reshape(-1)[:numel].tolist() == list(range(1, numel + 1))
    assert not rows.reshape(-1)[numel:].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 codec needs CUDA")
@pytest.mark.parametrize("numel", [1, 129, 1000, 1 << 20])
def test_fp8_codec_round_trip_keeps_wire_sizes(numel):
    from loongforge.embodied.optimizer.tensor_wise_optimizer import parameter_sync as sync

    if not sync._fp8_is_available():
        pytest.skip("TransformerEngine blockwise quantizer unavailable")
    x = torch.randn(numel, device="cuda") * 1e-3
    data, scales = sync._quantize_flat(x)
    assert data.numel() == numel and scales.numel() == sync._fp8_scale_count(numel)
    decoded = sync._dequantize_flat(data, scales, numel)
    assert decoded.shape == x.shape
    assert (decoded - x).norm() <= 0.05 * x.norm()
