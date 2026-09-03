# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CLI, install, and kernel tests for FSDP2 delta-FP8 AllGather."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param as _fsdp_param
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _param_group

from loongforge.embodied.distributed import delta_fp8_allgather as delta_mod
from loongforge.embodied.distributed import delta_fp8_comm
from loongforge.embodied.distributed.delta_fp8_comm import triton as delta_triton
from loongforge.embodied.train.training_args import TrainingArgs, build_arg_parser
from loongforge.embodied.train.validators import validate


def _training_args(**overrides):
    return replace(TrainingArgs(), **overrides)


class _RegistryKey:
    pass


def _fake_fsdp_model(key):
    param_group = SimpleNamespace(
        fsdp_params=[key],
        device=torch.device("cuda"),
        _all_gather_process_group=object(),
    )
    state = SimpleNamespace(_fsdp_param_groups=[param_group])
    module = SimpleNamespace(_get_fsdp_state=lambda: state)
    return SimpleNamespace(modules=lambda: iter([module, module]))


def test_delta_fp8_cli_defaults_off():
    parser = build_arg_parser()
    args = parser.parse_args(["--model-name", "dreamzero_full_wan22_5b"])
    assert not hasattr(args, "fsdp_delta_fp8_allgather")
    defaults = TrainingArgs()
    assert defaults.fsdp_delta_fp8_allgather is False
    assert defaults.fsdp_delta_fp8_block == 256
    assert defaults.fsdp_delta_fp8_prime_steps == 1
    assert defaults.fsdp_delta_fp8_reprime_interval == 0


def test_delta_fp8_cli_opt_in():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "dreamzero_full_wan22_5b",
            "--fsdp-delta-fp8-allgather",
            "--fsdp-delta-fp8-block",
            "128",
            "--fsdp-delta-fp8-prime-steps",
            "2",
            "--fsdp-delta-fp8-reprime-interval",
            "8",
        ]
    )
    assert args.fsdp_delta_fp8_allgather is True
    assert args.fsdp_delta_fp8_block == 128
    assert args.fsdp_delta_fp8_prime_steps == 2
    assert args.fsdp_delta_fp8_reprime_interval == 8


def test_delta_fp8_cli_opt_out_flag():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "dreamzero_full_wan22_5b",
            "--no-fsdp-delta-fp8-allgather",
        ]
    )
    assert args.fsdp_delta_fp8_allgather is False


@pytest.mark.parametrize("block", [0, -1, 3, 255, 1 << 21])
def test_validate_config_rejects_invalid_triton_blocks(block):
    with pytest.raises(ValueError, match="fsdp_delta_fp8_block"):
        delta_mod._validate_config(block, prime_steps=1, reprime_interval=0)


@pytest.mark.parametrize("block", [1, 256, 1 << 20])
def test_validate_config_accepts_power_of_two_triton_blocks(block):
    delta_mod._validate_config(block, prime_steps=1, reprime_interval=0)


def test_registration_preserves_per_group_scope_and_config(monkeypatch):
    monkeypatch.setattr(delta_mod, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(delta_mod.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(delta_mod, "install_delta_fp8_allgather", lambda **kwargs: None)
    delta_mod._GROUP_CONFIGS.clear()
    try:
        first_key = _RegistryKey()
        second_key = _RegistryKey()
        assert delta_mod.register_delta_fp8_allgather(
            _fake_fsdp_model(first_key), block=256
        ) == 1
        assert delta_mod.register_delta_fp8_allgather(
            _fake_fsdp_model(second_key), block=512
        ) == 1

        assert delta_mod._GROUP_CONFIGS[first_key].block == 256
        assert delta_mod._GROUP_CONFIGS[second_key].block == 512
        assert delta_mod._MODEL_SCOPED is True
    finally:
        delta_mod._GROUP_CONFIGS.clear()
        delta_mod._MODEL_SCOPED = False


def test_model_scoped_hook_falls_back_for_unregistered_group(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(delta_mod, "_ORIGINAL_FOREACH_ALL_GATHER", lambda *args: sentinel)
    monkeypatch.setattr(delta_mod, "_MODEL_SCOPED", True)
    delta_mod._GROUP_CONFIGS.clear()

    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    result = delta_mod._delta_foreach_all_gather(
        [_RegistryKey()], group, True, None, None, None, delta_mod.DefaultAllGather()
    )
    assert result is sentinel


@pytest.mark.parametrize(
    ("device", "backend", "message"),
    [
        (torch.device("cpu"), "gloo", "requires a CUDA device"),
        (torch.device("cuda"), "gloo", "requires the NCCL backend"),
    ],
)
def test_runtime_validation_rejects_unsupported_device_or_backend(
    device, backend, message
):
    with pytest.raises(RuntimeError, match=message):
        delta_fp8_comm.validate_runtime(device, backend)


def test_runtime_validation_rejects_missing_triton_fp8_type(monkeypatch):
    monkeypatch.setattr(delta_fp8_comm, "triton", object())
    monkeypatch.setattr(delta_fp8_comm, "tl", object())
    with pytest.raises(RuntimeError, match="Triton FP8 type"):
        delta_fp8_comm.validate_runtime(torch.device("cuda"), "nccl")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("nccl", "nccl"), ("cpu:gloo,cuda:nccl", "nccl"), ("cuda:gloo", "gloo")],
)
def test_backend_for_device_resolves_device_scoped_backend(backend, expected):
    assert delta_fp8_comm._backend_for_device(backend, "cuda") == expected


def test_shared_scratch_reuses_and_grows_views():
    stream = object()
    device = torch.device("cpu")
    delta_mod._SCRATCH_STATES.clear()
    try:
        first = delta_mod._scratch_for(stream, 8, 2, 2, device)
        second = delta_mod._scratch_for(stream, 4, 2, 1, device)
        assert [tensor.data_ptr() for tensor in first] == [
            tensor.data_ptr() for tensor in second
        ]
        larger = delta_mod._scratch_for(stream, 16, 2, 4, device)
        assert larger[0].numel() == 16
        assert larger[1].numel() == 32
        assert larger[2].numel() == 4
        assert larger[3].numel() == 8
    finally:
        delta_mod._SCRATCH_STATES.clear()


def test_flat_input_alias_requires_one_adjacent_contiguous_storage():
    flat = torch.arange(12, dtype=torch.bfloat16)
    inputs = [flat[:3], flat[3:8], flat[8:]]
    alias = delta_mod._as_contiguous_flat_input(inputs, flat.numel())
    assert alias is not None
    assert alias.data_ptr() == flat.data_ptr()
    assert torch.equal(alias, flat)

    disjoint = [flat[:3], flat[4:]]
    assert delta_mod._as_contiguous_flat_input(disjoint, 11) is None
    assert delta_mod._as_contiguous_flat_input([flat[::2]], 6) is None


def test_reuse_flat_input_avoids_persistent_shard_buffer():
    state = delta_mod._GroupState(
        shard_numel=8,
        world_size=2,
        device=torch.device("cpu"),
        block=4,
    )
    flat = torch.arange(8, dtype=torch.bfloat16)
    shard_input, reused = delta_mod._get_shard_input(
        state, [flat[:3], flat[3:]], [3, 5], torch.device("cpu")
    )
    assert reused is True
    assert state.shard_buffer is None
    assert shard_input.data_ptr() == flat.data_ptr()


def test_param_major_metadata():
    metadata = delta_mod._build_param_major_block_metadata(
        (3, 5), world_size=2, block=4, device=torch.device("cpu")
    )
    assert metadata.tolist() == [[0, 0, 3], [3, 6, 5], [7, 10, 5]]


def test_direct_param_prime_preserves_layout_async_work_and_scratch():
    calls = []

    class _FakeWork:
        def __init__(self, index):
            self.index = index

        def block_current_stream(self):
            calls.append(("block", self.index))

        def wait(self):
            calls.append(("wait", self.index))

    def fake_all_gather(*, output_tensor, input_tensor, group, async_op):
        index = sum(call[0] == "gather" for call in calls)
        calls.append(("gather", output_tensor, input_tensor, group, async_op))
        shard_numel = input_tensor.numel()
        for rank in range(2):
            output_tensor.narrow(0, rank * shard_numel, shard_numel).copy_(
                input_tensor + rank * 100
            )
        return _FakeWork(index)

    shard_input = torch.tensor(
        [10, 11, 12, 20, 21, 22, 23, 24], dtype=torch.bfloat16
    )
    parameter_major = torch.empty(16, dtype=torch.bfloat16)
    delta_mod._SCRATCH_STATES.clear()
    work = delta_mod._launch_param_major_prime(
        fake_all_gather,
        object(),
        shard_input,
        parameter_major,
        (3, 5),
        world_size=2,
        async_op=True,
    )

    assert work is not None
    gather_calls = [call for call in calls if call[0] == "gather"]
    assert all(call[4] is True for call in gather_calls)
    assert [call[3] for call in gather_calls] == [gather_calls[0][3]] * 2
    assert [call[2].tolist() for call in gather_calls] == [
        [10, 11, 12],
        [20, 21, 22, 23, 24],
    ]
    assert parameter_major.tolist() == [
        10,
        11,
        12,
        110,
        111,
        112,
        20,
        21,
        22,
        23,
        24,
        120,
        121,
        122,
        123,
        124,
    ]
    assert [(call[0], call[1]) for call in calls if call[0] == "block"] == [
        ("block", 0),
        ("block", 1),
    ]
    assert not any(call[0] == "wait" for call in calls)
    assert not delta_mod._SCRATCH_STATES


def test_aliased_group_state_owns_fsdp_outputs_and_skips_free(monkeypatch):
    params = [SimpleNamespace(all_gather_outputs=[]) for _ in range(2)]
    state = delta_mod._GroupState(
        shard_numel=8,
        world_size=2,
        device=torch.device("cpu"),
        block=4,
        fsdp_params=params,
        param_shard_numels=(3, 5),
    )
    try:
        assert state.reference.numel() == 16
        assert [output[0].numel() for output in (p.all_gather_outputs for p in params)] == [
            6,
            10,
        ]
        assert all(
            p.all_gather_outputs[0].untyped_storage().data_ptr()
            == state.reference.untyped_storage().data_ptr()
            for p in params
        )
        calls = []
        monkeypatch.setattr(
            delta_mod,
            "_ORIGINAL_FREE_UNSHARDED_PARAM",
            lambda fsdp_param: calls.append(fsdp_param),
        )
        delta_mod._delta_free_unsharded_param(params[0])
        assert calls == []
        params[0]._delta_fp8_persistent_reference = False
        delta_mod._delta_free_unsharded_param(params[0])
        assert calls == [params[0]]
    finally:
        delta_mod._ALIASED_RESULTS.pop(id(state.reference), None)


def test_delta_collectives_propagate_async_without_host_wait(monkeypatch):
    calls = []

    class _FakeWork:
        def __init__(self, name):
            self.name = name
            self.blocked = False
            self.waited = False

        def block_current_stream(self):
            self.blocked = True
            calls.append(f"block:{self.name}")

        def wait(self):
            self.waited = True
            calls.append(f"wait:{self.name}")

    payload_work = _FakeWork("payload")
    scale_work = _FakeWork("scale")

    def fake_payload(*, output_tensor, input_tensor, group, async_op):
        calls.append(("payload", async_op))
        return payload_work

    def fake_scale(output_tensor, input_tensor, *, group, async_op):
        calls.append(("scale", async_op))
        return scale_work

    monkeypatch.setattr(delta_mod.dist, "all_gather_into_tensor", fake_scale)
    result = delta_mod._launch_delta_collectives(
        fake_payload,
        torch.empty(8, dtype=torch.uint8),
        torch.empty(4, dtype=torch.uint8),
        torch.empty(2, dtype=torch.float32),
        torch.empty(1, dtype=torch.float32),
        group=object(),
        async_op=True,
    )

    assert result is payload_work
    assert calls == [("payload", True), ("scale", True), "block:payload", "block:scale"]
    assert not payload_work.waited
    assert not scale_work.waited


def test_validate_rejects_delta_fp8_without_fsdp():
    training_args = _training_args(
        distributed_strategy="ddp",
        fsdp_delta_fp8_allgather=True,
    )
    with pytest.raises(ValueError, match="--fsdp-delta-fp8-allgather requires"):
        validate(training_args, SimpleNamespace(model_type="dreamzero"), SimpleNamespace())


def test_validate_rejects_non_positive_delta_fp8_block():
    training_args = _training_args(
        distributed_strategy="fsdp",
        fsdp_delta_fp8_allgather=True,
        fsdp_delta_fp8_block=0,
    )
    with pytest.raises(ValueError, match="--fsdp-delta-fp8-block must be .*positive"):
        validate(training_args, SimpleNamespace(model_type="dreamzero"), SimpleNamespace())


@pytest.mark.skipif(delta_triton is None, reason="delta-FP8 install requires Triton")
def test_install_and_uninstall_patches_foreach_all_gather():
    original = _collectives.foreach_all_gather
    original_copy_out = _collectives.foreach_all_gather_copy_out
    original_free = _fsdp_param.FSDPParam.free_unsharded_param
    original_init = _fsdp_param.FSDPParam.init_unsharded_param
    try:
        delta_mod.install_delta_fp8_allgather(
            block=256,
            prime_steps=1,
            reprime_interval=0,
        )
        assert _collectives.foreach_all_gather is delta_mod._delta_foreach_all_gather
        assert _param_group.foreach_all_gather is delta_mod._delta_foreach_all_gather
        assert (
            _collectives.foreach_all_gather_copy_out
            is delta_mod._delta_foreach_all_gather_copy_out
        )
        assert _fsdp_param.FSDPParam.free_unsharded_param is delta_mod._delta_free_unsharded_param
        assert _fsdp_param.FSDPParam.init_unsharded_param is delta_mod._delta_init_unsharded_param
        delta_mod.install_delta_fp8_allgather(block=128, prime_steps=2, reprime_interval=4)
        assert delta_mod._CONFIG["block"] == 128
        assert delta_mod._CONFIG["prime_steps"] == 2
        assert delta_mod._CONFIG["reprime_interval"] == 4
    finally:
        delta_mod.uninstall_delta_fp8_allgather()
    assert _collectives.foreach_all_gather is original
    assert _param_group.foreach_all_gather is original
    assert _collectives.foreach_all_gather_copy_out is original_copy_out
    assert _param_group.foreach_all_gather_copy_out is original_copy_out
    assert _fsdp_param.FSDPParam.free_unsharded_param is original_free
    assert _fsdp_param.FSDPParam.init_unsharded_param is original_init
    assert delta_mod._INSTALLED is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="delta-FP8 kernels require CUDA")
def test_quantize_dequantize_round_trip():
    from loongforge.embodied.distributed.delta_fp8_comm import dequantize_add, quantize_delta

    torch.manual_seed(0)
    device = "cuda"
    numel, world_size = 4096, 1
    delta = (torch.randn(numel, device=device) * 1.0e-5).to(torch.bfloat16)
    base = (torch.randn(numel, device=device) * 0.02).to(torch.bfloat16)
    target = (base.to(torch.float32) + delta.to(torch.float32)).to(torch.bfloat16)
    y = base.clone()
    quantized, scales = quantize_delta(target, y)
    dequantize_add(y, quantized, scales, numel, world_size)
    applied = y.to(torch.float32) - base.to(torch.float32)
    want = target.to(torch.float32) - base.to(torch.float32)
    amax = want.abs().reshape(-1, 256).amax(dim=1, keepdim=True).expand(-1, 256).reshape(-1)
    tolerance = amax / 16.0 + target.to(torch.float32).abs() * (2.0 ** -8)
    assert int(((applied - want).abs() > tolerance).sum()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="delta-FP8 kernels require CUDA")
def test_error_feedback_does_not_drift():
    from loongforge.embodied.distributed.delta_fp8_comm import dequantize_add, quantize_delta

    torch.manual_seed(42)
    device = "cuda"
    world_size = 8
    shard_numel = (1 << 20) + 384
    steps = 20
    numel = shard_numel * world_size
    master = torch.randn(numel, device=device, dtype=torch.float32) * 0.02
    y = master.to(torch.bfloat16).clone()
    block = 256
    num_blocks = (shard_numel + block - 1) // block
    quantized = torch.empty(numel, dtype=torch.uint8, device=device)
    scales = torch.empty(num_blocks * world_size, dtype=torch.float32, device=device)
    lr = 1.0e-5
    worst_rms = 0.0
    for _ in range(steps):
        master = master - lr * torch.randn_like(master)
        exact = master.to(torch.bfloat16)
        for rank in range(world_size):
            lo, hi = rank * shard_numel, (rank + 1) * shard_numel
            q_r, s_r = quantize_delta(exact[lo:hi].contiguous(), y[lo:hi].contiguous())
            quantized[lo:hi].copy_(q_r)
            scales[rank * num_blocks : (rank + 1) * num_blocks].copy_(s_r)
        dequantize_add(y, quantized, scales, shard_numel, world_size)
        error = y.to(torch.float32) - exact.to(torch.float32)
        worst_rms = max(worst_rms, float(error.norm() / exact.to(torch.float32).norm()))
    exact = master.to(torch.bfloat16)
    final_rms = float(
        (y.to(torch.float32) - exact.to(torch.float32)).norm()
        / exact.to(torch.float32).norm()
    )
    assert torch.isfinite(y).all()
    assert worst_rms < 1.0e-4
    assert final_rms < 1.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="delta-FP8 kernels require CUDA")
def test_param_major_error_feedback_updates_fused_fsdp_storage():
    from loongforge.embodied.distributed.delta_fp8_comm import (
        dequantize_add_param_major,
        quantize_delta_param_major_into,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    world_size = 2
    param_numels = (259, 513)
    shard_numel = sum(param_numels)
    metadata = delta_mod._build_param_major_block_metadata(
        param_numels, world_size, 256, device
    )
    reference = torch.randn(shard_numel * world_size, device=device).to(torch.bfloat16)
    target = reference.clone()
    local_inputs = []
    reference_offset = 0
    for rank in range(world_size):
        pieces = []
        reference_offset = 0
        for param_numel in param_numels:
            piece = target.narrow(
                0, reference_offset + rank * param_numel, param_numel
            )
            piece.add_(
                (torch.randn_like(piece, dtype=torch.float32) * 1.0e-4).to(
                    torch.bfloat16
                )
            )
            pieces.append(piece.clone())
            reference_offset += param_numel * world_size
        local_inputs.append(torch.cat(pieces))

    quantized_all = torch.empty(
        shard_numel * world_size, dtype=torch.uint8, device=device
    )
    scales_all = torch.empty(
        metadata.shape[0] * world_size, dtype=torch.float32, device=device
    )
    for rank, local_input in enumerate(local_inputs):
        quantize_delta_param_major_into(
            local_input,
            reference,
            quantized_all.narrow(0, rank * shard_numel, shard_numel),
            scales_all.narrow(0, rank * metadata.shape[0], metadata.shape[0]),
            metadata,
            rank,
        )
    dequantize_add_param_major(
        reference,
        quantized_all,
        scales_all,
        metadata,
        shard_numel,
        world_size,
    )
    error = (reference.float() - target.float()).abs()
    assert torch.isfinite(reference).all()
    assert float(error.max()) < 2.0e-3


def _run_aliased_fsdp_training(rank, world_size, init_file):
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(11)
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 8),
        ).cuda()
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
        fully_shard(model[0], reshard_after_forward=False, mp_policy=mp_policy)
        fully_shard(model[2], reshard_after_forward=False, mp_policy=mp_policy)
        fully_shard(model, reshard_after_forward=False, mp_policy=mp_policy)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
        delta_mod.install_delta_fp8_allgather()

        inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            prediction = model(inputs)
            for state in delta_mod._STATES.values():
                for fsdp_param in state.fsdp_params:
                    output = fsdp_param.all_gather_outputs[0]
                    unsharded = fsdp_param._unsharded_param
                    local_unsharded = getattr(unsharded, "_local_tensor", unsharded)
                    assert (
                        local_unsharded.untyped_storage().data_ptr()
                        == output.untyped_storage().data_ptr()
                    )
                    assert local_unsharded.storage_offset() == output.storage_offset()
            loss = (prediction - target).float().square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for state in delta_mod._STATES.values():
                if state.block_metadata is None:
                    continue
                assert state.reference.untyped_storage().nbytes() > 0
                reference_offset = 0
                for fsdp_param, param_numel in zip(
                    state.fsdp_params, state.param_shard_numels
                ):
                    local_reference = state.reference.narrow(
                        0,
                        reference_offset + rank * param_numel,
                        param_numel,
                    )
                    expected = fsdp_param._sharded_param_data.to(torch.bfloat16)
                    assert float((local_reference - expected).abs().max()) < 2.0e-2
                    reference_offset += param_numel * world_size
        assert delta_mod._STATES
        assert all(
            state.block_metadata is not None for state in delta_mod._STATES.values()
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="aliased FSDP lifecycle test requires two CUDA devices",
)
def test_aliased_reference_survives_fsdp_optimizer_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.multiprocessing.spawn(
            _run_aliased_fsdp_training,
            args=(2, f"{tmpdir}/init"),
            nprocs=2,
            join=True,
        )
