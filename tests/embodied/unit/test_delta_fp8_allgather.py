# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for model-scoped FSDP2 Delta-FP8 registration."""

import gc
import os
import weakref
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from loongforge.embodied.distributed import delta_fp8_allgather as delta_fp8


class _Key:
    pass


class _ParamGroup:
    def __init__(self, key):
        self.fsdp_params = [key]
        self.device = torch.device("cuda")
        self._all_gather_process_group = object()


class _State:
    def __init__(self, key):
        self._fsdp_param_groups = [_ParamGroup(key)]


class _Module:
    def __init__(self, state):
        self._state = state

    def _get_fsdp_state(self):
        return self._state


class _Model:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


class _FakeStream:
    def __init__(self, calls):
        self.calls = calls

    def wait_stream(self, stream):
        self.calls.append(("wait_stream", stream))

    def record_event(self):
        self.calls.append(("record_event",))
        return "event"


class _FakeDeviceHandle:
    def __init__(self, calls):
        self.calls = calls

    def stream(self, stream):
        self.calls.append(("stream", stream))
        return nullcontext()


class _FakeWork(dist.distributed_c10d.Work):
    def __init__(self, calls, name):
        super().__init__()
        self.calls = calls
        self.name = name

    def wait(self, timeout=None):
        self.calls.append(("wait", self.name))
        return None

    def is_completed(self):
        return True


@pytest.fixture(autouse=True)
def _clear_registries():
    delta_fp8._GROUP_CONFIGS.clear()
    delta_fp8._STATES.clear()
    yield
    delta_fp8._GROUP_CONFIGS.clear()
    delta_fp8._STATES.clear()


@pytest.mark.parametrize("block", [0, -1, 3, 255, 1 << 21])
def test_validate_config_rejects_invalid_triton_blocks(block):
    with pytest.raises(ValueError, match="fsdp_delta_fp8_block"):
        delta_fp8._validate_config(block, prime_steps=1, reprime_interval=0)


@pytest.mark.parametrize("block", [1, 256, 1 << 20])
def test_validate_config_accepts_power_of_two_triton_blocks(block):
    delta_fp8._validate_config(block, prime_steps=1, reprime_interval=0)


def test_registration_is_scoped_per_fsdp_group(monkeypatch):
    monkeypatch.setattr(delta_fp8, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(delta_fp8.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(delta_fp8, "_install_delta_fp8_allgather", lambda: None)

    first_key = _Key()
    first_state = _State(first_key)
    first_module = _Module(first_state)
    # A grouped FSDP unit exposes the same state from each member module.
    first_model = _Model(first_module, first_module)
    assert delta_fp8.register_delta_fp8_allgather(first_model, block=256) == 1

    second_key = _Key()
    second_model = _Model(_Module(_State(second_key)))
    assert delta_fp8.register_delta_fp8_allgather(second_model, block=512) == 1

    assert delta_fp8._GROUP_CONFIGS[first_key].block == 256
    assert delta_fp8._GROUP_CONFIGS[second_key].block == 512
    assert _Key() not in delta_fp8._GROUP_CONFIGS


def test_unregistered_group_falls_back_to_native_allgather(monkeypatch):
    sentinel = object()
    calls = []

    def original(*args):
        calls.append(args)
        return sentinel

    class Group:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 0

    monkeypatch.setattr(delta_fp8, "_ORIGINAL_FOREACH_ALL_GATHER", original)
    result = delta_fp8._delta_foreach_all_gather(
        [_Key()], Group(), False, None, None, None, None
    )
    assert result is sentinel
    assert len(calls) == 1


def test_group_state_is_released_with_fsdp_param(monkeypatch):
    monkeypatch.setattr(delta_fp8, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(delta_fp8.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(delta_fp8, "_install_delta_fp8_allgather", lambda: None)

    key = _Key()
    state = _State(key)
    module = _Module(state)
    model = _Model(module)
    delta_fp8.register_delta_fp8_allgather(model)
    delta_fp8._STATES[key] = object()
    key_ref = weakref.ref(key)

    del key, state, module, model
    gc.collect()

    assert key_ref() is None
    assert len(delta_fp8._GROUP_CONFIGS) == 0
    assert len(delta_fp8._STATES) == 0


def test_async_delta_collectives_propagate_work_and_preserve_order(monkeypatch):
    calls = []
    key = _Key()
    state = type("State", (), {})()
    state.gathers = 1
    state.reference = torch.zeros(4, dtype=torch.bfloat16)
    state.shard_buffer = torch.ones(2, dtype=torch.bfloat16)
    state.quantized_local = torch.empty(2, dtype=torch.uint8)
    state.quantized_all = torch.empty(4, dtype=torch.uint8)
    state.scales_local = torch.empty(1, dtype=torch.float32)
    state.scales_all = torch.empty(2, dtype=torch.float32)
    _GROUP_CONFIGS = delta_fp8._GROUP_CONFIGS
    _GROUP_CONFIGS[key] = delta_fp8._GroupConfig(2, 1, 0)

    class Group:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 0

    comm = delta_fp8.DefaultAllGather()
    payload_work = _FakeWork(calls, "payload")
    scale_work = _FakeWork(calls, "scales")

    def fake_payload(self, **kwargs):
        calls.append(("payload", kwargs["async_op"]))
        return payload_work

    def fake_scales(*args, **kwargs):
        calls.append(("scales", kwargs["async_op"]))
        return scale_work

    monkeypatch.setattr(delta_fp8.DefaultAllGather, "__call__", fake_payload)
    monkeypatch.setattr(delta_fp8.dist, "all_gather_into_tensor", fake_scales)
    monkeypatch.setattr(delta_fp8, "_get_device_handle", lambda _: _FakeDeviceHandle(calls))
    monkeypatch.setattr(delta_fp8, "_state_for", lambda *args: state)
    monkeypatch.setattr(
        delta_fp8._collectives,
        "_get_param_all_gather_inputs",
        lambda _: [[torch.ones(2, dtype=torch.bfloat16)]],
    )
    monkeypatch.setattr(
        delta_fp8._collectives,
        "_get_all_gather_input_metadatas",
        lambda _: ([[torch.bfloat16]], [[2]], torch.bfloat16),
    )
    monkeypatch.setattr(
        torch.ops.fsdp,
        "all_gather_copy_in",
        lambda inputs, output, split_sizes, shard_numel, rank: output,
    )
    monkeypatch.setattr(delta_fp8, "quantize_delta_into", lambda *args: calls.append(("quantize",)))
    monkeypatch.setattr(delta_fp8, "dequantize_add", lambda *args: calls.append(("dequantize",)))

    copy_stream = _FakeStream(calls)
    gather_stream = _FakeStream(calls)
    result = delta_fp8._delta_foreach_all_gather(
        [key],
        Group(),
        True,
        copy_stream,
        gather_stream,
        torch.device("cpu"),
        comm,
    )

    assert isinstance(result.all_gather_work, dist.distributed_c10d.Work)
    result.all_gather_work.wait()
    assert calls[-2:] == [("wait", "payload"), ("wait", "scales")]
    assert calls.index(("payload", True)) < calls.index(("scales", True))
    assert calls.index(("scales", True)) < calls.index(("dequantize",))


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
    from loongforge.embodied.distributed import delta_fp8_comm

    with pytest.raises(RuntimeError, match=message):
        delta_fp8_comm.validate_runtime(device, backend)


def test_runtime_validation_rejects_missing_triton_fp8_type(monkeypatch):
    from loongforge.embodied.distributed import delta_fp8_comm

    monkeypatch.setattr(delta_fp8_comm, "triton", object())
    monkeypatch.setattr(delta_fp8_comm, "tl", object())
    with pytest.raises(RuntimeError, match="Triton FP8 type"):
        delta_fp8_comm.validate_runtime(torch.device("cuda"), "nccl")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("nccl", "nccl"), ("cpu:gloo,cuda:nccl", "nccl"), ("cuda:gloo", "gloo")],
)
def test_backend_for_device_resolves_device_scoped_backend(backend, expected):
    from loongforge.embodied.distributed import delta_fp8_comm

    assert delta_fp8_comm._backend_for_device(backend, "cuda") == expected


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="run with torchrun to exercise real FSDP collectives",
)
def test_real_fsdp_enabled_then_disabled_model():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    try:
        enabled_model = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)
        ).cuda().bfloat16()
        fully_shard(
            enabled_model,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=True,
        )
        assert delta_fp8.register_delta_fp8_allgather(enabled_model, block=256) == 1

        optimizer = torch.optim.SGD(enabled_model.parameters(), lr=1e-2)
        for _ in range(2):
            inputs = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
            enabled_model(inputs).float().square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        assert len(delta_fp8._STATES) == 1
        enabled_gathers = next(iter(delta_fp8._STATES.values())).gathers
        assert enabled_gathers >= 3

        disabled_model = nn.Linear(16, 8).cuda().bfloat16()
        fully_shard(
            disabled_model,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=True,
        )
        disabled_model(
            torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        ).float().sum().backward()
        torch.cuda.synchronize()

        # The global dispatcher remains installed, but an unregistered model
        # uses native FSDP and does not allocate a Delta-FP8 reference.
        assert len(delta_fp8._GROUP_CONFIGS) == 1
        assert len(delta_fp8._STATES) == 1

        del optimizer, enabled_model, inputs
        gc.collect()
        assert len(delta_fp8._GROUP_CONFIGS) == 0
        assert len(delta_fp8._STATES) == 0
    finally:
        torch.cuda.synchronize()
        delta_fp8.uninstall_delta_fp8_allgather()
        dist.destroy_process_group()
