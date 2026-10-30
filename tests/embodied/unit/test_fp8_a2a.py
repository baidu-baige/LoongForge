# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Config, layout, and kernel tests for the FP8 AllToAll+AllGather DDP hook.

The layout arithmetic is the risky part of this module: a wrong offset in the
fused ``[fp8 payload][fp32 scales]`` chunk silently corrupts gradients instead
of raising. So the round-trip tests use values that E4M3 represents exactly
(multiples of 448 / 2**k) and assert *exact* equality -- a tolerance test would
pass on a layout that merely shuffles blocks.
"""

from __future__ import annotations

import inspect
import tempfile

import pytest
import torch

from loongforge.embodied.distributed.ddp_utils import fp8_a2a_comm as mod
from loongforge.embodied.distributed.ddp_utils.ddp_comm_hook import resolve_comm_hook
from loongforge.embodied.train.training_args import TrainingArgs

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fp8 a2a kernels require CUDA"
)


class _FakeBucket:
    """Minimal stand-in for ``dist.GradBucket``: the hook only needs these two."""

    def __init__(self, buffer: torch.Tensor, index: int = 0):
        self._buffer = buffer
        self._index = index

    def buffer(self):
        return self._buffer

    def index(self):
        return self._index


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Reset the module globals so tests cannot leak scratch into each other."""
    mod.reset_scratch()
    mod.configure()
    yield
    mod.reset_scratch()
    mod.configure()


# --------------------------------------------------------------------------
# CLI defaults and hook registration
# --------------------------------------------------------------------------

def test_cli_defaults():
    defaults = TrainingArgs()
    assert defaults.ddp_comm_hook is None
    assert defaults.ddp_comm_hook_fp8_block == mod.DEFAULT_BLOCK
    assert defaults.ddp_comm_hook_fp8_min_mib == mod.DEFAULT_MIN_MIB
    assert defaults.ddp_comm_hook_fp8_max_scratch_gb == mod.DEFAULT_MAX_SCRATCH_GB


def test_hook_resolves_by_name():
    assert resolve_comm_hook("fp8_a2a_allgather_hook") is mod.fp8_a2a_allgather_hook


# --------------------------------------------------------------------------
# configure() validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("block", [1, 2, 4, 256, 512, mod.MAX_BLOCK])
def test_configure_accepts_powers_of_two(block):
    mod.configure(block=block)
    assert mod._config()[0] == block


@pytest.mark.parametrize("block", [0, -4, 3, 100, 255, 384, 768])
def test_configure_rejects_non_power_of_two(block):
    """Triton's tl.arange(0, BLOCK) needs a power of two, not merely an even int."""
    with pytest.raises(ValueError, match="power of two"):
        mod.configure(block=block)


def test_configure_rejects_block_above_max():
    with pytest.raises(ValueError, match=f"<= {mod.MAX_BLOCK}"):
        mod.configure(block=mod.MAX_BLOCK * 2)


def test_configure_rejects_negative_min_mib():
    with pytest.raises(ValueError, match="min_mib must be >= 0"):
        mod.configure(min_mib=-1.0)


def test_configure_rejects_negative_max_scratch_gb():
    with pytest.raises(ValueError, match="max_scratch_gb must be >= 0"):
        mod.configure(max_scratch_gb=-1.0)


def test_changing_block_drops_stale_scratch():
    """Scratch sized for the old block has a stale chunk_u8 and must not survive."""
    mod.configure(block=256)
    mod._SCRATCH[0] = object()
    mod.configure(block=512)
    assert not mod._SCRATCH

# --------------------------------------------------------------------------
# validate_runtime(): preflight once at install time, not once per bucket
# --------------------------------------------------------------------------

def test_validate_runtime_rejects_cpu():
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        mod.validate_runtime(torch.device("cpu"), "cpu:gloo,cuda:nccl")


def test_validate_runtime_names_the_ddp_flag():
    """The message must name the option the user passed, not the FSDP one."""
    with pytest.raises(RuntimeError, match=r"--ddp-comm-hook fp8_a2a_allgather_hook"):
        mod.validate_runtime(torch.device("cpu"), "gloo")


def test_hook_does_not_preflight_per_bucket():
    """The old require_triton() ran once per bucket per step inside the reducer."""
    src = inspect.getsource(mod.fp8_a2a_allgather_hook)
    assert "validate_runtime" not in src
    assert not hasattr(mod, "require_triton")


@requires_cuda
def test_validate_runtime_rejects_gloo_on_cuda():
    with pytest.raises(RuntimeError, match="requires the NCCL backend"):
        mod.validate_runtime(torch.device("cuda", 0), "gloo")


@requires_cuda
@pytest.mark.parametrize("backend", ["nccl", "cpu:gloo,cuda:nccl", "CUDA:NCCL"])
def test_validate_runtime_accepts_every_nccl_spelling(backend):
    """ctx.backend is device-scoped, so the resolver has to handle both forms."""
    mod.validate_runtime(torch.device("cuda", 0), backend)


# --------------------------------------------------------------------------
# plan(): the layout arithmetic, without allocating
# --------------------------------------------------------------------------


def test_plan_layout_matches_the_documented_formula():
    block, world_size = 256, 8
    align = block * mod.NUM_BLOCKS_PER_TILE
    numel = 106_560_711
    S, chunk_u8, total = mod._BucketScratch.plan(
        numel, torch.bfloat16, world_size, block
    )
    assert S % align == 0
    assert S >= (numel + world_size - 1) // world_size
    assert S - (numel + world_size - 1) // world_size < align
    assert chunk_u8 == S + 4 * (S // block)
    assert total == 2 * world_size * chunk_u8 + S * 2
    # Documented cost: about 1.14x the bucket at world_size=8.
    assert 1.13 < total / (numel * 2) < 1.15


@pytest.mark.parametrize("numel", [1, 2, 4095, 4096, 4097, 16_384, 16_385])
def test_plan_covers_every_element(numel):
    """Padding may overshoot but must never leave an element unassigned."""
    world_size, block = 8, 256
    S, chunk_u8, _ = mod._BucketScratch.plan(numel, torch.bfloat16, world_size, block)
    assert S * world_size >= numel
    assert chunk_u8 % 4 == 0, "the buffer is viewed as fp32, so chunks must stay aligned"


def test_plan_shard_is_the_only_term_that_scales_with_world_size():
    numel, block = 1 << 24, 256
    prev_payload = None
    for world_size in (2, 4, 8, 16):
        S, chunk_u8, total = mod._BucketScratch.plan(
            numel, torch.bfloat16, world_size, block
        )
        payload = 2 * world_size * chunk_u8
        if prev_payload is not None:
            assert payload == pytest.approx(prev_payload, rel=1e-3)
        prev_payload = payload
        assert total - payload == S * 2


def test_plan_fp32_gradients_halve_the_relative_cost():
    numel, world_size, block = 1 << 24, 8, 256
    bf16 = mod._BucketScratch.plan(numel, torch.bfloat16, world_size, block)[2]
    fp32 = mod._BucketScratch.plan(numel, torch.float32, world_size, block)[2]
    assert bf16 / (numel * 2) == pytest.approx(1.0156 + 1 / world_size, abs=2e-3)
    assert fp32 / (numel * 4) == pytest.approx(0.5078 + 1 / world_size, abs=2e-3)

# --------------------------------------------------------------------------
# Scratch budget bookkeeping (the flag exists to prevent an OOM, so the
# accounting has to be right *before* anything is allocated)
# --------------------------------------------------------------------------

def _scratch_for(index, identity, numel, budget, world_size=8, block=256,
                 device=torch.device("cpu")):
    return mod._scratch_for(
        index, identity, numel, torch.bfloat16, device, world_size, block, budget
    )


def test_zero_budget_degrades_every_bucket():
    """Regression: `if budget and ...` made 0 *remove* the cap instead of enforcing it."""
    mod.configure(max_scratch_gb=0.0)
    assert mod._config()[2] == 0
    assert _scratch_for(0, ("a",), 106_560_711, budget=0) is None
    assert mod._SCRATCH_BYTES == 0
    assert not mod._SCRATCH


def test_bucket_over_budget_degrades_without_allocating():
    need = mod._BucketScratch.plan(1 << 26, torch.bfloat16, 8, 256)[2]
    assert _scratch_for(0, ("a",), 1 << 26, budget=need - 1) is None
    assert mod._SCRATCH_BYTES == 0


def test_over_budget_warning_fires_once_per_bucket():
    mod._OVER_BUDGET.clear()
    for _ in range(3):
        _scratch_for(7, ("a",), 1 << 26, budget=1)
    assert mod._OVER_BUDGET == {7}


@requires_cuda
def test_same_bucket_reuses_scratch():
    device = torch.device("cuda")
    budget = 2 << 30
    first = _scratch_for(0, ("a",), 1 << 20, budget, device=device)
    after_first = mod._SCRATCH_BYTES
    second = _scratch_for(0, ("a",), 1 << 20, budget, device=device)
    assert second is first
    assert mod._SCRATCH_BYTES == after_first


@requires_cuda
def test_bucket_relayout_replaces_scratch_instead_of_accumulating():
    """DDP rebuilds buckets after iteration 0; a second generation would double the cost."""
    device = torch.device("cuda")
    budget = 4 << 30
    _scratch_for(0, ("a",), 1 << 20, budget, device=device)
    one_generation = mod._SCRATCH_BYTES
    _scratch_for(0, ("b",), 1 << 20, budget, device=device)
    assert mod._SCRATCH_BYTES == one_generation
    assert len(mod._SCRATCH) == 1


# --------------------------------------------------------------------------
# Kernel round-trips. E4M3 stores 448/2**k exactly with a power-of-two scale,
# so these assert exact equality: a shuffled block or a mis-sized scale region
# changes the result, while a tolerance test would not notice.
# --------------------------------------------------------------------------

_EXACT_VALUES = (448.0, 224.0, 112.0, 56.0)


def _exact_input(world_size, S, device):
    chunks = [
        torch.full((S,), _EXACT_VALUES[r % len(_EXACT_VALUES)],
                   dtype=torch.bfloat16, device=device)
        for r in range(world_size)
    ]
    return torch.cat(chunks)


@requires_cuda
def test_dequant_reduce_is_exact_and_ordered():
    """Sums chunk r of every rank; a swapped chunk or scale changes the mean."""
    device, world_size, block = torch.device("cuda"), 4, 256
    S = block * mod.NUM_BLOCKS_PER_TILE
    numel = S * world_size
    x = _exact_input(world_size, S, device)

    _, chunk_u8, _ = mod._BucketScratch.plan(numel, torch.bfloat16, world_size, block)
    send = torch.empty(world_size * chunk_u8, dtype=torch.uint8, device=device)
    shard = torch.empty(S, dtype=torch.bfloat16, device=device)

    mod.quantize_chunks(x, send, numel, S, chunk_u8, world_size, block)
    mod.dequant_reduce(send, shard, S, chunk_u8, world_size, block)

    expected = sum(_EXACT_VALUES) / world_size
    assert torch.equal(shard, torch.full_like(shard, expected))


@requires_cuda
def test_full_round_trip_is_exact():
    """quantize -> reduce -> quantize -> scatter, with the AllGather emulated."""
    device, world_size, block = torch.device("cuda"), 4, 256
    S = block * mod.NUM_BLOCKS_PER_TILE
    numel = S * world_size
    x = _exact_input(world_size, S, device)

    _, chunk_u8, _ = mod._BucketScratch.plan(numel, torch.bfloat16, world_size, block)
    send = torch.empty(world_size * chunk_u8, dtype=torch.uint8, device=device)
    recv = torch.empty(world_size * chunk_u8, dtype=torch.uint8, device=device)
    shard = torch.empty(S, dtype=torch.bfloat16, device=device)

    mod.quantize_chunks(x, send, numel, S, chunk_u8, world_size, block)
    mod.dequant_reduce(send, shard, S, chunk_u8, world_size, block)
    mod.quantize_chunks(shard, send[:chunk_u8], S, S, chunk_u8, 1, block)
    for r in range(world_size):  # emulate all_gather_into_tensor
        recv[r * chunk_u8:(r + 1) * chunk_u8].copy_(send[:chunk_u8])
    out = torch.zeros_like(x)
    mod.dequant_scatter(recv, out, numel, S, chunk_u8, world_size, block)

    expected = sum(_EXACT_VALUES) / world_size
    assert torch.equal(out, torch.full_like(out, expected))


def _emulated_round_trip(x, numel, world_size, block, device):
    S, chunk_u8, _ = mod._BucketScratch.plan(
        numel, torch.bfloat16, world_size, block
    )
    send = torch.empty(world_size * chunk_u8, dtype=torch.uint8, device=device)
    recv = torch.empty(world_size * chunk_u8, dtype=torch.uint8, device=device)
    shard = torch.empty(S, dtype=torch.bfloat16, device=device)
    mod.quantize_chunks(x, send, numel, S, chunk_u8, world_size, block)
    mod.dequant_reduce(send, shard, S, chunk_u8, world_size, block)
    mod.quantize_chunks(shard, send[:chunk_u8], S, S, chunk_u8, 1, block)
    for r in range(world_size):
        recv[r * chunk_u8:(r + 1) * chunk_u8].copy_(send[:chunk_u8])
    out = torch.zeros(numel, dtype=torch.bfloat16, device=device)
    mod.dequant_scatter(recv, out, numel, S, chunk_u8, world_size, block)
    return out


@requires_cuda
@pytest.mark.parametrize("block", [64, 256, 1024])
def test_round_trip_preserves_random_gradients(block):
    device, world_size = torch.device("cuda"), 8
    torch.manual_seed(0)
    numel = block * mod.NUM_BLOCKS_PER_TILE * world_size
    x = torch.randn(numel, dtype=torch.bfloat16, device=device)
    out = _emulated_round_trip(x, numel, world_size, block, device)
    # Every chunk holds the same slice here, so the mean of the slices is not x;
    # compare against what the reduce is meant to produce.
    assert torch.isfinite(out).all()
    reference = x.float().view(world_size, -1).mean(0).repeat(world_size)
    rel = (out.float() - reference).norm() / reference.norm()
    assert float(rel) < 0.08, f"block={block} rel_l2={float(rel)}"


@requires_cuda
@pytest.mark.parametrize("slack", [1, 7, 2047])
def test_round_trip_handles_the_ragged_tail(slack):
    """numel % (world_size * align) != 0 is masked, not padded into a copy."""
    device, world_size, block = torch.device("cuda"), 8, 256
    align = block * mod.NUM_BLOCKS_PER_TILE
    numel = align * world_size * 2 - slack
    torch.manual_seed(0)
    x = torch.randn(numel, dtype=torch.bfloat16, device=device)
    out = _emulated_round_trip(x, numel, world_size, block, device)
    assert out.numel() == numel
    assert torch.isfinite(out).all()
    assert float(out.float().abs().max()) > 0.0


# --------------------------------------------------------------------------
# Multi-rank: the hook must agree with the AllReduce it replaces
# --------------------------------------------------------------------------

def _run_two_rank_equivalence(rank, world_size, init_file):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        device = torch.device("cuda", rank)
        numel = 1 << 20
        torch.manual_seed(1234 + rank)
        x = torch.randn(numel, dtype=torch.bfloat16, device=device)

        reference = x.detach().float()
        dist.all_reduce(reference)
        reference /= world_size

        mod.reset_scratch()
        mod.configure(min_mib=0.0)
        quantized = x.clone()
        mod.fp8_a2a_allgather_hook(None, _FakeBucket(quantized, 0)).wait()
        torch.cuda.synchronize()

        assert torch.isfinite(quantized).all()
        rel = (quantized.float() - reference).norm() / reference.norm()
        assert float(rel) < 0.08, f"rank={rank} rel_l2={float(rel)}"
        # A layout bug decorrelates the result instead of merely adding noise.
        cos = torch.nn.functional.cosine_similarity(
            quantized.float().unsqueeze(0), reference.unsqueeze(0)
        )
        assert float(cos) > 0.995, f"rank={rank} cosine={float(cos)}"

        # The small-bucket fallback must stay bit-comparable to stock DDP.
        mod.reset_scratch()
        mod.configure(min_mib=1024.0)
        passthrough = x.clone()
        mod.fp8_a2a_allgather_hook(None, _FakeBucket(passthrough, 1)).wait()
        torch.cuda.synchronize()
        assert not mod._SCRATCH, "the fallback must not allocate scratch"
        assert torch.equal(passthrough, reference.to(torch.bfloat16))
    finally:
        mod.reset_scratch()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="AllToAll-vs-AllReduce equivalence requires two CUDA devices",
)
def test_hook_matches_allreduce_across_two_ranks():
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.multiprocessing.spawn(
            _run_two_rank_equivalence,
            args=(2, f"{tmpdir}/init"),
            nprocs=2,
            join=True,
        )

