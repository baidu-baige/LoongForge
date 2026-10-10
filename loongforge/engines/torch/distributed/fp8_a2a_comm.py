# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FP8 AllToAll + FP8 AllGather DDP gradient communication.

Replaces DDP's bf16 ring AllReduce with

    quantize -> AllToAll(fp8) -> local fp32 reduce -> quantize -> AllGather(fp8)
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - guarded by validate_runtime().
    triton = None
    tl = None

logger = logging.getLogger(__name__)

DEFAULT_BLOCK = 256
# Buckets below this size fall back to plain AllReduce: two collectives plus four
# kernels do not pay for themselves on a few MiB, and AllToAll efficiency drops
# off below roughly 64 MiB per rank.
DEFAULT_MIN_MIB = 8.0
# Total resident scratch across all buckets. Buckets that do not fit degrade to
# full precision; see _scratch_for.
DEFAULT_MAX_SCRATCH_GB = 24.0
# Quantization blocks handled per Triton program. Decouples the tile size from
# the quantization block size: a 220 MB bucket at BLOCK=256 would otherwise
# need ~430k programs each touching only 256 elements.
NUM_BLOCKS_PER_TILE = 8
# Upper bound on ``block``: a program materializes NUM_BLOCKS_PER_TILE * block
# fp32 values, so 1024 already costs 32 KiB of registers per program.
MAX_BLOCK = 1024

_BLOCK = DEFAULT_BLOCK
_MIN_BYTES = int(DEFAULT_MIN_MIB * 2**20)
_MAX_SCRATCH_BYTES = int(DEFAULT_MAX_SCRATCH_GB * 2**30)
# Error feedback: carry each quantization point's residual into the next step.
# Off by default; see EF_MODES for what each mode costs.
_ERROR_FEEDBACK = "none"
# "allgather" only keeps the shard residual, which is 1/world_size of a bucket.
# Measurement (experiment 3) puts 1.8 of the 2.9% end-to-end relative RMS on the
# AllGather requantization and the rest on the AllToAll one, so this mode buys
# the larger half of the error at 1/9 of the memory and of the residual work.
EF_MODES = ("none", "allgather", "both")
# Residuals are kept in fp32, not bf16. It is tempting to halve them: a residual
# is ~3% of the gradient, so bf16's 2^-8 relative step is only ~1e-4 of a
# gradient -- seemingly three decades below the fp8 error being corrected. That
# per-step view is wrong, and measurably so: storing the residual in bf16 pushed
# the 300-step accumulated bias from +0.40% back to +3.41% (baseline reference),
# nearly the +4.19% of no error feedback at all. The reason is that EF's job is
# to stop *systematic* drift, and the fp8 residual is systematically signed, so
# rounding it each step contributes a same-sign term that does not telescope and
# accumulates ~linearly over the run -- a smaller copy of the exact drift EF
# removes. The correction has to be carried at higher precision than the error.
EF_DTYPE = torch.float32
_ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", RuntimeError)


FLAG = "--ddp-comm-hook fp8_a2a_allgather_hook"


# ── Selective-precision exemption (ddp_comm_hook_fp8_exempt) ─────────────────
# id(param) -> qualified name, populated at DDP-wrap time so the comm hook can
# attribute each slice of a bucket buffer to its parameter and decide, per
# element, whether to keep it in exact precision instead of quantizing it.
_PARAM_NAMES: dict[int, str] = {}
# Structural membership sets, also populated at DDP-wrap time by walking the
# module tree. Matching by module type / leaf name (not name substring) keeps
# the exemption model-agnostic: 'embed' is exactly the params of an nn.Embedding,
# and 'head' is exactly the params of an output-head leaf, so a container module
# that merely has 'head'/'embed' in its qualified path (e.g. GR00T's top-level
# ``action_head``) is never swept in wholesale.
_EMBED_IDS: set[int] = set()
_HEAD_IDS: set[int] = set()
# Parsed exempt spec (set of tokens) or None when disabled. Set by configure().
_EXEMPT_SPEC: "set[str] | None" = None
# Valid exempt tokens and the leaf module names that count as an output head.
EXEMPT_TOKENS = ("1d", "embed", "head")
_HEAD_LEAF_NAMES = ("lm_head", "output", "score", "classifier")
# Per-bucket cache of the exempt element indices (LongTensor on device).
_EXEMPT_IDX: dict[int, "tuple[int, torch.Tensor | None]"] = {}
# Persistent per-bucket buffers for the exempt reduce, so the snapshot/all-reduce
# allocate nothing per step and the whole exempt correction is CUDA-graph
# capture-safe. Keyed by bucket index, rebuilt when the exempt count changes.
_EXEMPT_BUF: dict[int, "tuple[torch.Tensor, torch.Tensor]"] = {}
# Per-bucket cache of exempt positions remapped into this rank's all-gather
# shard window (used to zero the AllGather-leg EF residual at exempt offsets).
# Keyed by bucket index; stores (buf_numel, shard_numel, LongTensor | None) so it
# rebuilds when either the fused/steady-state buffer or the shard size changes.
_EXEMPT_SHARD_IDX: dict[int, "tuple[int, int, torch.Tensor | None]"] = {}


def set_param_names(model) -> None:
    """Record structural exemption metadata for the model DDP is about to wrap.

    Called from ``parallel.py`` before the comm hook is registered. The comm
    hook only receives ``(state, bucket)``; matching ``bucket.parameters()`` back
    to names (``_PARAM_NAMES``) and to their owning module class (``_EMBED_IDS`` /
    ``_HEAD_IDS``) needs the module tree, so it is walked once here while the
    model is still in hand.
    """
    import torch.nn as nn
    _PARAM_NAMES.clear()
    _EMBED_IDS.clear()
    _HEAD_IDS.clear()
    _EXEMPT_IDX.clear()
    _EXEMPT_BUF.clear()
    _EXEMPT_SHARD_IDX.clear()
    for name, param in model.named_parameters():
        _PARAM_NAMES[id(param)] = name
    for mod_name, module in model.named_modules():
        leaf = mod_name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Embedding):
            for param in module.parameters(recurse=False):
                _EMBED_IDS.add(id(param))
        if leaf in _HEAD_LEAF_NAMES:
            for param in module.parameters(recurse=False):
                _HEAD_IDS.add(id(param))


def _exempt_spec():
    """Return the parsed exempt token set (from ``configure``), or None if off."""
    return _EXEMPT_SPEC


def _is_exempt(name: str, param, spec) -> bool:
    """Decide whether a parameter is kept in exact precision.

    ``1d``    -> every 1-D tensor (RMSNorm/LayerNorm scales, biases): tiny byte
                 count, high loss sensitivity (they scale activations directly).
    ``embed`` -> params of an ``nn.Embedding`` module (structural, so the token
                 embedding is caught whatever it is named and an MLP that merely
                 has 'embed' in its path is not).
    ``head``  -> params of an output-head leaf module (``lm_head`` / ``output`` /
                 ``score`` / ``classifier``); structural, so a container named
                 ``action_head`` is not swept in as a whole.
    """
    if spec is None:
        return False
    if "1d" in spec and param.dim() <= 1:
        return True
    if "embed" in spec and id(param) in _EMBED_IDS:
        return True
    if "head" in spec and id(param) in _HEAD_IDS:
        return True
    return False


def _bucket_param_slices(bucket):
    """Yield (name, param, offset, numel) for each param in the bucket buffer.

    Offsets come from comparing each gradient view's data_ptr against the buffer
    base, so alignment padding between params is handled exactly rather than
    assumed contiguous.
    """
    buffer = bucket.buffer()
    base = buffer.data_ptr()
    esize = buffer.element_size()
    params = bucket.parameters()
    grads = bucket.gradients()
    for param, grad in zip(params, grads):
        name = _PARAM_NAMES.get(id(param), f"<unknown@{id(param)}>")
        offset = (grad.data_ptr() - base) // esize
        yield name, param, offset, grad.numel()


def _exempt_indices(bucket, device):
    """Return a cached LongTensor of buffer positions kept in exact precision.

    Built per bucket index from the parameter slices; ``None`` when the
    exemption is off or the bucket has no exempt element. The reduction keeps
    these positions at fp32 mean (a small extra AllReduce) and quantizes the
    rest, so sensitive 1-D scales / embeddings never take the fp8 rounding.

    The cache stores the buffer ``numel`` alongside the index tensor and
    rebuilds on mismatch, mirroring ``_scratch_for``'s ``identity`` guard: DDP
    reuses ``bucket.index() == 0`` for both the fused iteration-0 buffer
    (~1.6e9 elements) and the far smaller steady-state bucket after rebuild, so
    a cache keyed on index alone would hand the small buffer offsets past its
    end and trip the scatter/gather bounds assert.
    """
    index = bucket.index()
    buf_numel = bucket.buffer().numel()
    cached = _EXEMPT_IDX.get(index)
    if cached is not None and cached[0] == buf_numel:
        return cached[1]
    spec = _exempt_spec()
    if spec is None:
        _EXEMPT_IDX[index] = (buf_numel, None)
        return None
    ranges = []
    for name, param, offset, numel in _bucket_param_slices(bucket):
        if _is_exempt(name, param, spec):
            ranges.append(torch.arange(offset, offset + numel, device=device))
    idx = torch.cat(ranges) if ranges else None
    _EXEMPT_IDX[index] = (buf_numel, idx)
    # One-time visibility per (bucket, size): a large exempt fraction means the
    # extra fp32 AllReduce is no longer cheap (e.g. a big trainable vocab
    # embedding under 'embed'); logging it lets that surface without a hard cap.
    n_ex = idx.numel() if idx is not None else 0
    logger.info(
        "fp8_a2a exempt: bucket %d -> %d/%d elems exact (%.3f%%)",
        index, n_ex, buf_numel, 100.0 * n_ex / max(buf_numel, 1),
    )
    return idx


def _exempt_shard_indices(bucket, S, rank, device):
    """Exempt positions remapped into this rank's all-gather shard window.

    The AllGather-leg EF residual (``res2``) lives in shard space of length
    ``S = buf_numel // world_size`` and this rank owns the slice ``[rank*S,
    (rank+1)*S)`` of the bucket. ``patch_exempt`` overwrites the exact fp32 mean
    at every exempt offset, so the residual that the fp8 shard path tracked for
    those same offsets is stale and must be zeroed -- but only for the exempt
    elements that fall in *this* rank's shard. Returns their shard-local offsets
    (global index minus ``rank*S``), or ``None`` when exemption is off or no
    exempt element lands in this shard. Cached like ``_exempt_indices``, guarded
    on both ``buf_numel`` and ``S`` so a bucket rebuild rebuilds the map.
    """
    index = bucket.index()
    buf_numel = bucket.buffer().numel()
    cached = _EXEMPT_SHARD_IDX.get(index)
    if cached is not None and cached[0] == buf_numel and cached[1] == S:
        return cached[2]
    idx = _exempt_indices(bucket, device)
    if idx is None:
        _EXEMPT_SHARD_IDX[index] = (buf_numel, S, None)
        return None
    lo, hi = rank * S, (rank + 1) * S
    local = idx[(idx >= lo) & (idx < hi)] - lo
    if local.numel() == 0:
        local = None
    _EXEMPT_SHARD_IDX[index] = (buf_numel, S, local)
    return local


def _exempt_buffers(index, n, dtype, device):
    """Persistent (gather, fp32) buffers for the capture-safe exempt reduce.

    ``gather`` matches the bucket dtype and receives ``index_select(out=...)``;
    ``fp32`` is the AllReduce buffer (the exempt mean is always accumulated in
    fp32, matching the fp8 path's fp32 reduce). The two alias when the bucket is
    already fp32. Cached per bucket index and rebuilt when the exempt count
    changes, mirroring ``_scratch_for`` / ``_exempt_indices``: DDP reuses
    ``bucket.index() == 0`` for the fused iteration-0 buffer and the smaller
    steady-state bucket, so a stale-length buffer would misalign the copy.

    Allocating these once (instead of ``index_select().float()`` per step) is
    what lets the exempt correction be captured into a CUDA graph: capture
    forbids fresh allocations on the capture stream.
    """
    buf = _EXEMPT_BUF.get(index)
    if buf is None or buf[0].numel() != n:
        gather = torch.empty(n, dtype=dtype, device=device)
        fp32 = gather if dtype == torch.float32 else torch.empty(
            n, dtype=torch.float32, device=device
        )
        buf = (gather, fp32)
        _EXEMPT_BUF[index] = buf
    return buf


def validate_runtime(device, backend: str) -> None:
    """Fail at install time when the FP8 AllToAll kernels cannot run.

    Called once from ``parallel.py`` before the hook is registered. Checking
    here rather than inside the hook means a CPU run, a gloo run, or a pre-Ada
    GPU fails before the model and dataloader are built, instead of surfacing as
    a Triton compile error from inside the DDP reducer at the first backward.
    """
    device = torch.device(device)
    entries = [e.strip() for e in str(backend).lower().split(",") if e.strip()]
    scoped = dict(e.rsplit(":", 1) for e in entries if ":" in e)
    plain = next((e for e in entries if ":" not in e), "")
    resolved = scoped.get(device.type, plain)
    context = f"device={device}, backend={str(backend).lower()}"

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"{FLAG} requires a CUDA device; got {context}")
    if resolved != "nccl":
        raise RuntimeError(
            f"{FLAG} requires the NCCL backend; got {context} "
            f"(resolved {device.type}:{resolved or 'unknown'})"
        )
    if triton is None or tl is None or not hasattr(tl, "float8e4nv"):
        raise RuntimeError(
            f"{FLAG} requires Triton with the FP8 type tl.float8e4nv; got {context}"
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(f"{FLAG} requires PyTorch FP8 E4M3 support; got {context}")
    # tl.float8e4nv needs Ada or newer. On cc 8.0 the import and the type both
    # look fine, and only Triton's first compile -- i.e. the first backward --
    # fails, which is the late failure this function exists to move forward.
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        raise RuntimeError(
            f"{FLAG} requires compute capability >= 8.9 for tl.float8e4nv; "
            f"got {capability} ({context})"
        )
    try:
        target = triton.runtime.driver.active.get_current_target()
        triton_backend = str(target.backend).lower()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize Triton's CUDA backend for {FLAG} ({context})"
        ) from exc
    if triton_backend != "cuda":
        raise RuntimeError(
            f"{FLAG} requires Triton's CUDA backend; "
            f"got triton_backend={triton_backend!r} ({context})"
        )


if triton is not None:

    @triton.jit
    def _quantize_fused_kernel(
        X,                      # input, num_chunks * S elements
        OUT_U8,                 # fused uint8 buffer
        OUT_F32,                # same storage viewed as fp32
        numel,                  # valid elements in X; the rest is padding
        S,                      # elements per chunk
        chunk_u8,               # bytes per chunk = S + 4 * S // BLOCK
        chunk_f32,              # chunk_u8 // 4
        scale_base_f32,         # S // 4, fp32-offset of the scale region
        tiles_per_chunk,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Blockwise-quantize ``X`` into ``num_chunks`` fused fp8+scale chunks.

        ``X`` is read straight out of ``bucket.buffer()``. The padding needed to
        make ``num_chunks * S`` a clean multiple is handled by masking the load,
        not by materializing a padded copy -- that copy would cost another full
        bucket of scratch (and HBM traffic) per bucket.

        Offsets are int64: DDP hands us buckets of up to 6.0e9 elements, and the
        fused buffer is 8x that in bytes, so int32 indexing silently wraps and
        lands as cudaErrorIllegalAddress.
        """
        pid = tl.program_id(0).to(tl.int64)
        chunk = pid // tiles_per_chunk
        blk0 = (pid % tiles_per_chunk) * NB

        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]

        idx = chunk * S + within
        x = tl.load(X + idx, mask=idx < numel, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        scale = amax / 448.0  # E4M3_MAX; Triton cannot read non-constexpr globals
        # Guard on the normal range, not on zero: a denormal scale makes
        # ``1 / scale`` overflow to +inf, every non-zero element of the block
        # convert to fp8 NaN, and (with error feedback) that NaN latch into the
        # residual forever. Flushing such a block costs nothing -- its largest
        # element is below 5e-36, i.e. 27 decades under a typical gradient.
        # Error feedback reaches this range on its own: a block whose gradient
        # stays zero keeps requantizing its own residual, which shrinks by
        # ~2% per step and lands in the denormals after a dozen steps.
        inv = tl.where(scale > 1.1754944e-38, 1.0 / scale, 0.0)
        q = (x * inv[:, None]).to(tl.float8e4nv)

        tl.store(OUT_U8 + chunk * chunk_u8 + within, q.to(tl.uint8, bitcast=True))
        tl.store(OUT_F32 + chunk * chunk_f32 + scale_base_f32 + blk0 + ob,
                 tl.where(inv > 0.0, scale, 0.0))

    @triton.jit
    def _dequant_reduce_kernel(
        RECV_U8,
        RECV_F32,
        OUT,                    # averaged shard, S elements
        chunk_u8,
        chunk_f32,
        scale_base_f32,
        world_size,
        inv_world,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Sum all ranks' fp8 chunks in fp32 and write the averaged shard.

        This is where DDP's ``/world_size`` happens, in fp32, exactly once.
        """
        cu = chunk_u8.to(tl.int64)
        cf = chunk_f32.to(tl.int64)
        blk0 = tl.program_id(0).to(tl.int64) * NB
        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]
        s_within = blk0 + ob

        acc = tl.zeros((NB, BLOCK), dtype=tl.float32)
        for r in range(world_size):
            q = tl.load(RECV_U8 + r * cu + within)
            d = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc = tl.load(RECV_F32 + r * cf + scale_base_f32 + s_within)
            acc += d * sc[:, None]

        tl.store(OUT + within, acc * inv_world)

    @triton.jit
    def _dequant_scatter_kernel(
        AG_U8,
        AG_F32,
        OUT,                    # full gradient bucket, written in place
        numel,                  # valid elements in OUT
        S,
        chunk_u8,
        chunk_f32,
        scale_base_f32,
        tiles_per_chunk,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Dequantize each rank's gathered shard into its slice of the bucket."""
        pid = tl.program_id(0).to(tl.int64)
        chunk = pid // tiles_per_chunk
        blk0 = (pid % tiles_per_chunk) * NB

        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]

        q = tl.load(AG_U8 + chunk * chunk_u8 + within)
        d = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sc = tl.load(AG_F32 + chunk * chunk_f32 + scale_base_f32 + blk0 + ob)
        idx = chunk * S + within
        tl.store(OUT + idx, d * sc[:, None], mask=idx < numel)

    @triton.jit
    def _ef_residual_kernel(
        Q_U8,
        Q_F32,
        VALUE,                  # what was quantized: gradient + previous residual
        RES,                    # residual, numel elements, any float dtype
        numel,
        S,
        chunk_u8,
        chunk_f32,
        scale_base_f32,
        tiles_per_chunk,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Close one error-feedback step: ``residual = value - dequant(q)``.

        Same dataflow as ``_dequant_scatter_kernel`` plus one read of ``VALUE``,
        which is what makes this worth its own kernel: dequantizing into ``RES``
        and then fixing it up with ``neg_()``/``add_()`` costs two extra
        full-bucket round trips through HBM per quantization point, and the
        residual is bucket-sized. Keeping the subtraction inside the same program
        also decouples ``RES``'s dtype from the arithmetic: the difference is
        formed in fp32 regardless of how ``RES`` is stored.
        """
        pid = tl.program_id(0).to(tl.int64)
        chunk = pid // tiles_per_chunk
        blk0 = (pid % tiles_per_chunk) * NB

        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]

        q = tl.load(Q_U8 + chunk * chunk_u8 + within)
        d = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sc = tl.load(Q_F32 + chunk * chunk_f32 + scale_base_f32 + blk0 + ob)
        idx = chunk * S + within
        mask = idx < numel
        v = tl.load(VALUE + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(RES + idx, v - d * sc[:, None], mask=mask)


class _BucketScratch:
    """Persistent per-bucket scratch, keyed by bucket identity.

    Both collectives move one fused ``uint8`` buffer of ``world_size`` equal
    chunks::

        chunk_j = [ S bytes fp8 payload ][ 4 * S/block bytes fp32 block scales ]

    Fusing the scales into the payload keeps this at one collective per hop
    instead of two. ``S`` is a multiple of ``block * NUM_BLOCKS_PER_TILE``, so
    the fp32 region is 4-byte aligned and the buffer can be viewed as fp32.

    Allocated once and never freed. That is deliberate: allocating comm scratch
    inside the hook every step is what makes the caching allocator hand out
    storage that NCCL is still reading, which shows up as an OOM/FATAL a few
    steps in rather than as a clean error.

    Cost per bucket is ``2 * padded_bytes * 1.0156 + shard_bytes``, i.e. about
    ``1.15x`` the bucket itself. Buffers are aliased where the dataflow allows:
    ``send`` is reused as the AllGather input once AllToAll has consumed it, and
    ``recv`` is reused as the AllGather output once the local reduce has drained
    it.
    """

    __slots__ = ("send", "recv", "shard", "res1", "res2", "S", "chunk_u8",
                 "tiles_per_chunk", "numel", "identity")

    def __init__(self, identity, numel: int, dtype, device, world_size, block,
                 error_feedback="none"):
        self.S, self.chunk_u8, _ = self.plan(numel, dtype, world_size, block)
        self.numel = numel
        self.identity = identity
        self.tiles_per_chunk = self.S // (block * NUM_BLOCKS_PER_TILE)

        total = world_size * self.chunk_u8
        self.send = torch.empty(total, dtype=torch.uint8, device=device)
        self.recv = torch.empty(total, dtype=torch.uint8, device=device)
        self.shard = torch.empty(self.S, dtype=dtype, device=device)

        # Error-feedback residuals, one per quantization point. These are the
        # only state this hook carries across steps, so they deliberately live
        # with the scratch: _scratch_for drops the whole object when the bucket
        # layout changes, and a residual held against a stale
        # parameter-to-offset mapping would inject a full bucket of noise.
        self.res1 = self.res2 = None
        if error_feedback == "both":
            self.res1 = torch.zeros(numel, dtype=EF_DTYPE, device=device)
        if error_feedback in ("both", "allgather"):
            self.res2 = torch.zeros(self.S, dtype=EF_DTYPE, device=device)

    @staticmethod
    def plan(numel: int, dtype, world_size: int, block: int, error_feedback="none"):
        """Size the scratch without allocating it.

        Kept as the single source of truth for the layout so the budget check
        can run *before* allocation. Sizing after allocating cannot prevent the
        OOM it exists to prevent.
        """
        align = block * NUM_BLOCKS_PER_TILE
        # Elements per chunk, rounded up so every chunk is a whole number of
        # tiles. Wasted elements are at most world_size * align - 1.
        per_rank = (numel + world_size - 1) // world_size
        S = (per_rank + align - 1) // align * align
        chunk_u8 = S + 4 * (S // block)
        total = 2 * world_size * chunk_u8 + S * dtype.itemsize
        if error_feedback == "both":
            total += (numel + S) * EF_DTYPE.itemsize
        elif error_feedback == "allgather":
            total += S * EF_DTYPE.itemsize
        return S, chunk_u8, total

    def bytes(self) -> int:
        residual = sum(r.nbytes for r in (self.res1, self.res2) if r is not None)
        return self.send.numel() + self.recv.numel() + self.shard.nbytes + residual


_SCRATCH: dict[int, _BucketScratch] = {}
_SCRATCH_BYTES = 0
# Buckets already reported as over budget, so the warning fires once per bucket
# instead of once per step.
_OVER_BUDGET: set[int] = set()


def _scratch_for(index, identity, numel, dtype, device, world_size, block, budget,
                 error_feedback="none"):
    """Get or (re)allocate scratch for one bucket, keyed by bucket index.

    Keyed by ``bucket.index()`` rather than by full identity so that DDP's
    bucket rebuild after the first iteration *replaces* the old scratch instead
    of accumulating a second full generation of it (~14 GiB each at
    bucket_cap_mb=200).

    Rebuild is otherwise harmless here: the comm buffers are pure scratch with
    no state carried across steps, so a changed parameter-to-offset mapping
    cannot silently corrupt anything. The error-feedback residuals *are* state,
    which is why they are allocated as part of this object and therefore
    discarded by the same reallocation.
    """
    global _SCRATCH_BYTES
    scratch = _SCRATCH.get(index)
    if scratch is not None:
        if scratch.identity == identity:
            return scratch
        _SCRATCH_BYTES -= scratch.bytes()
        del _SCRATCH[index]
        del scratch
        logger.info("fp8_a2a: bucket %d layout changed, reallocating scratch", index)

    need = _BucketScratch.plan(numel, dtype, world_size, block, error_feedback)[2]
    if _SCRATCH_BYTES + need > budget:
        # Degrade, do not abort. A bucket layout we cannot afford is a reason to
        # send this bucket at full precision, not to kill the training job: the
        # hook is an optimisation and every caller has a correct fallback. DDP's
        # iteration-0 bucket alone is 12 GiB of bf16 on FastWAM, and it appears
        # exactly once, so passing it through costs nothing in steady state.
        if index not in _OVER_BUDGET:
            _OVER_BUDGET.add(index)
            logger.warning(
                "fp8_a2a: bucket %d needs %.2f GiB scratch, budget %.2f GiB "
                "with %.2f GiB already held -- falling back to full-precision "
                "AllReduce for this bucket. Raise --ddp-comm-hook-fp8-max-scratch-gb "
                "or use a larger ddp_bucket_cap_mb to quantize it.",
                index, need / 2**30, budget / 2**30, _SCRATCH_BYTES / 2**30,
            )
        return None

    scratch = _BucketScratch(identity, numel, dtype, device, world_size, block,
                             error_feedback)
    _SCRATCH[index] = scratch
    _SCRATCH_BYTES += scratch.bytes()
    logger.info(
        "fp8_a2a: bucket %d numel=%d S=%d chunk_u8=%d scratch=%.1f MiB "
        "(total %.2f GiB over %d buckets)",
        index, numel, scratch.S, scratch.chunk_u8,
        scratch.bytes() / 2**20, _SCRATCH_BYTES / 2**30, len(_SCRATCH),
    )
    return scratch


def reset_scratch() -> None:
    """Drop all cached scratch. For tests and for bucket-layout changes."""
    global _SCRATCH_BYTES
    _SCRATCH.clear()
    _SCRATCH_BYTES = 0
    _OVER_BUDGET.clear()
    _EXEMPT_BUF.clear()
    _EXEMPT_SHARD_IDX.clear()


def quantize_chunks(x, out_u8, numel, S, chunk_u8, num_chunks, block):
    """Quantize ``x`` into ``num_chunks`` fused fp8+scale chunks of ``out_u8``."""
    tiles_per_chunk = S // (block * NUM_BLOCKS_PER_TILE)
    _quantize_fused_kernel[(num_chunks * tiles_per_chunk,)](
        x, out_u8, out_u8.view(torch.float32),
        numel, S, chunk_u8, chunk_u8 // 4, S // 4, tiles_per_chunk,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def dequant_reduce(recv_u8, out, S, chunk_u8, world_size, block):
    """Sum every rank's fp8 chunk in fp32 and write the averaged shard."""
    tiles = S // (block * NUM_BLOCKS_PER_TILE)
    _dequant_reduce_kernel[(tiles,)](
        recv_u8, recv_u8.view(torch.float32), out,
        chunk_u8, chunk_u8 // 4, S // 4, world_size, 1.0 / world_size,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def dequant_scatter(ag_u8, out, numel, S, chunk_u8, num_chunks, block):
    """Dequantize gathered shards back into the gradient bucket, in place."""
    tiles_per_chunk = S // (block * NUM_BLOCKS_PER_TILE)
    _dequant_scatter_kernel[(num_chunks * tiles_per_chunk,)](
        ag_u8, ag_u8.view(torch.float32), out,
        numel, S, chunk_u8, chunk_u8 // 4, S // 4, tiles_per_chunk,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def ef_residual(quantized, residual, value, numel, S, chunk_u8, num_chunks, block):
    """Close one error-feedback step: ``residual = value - dequant(quantized)``.

    ``value`` already holds ``gradient + previous residual`` and ``quantized``
    its fp8 form. One kernel, one pass: the residual is bucket-sized, so folding
    the subtraction into the dequantization instead of post-processing it saves
    two full-bucket HBM round trips per quantization point.
    """
    tiles_per_chunk = S // (block * NUM_BLOCKS_PER_TILE)
    _ef_residual_kernel[(num_chunks * tiles_per_chunk,)](
        quantized, quantized.view(torch.float32), value, residual,
        numel, S, chunk_u8, chunk_u8 // 4, S // 4, tiles_per_chunk,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def configure(block: int = DEFAULT_BLOCK, min_mib: float = DEFAULT_MIN_MIB,
              max_scratch_gb: float = DEFAULT_MAX_SCRATCH_GB,
              error_feedback: str = "none", exempt: str = "") -> None:
    """Set the hook's tunables. Called once from ``parallel.py`` at install time.

    DDP fixes the comm-hook signature at ``(state, bucket)``, so the knobs cannot
    be passed per call and have to live in module state.

    Changing ``block`` changes the wire layout and changing ``error_feedback``
    changes which buffers exist, so any scratch allocated under the previous
    value is dropped rather than silently reused with a stale ``chunk_u8`` or a
    missing residual.

    ``exempt`` is a comma list of ``EXEMPT_TOKENS`` (empty -> off); it drives the
    selective-precision exemption evaluated per element in the hook.
    """
    global _BLOCK, _MIN_BYTES, _MAX_SCRATCH_BYTES, _ERROR_FEEDBACK, _EXEMPT_SPEC
    # Power of two because all three kernels index with tl.arange(0, BLOCK).
    if block <= 0 or block & (block - 1):
        raise ValueError(
            f"ddp_comm_hook_fp8_block must be a positive power of two, got {block}"
        )
    # Each program holds an NUM_BLOCKS_PER_TILE x block fp32 tile in registers,
    # so the usable ceiling is far below Triton's own tl.arange limit.
    if block > MAX_BLOCK:
        raise ValueError(
            f"ddp_comm_hook_fp8_block must be <= {MAX_BLOCK}, got {block}"
        )
    if min_mib < 0:
        raise ValueError(
            f"ddp_comm_hook_fp8_min_mib must be >= 0, got {min_mib}"
        )
    if max_scratch_gb < 0:
        raise ValueError(
            f"ddp_comm_hook_fp8_max_scratch_gb must be >= 0, got {max_scratch_gb}"
        )
    if error_feedback not in EF_MODES:
        raise ValueError(
            f"ddp_comm_hook_fp8_error_feedback must be one of {EF_MODES}, "
            f"got {error_feedback!r}"
        )
    tokens = {tok.strip() for tok in exempt.split(",") if tok.strip()}
    bad = tokens - set(EXEMPT_TOKENS)
    if bad:
        raise ValueError(
            f"ddp_comm_hook_fp8_exempt tokens must be a subset of {EXEMPT_TOKENS}, "
            f"got unknown {sorted(bad)}"
        )
    if block != _BLOCK or error_feedback != _ERROR_FEEDBACK:
        reset_scratch()
    _BLOCK = block
    _MIN_BYTES = int(min_mib * 2**20)
    _MAX_SCRATCH_BYTES = int(max_scratch_gb * 2**30)
    _ERROR_FEEDBACK = error_feedback
    _EXEMPT_SPEC = tokens or None
    logger.info(
        "fp8_a2a: block=%d min_mib=%g max_scratch_gb=%g error_feedback=%s exempt=%s",
        block, min_mib, max_scratch_gb, _ERROR_FEEDBACK,
        ",".join(sorted(_EXEMPT_SPEC)) if _EXEMPT_SPEC else "off",
    )


def _config():
    return _BLOCK, _MIN_BYTES, _MAX_SCRATCH_BYTES, _ERROR_FEEDBACK


def _is_cuda_graph_capturing() -> bool:
    """Return whether the current CUDA stream is inside graph capture.

    The normal hook path deliberately uses NCCL ``Work`` futures so gradient
    communication can overlap the rest of backward.  A future continuation is
    not capture-safe, though: its callback can run on a host progress thread
    (and therefore launch work on a different stream), leaving the capture
    stream with unjoined work.  Capture mode uses the synchronous enqueue path
    below, which keeps every operation on the stream being captured.
    """
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (RuntimeError, _ACCELERATOR_ERROR):
        # This can be queried before CUDA has been initialized on a few PyTorch
        # builds.  In that case the eager path is the only possible path.
        return False


def _completed_future(tensor):
    """Return a DDP-compatible Future already resolved to ``tensor``."""
    result = torch.futures.Future()
    result.set_result(tensor)
    return result


def fp8_a2a_allgather_hook(process_group, bucket):
    """DDP comm hook: fp8 AllToAll + local fp32 reduce + fp8 AllGather.

    Egress is ``0.508x`` of ring AllReduce. Every element is quantized once per
    half, so the error is comparable to a single fp8 round-trip rather than
    compounding across ring hops.

    Small buckets fall back to plain AllReduce: two collectives plus four
    kernels do not pay for themselves when the payload is only a few MiB, and
    the AllToAll efficiency drops off below ~64 MiB per rank. Buckets whose
    scratch does not fit the budget take the same fallback.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    tensor = bucket.buffer()
    block, min_bytes, budget, error_feedback = _config()

    if world_size < 2 or tensor.nbytes < min_bytes:
        return allreduce_hook(process_group, bucket)

    identity = (tensor.numel(), tensor.dtype, tensor.device, id(group), block)
    st = _scratch_for(
        bucket.index(), identity, tensor.numel(), tensor.dtype, tensor.device,
        world_size, block, budget, error_feedback,
    )
    if st is None:
        return allreduce_hook(process_group, bucket)
    S, chunk_u8 = st.S, st.chunk_u8

    # Selective precision: snapshot the exempt positions before the fp8 path
    # mutates the buffer and reduce them to the exact fp32 cross-rank mean *now*.
    # The all_reduce runs synchronously in the hook body so every rank issues it
    # in the same deterministic bucket order as the a2a/all-gather below; issuing
    # it from the async all-gather callback let ranks enqueue this third
    # collective in divergent orders on the shared NCCL communicator and
    # cross-wired the reduction. Because exempt always takes the synchronous
    # inline path below (see the branch guard), the snapshot is produced and
    # consumed on the same stream -- no cross-stream event is needed.
    #
    # Every buffer here is pre-allocated and reused (``_exempt_buffers``,
    # ``_exempt_indices``): the ``index_select`` writes into ``gather`` via
    # ``out=`` and the reduce is in place, so the whole correction allocates
    # nothing per step and is CUDA-graph capture-safe. Only the scatter is
    # deferred, because the fp8 all-gather overwrites the whole buffer and would
    # otherwise clobber the exact values.
    exempt_idx = _exempt_indices(bucket, tensor.device)
    exempt_snapshot = None
    exempt_gather = None
    exempt_shard_idx = None
    if exempt_idx is not None:
        exempt_gather, exempt_snapshot = _exempt_buffers(
            bucket.index(), exempt_idx.numel(), tensor.dtype, tensor.device
        )
        torch.index_select(tensor, 0, exempt_idx, out=exempt_gather)
        if exempt_snapshot is not exempt_gather:
            exempt_snapshot.copy_(exempt_gather)
        dist.all_reduce(exempt_snapshot, group=group, async_op=False)
        exempt_snapshot.div_(world_size)
        # Cache exempt offsets in shard space once (eager warm-up ⇒ capture-safe)
        # so the AllGather-leg residual can be zeroed at those positions below.
        if st.res2 is not None:
            exempt_shard_idx = _exempt_shard_indices(
                bucket, S, dist.get_rank(group), tensor.device
            )

    def patch_exempt():
        """Write the exact cross-rank mean back over the exempt positions.

        Also drop the EF residual there: both quantize legs tracked a residual
        for the exempt offsets, but this exact mean overwrites them, so that
        residual is phantom -- carrying it forward would inject a growing bias
        into exactly the params exemption is meant to protect, and it perturbs
        the fp8 block scale shared with real neighbours. Zeroing keeps EF and
        exemption orthogonal. ``res1`` is in bucket space (same indices as
        ``exempt_idx``); ``res2`` is in this rank's shard space.
        """
        if exempt_snapshot is None:
            return
        if exempt_snapshot is exempt_gather:
            tensor.index_copy_(0, exempt_idx, exempt_snapshot)
        else:
            # fp32 mean -> bucket dtype, into the persistent gather buffer.
            exempt_gather.copy_(exempt_snapshot)
            tensor.index_copy_(0, exempt_idx, exempt_gather)
        if st.res1 is not None:
            st.res1.index_fill_(0, exempt_idx, 0.0)
        if st.res2 is not None and exempt_shard_idx is not None:
            st.res2.index_fill_(0, exempt_shard_idx, 0.0)

    def quantize_send():
        """Quantize the bucket into ``st.send``, applying error feedback first.

        ``tensor`` is safe to modify in place: it is the bucket buffer, and the
        AllGather result overwrites it wholesale at the end of the hook.
        """
        if st.res1 is not None:
            tensor.add_(st.res1)
        quantize_chunks(tensor, st.send, st.numel, S, chunk_u8, world_size, block)
        if st.res1 is not None:
            ef_residual(st.send, st.res1, tensor, st.numel, S, chunk_u8,
                        world_size, block)

    def quantize_shard():
        """Requantize the averaged shard for the AllGather, with error feedback.

        Each rank reduces only its own chunk, so ``res2`` is legitimately
        per-rank state; every rank still receives every quantized chunk, so the
        final bucket stays bit-identical across ranks.
        """
        if st.res2 is not None:
            st.shard.add_(st.res2)
        ag_send = st.send[:chunk_u8]
        quantize_chunks(st.shard, ag_send, S, S, chunk_u8, 1, block)
        if st.res2 is not None:
            ef_residual(ag_send, st.res2, st.shard, S, S, chunk_u8, 1, block)
        return ag_send

    def launch_allgather(async_op):
        """Second collective: requantize the reduced shard and gather it as fp8.
        Returns the NCCL Work (or None if sync)."""
        ag_send = quantize_shard()
        return dist.all_gather_into_tensor(
            st.recv, ag_send, group=group, async_op=async_op
        )

    def finish_allgather():
        """Write the gathered shards back into the bucket, in shard order."""
        dequant_scatter(st.recv, tensor, st.numel, S, chunk_u8, world_size, block)

    # Selective precision also forces the synchronous inline path. The exempt
    # correction adds a third collective (the fp32 all_reduce above) that has to
    # stay ordered against this bucket's a2a/all-gather on the shared NCCL
    # communicator, and its result is consumed by ``patch_exempt``. Driving that
    # through the async continuation chain let the exempt all_reduce interleave
    # with other buckets' collectives differently on each rank -- provably
    # nondeterministic (identical configs diverged by iter 3 and NaN'd within a
    # few steps, while the plain hook stayed bit-reproducible). Running the whole
    # dataflow inline on one stream, one bucket at a time, removes the
    # interleaving and the cross-stream read; the cost is losing comm/backward
    # overlap on exempt buckets.
    #
    # This inline path *is* CUDA-graph capturable: every exempt buffer is
    # pre-allocated (``_exempt_buffers``), so capture -- which forbids fresh
    # allocations and cannot join the host-side continuation chain the overlap
    # path uses -- sees only pre-allocated tensors and stream-ordered kernels
    # plus a synchronous NCCL enqueue.
    if _is_cuda_graph_capturing() or exempt_snapshot is not None:
        # CUDA Graph capture cannot join the host-side continuation chain used
        # by the overlap path.  Enqueue the complete dataflow on the capture
        # stream and hand DDP an already-resolved Future; NCCL's synchronous
        # Python API waits only for enqueue completion, while stream ordering
        # preserves the dependencies between each operation.
        quantize_send()
        dist.all_to_all_single(st.recv, st.send, group=group, async_op=False)
        dequant_reduce(st.recv, st.shard, S, chunk_u8, world_size, block)
        launch_allgather(async_op=False)
        finish_allgather()
        patch_exempt()
        return _completed_future(tensor)

    quantize_send()
    a2a = dist.all_to_all_single(st.recv, st.send, group=group, async_op=True)

    result = torch.futures.Future()

    def after_a2a(fut):
        try:
            fut.wait()
            # fp32 accumulate across ranks, divide by world_size, requantize.
            dequant_reduce(st.recv, st.shard, S, chunk_u8, world_size, block)
            work = launch_allgather(async_op=True)
        except Exception as exc:
            result.set_exception(exc)
            return

        def after_ag(fut):
            try:
                fut.wait()
                finish_allgather()
                result.set_result(tensor)
            except Exception as exc:
                result.set_exception(exc)

        work.get_future().then(after_ag)

    a2a.get_future().then(after_a2a)
    return result
