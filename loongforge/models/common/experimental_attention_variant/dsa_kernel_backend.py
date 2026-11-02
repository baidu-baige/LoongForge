# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backend selection for the fused-DSA kernels.

The fused DSA path binds five kernel entry points.  Only three of them have an
implementation for every supported GPU:

===================  =========================  =========================
entry point          SM90/SM100                 SM80 (Ampere)
===================  =========================  =========================
indexer forward      ``deep_gemm.fp8_mqa_logits``   ``lightning_indexer_sm80``
indexer backward     ``lightning_indexer_bwd``      ``lightning_indexer_sm80``
top-k                ``flashinfer.top_k``           ``flashinfer.top_k``
sparse MLA forward   ``flash_mla_fwd``              ``flash_mla_sm80`` / reference
sparse MLA backward  ``flash_mla_bwd``              ``flash_mla_sm80`` / TileLang
===================  =========================  =========================

``flash_mla_bwd`` is SM100-only (it keeps its accumulators in TMEM and issues
2-SM cluster UMMA), so on SM90 the sm100 backend falls back to the TileLang
sparse-MLA backward -- matching the original ``major != 10`` behaviour.  The
FlashMLA *forward* does run on SM90, so only the backward differs there.

Before this module the kernels were bound by top-level ``import`` in
``dsa_fused_kernels`` with a single ``if major == 10`` branch selecting between two
backward implementations, so importing ``dsa_fused`` failed outright on any machine
without the SM90/SM100 wheels -- including the ``COMPILE_ENV=ampere`` image, which
deliberately skips FlashMLA.  Everything is therefore resolved lazily here.

The indexer differs semantically between backends: SM90/SM100 quantise q/k to FP8
e4m3 (TransformerEngine ``Float8BlockQuantizer``, which needs SM89+), while SM80
runs BF16 because Ampere has no FP8 tensor cores.  Callers must branch on
:attr:`DSAKernelBackend.indexer_is_fp8` rather than assuming FP8.

Selection order: explicit ``--dsa-kernel-backend`` > ``LOONGFORGE_DSA_KERNEL_BACKEND``
env var > compute capability.  ``torch_ref`` is intentionally kept as a permanent
option: it is the numerical ground truth used to validate the CUDA kernels, and it
is the only backend that runs without any custom kernel installed.
"""

import sys
from pathlib import Path
from typing import Optional

import torch

# The SM80 operator packages (``flash_mla_sm80`` / ``lightning_indexer_sm80``) are
# vendored in-tree under ``ops/cuda_source/`` and built in place, so LoongForge can
# call them without depending on DeepTraining or a manually-set PYTHONPATH.  Their
# parent directories are appended (not prepended) to ``sys.path`` so that a
# pip-installed copy or an explicit ``DSA_SM80_OPS_PATH`` on ``PYTHONPATH`` still
# takes precedence as an override.
_SM80_OPS_ADDED = False


def _ensure_sm80_ops_on_path() -> None:
    global _SM80_OPS_ADDED
    if _SM80_OPS_ADDED:
        return
    _SM80_OPS_ADDED = True
    # dsa_kernel_backend.py -> experimental_attention_variant -> common -> models
    # -> loongforge -> <repo root>
    repo_root = Path(__file__).resolve().parents[4]
    ops_dir = repo_root / "ops" / "cuda_source"
    for pkg in ("sparse_mla_sm80", "lightning_indexer_sm80"):
        pkg_dir = ops_dir / pkg
        if pkg_dir.is_dir():
            entry = str(pkg_dir)
            if entry not in sys.path:
                sys.path.append(entry)


BACKEND_SM100 = "sm100"
BACKEND_SM80 = "sm80"
BACKEND_TORCH_REF = "torch_ref"
BACKEND_TILELANG = "tilelang"
BACKEND_AUTO = "auto"

VALID_BACKENDS = (BACKEND_AUTO, BACKEND_SM100, BACKEND_SM80, BACKEND_TORCH_REF, BACKEND_TILELANG)

_MISSING_HINT = (
    "fused DSA backend {backend!r} needs {what}. "
    "Install the matching kernel package, or fall back with "
    "--dsa-kernel-backend torch_ref (correct but slow)."
)


def _auto_backend() -> str:
    if not torch.cuda.is_available():
        return BACKEND_TORCH_REF
    major, minor = torch.cuda.get_device_capability()
    if major == 10:
        return BACKEND_SM100
    if (major, minor) == (8, 0) or (major == 8 and minor < 9):
        return BACKEND_SM80
    # SM89 / SM90 keep the FP8 path: they have FP8 tensor cores and the shipped
    # DeepGEMM kernel covers SM90.  FlashMLA is SM100-only, so sparse MLA still
    # needs the TileLang fallback there, which is what the old code did.
    return BACKEND_SM100


def _requested_backend(args=None) -> str:
    import os

    if args is None:
        try:
            from megatron.training import get_args

            args = get_args()
        except Exception:  # noqa: BLE001 - args not initialised outside training
            args = None
    value = getattr(args, "dsa_kernel_backend", None) if args is not None else None
    if not value:
        value = os.environ.get("LOONGFORGE_DSA_KERNEL_BACKEND")
    if not value:
        return BACKEND_AUTO
    value = value.strip().lower()
    if value not in VALID_BACKENDS:
        raise ValueError(
            f"invalid dsa kernel backend {value!r}; expected one of {VALID_BACKENDS}"
        )
    return value


class DSAKernelBackend:
    """Lazily-bound kernel set for one backend name."""

    def __init__(self, name: str):
        self.name = name
        self._cache = {}
        if name == BACKEND_SM80:
            _ensure_sm80_ops_on_path()

    # -- capability flags -------------------------------------------------
    @property
    def indexer_is_fp8(self) -> bool:
        """Whether the indexer expects FP8 e4m3 q/k plus per-token dequant scales."""
        return self.name in (BACKEND_SM100,)

    @property
    def has_sparse_mla_fwd_kernel(self) -> bool:
        return self.name in (BACKEND_SM100, BACKEND_SM80)

    def _get(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def _missing(self, what) -> ImportError:
        return ImportError(_MISSING_HINT.format(backend=self.name, what=what))

    # -- top-k ------------------------------------------------------------
    def top_k(self, logits, k, sorted=False):  # noqa: A002 - mirrors flashinfer
        """Top-k over indexer logits.

        ``flashinfer.top_k`` is a radix select with no tensor-core code and no
        arch gating, so it works unchanged on SM80; verified against
        ``torch.topk`` on A800.  ``radix_topk`` (DeepTraining ``cuda_source/topk``)
        is an equivalent drop-in should flashinfer ever be unavailable.
        """

        def _load():
            try:
                import flashinfer

                return flashinfer.top_k
            except ImportError:
                try:
                    import radix_topk

                    return radix_topk.top_k
                except ImportError as exc:
                    raise self._missing("flashinfer or radix_topk") from exc

        return self._get("top_k", _load)(logits, k, sorted=sorted)

    # -- indexer ----------------------------------------------------------
    def mqa_logits(self, q, k, weights, cu_seqlen_ks, cu_seqlen_ke, *, k_scale=None,
                   clean_logits=True, max_seqlen_k=0):
        """Lightning-indexer logits, ``[s_q, s_kv]`` float32.

        FP8 backends take ``q``/``k`` as ``float8_e4m3fn`` plus ``k_scale``; BF16
        backends take bfloat16 and ignore ``k_scale``.
        """
        if self.indexer_is_fp8:

            def _load():
                try:
                    import deep_gemm

                    return deep_gemm.fp8_mqa_logits
                except ImportError as exc:
                    raise self._missing("deep_gemm") from exc

            fn = self._get("mqa_logits", _load)
            return fn(q, (k, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke,
                      clean_logits=clean_logits, max_seqlen_k=max_seqlen_k)

        if self.name == BACKEND_SM80:

            def _load():
                try:
                    from lightning_indexer_sm80 import bf16_mqa_logits

                    return bf16_mqa_logits
                except ImportError as exc:
                    raise self._missing("lightning_indexer_sm80") from exc

            logits = self._get("mqa_logits", _load)(q, k, weights, cu_seqlen_ks, cu_seqlen_ke)
        else:
            logits = _ref_bf16_mqa_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke)

        if not clean_logits and max_seqlen_k:
            # Packed path: the caller re-masks per sequence, and only the first
            # max_seqlen_k columns are meaningful.
            logits = logits[:, :max_seqlen_k]
        return logits

    def mqa_logits_bwd(self, grad_logits, q, k, weights, cu_seqlen_ks, cu_seqlen_ke,
                       topk_indices, topk, *, k_scale=None, clean_logits=True,
                       max_seqlen_k=0):
        """Indexer backward, returning ``(grad_q, grad_k, grad_weights)`` float32."""
        if self.indexer_is_fp8:

            def _load():
                try:
                    import lightning_indexer_bwd

                    return lightning_indexer_bwd.fp8_mqa_logits_bwd
                except ImportError as exc:
                    raise self._missing("lightning_indexer_bwd") from exc

            return self._get("mqa_logits_bwd", _load)(
                grad_logits, q, (k, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke,
                topk_indices=topk_indices, clean_logits=clean_logits,
                max_seqlen_k=max_seqlen_k, topk=topk,
            )

        if self.name == BACKEND_SM80:
            def _load():
                try:
                    from lightning_indexer_sm80 import bf16_mqa_logits_bwd

                    return bf16_mqa_logits_bwd
                except ImportError:
                    _warn_once(
                        "lightning_indexer_sm80 has no bf16_mqa_logits_bwd yet; "
                        "using the PyTorch reference for the indexer backward. "
                        "This is correct but slow."
                    )
                    return None

            fn = self._get("mqa_logits_bwd", _load)
            if fn is not None:
                return fn(grad_logits, q, k, weights, cu_seqlen_ks, cu_seqlen_ke,
                          topk_indices)

        return _ref_bf16_mqa_logits_bwd(grad_logits, q, k, weights, cu_seqlen_ks,
                                        cu_seqlen_ke, topk_indices)

    # -- sparse MLA -------------------------------------------------------
    def sparse_mla_fwd(self, q, kv, indices, sm_scale, d_v, *, q_start_index_s=0,
                       write_p_out=False, topk_length=None, attn_sink=None,
                       window_size=0):
        """Sparse MLA forward.

        Returns ``(out, max_logits, lse, p_out_or_None)`` with 3-D ``out``
        ``[s_q, h_q, d_v]`` and **natural-log** ``lse``.  Note the CUDA
        docstring claims base-2; the kernel actually writes
        ``logf(li) + mi * ln2`` and the reference uses ``torch.logsumexp``.
        """
        if self.has_sparse_mla_fwd_kernel:
            def _load():
                if self.name == BACKEND_SM100:
                    try:
                        from flash_mla_fwd import flash_mla_sparse_fwd

                        return flash_mla_sparse_fwd
                    except ImportError as exc:
                        raise self._missing("flash_mla_fwd") from exc
                try:
                    from flash_mla_sm80 import flash_mla_sparse_fwd_sm80

                    return flash_mla_sparse_fwd_sm80
                except ImportError:
                    _warn_once(
                        "flash_mla_sm80 is not installed; using the PyTorch "
                        "reference for the sparse MLA forward. This is correct "
                        "but slow and memory hungry (it materialises the gathered "
                        "KV tensor)."
                    )
                    return None

            fn = self._get("sparse_mla_fwd", _load)
            if fn is not None:
                out = fn(q, kv, indices, sm_scale, d_v, q_start_index_s=q_start_index_s,
                         write_p_out=write_p_out, topk_length=topk_length,
                         attn_sink=attn_sink, window_size=window_size)
                return out[0], out[1], out[2], (out[3] if len(out) > 3 else None)

        return _ref_sparse_mla_fwd(q, kv, indices, sm_scale, d_v, q_start_index_s,
                                   attn_sink, write_p_out)

    def sparse_mla_bwd(self, q, kv, out, grad_out, indices, lse, *, sm_scale,
                       q_start_index_s=0, topk_length=None, fast_mode=False,
                       attn_sink=None):
        """Sparse MLA backward, returning ``(grad_q, grad_kv)``."""
        if self.name == BACKEND_SM100:
            # ``flash_mla_sparse_bwd`` is SM100-only: it keeps its accumulators
            # in TMEM and issues 2-SM cluster UMMA, neither of which exists
            # before Blackwell.  The sm100 backend also serves SM89/SM90, where
            # the FlashMLA *forward* runs but this backward does not, so fall
            # back to the TileLang backward there -- exactly what the
            # pre-refactor ``major != 10`` branch did.
            if torch.cuda.get_device_capability()[0] == 10:
                def _load():
                    try:
                        from flash_mla_bwd import flash_mla_sparse_bwd

                        return flash_mla_sparse_bwd
                    except ImportError as exc:
                        raise self._missing("flash_mla_bwd") from exc

                return self._get("sparse_mla_bwd", _load)(
                    q, kv, out, grad_out, indices, lse, sm_scale=sm_scale,
                    q_start_index_s=q_start_index_s, topk_length=topk_length,
                    fast_mode=fast_mode, attn_sink=attn_sink,
                )
            return _tilelang_sparse_mla_bwd(q, kv, out, grad_out, indices, lse,
                                            sm_scale, q_start_index_s)

        if self.name in (BACKEND_SM80, BACKEND_TILELANG):
            if self.name == BACKEND_SM80:
                try:
                    from flash_mla_sm80 import flash_mla_sparse_bwd_sm80
                except ImportError:
                    # Deliberately *not* falling back to TileLang here.  The
                    # pre-existing `major != 10` TileLang path does not run on
                    # A100/A800: with h_q=128 it splits the heads in two and then
                    # asks for 216 KB of dynamic shared memory (Q, dO and the dkv
                    # accumulators for 64 heads x 512), against a 164 KB limit, and
                    # fails with "Failed to set the allowed dynamic shared memory
                    # size to 221184".  Splitting the heads four ways would fit, but
                    # that is a change to the TileLang op.  Select `tilelang`
                    # explicitly if you want to try it.
                    _warn_once(
                        "flash_mla_sm80 is not installed; using the PyTorch "
                        "reference for the sparse MLA backward. This is correct but "
                        "slow, and it materialises the gathered KV tensor, so keep "
                        "sequence lengths small."
                    )
                    return _ref_sparse_mla_bwd(q, kv, out, grad_out, indices, lse,
                                               sm_scale, q_start_index_s, attn_sink)
                return flash_mla_sparse_bwd_sm80(
                    q, kv, out, grad_out, indices, lse, sm_scale=sm_scale,
                    q_start_index_s=q_start_index_s, topk_length=topk_length,
                    fast_mode=fast_mode, attn_sink=attn_sink,
                )
            return _tilelang_sparse_mla_bwd(q, kv, out, grad_out, indices, lse,
                                            sm_scale, q_start_index_s)

        return _ref_sparse_mla_bwd(q, kv, out, grad_out, indices, lse, sm_scale,
                                   q_start_index_s, attn_sink)


# ---------------------------------------------------------------------------
# PyTorch references.  These are the numerical ground truth for the CUDA
# kernels and the only path that needs no custom kernel at all.  They are
# memory hungry by construction, so keep sequence lengths small when using them.
# ---------------------------------------------------------------------------

_WARNED = set()


def _warn_once(msg: str) -> None:
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    try:
        from megatron.core.utils import log_single_rank
        import logging

        log_single_rank(logging.getLogger(__name__), logging.WARNING, f"[fused DSA] {msg}")
    except Exception:  # noqa: BLE001
        print(f"[fused DSA] {msg}", flush=True)


def _ref_bf16_mqa_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, head_chunk=8):
    """``logits[m, n] = sum_h w[m, h] * relu(q[m, h, :] . k[n, :])``, -inf outside the window.

    Accumulated over head chunks: the direct ``einsum('mhd,nd->hmn')`` in the
    upstream reference materialises a tensor ``num_heads`` times larger than the
    output (8.6 GB bf16 at s_q=4096, s_kv=32768 under ChunkPipe).
    """
    s_kv = k.shape[0]
    num_heads = q.shape[1]
    kf = k.float()
    logits = torch.zeros(q.shape[0], s_kv, device=q.device, dtype=torch.float32)
    for h0 in range(0, num_heads, head_chunk):
        h1 = min(h0 + head_chunk, num_heads)
        score = torch.einsum("mhd,nd->mhn", q[:, h0:h1].float(), kf)
        logits += (score.relu() * weights[:, h0:h1].float().unsqueeze(-1)).sum(dim=1)
    pos = torch.arange(s_kv, device=q.device)
    mask = (pos[None, :] >= cu_seqlen_ks[:, None]) & (pos[None, :] < cu_seqlen_ke[:, None])
    return logits.masked_fill(~mask, float("-inf"))


def _ref_bf16_mqa_logits_bwd(grad_logits, q, k, weights, cu_seqlen_ks, cu_seqlen_ke,
                             topk_indices, token_chunk=256):
    """Indexer backward restricted to ``topk_indices``.

    ``grad_logits`` is ``[s_q, topk]``; entries whose index is negative (top-k
    padding) or outside the causal window contribute nothing.
    """
    s_q, num_heads, head_dim = q.shape
    s_kv = k.shape[0]
    qf, kf, wf = q.float(), k.float(), weights.float()
    grad_q = torch.zeros_like(qf)
    grad_k = torch.zeros_like(kf)
    grad_w = torch.zeros_like(wf)

    for m0 in range(0, s_q, token_chunk):
        m1 = min(m0 + token_chunk, s_q)
        idx = topk_indices[m0:m1].long()
        valid = idx >= 0
        pos = idx.clamp(min=0, max=max(s_kv - 1, 0))
        valid &= pos >= cu_seqlen_ks[m0:m1, None]
        valid &= pos < cu_seqlen_ke[m0:m1, None]

        kg = kf[pos]                                        # [m, topk, d]
        s = torch.einsum("mhd,mtd->mht", qf[m0:m1], kg)     # [m, h, topk]
        relu_s = s.relu()
        gl = (grad_logits[m0:m1].float() * valid).unsqueeze(1)  # [m, 1, topk]

        grad_w[m0:m1] = (gl * relu_s).sum(dim=-1)
        ds = gl * wf[m0:m1].unsqueeze(-1) * (s > 0)         # [m, h, topk]
        grad_q[m0:m1] = torch.einsum("mht,mtd->mhd", ds, kg)
        contrib = torch.einsum("mht,mhd->mtd", ds, qf[m0:m1])  # [m, topk, d]
        grad_k.index_add_(0, pos.reshape(-1), (contrib * valid.unsqueeze(-1)).reshape(-1, head_dim))

    return grad_q, grad_k, grad_w


def _ref_sparse_mla_core(q, kv, indices, sm_scale, d_v, q_start_index_s, attn_sink,
                         token_chunk=256):
    """Autograd-safe sparse MLA forward reference, 3-D in/out.

    Semantics follow ``flash_mla_fwd.ref_sparse_mla_fwd_interface`` (the function the
    kernel's own ``FLASH_MLA_CHECK_CORRECTNESS`` compares against), but neither
    upstream reference can be reused directly:

    * ``ref_sparse_mla_fwd_interface`` writes ``lse[lonely_q_mask] = inf`` in place on
      the output of ``logsumexp``, so autograd refuses to differentiate through it;
    * ``ref_sparse_mla_bwd_interface`` calls ``.backward()`` on the 4-tuple the
      forward returns, and passes its ``is_casual`` flag into the forward's ``d_v``
      positional slot.

    ``p_out`` is the **unscaled** masked ``qk``; ``triton_attn_dist`` applies
    ``softmax_scale`` itself.  ``lse`` is natural log, ``+inf`` for query tokens with
    no valid key.
    """
    s_q, h_q, _ = q.shape
    s_kv, h_kv, _ = kv.shape
    assert h_kv == 1, f"reference sparse MLA supports h_kv=1 only, got {h_kv}"
    kvf = kv[:, 0].float()
    idx_all = indices[:, 0]

    outs, max_logits_l, lses, p_outs = [], [], [], []
    for m0 in range(0, s_q, token_chunk):
        m1 = min(m0 + token_chunk, s_q)
        idx = idx_all[m0:m1].long()
        gathered = kvf[idx.clamp(min=0, max=max(s_kv - 1, 0))]  # [m, topk, d_qk]
        qk = torch.einsum("shd,std->sht", q[m0:m1].float(), gathered)

        causal_limit = torch.arange(
            q_start_index_s + m0, q_start_index_s + m1, device=q.device, dtype=idx.dtype
        ).unsqueeze(-1)
        valid = ((idx >= 0) & (idx < s_kv) & (idx <= causal_limit)).unsqueeze(1)

        neg_inf = torch.full_like(qk, float("-inf"))
        p_outs.append(torch.where(valid, qk, neg_inf))
        score = torch.where(valid, qk * sm_scale, neg_inf)

        max_logits_l.append(score.amax(dim=-1))
        lse = torch.logsumexp(score, dim=-1)  # [m, h]
        lonely = torch.isneginf(lse)
        pos_inf = torch.full_like(lse, float("inf"))
        p = torch.exp(score - torch.where(lonely, pos_inf, lse).unsqueeze(-1))
        outs.append(torch.einsum("sht,std->shd", p, gathered[..., :d_v]))
        lses.append(torch.where(lonely, pos_inf, lse))

    o = torch.cat(outs, dim=0)
    max_logits = torch.cat(max_logits_l, dim=0)
    lse = torch.cat(lses, dim=0)
    p_out = torch.cat(p_outs, dim=0)
    if attn_sink is not None:
        sink = attn_sink.view(1, h_q).float().to(o.device)
        finite_lse = torch.where(torch.isposinf(lse), torch.full_like(lse, float("-inf")), lse)
        s = torch.exp(finite_lse) / (torch.exp(finite_lse) + torch.exp(sink))
        o = o * s.unsqueeze(-1)
    return o.to(torch.bfloat16), max_logits, lse, p_out


def _ref_sparse_mla_fwd(q, kv, indices, sm_scale, d_v, q_start_index_s, attn_sink,
                        write_p_out):
    o, max_logits, lse, p_out = _ref_sparse_mla_core(
        q, kv, indices, sm_scale, d_v, q_start_index_s, attn_sink)
    return o, max_logits, lse, (p_out if write_p_out else None)


def _ref_sparse_mla_bwd(q, kv, out, grad_out, indices, lse, sm_scale, q_start_index_s,
                        attn_sink):
    """Sparse MLA backward reference: autograd through :func:`_ref_sparse_mla_core`.

    Grad mode is disabled inside an autograd backward, hence the explicit
    ``enable_grad``.
    """
    d_v = out.shape[-1]
    with torch.enable_grad():
        q_ = q.detach().clone().requires_grad_(True)
        kv_ = kv.detach().clone().requires_grad_(True)
        o = _ref_sparse_mla_core(q_, kv_, indices, sm_scale, d_v, q_start_index_s,
                                 attn_sink)[0]
        o.backward(grad_out)
    return q_.grad, kv_.grad


def _tilelang_sparse_mla_bwd(q, kv, out, grad_out, indices, lse, sm_scale,
                             q_start_index_s):
    """TileLang sparse MLA backward, the pre-existing ``major != 10`` fallback.

    The TileLang kernel wants natural-log ``lse``; the CUDA forward already
    returns natural log, so no conversion is applied here.  The historical
    ``lse / log2e`` in ``dsa_fused_kernels`` compensated for a base-2 reading of
    the FlashMLA docstring that does not match the kernel.
    """
    from .sparse_mla_bwd import sparse_mla_bwd_interface

    offsets = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    return sparse_mla_bwd_interface(
        q, kv, out, grad_out, indices, lse, offsets, chunk_offset=q_start_index_s,
        sm_scale=sm_scale, return_kernel=False, delta=None,
    )


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_BACKEND: Optional[DSAKernelBackend] = None


def resolve_backend_name(args=None) -> str:
    """Backend name :func:`get_dsa_backend` would resolve to for ``args``.

    Name resolution only -- no kernels are bound and no operator packages are
    imported -- so this is safe to call from argument validation.  ``args``
    defaults to the megatron global args; pass it explicitly when they are not
    yet installed (e.g. during early arg validation).
    """
    name = _requested_backend(args)
    if name == BACKEND_AUTO:
        name = _auto_backend()
    return name


def get_dsa_backend() -> DSAKernelBackend:
    """Resolve (once) and return the active fused-DSA kernel backend."""
    global _BACKEND
    if _BACKEND is None:
        name = resolve_backend_name()
        _BACKEND = DSAKernelBackend(name)
        _warn_once(f"kernel backend = {name}")
    return _BACKEND


def reset_dsa_backend() -> None:
    """Drop the cached backend.  Used by tests that sweep backends."""
    global _BACKEND
    _BACKEND = None
