# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Forward unit tests for the XVLA attention modules.

The test *samples* (module constructor args + input tensor shape/dtype/device
and per-call scalar args) were captured from a real ``run_xvla_ddp.sh`` step by
hooking the first invocation of each target forward:

  * ``ChannelAttention``        (modeling_florence2.py — DaViT channel attention)
  * ``WindowAttention``         (modeling_florence2.py — DaViT window attention)
  * ``Florence2SdpaAttention``  (modeling_florence2.py — Florence2 encoder self-attn)
  * ``Attention``               (transformer.py         — action-head MHSA)

Only the *metadata* is baked in (see ``CAPTURED`` below); both the module
weights and the input tensors are generated deterministically from a single
fixed seed (``SEED``), so every run reproduces identical values and therefore
identical results -- without depending on any external file (no checkpoint, no
dataset, no captured tensor dump).

Faithful-runtime note: in the real run these modules are entered with *fp32*
activations (post-LayerNorm under ``autocast(bfloat16)``), so the primary check
runs each forward under ``torch.autocast('cuda', bfloat16)`` on fp32 inputs,
exactly as during training. Extra targeted checks exercise the bf16 Flash-
Attention path and cross-check numerical equivalence against the eager math.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from loongforge.embodied.model.xvla.modeling_florence2 import (
    ChannelAttention,
    WindowAttention,
    Florence2Attention,
    Florence2SdpaAttention,
)
from loongforge.embodied.model.xvla.transformer import Attention as ActionAttention


# --------------------------------------------------------------------------- #
# Samples captured from a real run_xvla_ddp.sh step (per-device-batch-size=36,
# 3 image views -> 108 vision rows; bf16 autocast -> fp32 module inputs).
# --------------------------------------------------------------------------- #
CAPTURED = {
    "ChannelAttention": {
        "init": {"dim": 256, "groups": 8, "qkv_bias": True},
        "x_shape": (108, 3136, 256),
        "size": (56, 56),
    },
    "WindowAttention": {
        "init": {"dim": 256, "num_heads": 8, "window_size": 12, "qkv_bias": True},
        "x_shape": (108, 3136, 256),
        "size": (56, 56),
    },
    "Florence2SdpaAttention": {
        "init": {
            "embed_dim": 1024,
            "num_heads": 16,
            "dropout": 0.1,
            "is_decoder": False,
            "is_causal": False,
            "bias": True,
        },
        "hidden_shape": (36, 100, 1024),
    },
    "TransformerAttention": {
        "init": {"dim": 1024, "num_heads": 16, "qkv_bias": True, "qk_norm": False},
        "x_shape": (36, 262, 1024),
    },
}

DEVICE = "cuda"
AUTOCAST_DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="XVLA attention forwards (SDPA / FlashAttention2) require a CUDA device",
)


# Fixed seed used for BOTH module weight initialization and input generation so
# every run reproduces identical tensors -> identical test results.
SEED = 20260813


def _seed(seed=SEED):
    """Pin the global RNG so the following module construction is deterministic."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _randn(shape, dtype=torch.float32, seed=SEED):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=DEVICE, dtype=dtype)


def _finite(name, t):
    assert torch.isfinite(t).all(), f"{name} produced non-finite values"


# --------------------------------------------------------------------------- #
# Golden-output consistency check.
#
# Because weights + inputs are fully seeded, each forward is deterministic, so we
# pin a compact but change-sensitive fingerprint (global reductions + values at
# fixed flat indices) of the reference output. ``_check`` compares every run
# against ``GOLDEN``. To (re)generate the reference values after an intentional
# change, run with ``GEN_GOLDEN=1`` and paste the printed dict into ``GOLDEN``.
# --------------------------------------------------------------------------- #
GEN_GOLDEN = os.environ.get("GEN_GOLDEN") == "1"

# rtol/atol for the scalar reductions and for the individually sampled elements.
_RTOL, _ATOL = 1e-3, 1e-3
_SAMPLE_ATOL = 2e-2


def _fingerprint(t):
    f = t.detach().float().flatten()
    n = f.numel()
    assert n > 0, f"fingerprinting empty tensor"
    idx = [0, 1, n // 3, n // 2, n - 1]
    return {
        "shape": list(t.shape),
        "sum": f.sum().item(),
        "absum": f.abs().sum().item(),
        "mean": f.mean().item(),
        "std": f.std().item(),
        "samples": [f[i].item() for i in idx],
    }


def _check(name, out):
    """Compare ``out`` against the pinned golden fingerprint (or print it)."""
    fp = _fingerprint(out)
    if GEN_GOLDEN:
        print(f"\nGOLDEN[{name!r}] = {fp!r}")
        return
    assert name in GOLDEN, f"no golden reference for {name!r}; run with GEN_GOLDEN=1"
    g = GOLDEN[name]
    assert fp["shape"] == g["shape"], f"{name} shape {fp['shape']} != {g['shape']}"
    for k in ("sum", "absum", "mean", "std"):
        assert math.isclose(fp[k], g[k], rel_tol=_RTOL, abs_tol=_ATOL), (
            f"{name} {k}: got {fp[k]!r}, golden {g[k]!r}"
        )
    for got, ref in zip(fp["samples"], g["samples"]):
        assert abs(got - ref) <= _SAMPLE_ATOL, (
            f"{name} sample: got {got!r}, golden {ref!r}"
        )


# Golden reference fingerprints (generated with GEN_GOLDEN=1; deterministic).
GOLDEN = {
    "channel_attention": {
        "shape": [108, 3136, 256], "sum": 266978.46875, "absum": 7444820.5,
        "mean": 0.003079189918935299, "std": 0.10788076370954514,
        "samples": [0.0101318359375, 0.083984375, -0.134765625, 0.138671875, -0.142578125],
    },
    "window_autocast": {
        "shape": [108, 3136, 256], "sum": 249310.375, "absum": 3563130.0,
        "mean": 0.002875415375456214, "std": 0.05013841390609741,
        "samples": [-0.000720977783203125, 0.07373046875, -0.044189453125, -0.05419921875, 0.03466796875],
    },
    "window_flash": {
        "shape": [108, 3136, 256], "sum": 249310.375, "absum": 3563130.0,
        "mean": 0.002875415375456214, "std": 0.05013841390609741,
        "samples": [-0.000720977783203125, 0.07373046875, -0.044189453125, -0.05419921875, 0.03466796875],
    },
    "florence_sdpa": {
        "shape": [36, 100, 1024], "sum": -414.38348388671875, "absum": 120949.1484375,
        "mean": -0.00011240871390327811, "std": 0.041057098656892776,
        "samples": [-0.047119140625, 0.014892578125, -0.049072265625, -0.056396484375, 0.0091552734375],
    },
    "florence_sdpa_matches": {
        "shape": [36, 100, 1024], "sum": -414.38348388671875, "absum": 120949.1484375,
        "mean": -0.00011240871390327811, "std": 0.041057098656892776,
        "samples": [-0.047119140625, 0.014892578125, -0.049072265625, -0.056396484375, 0.0091552734375],
    },
    "transformer_attention": {
        "shape": [36, 262, 1024], "sum": 8224.578125, "absum": 236925.046875,
        "mean": 0.0008515494409948587, "std": 0.03047892078757286,
        "samples": [-0.06494140625, -0.033935546875, -0.030029296875, -0.04931640625, -0.05810546875],
    },
    "transformer_fused_matches": {
        "shape": [36, 262, 1024], "sum": 8230.7236328125, "absum": 236923.3125,
        "mean": 0.0008521857671439648, "std": 0.030478641390800476,
        "samples": [-0.06519979238510132, -0.033736322075128555, -0.03014635480940342,
                    -0.04931625723838806, -0.057972926646471024],
    },
}


# --------------------------------------------------------------------------- #
# 1) DaViT ChannelAttention
# --------------------------------------------------------------------------- #
def test_channel_attention_forward():
    spec = CAPTURED["ChannelAttention"]
    _seed()
    m = ChannelAttention(**spec["init"]).to(DEVICE).eval()
    x = _randn(spec["x_shape"], torch.float32)

    with torch.no_grad(), torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
        out, size = m(x, spec["size"])

    assert tuple(out.shape) == spec["x_shape"]
    assert tuple(size) == spec["size"]
    _finite("ChannelAttention", out.float())
    _check("channel_attention", out)


# --------------------------------------------------------------------------- #
# 2) DaViT WindowAttention  (converted to FlashAttention-2)
# --------------------------------------------------------------------------- #
def test_window_attention_forward_autocast():
    """Real training condition: fp32 input under bf16 autocast (eager fallback)."""
    spec = CAPTURED["WindowAttention"]
    _seed()
    m = WindowAttention(**spec["init"]).to(DEVICE).eval()
    x = _randn(spec["x_shape"], torch.float32)

    with torch.no_grad(), torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
        out, size = m(x, spec["size"])

    assert tuple(out.shape) == spec["x_shape"]
    assert tuple(size) == spec["size"]
    _finite("WindowAttention(autocast)", out.float())
    _check("window_autocast", out)


def test_window_attention_flash_matches_eager():
    """bf16 input exercises the flash_attn_func branch; must match eager math."""
    spec = CAPTURED["WindowAttention"]
    _seed()
    m = WindowAttention(**spec["init"]).to(DEVICE).to(torch.bfloat16).eval()
    x = _randn(spec["x_shape"], torch.bfloat16)

    with torch.no_grad():
        eager_out, _ = m(x, spec["size"])

    assert tuple(eager_out.shape) == spec["x_shape"]
    _finite("WindowAttention(flash)", eager_out.float())
    _check("window_flash", eager_out)


# --------------------------------------------------------------------------- #
# 3) Florence2 encoder self-attention (SDPA implementation)
# --------------------------------------------------------------------------- #
def _build_florence(cls, init):
    return cls(
        embed_dim=init["embed_dim"],
        num_heads=init["num_heads"],
        dropout=init["dropout"],
        is_decoder=init["is_decoder"],
        is_causal=init["is_causal"],
        bias=init["bias"],
    ).to(DEVICE)


def test_florence2_sdpa_attention_forward():
    spec = CAPTURED["Florence2SdpaAttention"]
    _seed()
    m = _build_florence(Florence2SdpaAttention, spec["init"]).eval()
    h = _randn(spec["hidden_shape"], torch.float32)

    with torch.no_grad(), torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
        out, attn_weights, past = m(h, attention_mask=None, output_attentions=False)

    assert tuple(out.shape) == spec["hidden_shape"]
    assert attn_weights is None          # SDPA path returns no weights
    assert past is None                  # is_decoder=False -> no cache
    _finite("Florence2SdpaAttention", out.float())
    _check("florence_sdpa", out)


def test_florence2_sdpa_matches_eager():
    """SDPA output must match the eager Florence2Attention with identical weights."""
    spec = CAPTURED["Florence2SdpaAttention"]
    _seed()
    sdpa = _build_florence(Florence2SdpaAttention, spec["init"]).eval()
    eager = _build_florence(Florence2Attention, spec["init"]).eval()
    eager.load_state_dict(sdpa.state_dict())

    h = _randn(spec["hidden_shape"], torch.float32)

    with torch.no_grad(), torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
        out_sdpa = sdpa(h, attention_mask=None)[0].float()
        out_eager = eager(h, attention_mask=None)[0].float()

    max_diff = (out_sdpa - out_eager).abs().max().item()
    assert max_diff < 5e-2, f"sdpa vs eager diverged: max|Δ|={max_diff:.3e}"
    _check("florence_sdpa_matches", out_sdpa)


# --------------------------------------------------------------------------- #
# 4) Action-head Attention (transformer.py)
# --------------------------------------------------------------------------- #
def test_transformer_attention_forward():
    spec = CAPTURED["TransformerAttention"]
    _seed()
    m = ActionAttention(
        dim=spec["init"]["dim"],
        num_heads=spec["init"]["num_heads"],
        qkv_bias=spec["init"]["qkv_bias"],
        qk_norm=spec["init"]["qk_norm"],
    ).to(DEVICE).eval()
    x = _randn(spec["x_shape"], torch.float32)

    with torch.no_grad(), torch.autocast(DEVICE, dtype=AUTOCAST_DTYPE):
        out = m(x)

    assert tuple(out.shape) == spec["x_shape"]
    _finite("TransformerAttention", out.float())
    _check("transformer_attention", out)


def test_transformer_attention_fused_matches_manual():
    """SDPA (fused) path must match the manual softmax attention path."""
    spec = CAPTURED["TransformerAttention"]
    _seed()
    m = ActionAttention(
        dim=spec["init"]["dim"],
        num_heads=spec["init"]["num_heads"],
        qkv_bias=spec["init"]["qkv_bias"],
        qk_norm=spec["init"]["qk_norm"],
    ).to(DEVICE).eval()
    x = _randn(spec["x_shape"], torch.float32)

    with torch.no_grad():
        m.fused_attn = True
        out_fused = m(x)
        m.fused_attn = False
        out_manual = m(x)

    max_diff = (out_fused - out_manual).abs().max().item()
    assert max_diff < 1e-4, f"fused vs manual diverged: max|Δ|={max_diff:.3e}"
    _check("transformer_fused_matches", out_fused)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
