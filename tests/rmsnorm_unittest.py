import os
import sys
import unittest
import torch
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
MEGATRON_PATH = Path("/workspace/AIAK-Megatron")


def _ensure_megatron_path():
    """Allow running tests without export by adding Megatron to sys.path."""
    fallback = REPO_ROOT / "AIAK-Megatron"

    # Ensure the repo itself is importable for aiak_training_omni.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for candidate in [MEGATRON_PATH, fallback]:
        if candidate and str(candidate) not in sys.path and candidate.exists():
            sys.path.insert(0, str(candidate))


_ensure_megatron_path()

from apex.normalization.fused_layer_norm import FusedRMSNorm as ApexFusedRMSNorm

# Optional override: set to a float (e.g., 2.5) to fix gamma scaling; None uses random 1-4 range.
GAMMA_SCALE = 3.5


class _AutoConfig(SimpleNamespace):
    """Config shim that supplies False/float defaults for unknown attrs on demand."""

    def __getattr__(self, name):
        default_val = torch.float32 if name.endswith("dtype") else False
        setattr(self, name, default_val)
        return default_val


class RmsNormParityTest(unittest.TestCase):
    """Compare outputs between TransformerEngine TENorm and Apex FusedRMSNorm."""

    def setUp(self):

        from megatron.core.extensions.transformer_engine import TENorm
        self.te_norm_cls = TENorm
        print("[setup] using megatron.core.extensions.transformer_engine.TENorm")

        if self.te_norm_cls is None:
            print("[setup] multiacc_modules.TENorm is None; skipping")
            self.skipTest("TENorm unavailable on this system.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[setup] device={self.device}, cuda_available={torch.cuda.is_available()}")
        print(f"[setup] te_norm_cls={self.te_norm_cls}")

    def _build_norms(self, hidden_size: int, eps: float, dtype: torch.dtype):
        try:
            apex_norm = ApexFusedRMSNorm(hidden_size, eps=eps, elementwise_affine=True).to(
                device=self.device, dtype=dtype
            )
        except Exception as exc:
            print(f"[_build_norms] Apex FusedRMSNorm init failed: {exc}")
            self.skipTest(f"Apex FusedRMSNorm unavailable: {exc}")

        try:
            te_norm = self._build_te_norm(hidden_size, eps)
        except Exception as exc:
            print(f"[ _build_norms] TENorm init failed: {exc}")
            self.skipTest(f"TENorm instantiation failed: {exc}")

        te_norm = te_norm.to(device=self.device, dtype=dtype)

        # Keep parameters identical so output diffs reflect implementation only.
        with torch.no_grad():
            # Use a shared random gamma each run to avoid training-time weights skewing the comparison.
            # Previously we copied Apex->TE, which inherited whatever weights Apex initialized with.
            # Now both get the exact same freshly-sampled gamma so differences reflect implementation only.
            if GAMMA_SCALE is not None:
                gamma_scale = GAMMA_SCALE
            else:
                # Randomly amplify to make differences more visible; reproducible under manual_seed.
                gamma_scale = 1.0 + 3.0 * torch.rand(1, device=self.device).item()
            print(f"[_build_norms] gamma_scale={gamma_scale:.4f}")
            gamma = torch.randn(hidden_size, device=self.device, dtype=dtype) * gamma_scale
            apex_norm.weight.copy_(gamma)
            if hasattr(te_norm, "weight"):
                te_norm.weight.copy_(gamma)
            if hasattr(te_norm, "bias") and te_norm.bias is not None:
                te_norm.bias.zero_()

        return apex_norm, te_norm

    def _build_te_norm(self, hidden_size: int, eps: float):
        try:
            params = list(inspect.signature(self.te_norm_cls).parameters.values())
        except (TypeError, ValueError):
            params = []

        names = [p.name for p in params]
        expects_config_first = bool(names) and names[0] in ("config", "cfg")
        shape_kw = "normalized_shape" if "normalized_shape" in names else "hidden_size"
        eps_kw = "eps" in names

        if not expects_config_first:
            kwargs = {shape_kw: hidden_size}
            if eps_kw:
                kwargs["eps"] = eps
            if "sequence_parallel" in names:
                kwargs["sequence_parallel"] = False
            try:
                return self.te_norm_cls(**kwargs)
            except TypeError:
                return self.te_norm_cls(hidden_size, eps)

        # Config-first signatures; _AutoConfig fills any unexpected attrs on access.
        config = _AutoConfig(
            sequence_parallel=False,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=False,
            return_3d_tensor=False,
            params_dtype=torch.float32,
            use_cpu_initialization=False,
            init_model_with_meta_device=False,
        )

        kwargs = {shape_kw: hidden_size}
        if eps_kw:
            kwargs["eps"] = eps

        try:
            return self.te_norm_cls(config, **kwargs)
        except TypeError:
            return self.te_norm_cls(config, hidden_size, eps)

    def test_bf16_output_matches(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 not supported on this device.")
        if self.device != "cuda":  # Apex FusedRMSNorm is CUDA-only
            self.skipTest("CUDA required for BF16 parity test.")

        self._run_parity_case(shape=(325, 1, 32, 128), eps=1e-6, dtype=torch.bfloat16, seed=1)

    def test_output_matches_multiple_shapes_and_eps(self):
        if self.device != "cuda":  # Apex FusedRMSNorm is CUDA-only
            self.skipTest("CUDA required for parity tests.")

        cases = [
            {"shape": (2, 128), "eps": 1e-5, "dtype": torch.float16, "seed": 2},
            {"shape": (4, 3, 64), "eps": 1e-6, "dtype": torch.float16, "seed": 3},
            {"shape": (1, 2, 2, 256), "eps": 1e-8, "dtype": torch.float32, "seed": 4},
        ]

        # Add a bf16 coverage point when supported.
        if torch.cuda.is_bf16_supported():
            cases.append({"shape": (8, 16, 128), "eps": 1e-6, "dtype": torch.bfloat16, "seed": 5})

        for case in cases:
            with self.subTest(case=case):
                self._run_parity_case(**case)

    def test_parity_multiple_seeds(self):
        if self.device != "cuda":  # Apex FusedRMSNorm is CUDA-only
            self.skipTest("CUDA required for parity tests.")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 not supported on this device.")

        seeds = [1, 2, 3, 4, 5, 1224,232,232,9999]
        case = {"shape": (325, 1, 32, 128), "eps": 1e-6, "dtype": torch.bfloat16}

        for seed in seeds:
            with self.subTest(seed=seed):
                self._run_parity_case(seed=seed, **case)

    def _run_parity_case(self, shape, eps, dtype, seed):
        torch.manual_seed(seed)
        hidden_size = shape[-1]

        apex_norm, te_norm = self._build_norms(hidden_size, eps, dtype)

        # Use non-zero mean/variance to exercise scaling paths.
        mean = 0.017822265625
        var = 0.609375
        std = math.sqrt(var)
        x = torch.normal(mean=mean, std=std, size=shape, device=self.device, dtype=dtype)

        apex_out = apex_norm(x)
        te_out = te_norm(x)

        # Float32 reference to pinpoint which side drifts.
        ref_out = self._rms_ref(x, eps, apex_norm.weight)

        tag = f"{str(dtype).split('.')[-1]}_eps{eps}_shape{shape}_seed{seed}"
        self._log_example(x, apex_out, te_out, ref_out, tag)

        # Relax tolerances for lower precision dtypes.
        if dtype == torch.float16:
            atol, rtol = 1e-3, 1e-3
        elif dtype == torch.bfloat16:
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 1e-6, 1e-6

        # Capture stats so failures still show mean/std info.
        apex_mean = apex_out.mean().item()
        apex_var = apex_out.var(unbiased=False).item()
        te_mean = te_out.mean().item()
        te_var = te_out.var(unbiased=False).item()
        ref_mean = ref_out.mean().item()
        ref_var = ref_out.var(unbiased=False).item()

        try:
            torch.testing.assert_close(te_out, apex_out, rtol=rtol, atol=atol)
        except AssertionError as err:
            diff = (te_out - apex_out).abs()
            max_val = diff.max().item()
            max_idx = torch.nonzero(diff == max_val, as_tuple=False)[0].tolist()

            def grab_slice(t):
                if t.dim() >= 4:
                    sl = t[max_idx[0], max_idx[1], max_idx[2], :8]
                elif t.dim() == 3:
                    sl = t[max_idx[0], max_idx[1], :8]
                elif t.dim() == 2:
                    sl = t[max_idx[0], :8]
                else:
                    sl = t.view(-1)[:8]
                return [f"{v:.6f}" for v in sl.detach().cpu().tolist()]

            extra = "\n" + "\n".join(
                [
                    f"[{tag}] apex: mean={apex_mean:.9f} std={math.sqrt(apex_var):.9f}",
                    f"[{tag}] te  : mean={te_mean:.9f} std={math.sqrt(te_var):.9f}",
                    f"[{tag}] ref : mean={ref_mean:.9f} std={math.sqrt(ref_var):.9f}",
                    f"[{tag}] apex_vs_ref_max={(apex_out - ref_out).abs().max().item():.9f}",
                    f"[{tag}]   te_vs_ref_max={(te_out - ref_out).abs().max().item():.9f}",
                    f"[{tag}] max_abs_diff={max_val:.9f} at idx={tuple(max_idx)}",
                    f"[{tag}] apex_slice: {grab_slice(apex_out)}",
                    f"[{tag}]   te_slice: {grab_slice(te_out)}",
                    f"[{tag}]  ref_slice: {grab_slice(ref_out)}",
                ]
            )
            err.args = (str(err) + extra, *err.args[1:])
            raise

    def _rms_ref(self, x, eps, gamma):
        ref = x.float()
        rms = torch.rsqrt(ref.pow(2).mean(dim=-1, keepdim=True) + eps)
        ref = ref * rms
        ref = ref * gamma.float()
        return ref.to(dtype=x.dtype)

    def _log_example(self, x, apex_out, te_out, ref_out, tag):
        # Print a small slice to keep output concise.
        with torch.no_grad():
            def slice_last(t):
                if t.dim() >= 4:
                    return t[0, 0, 0, :5], "[0,0,0,:5]"
                if t.dim() == 3:
                    return t[0, 0, :5], "[0,0,:5]"
                if t.dim() == 2:
                    return t[0, :5], "[0,:5]"
                return t.view(-1)[:5], "[:5]"

            def fmt_slice(t):
                sl, idx = slice_last(t)
                return [f"{v:.6f}" for v in sl.detach().cpu().tolist()], idx

            input_slice, input_idx = fmt_slice(x)
            apex_slice, apex_idx = fmt_slice(apex_out)
            te_slice, te_idx = fmt_slice(te_out)

            print(f"[{tag}] input{input_idx}: {input_slice} shape={tuple(x.shape)}")
            print(
                f"[{tag}] input_mean={x.mean().item():.6f} input_var={x.var(unbiased=False).item():.6f}"
            )
            print(
                f"[{tag}] ref_mean={ref_out.mean().item():.6f} ref_var={ref_out.var(unbiased=False).item():.6f}"
            )
            print(f"[{tag}] apex_out{apex_idx}: {apex_slice}")
            print(f"[{tag}] te_out{te_idx}: {te_slice}")
            print(
                f"[{tag}] apex_mean={apex_out.mean().item():.6f} apex_var={apex_out.var(unbiased=False).item():.6f}"
            )
            print(
                f"[{tag}] te_mean={te_out.mean().item():.6f} te_var={te_out.var(unbiased=False).item():.6f}"
            )
            diff = (te_out - apex_out).abs().max()
            print(f"[{tag}] max_abs_diff: {diff.item():.6f}")
            print(
                f"[{tag}] apex_vs_ref_max: {(apex_out - ref_out).abs().max().item():.6f} "
                f"te_vs_ref_max: {(te_out - ref_out).abs().max().item():.6f}"
            )


if __name__ == "__main__":
    unittest.main()
