# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the wall_oss_0_5 CUDA operator extension.
#
# Usage (from this directory):
#   pip install --no-build-isolation -e .
#
# Environment variables:
#   TORCH_CUDA_ARCH_LIST  Override target architectures (default: "8.0 12.0")
#   NVCC_THREADS          NVCC parallel compile threads  (default: 8)
#
"""Standalone setup to compile the wall_oss_0_5 CUDA operators."""

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC_DIR = Path("csrc")
PKG_DIR = ROOT / "wall_oss_05_op"

# Restrict to sm_80 and sm_120 only.
if not os.getenv("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 12.0"
    print("Targeting sm_80 and sm_120 only.")
else:
    print(f"Using TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

# Parallel NVCC compilation.
_NVCC_THREADS = ["--threads", os.getenv("NVCC_THREADS", "8")]

# Common NVCC flags shared by both modules.
# Undefine the CUDA "no-half" macros so bf16/fp16 built-in operators work.
_NVCC_COMMON = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-DNDEBUG",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
] + _NVCC_THREADS

_CXX_FLAGS = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

# Bitwise-exact kernels are built as a SEPARATE module: they must not see
# --use_fast_math (it substitutes the approximate __expf for expf and enables
# ftz, either of which silently breaks the bitwise match with eager PyTorch).
EXACT_DIRS = ("swiglu_exact", "rmsnorm_exact")
EXACT_BINDING = CSRC_DIR / "binding_exact.cu"


def _is_exact_source(path):
    """Return whether a CUDA source belongs to the exact-math extension."""
    return path.name == EXACT_BINDING.name or path.parent.name in EXACT_DIRS


def build_ext_modules():
    """Construct the standard and optional exact CUDA extension modules."""
    binding = CSRC_DIR / "binding.cu"
    if not (ROOT / binding).exists():
        raise RuntimeError(
            "CUDA operator sources are missing. Expected "
            f"{ROOT / binding} to exist."
        )

    cuda_sources = [binding] + sorted(
        path
        for path in (ROOT / CSRC_DIR).rglob("*.cu")
        if path.name != "binding.cu" and not _is_exact_source(path)
    )
    cuda_sources = [
        path if path == binding else path.relative_to(ROOT) for path in cuda_sources
    ]

    modules = [
        CUDAExtension(
            name="wall_oss_05_op._cuda_ext_bin",
            sources=[str(path) for path in cuda_sources],
            include_dirs=[str(CSRC_DIR), str(CSRC_DIR / "common")],
            extra_compile_args={
                "cxx": _CXX_FLAGS,
                "nvcc": _NVCC_COMMON,
            },
        )
    ]

    if (ROOT / EXACT_BINDING).exists():
        exact_sources = [EXACT_BINDING] + sorted(
            path.relative_to(ROOT)
            for path in (ROOT / CSRC_DIR).rglob("*.cu")
            if path.parent.name in EXACT_DIRS
        )
        # Exact kernels: no fast-math; keep full IEEE precision.
        _nvcc_exact = [
            "-O3",
            "-std=c++17",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            # belt-and-braces: kernels already use explicit rounding intrinsics
            "-fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ] + _NVCC_THREADS
        modules.append(
            CUDAExtension(
                name="wall_oss_05_op._cuda_ext_exact_bin",
                sources=[str(path) for path in exact_sources],
                include_dirs=[str(CSRC_DIR), str(CSRC_DIR / "common")],
                extra_compile_args={
                    "cxx": _CXX_FLAGS,
                    "nvcc": _nvcc_exact,
                },
            )
        )

    return modules


setup(
    name="wall_oss_05_op",
    version="1.0.0",
    description=(
        "CUDA operator kernels for wall_oss_0_5 "
        "(rope, m_rope, permute, rot_pos, window_index, get_rope_index, "
        "swiglu_exact, rmsnorm_exact)."
    ),
    license="Apache-2.0",
    classifiers=["License :: OSI Approved :: Apache Software License"],
    package_dir={"wall_oss_05_op": "wall_oss_05_op"},
    packages=["wall_oss_05_op"],
    ext_modules=build_ext_modules(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
