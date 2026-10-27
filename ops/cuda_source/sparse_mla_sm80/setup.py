# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build script for the SM80 sparse MLA extension.

Targets sm_80 by default and honours TORCH_CUDA_ARCH_LIST, following the
convention of cuda_source/gr00t_n1_7_op/setup.py rather than the hard-coded
`-gencode` used by the SM100-only attention ops in this repository.

Unlike gr00t_n1_7_op the arch list is *filtered*: the kernel uses
`mma.sync.aligned.m16n8k16...bf16`, which ptxas rejects below sm_80.  Many images
ship a broad TORCH_CUDA_ARCH_LIST (e.g. "7.5 8.0 8.6 9.0 10.0 12.0+PTX"), so
honouring it verbatim breaks the build.
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

MIN_ARCH = 80


def _resolve_arch_list() -> str:
    requested = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if not requested:
        return "8.0"
    kept, dropped = [], []
    for entry in requested.replace(",", " ").split():
        try:
            major, minor = entry.split("+")[0].split(".")[:2]
            sm = int(major) * 10 + int(minor)
        except ValueError:
            dropped.append(entry)
            continue
        (kept if sm >= MIN_ARCH else dropped).append(entry)
    if dropped:
        print(f"[flash_mla_sm80] dropping arch(s) below sm_{MIN_ARCH}: {' '.join(dropped)}")
    if not kept:
        raise SystemExit(
            f"TORCH_CUDA_ARCH_LIST={requested!r} contains no architecture >= sm_{MIN_ARCH}; "
            "this kernel requires bf16 mma."
        )
    return " ".join(kept)


os.environ["TORCH_CUDA_ARCH_LIST"] = _resolve_arch_list()
print(f"[flash_mla_sm80] TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")


setup(
    name="flash_mla_sm80",
    version="0.1.0",
    description="SM80 sparse MLA (forward) for DeepSeek Sparse Attention",
    packages=find_packages(include=["flash_mla_sm80", "flash_mla_sm80.*"]),
    ext_modules=[
        CUDAExtension(
            name="flash_mla_sm80.cuda",
            sources=[
                "src/pybind.cpp",
                "src/sparse_mla_fwd_sm80.cu",
                "src/sparse_mla_bwd_sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--ptxas-options=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)