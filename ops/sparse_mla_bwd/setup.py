"""
setup sparse_mla_backward.
"""
import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


def is_flag_set(flag: str) -> bool:
    """
    is_flag_set.
    """
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]

def get_features_args():
    """
    get_features_args.
    """
    features_args = []
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args

def get_arch_flags():
    """
    check NVCC Version.
    """
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable.
    # For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), '--version'], stderr=subprocess.STDOUT
    ).decode('utf-8')
    nvcc_version_number = nvcc_version.split('release ')[1].split(',')[0].strip()
    major, minor = map(int, nvcc_version_number.split('.'))
    print(f'Compiling using NVCC {major}.{minor}')

    if major < 12 or (major == 12 and minor <= 8):
        raise RuntimeError("sm100 compilation for Flash MLA BWD requires NVCC 12.9 or higher. "
                           "Please update your environment.")

    arch_flags = ["-gencode", "arch=compute_100f,code=sm_100f"]
    return arch_flags

def get_nvcc_thread_args():
    """
    get_nvcc_thread_args.
    """
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]

subprocess.run(["git", "submodule", "update", "--init", "FlashMLA/csrc/cutlass"])

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_bwd.cuda",
        sources=[
              "src/pybind.cpp",
              "src/sm100/sparse_mla_bwd.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
                "-O3",
                "-std=c++20",
                "-DNDEBUG",
                "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v,--register-usage-level=10"
            ] + get_features_args() + get_arch_flags() + get_nvcc_thread_args(),
        },
        include_dirs=[
            Path(this_dir) / "src",
            Path(this_dir) / "src" / "sm100",
            Path(this_dir) / "FlashMLA" / "csrc",
            Path(this_dir) / "FlashMLA" / "csrc" / "kerutils" / "include",
            Path(this_dir) / "FlashMLA" / "csrc" / "cutlass" / "include",
            Path(this_dir) / "FlashMLA" / "csrc" / "cutlass" / "tools" / "util" / "include",
        ],
    )
)

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash_mla_bwd",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla_bwd']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)