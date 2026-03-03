"""Lightning Indexer Backward - Setup Script"""

import os
import sys
import setuptools
import subprocess
import torch
import shutil
from pathlib import Path
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

SKIP_CUDA_BUILD = int(os.getenv('SKIP_CUDA_BUILD', '0')) == 1
SKIP_DEEP_GEMM_CLONE = int(os.getenv('SKIP_DEEP_GEMM_CLONE', '1')) == 1
current_dir = os.path.dirname(os.path.realpath(__file__))

# DeepGEMM source configuration
DEEP_GEMM_REPO = "https://github.com/deepseek-ai/DeepGEMM.git"
VENDOR_DIR = os.path.join(current_dir, "vendor")
DEEP_GEMM_DIR = os.path.join(VENDOR_DIR, "DeepGEMM")
DEEP_GEMM_CSRCS_DIR = os.path.join(VENDOR_DIR, "deep_gemm_csrc")


def setup_deep_gemm():
    """Setup DeepGEMM: check/clone, install, and copy csrc."""

    if SKIP_CUDA_BUILD:
        print("SKIP_CUDA_BUILD is set, skipping DeepGEMM setup.")
        return

    # Check or clone DeepGEMM
    if SKIP_DEEP_GEMM_CLONE:
        if not os.path.exists(DEEP_GEMM_DIR):
            print("=" * 60)
            print("DeepGEMM directory not found!")
            print("=" * 60)
            print(f"\nExpected location: {DEEP_GEMM_DIR}")
            print("\nTo manually clone DeepGEMM, run:")
            print(f"  mkdir -p {VENDOR_DIR}")
            print(f"  git clone --recurse-submodules {DEEP_GEMM_REPO} {DEEP_GEMM_DIR}")
            print("\nOr set SKIP_DEEP_GEMM_CLONE=0 to auto-clone.")
            print("=" * 60)
            raise FileNotFoundError("DeepGEMM directory not found")
        print(f"Using manually cloned DeepGEMM at: {DEEP_GEMM_DIR}")
    else:
        os.makedirs(VENDOR_DIR, exist_ok=True)
        if os.path.exists(DEEP_GEMM_DIR):
            print(f"Removing existing DeepGEMM directory: {DEEP_GEMM_DIR}")
            shutil.rmtree(DEEP_GEMM_DIR)
        print(f"Cloning DeepGEMM from {DEEP_GEMM_REPO} with submodules...")
        subprocess.run(
            ["git", "clone", "--recurse-submodules", DEEP_GEMM_REPO, DEEP_GEMM_DIR],
            check=True
        )

    # Install DeepGEMM
    install_sh = os.path.join(DEEP_GEMM_DIR, "install.sh")
    setup_py = os.path.join(DEEP_GEMM_DIR, "setup.py")

    if os.path.exists(install_sh):
        print("Installing DeepGEMM using install.sh...")
        subprocess.run(["bash", "install.sh"], cwd=DEEP_GEMM_DIR, check=True)
    elif os.path.exists(setup_py):
        print("Installing DeepGEMM using setup.py...")
        subprocess.run([sys.executable, "-m", "pip", "install", "."], cwd=DEEP_GEMM_DIR, check=True)
    else:
        raise FileNotFoundError(f"Neither install.sh nor setup.py found in {DEEP_GEMM_DIR}")

    # Copy csrc
    src_csrc = os.path.join(DEEP_GEMM_DIR, "csrc")
    if os.path.exists(src_csrc):
        shutil.rmtree(DEEP_GEMM_CSRCS_DIR, ignore_errors=True)
        shutil.copytree(src_csrc, DEEP_GEMM_CSRCS_DIR)
        print(f"Copied {src_csrc} -> {DEEP_GEMM_CSRCS_DIR}")


def get_deep_gemm_include_dir():
    """Return the include directory path of the deep_gemm package."""

    try:
        import deep_gemm
        return os.path.join(os.path.dirname(deep_gemm.__file__), 'include')
    except ImportError:
        raise ImportError("deep_gemm is not installed. Please check the installation logs.")


# Compiler flags
cxx_flags = [
    '-std=c++17',
    '-O3',
    '-fPIC',
    '-Wno-psabi',
    '-Wno-deprecated-declarations',
    f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
]

sources = ['csrc/python_api.cpp']

build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    'csrc',
    'vendor/deep_gemm_csrc',
]

if not SKIP_CUDA_BUILD:
    build_include_dirs.append(get_deep_gemm_include_dir())

build_libraries = ['cudart', 'nvrtc']
build_library_dirs = [f'{CUDA_HOME}/lib64']


def get_ext_modules():
    """Return a list of CUDAExtension modules for building."""
    if SKIP_CUDA_BUILD:
        return []
    return [
        CUDAExtension(
            name='lightning_indexer_bwd._C',
            sources=sources,
            include_dirs=build_include_dirs,
            libraries=build_libraries,
            library_dirs=build_library_dirs,
            extra_compile_args=cxx_flags
        )
    ]


class CustomBuildPy(build_py):
    """Custom build command that sets up DeepGEMM and copies include files."""

    def run(self):
        """Execute the build process with DeepGEMM setup and include preparation."""
        setup_deep_gemm()
        self.prepare_includes()
        build_py.run(self)

    def prepare_includes(self):
        """Copy DeepGEMM include subdirectories to build directory."""
        build_include_dir = os.path.join(self.build_lib, 'lightning_indexer_bwd', 'include')
        os.makedirs(build_include_dir, exist_ok=True)

        dg_include_src = get_deep_gemm_include_dir()
        for subdir in ['deep_gemm', 'cute', 'cutlass']:
            src = os.path.join(dg_include_src, subdir)
            dst = os.path.join(build_include_dir, subdir)

            if not os.path.exists(src):
                print(f"Warning: {src} does not exist, skipping...")
                continue

            if os.path.exists(dst):
                shutil.rmtree(dst)

            shutil.copytree(src, dst)


def get_package_version():
    """Extract version string from the package __init__.py file."""
    init_file = Path(current_dir) / 'lightning_indexer_bwd' / '__init__.py'
    if init_file.exists():
        import re
        content = init_file.read_text()
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content, re.MULTILINE)
        if match:
            return match.group(1)
    return '0.0.1'


if __name__ == '__main__':
    setuptools.setup(
        name='lightning_indexer_bwd',
        version=get_package_version(),
        packages=find_packages('.'),
        package_data={
            'lightning_indexer_bwd': ['include/**/*'],
        },
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass={'build_py': CustomBuildPy},
    )
