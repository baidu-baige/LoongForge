#! /bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime identity.
PYTHON_BIN=${PYTHON_BIN:-"python"}

# Python dependencies.
MODELOPT_EXPECTED_VERSION=${MODELOPT_EXPECTED_VERSION:-}
MODELOPT_INSTALL_SOURCE=${MODELOPT_INSTALL_SOURCE:-"local_repo"}
MODELOPT_PIP_SPEC=${MODELOPT_PIP_SPEC:-}
TRANSFORMERS_PIP_SPEC=${TRANSFORMERS_PIP_SPEC:-"transformers==4.57.3"}
DATASETS_PIP_SPEC=${DATASETS_PIP_SPEC:-"datasets>=3.0.0"}
ACCELERATE_PIP_SPEC=${ACCELERATE_PIP_SPEC:-"accelerate>=1.0.0"}
BLOBFILE_PIP_SPEC=${BLOBFILE_PIP_SPEC:-"blobfile"}
MODELOPT_EXTRA_PIP_PACKAGES=${MODELOPT_EXTRA_PIP_PACKAGES:-}
INSTALL_MODELOPT=${INSTALL_MODELOPT:-1}

# blobfile is installed from the internal mirror because it is required by Kimi remote code.
BLOBFILE_PIP_INDEX=${BLOBFILE_PIP_INDEX:-"http://mirrors.baidubce.com/pypi/simple/"}
BLOBFILE_PIP_TRUSTED_HOST=${BLOBFILE_PIP_TRUSTED_HOST:-"mirrors.baidubce.com"}

# Patch noisy/unpicklable Transformers processor logging if needed by the runtime.
PATCH_TRANSFORMERS_PROCESSING_UTILS=${PATCH_TRANSFORMERS_PROCESSING_UTILS:-1}

# Runtime ModelOpt script checkout. Pin by default to the commit that completed
# Kimi K2.6 NVFP4 export in convert_nvfp4_20260522-175104.log.
MODELOPT_REPO_REF=${MODELOPT_REPO_REF:-"b02e8885509c53b4e187f9fd5f56c5497e937d7e"}
MODELOPT_REPO_BASE=${MODELOPT_REPO_BASE:-"${TMPDIR:-/tmp}/kimi_modelopt_repos"}
MODELOPT_REPO=${MODELOPT_REPO:-"${MODELOPT_REPO_BASE}/Model-Optimizer-kimi-k2.6-b02e888"}
MODELOPT_REPO_URL=${MODELOPT_REPO_URL:-"https://github.com/NVIDIA/Model-Optimizer.git"}
CLONE_MODELOPT_REPO=${CLONE_MODELOPT_REPO:-1}
RECLONE_MODELOPT_REPO=${RECLONE_MODELOPT_REPO:-0}
UPDATE_MODELOPT_REPO=${UPDATE_MODELOPT_REPO:-1}

if [ -z "$MODELOPT_PIP_SPEC" ]; then
    case "$MODELOPT_INSTALL_SOURCE" in
        local_repo)
            MODELOPT_PIP_SPEC="${MODELOPT_REPO}[hf]"
            ;;
        pypi)
            if [ -z "$MODELOPT_EXPECTED_VERSION" ]; then
                MODELOPT_PIP_SPEC="nvidia-modelopt[hf]"
            else
                MODELOPT_PIP_SPEC="nvidia-modelopt[hf]==${MODELOPT_EXPECTED_VERSION}"
            fi
            ;;
        *)
            echo "Unsupported MODELOPT_INSTALL_SOURCE=$MODELOPT_INSTALL_SOURCE; expected local_repo or pypi." >&2
            exit 1
            ;;
    esac
fi

# GitHub access may need proxy in the cluster pods.
ENABLE_GIT_PROXY=${ENABLE_GIT_PROXY:-1}
GIT_PROXY_URL=${GIT_PROXY_URL:-"http://agent.baidu.com:8891"}

# Checkout validation knobs.
CHECK_MODELOPT_REPO_REF=${CHECK_MODELOPT_REPO_REF:-0}
PATCH_MODELOPT_TOKENIZER_DEEPCOPY=${PATCH_MODELOPT_TOKENIZER_DEEPCOPY:-1}

git_with_optional_proxy() {
    if [ "$ENABLE_GIT_PROXY" = "1" ]; then
        env \
            http_proxy="${http_proxy:-$GIT_PROXY_URL}" \
            https_proxy="${https_proxy:-$GIT_PROXY_URL}" \
            HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-$GIT_PROXY_URL}}" \
            HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-$GIT_PROXY_URL}}" \
            git "$@"
    else
        git "$@"
    fi
}

clone_modelopt_repo_if_needed() {
    if [ "$CLONE_MODELOPT_REPO" != "1" ]; then
        echo "CLONE_MODELOPT_REPO=$CLONE_MODELOPT_REPO: skip ModelOpt git clone."
        return
    fi

    mkdir -p "$(dirname "$MODELOPT_REPO")"
    if [ -d "$MODELOPT_REPO/.git" ]; then
        echo "ModelOpt repo already exists: $MODELOPT_REPO"
        return
    fi
    if [ -e "$MODELOPT_REPO" ]; then
        if [ "$RECLONE_MODELOPT_REPO" != "1" ]; then
            echo "MODELOPT_REPO exists but is not a git repo: $MODELOPT_REPO" >&2
            echo "Set RECLONE_MODELOPT_REPO=1 to remove it and clone again." >&2
            exit 1
        fi
        echo "RECLONE_MODELOPT_REPO=1: removing non-git path $MODELOPT_REPO"
        rm -rf "$MODELOPT_REPO"
    fi

    echo "Cloning ModelOpt repo:"
    echo "  url:  $MODELOPT_REPO_URL"
    echo "  dest: $MODELOPT_REPO"
    if [ "$ENABLE_GIT_PROXY" = "1" ]; then
        echo "  proxy: $GIT_PROXY_URL"
    else
        echo "  proxy: disabled"
    fi
    git_with_optional_proxy clone "$MODELOPT_REPO_URL" "$MODELOPT_REPO"
}

modelopt_repo_current_ref() {
    git -C "$MODELOPT_REPO" branch --show-current 2>/dev/null | grep -v '^$' || \
        git -C "$MODELOPT_REPO" describe --tags --exact-match 2>/dev/null || \
        git -C "$MODELOPT_REPO" rev-parse --short HEAD
}

resolve_modelopt_checkout_ref() {
    if git -C "$MODELOPT_REPO" rev-parse --verify --quiet "origin/${MODELOPT_REPO_REF}^{commit}" >/dev/null; then
        printf 'origin/%s\n' "$MODELOPT_REPO_REF"
    else
        printf '%s\n' "$MODELOPT_REPO_REF"
    fi
}

fetch_modelopt_checkout_ref() {
    if git_with_optional_proxy -C "$MODELOPT_REPO" fetch origin \
        "+refs/heads/${MODELOPT_REPO_REF}:refs/remotes/origin/${MODELOPT_REPO_REF}" 2>/dev/null; then
        printf 'origin/%s\n' "$MODELOPT_REPO_REF"
        return
    fi

    if git_with_optional_proxy -C "$MODELOPT_REPO" fetch origin \
        "+refs/tags/${MODELOPT_REPO_REF}:refs/tags/${MODELOPT_REPO_REF}" 2>/dev/null; then
        printf '%s\n' "$MODELOPT_REPO_REF"
        return
    fi

    if git_with_optional_proxy -C "$MODELOPT_REPO" fetch origin "$MODELOPT_REPO_REF"; then
        printf '%s\n' "$MODELOPT_REPO_REF"
        return
    fi

    echo "Direct fetch for $MODELOPT_REPO_REF failed; fetching origin refs and resolving locally." >&2
    git_with_optional_proxy -C "$MODELOPT_REPO" fetch origin
    resolve_modelopt_checkout_ref
}

update_modelopt_repo_to_ref() {
    clone_modelopt_repo_if_needed
    if [ ! -d "$MODELOPT_REPO/.git" ]; then
        echo "MODELOPT_REPO is not a git repo: $MODELOPT_REPO" >&2
        exit 1
    fi

    if [ "$UPDATE_MODELOPT_REPO" != "1" ]; then
        echo "UPDATE_MODELOPT_REPO=$UPDATE_MODELOPT_REPO: keep existing ModelOpt checkout."
        return
    fi

    echo "Updating ModelOpt repo to ${MODELOPT_REPO_REF}"
    git -C "$MODELOPT_REPO" reset --hard
    git -C "$MODELOPT_REPO" clean -fd

    local checkout_ref
    checkout_ref=$(fetch_modelopt_checkout_ref)
    git -C "$MODELOPT_REPO" checkout --detach "$checkout_ref"
    git -C "$MODELOPT_REPO" reset --hard "$checkout_ref"
}

patch_modelopt_tokenizer_deepcopy() {
    if [ "$PATCH_MODELOPT_TOKENIZER_DEEPCOPY" != "1" ]; then
        echo "PATCH_MODELOPT_TOKENIZER_DEEPCOPY=$PATCH_MODELOPT_TOKENIZER_DEEPCOPY: skip ModelOpt tokenizer deepcopy patch."
        return
    fi

    local dataset_utils_path before_matches before_count after_count removed_count
    dataset_utils_path="$MODELOPT_REPO/modelopt/torch/utils/dataset_utils.py"
    if [ ! -f "$dataset_utils_path" ]; then
        echo "Missing ModelOpt dataset_utils.py: $dataset_utils_path" >&2
        exit 1
    fi

    before_matches=$(grep -n 'tokenizer = copy\.deepcopy(tokenizer)' "$dataset_utils_path" || true)
    before_count=$(grep -c 'tokenizer = copy\.deepcopy(tokenizer)' "$dataset_utils_path" || true)
    if [ "$before_count" -eq 0 ]; then
        echo "ModelOpt tokenizer deepcopy patch already clean: $dataset_utils_path"
        return
    fi

    echo "ModelOpt tokenizer deepcopy patch removing $before_count line(s) from $dataset_utils_path:"
    printf '%s\n' "$before_matches" | sed 's/^/  /'
    sed -i '/Tokenizer encoding may modify the tokenizer in place, so we need to clone it\./d' "$dataset_utils_path"
    sed -i '/batch_encode_plus will modify the tokenizer in place, so we need to clone it\./d' "$dataset_utils_path"
    sed -i '/tokenizer = copy\.deepcopy(tokenizer)/d' "$dataset_utils_path"
    after_count=$(grep -c 'tokenizer = copy\.deepcopy(tokenizer)' "$dataset_utils_path" || true)
    removed_count=$((before_count - after_count))
    echo "ModelOpt tokenizer deepcopy patch applied: removed $removed_count line(s), remaining_matches=$after_count"
}

update_modelopt_repo_to_ref
patch_modelopt_tokenizer_deepcopy

echo "ModelOpt repo ready for runtime use: $MODELOPT_REPO"

if [ -d "$MODELOPT_REPO/.git" ]; then
    MODELOPT_REPO_CURRENT_REF=$(modelopt_repo_current_ref)
    MODELOPT_REPO_COMMIT=$(git -C "$MODELOPT_REPO" rev-parse HEAD)
    MODELOPT_EXPECTED_COMMIT=$(
        git -C "$MODELOPT_REPO" rev-parse "${MODELOPT_REPO_REF}^{commit}" 2>/dev/null || \
            git -C "$MODELOPT_REPO" rev-parse "origin/${MODELOPT_REPO_REF}^{commit}" 2>/dev/null || \
            true
    )
    echo "ModelOpt repo ref: $MODELOPT_REPO_CURRENT_REF"
    echo "ModelOpt repo commit: $MODELOPT_REPO_COMMIT"
    if [ "$CHECK_MODELOPT_REPO_REF" = "1" ] && [ -n "$MODELOPT_EXPECTED_COMMIT" ] && [ "$MODELOPT_REPO_COMMIT" != "$MODELOPT_EXPECTED_COMMIT" ]; then
        echo "WARNING: ModelOpt HEAD $MODELOPT_REPO_COMMIT does not match MODELOPT_REPO_REF=$MODELOPT_REPO_REF ($MODELOPT_EXPECTED_COMMIT)."
    fi
fi

modelopt_version_report() {
    "$PYTHON_BIN" - "$MODELOPT_EXPECTED_VERSION" <<'PY'
import importlib.metadata as md
import sys

expected = sys.argv[1]
try:
    got = md.version("nvidia-modelopt")
except md.PackageNotFoundError:
    print("nvidia-modelopt is not installed.")
    raise SystemExit(1)

if expected and got != expected:
    print(f"Installed nvidia-modelopt version {got} does not match expected {expected}.")
    raise SystemExit(1)

if expected:
    print(f"Installed nvidia-modelopt version matches expected {expected}.")
else:
    print(f"Installed nvidia-modelopt version: {got}")
PY
}

hf_dependencies_match() {
    "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
from packaging.version import Version

versions = {}
for pkg in ("transformers", "datasets", "accelerate", "blobfile"):
    try:
        versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        print(f"{pkg} is not installed.")
        raise SystemExit(1)

if Version(versions["transformers"]) >= Version("5.0.0"):
    print(
        "Installed transformers version "
        f"{versions['transformers']} is incompatible with Kimi remote code; expected <5.0."
    )
    raise SystemExit(1)

if Version(versions["datasets"]) < Version("3.0.0"):
    print(
        "Installed datasets version "
        f"{versions['datasets']} is too old for ModelOpt HF PTQ; expected >=3.0.0."
    )
    raise SystemExit(1)

print("Installed HF dependency versions:", versions)

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import transformers
import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils._torchvision_available = False
transformers_import_utils._torchvision_version = "N/A"

from transformers import (  # noqa: F401
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

print("Transformers symbol import check passed.")
PY
}

patch_transformers_processing_utils() {
    if [ "$PATCH_TRANSFORMERS_PROCESSING_UTILS" != "1" ]; then
        echo "PATCH_TRANSFORMERS_PROCESSING_UTILS=$PATCH_TRANSFORMERS_PROCESSING_UTILS: skip Transformers processing_utils patch."
        return
    fi

    local processing_utils
    if ! processing_utils=$(
        "$PYTHON_BIN" -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'processing_utils.py'))"
    ); then
        echo "processing_utils patch failed: could not locate transformers processing_utils.py, continuing"
        return
    fi

    if [ ! -f "$processing_utils" ]; then
        echo "processing_utils patch failed: missing file $processing_utils, continuing"
        return
    fi

    local before_matches before_count after_count removed_count
    before_matches=$(grep -n 'logger\.info.*Processor.*processor' "$processing_utils" || true)
    before_count=$(grep -c 'logger\.info.*Processor.*processor' "$processing_utils" || true)

    if [ "$before_count" -eq 0 ]; then
        echo "processing_utils patch already clean: $processing_utils"
        return
    fi

    echo "processing_utils patch removing $before_count line(s) from $processing_utils:"
    printf '%s\n' "$before_matches" | sed 's/^/  /'

    if ! sed -i '/logger\.info.*Processor.*processor/d' "$processing_utils"; then
        echo "processing_utils patch failed, continuing"
        return
    fi

    after_count=$(grep -c 'logger\.info.*Processor.*processor' "$processing_utils" || true)
    removed_count=$((before_count - after_count))
    echo "processing_utils patch applied: removed $removed_count line(s), remaining_matches=$after_count"
}

if [ "$INSTALL_MODELOPT" != "1" ]; then
    echo "INSTALL_MODELOPT=$INSTALL_MODELOPT: skip pip install."
    modelopt_version_report
    hf_dependencies_match
    patch_transformers_processing_utils
    exit 0
fi

echo "Installing ModelOpt/HF packages:"
echo "  MODELOPT_INSTALL_SOURCE=$MODELOPT_INSTALL_SOURCE"
echo "  $MODELOPT_PIP_SPEC"
echo "  $TRANSFORMERS_PIP_SPEC"
echo "  $DATASETS_PIP_SPEC"
echo "  $ACCELERATE_PIP_SPEC"
echo "  $BLOBFILE_PIP_SPEC"

if [ -n "$MODELOPT_EXTRA_PIP_PACKAGES" ]; then
    "$PYTHON_BIN" -m pip install \
        "$MODELOPT_PIP_SPEC" \
        "$TRANSFORMERS_PIP_SPEC" \
        "$DATASETS_PIP_SPEC" \
        "$ACCELERATE_PIP_SPEC" \
        $MODELOPT_EXTRA_PIP_PACKAGES
else
    "$PYTHON_BIN" -m pip install \
        "$MODELOPT_PIP_SPEC" \
        "$TRANSFORMERS_PIP_SPEC" \
        "$DATASETS_PIP_SPEC" \
        "$ACCELERATE_PIP_SPEC"
fi

"$PYTHON_BIN" -m pip install "$BLOBFILE_PIP_SPEC" \
    -i "$BLOBFILE_PIP_INDEX" \
    --trusted-host "$BLOBFILE_PIP_TRUSTED_HOST" \
    -q

modelopt_version_report
hf_dependencies_match
patch_transformers_processing_utils
