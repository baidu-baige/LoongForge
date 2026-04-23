"""Transformer Engine parallel layer performance benchmark.

Benchmarks forward + backward timing for four TE Linear layer types:
TELayerNormColumnParallelLinear, TERowParallelLinear,
TEColumnParallelGroupedLinear, and TERowParallelGroupedLinear.
Supports BF16 and FP8 precisions. Reports per-case latency, throughput,
and achieved TFLOPS.

Usage
-----

1. Run with built-in defaults (qwen3_vl_vit + qwen3_vl_llm_30b +
   qwen3_vl_llm_235b, each with its own shape sweep)::

    # via pytest
    pytest tools/benchmark_te_parallel_layers.py -s

    # direct execution
    python tools/benchmark_te_parallel_layers.py

2. Run a specific model subset (vision / llm / llm_235b)::

    TE_LAYER_PERF_CASESET=vision pytest ... -s
    TE_LAYER_PERF_CASESET=llm   pytest ... -s

3. Load model config from an LoongForge Hydra YAML::

    # VL model (auto-resolves Hydra defaults for image_encoder / foundation)
    TE_LAYER_PERF_OMNI_CONFIG_PATH=path/to/configs/models/qwen3_vl/qwen3_vl_235b_a22b.yaml \
        pytest ... -s

    # Standalone LLM (dense or MoE)
    TE_LAYER_PERF_OMNI_CONFIG_PATH=path/to/configs/models/qwen3/qwen3_8b.yaml \
        pytest ... -s

    # Standalone ViT / image encoder
    TE_LAYER_PERF_OMNI_CONFIG_PATH=path/to/configs/models/image_encoder/qwen3_vit.yaml \
        pytest ... -s

   Supports three config types:
   - Top-level VL configs with Hydra ``defaults`` (extracts both ViT and LLM)
   - Standalone LLM configs (dense models produce fc1/fc2 cases;
     MoE models produce grouped GEMM cases)
   - Standalone ViT / image encoder configs

4. Override shape sweep (sequence_length x micro_batch_size)::

    TE_LAYER_PERF_SHAPE_SWEEP="1024x1,4096x2,8192x1" pytest ... -s

5. Override precisions (default: "bf16,fp8")::

    # BF16 only
    TE_LAYER_PERF_PRECISIONS=bf16 pytest ... -s

    # FP8 only
    TE_LAYER_PERF_PRECISIONS=fp8 pytest ... -s

6. Override tensor/expert parallelism (requires torchrun with matching GPU count)::

    # TP=2 (needs 2 GPUs)
    TE_LAYER_PERF_TP_SIZE=2 torchrun --nproc_per_node 2 \
        tools/benchmark_te_parallel_layers.py

    # TP=2 + EP=4 (needs 8 GPUs, since world_size must be divisible by tp*ep)
    TE_LAYER_PERF_TP_SIZE=2 TE_LAYER_PERF_EP_SIZE=4 \
        torchrun --nproc_per_node 8 tools/benchmark_te_parallel_layers.py

Environment Variables
---------------------

Model / case configuration:
    TE_LAYER_PERF_OMNI_CONFIG_PATH  Load model config from an Omni Training YAML
    TE_LAYER_PERF_CASES_PATH        Load model config from a JSON/YAML file
    TE_LAYER_PERF_CASES_JSON        Load model config from a JSON string
    TE_LAYER_PERF_CASESET           Built-in subset: all|vision|llm|llm_235b (default: all)
    TE_LAYER_PERF_SHAPE_SWEEP       Override shape sweep, e.g. "1024x1,4096x2"

    Priority: OMNI_CONFIG_PATH > CASES_PATH > CASES_JSON > built-in defaults

Parallelism overrides (applied after model loading, requires multi-GPU launch):
    TE_LAYER_PERF_TP_SIZE           Override tensor_model_parallel_size (0 = model default)
    TE_LAYER_PERF_EP_SIZE           Override expert_model_parallel_size (0 = model default)
    TE_LAYER_PERF_ETP_SIZE          Override expert_tensor_parallel_size (0 = model default)
    TE_LAYER_PERF_SEQ_PARALLEL      Override sequence_parallel ("true"/"false", unset = model default)

Benchmark parameters:
    TE_LAYER_PERF_WARMUP            Number of warmup iterations (default: 10)
    TE_LAYER_PERF_ITERS             Number of timed iterations (default: 10)
    TE_LAYER_PERF_PRECISIONS        Precisions to test, comma-separated (default: "bf16,fp8")
    TE_LAYER_PERF_FP8_RECIPE        FP8 recipe: blockwise|delayed|current (default: blockwise)
    TE_LAYER_PERF_RECOMPUTE         Enable activation recompute (default: true)
    TE_LAYER_PERF_SPLIT_SKEW        Zipf skew for grouped GEMM token splits (default: 1.2)

Output:
    TE_LAYER_PERF_REPORT_PATH       Write results to a JSON file (optional)
    TE_LAYER_PERF_FP8_POLICY_PATH   Export FP8 dynamic policy JSON for training (optional)
    TE_LAYER_PERF_SPEEDUP_THRESHOLD Minimum speedup to consider FP8 beneficial (default: 1.0)
"""

import gc
import json
import math
import os
import re
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pytest
import torch
import torch.utils.checkpoint
import yaml

os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("GLOO_LOG_LEVEL", "WARN")

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.fp8_utils import get_fp8_align_size, get_fp8_context
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support

    FP8_AVAILABLE, FP8_UNAVAILABLE_REASON = check_fp8_support()
except Exception as error:
    FP8_AVAILABLE = False
    FP8_UNAVAILABLE_REASON = str(error)


# ============================================================================
# Section 1: Environment variables & constants
# ============================================================================

PERF_WARMUP = int(os.getenv("TE_LAYER_PERF_WARMUP", "10"))
PERF_ITERS = int(os.getenv("TE_LAYER_PERF_ITERS", "10"))
PERF_REPORT_PATH = os.getenv("TE_LAYER_PERF_REPORT_PATH")
PERF_CASESET = os.getenv("TE_LAYER_PERF_CASESET", "all").strip().lower()
PERF_CASES_PATH = os.getenv("TE_LAYER_PERF_CASES_PATH")
PERF_CASES_JSON = os.getenv("TE_LAYER_PERF_CASES_JSON")
PERF_FP8_RECIPE = os.getenv("TE_LAYER_PERF_FP8_RECIPE", "blockwise").strip().lower()
PERF_PRECISIONS = tuple(
    item.strip().lower()
    for item in os.getenv("TE_LAYER_PERF_PRECISIONS", "bf16,fp8").split(",")
    if item.strip()
)
PERF_SPLIT_SKEW = float(os.getenv("TE_LAYER_PERF_SPLIT_SKEW", "1.2"))
PERF_RECOMPUTE = os.getenv("TE_LAYER_PERF_RECOMPUTE", "true").strip().lower() in ("true", "1", "yes")
PERF_SHAPE_SWEEP = os.getenv("TE_LAYER_PERF_SHAPE_SWEEP")  # e.g. "1024x1,2048x4,8192x2"
PERF_OMNI_CONFIG_PATH = os.getenv("TE_LAYER_PERF_OMNI_CONFIG_PATH")  # Omni Training YAML path
PERF_TP_SIZE = int(os.getenv("TE_LAYER_PERF_TP_SIZE", "0"))  # override tensor_model_parallel_size (0 = use model default)
PERF_EP_SIZE = int(os.getenv("TE_LAYER_PERF_EP_SIZE", "0"))  # override expert_model_parallel_size (0 = use model default)
PERF_ETP_SIZE = int(os.getenv("TE_LAYER_PERF_ETP_SIZE", "0"))  # override expert_tensor_parallel_size (0 = use model default)
PERF_SEQ_PARALLEL = os.getenv("TE_LAYER_PERF_SEQ_PARALLEL")  # override sequence_parallel ("true"/"false", None = use model default)
PERF_FP8_POLICY_PATH = os.getenv("TE_LAYER_PERF_FP8_POLICY_PATH")  # export FP8 dynamic policy JSON
PERF_SPEEDUP_THRESHOLD = float(os.getenv("TE_LAYER_PERF_SPEEDUP_THRESHOLD", "1.0"))

# Module kind constants (shared by case building, FP8 policy, and reporting).
_MODULE_KIND_TO_CLASS = {
    "layernorm_column": TELayerNormColumnParallelLinear,
    "column": TEColumnParallelLinear,
    "row": TERowParallelLinear,
    "column_grouped": TEColumnParallelGroupedLinear,
    "row_grouped": TERowParallelGroupedLinear,
}
_MOE_MODULE_KINDS = {"column_grouped", "row_grouped"}
_SWEEP_SUFFIX_RE = re.compile(r"_s\d+_b\d+$")
_DEFAULT_SWEEP_BY_VARIANT = {"vit": None, "llm": None}  # populated after shape sweep definitions


# ============================================================================
# Section 2: Data classes
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    variant: str
    name: str
    sequence_length: int
    micro_batch_size: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    normalization: str = "LayerNorm"
    swiglu: bool = False
    gated_linear_unit: bool = False
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False
    gradient_accumulation_fusion: bool = True
    layernorm_epsilon: float = 1e-6
    layernorm_zero_centered_gamma: bool = False
    tp_comm_overlap: bool = False
    tp_comm_bulk_wgrad: bool = True
    tp_comm_bulk_dgrad: bool = True
    tp_comm_overlap_ag: bool = True
    tp_comm_overlap_rs_dgrad: bool = False
    tp_comm_overlap_disable_qkv: bool = False
    tp_comm_overlap_disable_fc1: bool = False
    tp_comm_split_ag: bool = True
    tp_comm_atomic_ag: bool = False
    symmetric_ar_type: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    patch_size: Optional[int] = None
    num_experts: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_router_topk: int = 1

    @property
    def dense_num_tokens(self) -> int:
        return self.sequence_length * self.micro_batch_size

    @property
    def moe_num_tokens(self) -> int:
        return self.dense_num_tokens * self.moe_router_topk

    @property
    def qkv_output_size(self) -> int:
        return self.kv_channels * self.num_attention_heads + 2 * (
            self.kv_channels * self.num_query_groups
        )

    @property
    def attention_projection_size(self) -> int:
        return self.kv_channels * self.num_attention_heads

    @property
    def grouped_num_gemms(self) -> int:
        if self.num_experts is None:
            return 1
        if self.num_experts % self.expert_model_parallel_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"expert_model_parallel_size ({self.expert_model_parallel_size})."
            )
        return self.num_experts // self.expert_model_parallel_size

    @property
    def image_num_tokens(self) -> Optional[int]:
        if self.image_size is None or self.patch_size is None:
            return None
        return (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)


@dataclass(frozen=True)
class ModuleCase:
    case_name: str
    model_name: str
    module_name: str
    module_kind: str
    input_shape: Tuple[int, ...]
    input_size: int
    output_size: int
    num_tokens: int
    bias: bool
    skip_bias_add: bool
    is_expert: bool
    input_is_parallel: bool = False
    num_gemms: int = 1
    tp_comm_buffer_name: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkResult:
    case_name: str
    model_name: str
    module_name: str
    module_kind: str
    precision: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_tokens: int
    forward_ms: float
    backward_ms: float
    total_ms: float
    tokens_per_second: float
    forward_flops: float
    backward_flops: float
    total_flops: float
    achieved_tflops: float
    tp_size: int = 1
    etp_size: int = 1
    num_gemms: int = 1
    ub_name: Optional[str] = None


# ============================================================================
# Section 3: Built-in model definitions & shape sweeps
# ============================================================================

BASE_VIT_MODEL = ModelSpec(
    variant="vit",
    name="qwen3_vl_vit",
    sequence_length=1024,
    micro_batch_size=1,
    hidden_size=1152,
    ffn_hidden_size=4304,
    num_attention_heads=16,
    num_query_groups=16,
    kv_channels=72,
    add_bias_linear=True,
    add_qkv_bias=True,
    normalization="LayerNorm",
    swiglu=False,
    gated_linear_unit=False,
    image_size=(1344, 1344),
    patch_size=16,
)

BASE_LLM_MODEL = ModelSpec(
    variant="llm",
    name="qwen3_vl_llm_30b",
    sequence_length=424,
    micro_batch_size=1,
    hidden_size=2048,
    ffn_hidden_size=6144,
    num_attention_heads=32,
    num_query_groups=4,
    kv_channels=128,
    add_bias_linear=False,
    add_qkv_bias=False,
    normalization="RMSNorm",
    swiglu=True,
    num_experts=128,
    moe_ffn_hidden_size=768,
    moe_router_topk=8,
)

BASE_LLM_235B_MODEL = ModelSpec(
    variant="llm",
    name="qwen3_vl_llm_235b",
    sequence_length=424,
    micro_batch_size=1,
    hidden_size=4096,
    ffn_hidden_size=12288,
    num_attention_heads=64,
    num_query_groups=4,
    kv_channels=128,
    add_bias_linear=False,
    add_qkv_bias=False,
    normalization="RMSNorm",
    swiglu=True,
    num_experts=128,
    moe_ffn_hidden_size=1536,
    moe_router_topk=8,
)

VIT_SHAPE_SWEEP: Sequence[Tuple[int, int]] = (
    (1024, 1),
    (1024, 4),
    (1024, 8),
    (2048, 2),
    (2048, 4),
    (2048, 8),
    (4096, 2),
    (4096, 4),
    (8192, 2),
    (16384, 1),
    (32768, 1),
    (65536, 1),
    (131072, 1),
)

LLM_SHAPE_SWEEP: Sequence[Tuple[int, int]] = (
    (424, 1),
    (1024, 1),
    (1024, 2),
    (2048, 4),
    (4096, 2),
    (8192, 1),
    (16384, 1),
    (32768, 1),
    (65536, 1),
    (131072, 1),
)

LLM_235B_SHAPE_SWEEP: Sequence[Tuple[int, int]] = (
    (424, 1),
    (1024, 1),
    (4096, 1),
    (8192, 1),
    (16384, 1),
    (32768, 1),
    (65536, 1),
    (131072, 1),
)

# Now that shape sweeps are defined, populate the variant -> default sweep mapping.
_DEFAULT_SWEEP_BY_VARIANT.update({"vit": VIT_SHAPE_SWEEP, "llm": LLM_SHAPE_SWEEP})


def _build_shape_sweep_models(
    base_model: ModelSpec,
    shape_sweep: Sequence[Tuple[int, int]],
) -> List[ModelSpec]:
    models: List[ModelSpec] = []
    for sequence_length, micro_batch_size in shape_sweep:
        models.append(
            replace(
                base_model,
                name=(
                    f"{base_model.name}_s{sequence_length}_b{micro_batch_size}"
                ),
                sequence_length=sequence_length,
                micro_batch_size=micro_batch_size,
            )
        )
    return models


DEFAULT_MODELS: Sequence[ModelSpec] = (
    *_build_shape_sweep_models(BASE_VIT_MODEL, VIT_SHAPE_SWEEP),
    *_build_shape_sweep_models(BASE_LLM_MODEL, LLM_SHAPE_SWEEP),
    *_build_shape_sweep_models(BASE_LLM_235B_MODEL, LLM_235B_SHAPE_SWEEP),
)


# ============================================================================
# Section 4: Case building — generate ModuleCase lists from ModelSpec
# ============================================================================

def _dense_input_shape(model: ModelSpec) -> Tuple[int, ...]:
    """Return the input shape for a dense (non-expert) layer, accounting for sequence parallel."""
    seq_len = model.sequence_length
    if model.sequence_parallel and model.tensor_model_parallel_size > 1:
        seq_len = seq_len // model.tensor_model_parallel_size
    return (seq_len, model.micro_batch_size, model.hidden_size)


def _row_parallel_input_dim(full_dim: int, tp_size: int) -> int:
    """Return the per-GPU input dimension for a row-parallel layer (input_is_parallel=True)."""
    return full_dim // tp_size if tp_size > 1 else full_dim


def _build_attention_cases(model: ModelSpec, prefix: str) -> List[ModuleCase]:
    """Build QKV and projection cases shared by both ViT and LLM variants."""
    dense_shape = _dense_input_shape(model)
    tp = model.tensor_model_parallel_size
    # Row-parallel proj: input feature dim is sharded by TP
    proj_input_dim = _row_parallel_input_dim(model.attention_projection_size, tp)
    proj_shape = (dense_shape[0], dense_shape[1], proj_input_dim)
    return [
        ModuleCase(
            case_name=f"{prefix}_self_attention_qkv",
            model_name=model.name,
            module_name="TELayerNormColumnParallelLinear",
            module_kind="layernorm_column",
            input_shape=dense_shape,
            input_size=model.hidden_size,
            output_size=model.qkv_output_size,
            num_tokens=model.dense_num_tokens,
            bias=model.add_bias_linear or model.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        ),
        ModuleCase(
            case_name=f"{prefix}_self_attention_proj",
            model_name=model.name,
            module_name="TERowParallelLinear",
            module_kind="row",
            input_shape=proj_shape,
            input_size=model.attention_projection_size,
            output_size=model.hidden_size,
            num_tokens=model.dense_num_tokens,
            bias=model.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            input_is_parallel=True,
            tp_comm_buffer_name="proj",
        ),
    ]


def _build_dense_ffn_cases(model: ModelSpec, prefix: str, ffn_hidden: int, ffn_output: int) -> List[ModuleCase]:
    """Build dense fc1/fc2 cases shared by ViT and dense-LLM variants."""
    dense_shape = _dense_input_shape(model)
    tp = model.tensor_model_parallel_size
    # Row-parallel fc2: input feature dim is sharded by TP
    fc2_input_dim = _row_parallel_input_dim(ffn_hidden, tp)
    return [
        ModuleCase(
            case_name=f"{prefix}_fc1",
            model_name=model.name,
            module_name="TELayerNormColumnParallelLinear",
            module_kind="layernorm_column",
            input_shape=dense_shape,
            input_size=model.hidden_size,
            output_size=ffn_output,
            num_tokens=model.dense_num_tokens,
            bias=model.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        ),
        ModuleCase(
            case_name=f"{prefix}_fc2",
            model_name=model.name,
            module_name="TERowParallelLinear",
            module_kind="row",
            input_shape=(dense_shape[0], dense_shape[1], fc2_input_dim),
            input_size=ffn_hidden,
            output_size=model.hidden_size,
            num_tokens=model.dense_num_tokens,
            bias=model.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            input_is_parallel=True,
            tp_comm_buffer_name="fc2",
        ),
    ]


def _build_cases_for_model(model: ModelSpec) -> List[ModuleCase]:
    is_gated = model.swiglu or model.gated_linear_unit

    if model.variant == "vit":
        return (
            _build_attention_cases(model, "vision")
            + _build_dense_ffn_cases(model, "vision_mlp", model.ffn_hidden_size, model.ffn_hidden_size)
        )

    if model.variant == "llm":
        cases = _build_attention_cases(model, "llm")

        if model.num_experts is not None and model.moe_ffn_hidden_size is not None:
            ffn_output_size = model.moe_ffn_hidden_size * (2 if is_gated else 1)
            etp = model.expert_tensor_parallel_size
            # Row-parallel grouped fc2: input feature dim is sharded by expert TP
            fc2_input_dim = _row_parallel_input_dim(model.moe_ffn_hidden_size, etp)
            cases.extend([
                ModuleCase(
                    case_name="llm_moe_fc1",
                    model_name=model.name,
                    module_name="TEColumnParallelGroupedLinear",
                    module_kind="column_grouped",
                    input_shape=(model.moe_num_tokens, model.hidden_size),
                    input_size=model.hidden_size,
                    output_size=ffn_output_size,
                    num_tokens=model.moe_num_tokens,
                    bias=model.add_bias_linear,
                    skip_bias_add=False,
                    is_expert=True,
                    num_gemms=model.grouped_num_gemms,
                    tp_comm_buffer_name="fc1",
                ),
                ModuleCase(
                    case_name="llm_moe_fc2",
                    model_name=model.name,
                    module_name="TERowParallelGroupedLinear",
                    module_kind="row_grouped",
                    input_shape=(model.moe_num_tokens, fc2_input_dim),
                    input_size=model.moe_ffn_hidden_size,
                    output_size=model.hidden_size,
                    num_tokens=model.moe_num_tokens,
                    bias=model.add_bias_linear,
                    skip_bias_add=True,
                    is_expert=True,
                    num_gemms=model.grouped_num_gemms,
                    tp_comm_buffer_name="fc2",
                ),
            ])
        else:
            ffn_output_size = model.ffn_hidden_size * (2 if is_gated else 1)
            cases.extend(_build_dense_ffn_cases(model, "llm_dense", model.ffn_hidden_size, ffn_output_size))

        return cases

    raise ValueError(f"Unsupported model variant: {model.variant}")


# ============================================================================
# Section 5: Configuration loading & parsing (Hydra YAML, JSON, env vars)
# ============================================================================

def _parse_shape_sweep(raw: str) -> List[Tuple[int, int]]:
    """Parse a shape sweep string like '1024x1,2048x4' into a list of (seq_len, mbs) tuples."""
    result: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid shape_sweep entry '{item}': expected format 'seq_lenxmicro_batch_size' (e.g. '1024x1')."
            )
        result.append((int(parts[0]), int(parts[1])))
    return result


def _load_config_file(path: str):
    """Load a configuration file, auto-detecting JSON or YAML by extension."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    text = file_path.read_text()
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_hydra_default_path(config_dir: Path, relative_path_with_target: str, config_name: str) -> Tuple[str, Path]:
    """Resolve a Hydra defaults entry.

    Args:
        config_dir: Directory of the top-level config file.
        relative_path_with_target: e.g. '../../models/qwen3@model.foundation'
        config_name: e.g. 'qwen3_235b_a22b'

    Returns (target_key, resolved_yaml_path).
    """
    relative_path, target_key = relative_path_with_target.split("@")
    relative_path = relative_path.strip()
    target_key = target_key.strip()

    yaml_path = (config_dir / relative_path / f"{config_name}.yaml").resolve()
    return target_key, yaml_path


def _te_normalization(norm: str) -> str:
    """Map Omni normalization names to TE-supported names (LayerNorm / RMSNorm)."""
    if "RMSNorm" in norm:
        return "RMSNorm"
    return norm


def _model_spec_from_cfg(cfg: dict, *, variant: str, name: str) -> ModelSpec:
    """Build a ModelSpec from a raw config dict (Omni YAML section or standalone)."""
    is_vit = variant == "vit"
    image_size = cfg.get("image_size")
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    return ModelSpec(
        variant=variant,
        name=name,
        sequence_length=1024 if is_vit else 424,
        micro_batch_size=1,
        hidden_size=cfg["hidden_size"],
        ffn_hidden_size=cfg["ffn_hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_query_groups=cfg.get("num_query_groups") or cfg["num_attention_heads"],
        kv_channels=cfg.get("kv_channels") or (cfg["hidden_size"] // cfg["num_attention_heads"]),
        add_bias_linear=cfg.get("add_bias_linear", False),
        add_qkv_bias=cfg.get("add_qkv_bias", False),
        normalization=_te_normalization(cfg.get("normalization", "LayerNorm" if is_vit else "RMSNorm")),
        swiglu=cfg.get("swiglu", False),
        gated_linear_unit=cfg.get("gated_linear_unit", False),
        image_size=image_size,
        patch_size=cfg.get("patch_size"),
        num_experts=cfg.get("num_experts"),
        moe_ffn_hidden_size=cfg.get("moe_ffn_hidden_size"),
        moe_router_topk=cfg.get("moe_router_topk", 8 if cfg.get("num_experts") else 1),
        tensor_model_parallel_size=cfg.get("tensor_model_parallel_size", 1),
        expert_model_parallel_size=cfg.get("expert_model_parallel_size", 1),
        expert_tensor_parallel_size=cfg.get("expert_tensor_parallel_size", 1),
        sequence_parallel=cfg.get("sequence_parallel", False),
    )


def _parse_omni_config(config_path: str) -> List[ModelSpec]:
    """Parse an LoongForge model YAML (Hydra-style) into ModelSpec instances.

    Supports three config types:
    - Top-level VL configs (e.g. qwen3_vl/qwen3_vl_235b_a22b.yaml) that reference
      sub-configs via Hydra ``defaults`` -- extracts both ViT and LLM ModelSpecs
    - Standalone LLM configs (e.g. qwen3/qwen3_8b.yaml or qwen3/qwen3_30b_a3b.yaml)
      with flat fields -- works for both dense and MoE models
    - Standalone ViT / image encoder configs (e.g. image_encoder/qwen3_vit.yaml)
      identified by ``model_type`` ending with "vit"
    """
    file_path = Path(config_path).resolve()
    config_dir = file_path.parent
    data = yaml.safe_load(file_path.read_text())

    model_name = file_path.stem

    # Check if this is a top-level VL config with Hydra defaults.
    defaults = data.get("defaults", [])
    sub_configs: dict = {}  # target_key -> loaded yaml dict
    for entry in defaults:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key == "_self_" or "@" not in key:
                    continue
                target_key, yaml_path = _resolve_hydra_default_path(config_dir, key, value)
                if yaml_path.exists():
                    sub_configs[target_key] = yaml.safe_load(yaml_path.read_text())

    # Merge top-level model overrides into sub-configs (Hydra semantics: _self_ overrides).
    model_overrides = data.get("model", {})
    for sub_key, override_values in model_overrides.items():
        full_key = f"model.{sub_key}"
        if isinstance(override_values, dict) and full_key in sub_configs:
            sub_configs[full_key].update(override_values)

    results: List[ModelSpec] = []

    vit_cfg = sub_configs.get("model.image_encoder")
    if vit_cfg:
        results.append(_model_spec_from_cfg(vit_cfg, variant="vit", name=f"{model_name}_vit"))

    llm_cfg = sub_configs.get("model.foundation")
    if llm_cfg:
        results.append(_model_spec_from_cfg(llm_cfg, variant="llm", name=f"{model_name}_llm"))

    # Fallback: if no sub-configs found, treat as a direct model config.
    if not results:
        is_vit = data.get("model_type", "").endswith("vit")
        results.append(_model_spec_from_cfg(data, variant="vit" if is_vit else "llm", name=model_name))

    return results


def _parse_model_specs(data) -> List[ModelSpec]:
    """Build a list of ModelSpec from parsed JSON/YAML data.

    Supported file formats::

        # Format 1: object with a models list and optional global shape_sweep
        models:
          - variant: vit
            name: my_vit
            sequence_length: 1024
            micro_batch_size: 1
            hidden_size: 1152
            ffn_hidden_size: 4304
            num_attention_heads: 16
            num_query_groups: 16
            kv_channels: 72
            # optional: per-model shape_sweep
            shape_sweep:
              - [1024, 1]
              - [4096, 2]
          - variant: llm
            name: my_llm
            ...
        # optional: global shape_sweep (applies to models without their own)
        shape_sweep:
          - [1024, 1]
          - [8192, 1]

        # Format 2: a bare list of models
        - variant: vit
          ...

    shape_sweep priority: TE_LAYER_PERF_SHAPE_SWEEP env var > per-model
    shape_sweep > global shape_sweep > no expansion (uses the model's own
    sequence_length / micro_batch_size directly).
    """
    if isinstance(data, dict):
        models = data.get("models")
        if models is None:
            raise ValueError("Configured benchmark JSON/YAML must contain a `models` field.")
    elif isinstance(data, list):
        models = data
    else:
        raise ValueError("Configured benchmark JSON/YAML must be a list or an object containing `models`.")

    # Determine global shape_sweep override: env var > config-level field > None
    global_shape_sweep: Optional[List[Tuple[int, int]]] = None
    if PERF_SHAPE_SWEEP:
        global_shape_sweep = _parse_shape_sweep(PERF_SHAPE_SWEEP)
    elif isinstance(data, dict) and "shape_sweep" in data:
        global_shape_sweep = [(s, b) for s, b in data["shape_sweep"]]

    result: List[ModelSpec] = []
    for item in models:
        # Pop per-model shape_sweep before constructing ModelSpec (it's not a ModelSpec field).
        per_model_sweep_raw = item.pop("shape_sweep", None)
        base_model = ModelSpec(**item)

        # Priority: env var > per-model > global config-level > no sweep (use model as-is).
        sweep = global_shape_sweep
        if sweep is None and per_model_sweep_raw is not None:
            sweep = [(s, b) for s, b in per_model_sweep_raw]

        if sweep:
            result.extend(_build_shape_sweep_models(base_model, sweep))
        else:
            result.append(base_model)
    return result


def _expand_models_with_sweep(
    base_models: List[ModelSpec],
    sweep_override: Optional[List[Tuple[int, int]]] = None,
) -> List[ModelSpec]:
    """Expand base models with a shape sweep. Uses override if given, else variant defaults."""
    models: List[ModelSpec] = []
    for base in base_models:
        sweep = sweep_override or _DEFAULT_SWEEP_BY_VARIANT.get(base.variant, LLM_SHAPE_SWEEP)
        models.extend(_build_shape_sweep_models(base, sweep))
    return models


def _apply_parallel_overrides(models: List[ModelSpec]) -> List[ModelSpec]:
    """Apply TE_LAYER_PERF_TP_SIZE / EP_SIZE / ETP_SIZE / SEQ_PARALLEL env var overrides.

    EP and ETP overrides are only applied to models that have num_experts set,
    since expert parallelism requires MoE layers.
    """
    common_overrides = {}
    expert_overrides = {}
    if PERF_TP_SIZE > 0:
        common_overrides["tensor_model_parallel_size"] = PERF_TP_SIZE
    if PERF_EP_SIZE > 0:
        expert_overrides["expert_model_parallel_size"] = PERF_EP_SIZE
    if PERF_ETP_SIZE > 0:
        expert_overrides["expert_tensor_parallel_size"] = PERF_ETP_SIZE
    if PERF_SEQ_PARALLEL is not None:
        common_overrides["sequence_parallel"] = PERF_SEQ_PARALLEL.strip().lower() in ("true", "1", "yes")
    if not common_overrides and not expert_overrides:
        return models
    result = []
    for m in models:
        overrides = dict(common_overrides)
        if m.num_experts is not None:
            overrides.update(expert_overrides)
        result.append(replace(m, **overrides) if overrides else m)
    return result


def _get_base_models_for_caseset() -> List[ModelSpec]:
    """Return the base (un-swept) model specs matching the current PERF_CASESET."""
    if PERF_CASESET == "vision":
        return [BASE_VIT_MODEL]
    if PERF_CASESET == "llm":
        return [BASE_LLM_MODEL]
    if PERF_CASESET == "llm_235b":
        return [BASE_LLM_235B_MODEL]
    return [BASE_VIT_MODEL, BASE_LLM_MODEL, BASE_LLM_235B_MODEL]


def _load_models() -> List[ModelSpec]:
    env_sweep = _parse_shape_sweep(PERF_SHAPE_SWEEP) if PERF_SHAPE_SWEEP else None

    if PERF_OMNI_CONFIG_PATH:
        models = _expand_models_with_sweep(_parse_omni_config(PERF_OMNI_CONFIG_PATH), env_sweep)
    elif PERF_CASES_PATH:
        models = _parse_model_specs(_load_config_file(PERF_CASES_PATH))
    elif PERF_CASES_JSON:
        models = _parse_model_specs(json.loads(PERF_CASES_JSON))
    elif env_sweep:
        models = _expand_models_with_sweep(_get_base_models_for_caseset(), env_sweep)
    elif PERF_CASESET == "vision":
        models = [model for model in DEFAULT_MODELS if model.variant == "vit"]
    elif PERF_CASESET == "llm":
        models = [model for model in DEFAULT_MODELS if model.name.startswith("qwen3_vl_llm_30b")]
    elif PERF_CASESET == "llm_235b":
        models = [model for model in DEFAULT_MODELS if model.name.startswith("qwen3_vl_llm_235b")]
    else:
        models = list(DEFAULT_MODELS)

    return _apply_parallel_overrides(models)


def _load_cases() -> List[Tuple[ModelSpec, ModuleCase]]:
    cases: List[Tuple[ModelSpec, ModuleCase]] = []
    for model in _load_models():
        for case in _build_cases_for_model(model):
            cases.append((model, case))
    return cases


DEFAULT_CASES = _load_cases()


# ============================================================================
# Section 6: Benchmark engine — module instantiation, input building, timing
# ============================================================================

def _require_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark Transformer Engine layers.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Current GPU does not support BF16.")
    if not HAVE_TE:
        raise RuntimeError("Transformer Engine is not available in the current environment.")
    if TEColumnParallelGroupedLinear is None or TERowParallelGroupedLinear is None:
        raise RuntimeError("GroupedLinear benchmarks require Transformer Engine >= 1.9.0.dev0.")
    if "fp8" in PERF_PRECISIONS and not FP8_AVAILABLE:
        raise RuntimeError(f"FP8 is not available: {FP8_UNAVAILABLE_REASON}")


def _get_quantization_context(config: TransformerConfig, *, is_init: bool = False):
    try:
        return get_fp8_context(config, is_init=is_init)
    except (AssertionError, ValueError, RuntimeError) as error:
        raise RuntimeError(str(error)) from error


def _make_config(model: ModelSpec, precision: str) -> TransformerConfig:
    config_kwargs = dict(
        num_layers=1,
        hidden_size=model.hidden_size,
        ffn_hidden_size=model.ffn_hidden_size,
        num_attention_heads=model.num_attention_heads,
        num_query_groups=model.num_query_groups,
        kv_channels=model.kv_channels,
        use_cpu_initialization=False,
        perform_initialization=True,
        sequence_parallel=model.sequence_parallel,
        gradient_accumulation_fusion=model.gradient_accumulation_fusion,
        normalization=model.normalization,
        layernorm_epsilon=model.layernorm_epsilon,
        layernorm_zero_centered_gamma=model.layernorm_zero_centered_gamma,
        tp_comm_overlap=model.tp_comm_overlap,
        tp_comm_bulk_wgrad=model.tp_comm_bulk_wgrad,
        tp_comm_bulk_dgrad=model.tp_comm_bulk_dgrad,
        tp_comm_overlap_ag=model.tp_comm_overlap_ag,
        tp_comm_overlap_rs_dgrad=model.tp_comm_overlap_rs_dgrad,
        tp_comm_overlap_disable_qkv=model.tp_comm_overlap_disable_qkv,
        tp_comm_overlap_disable_fc1=model.tp_comm_overlap_disable_fc1,
        tp_comm_split_ag=model.tp_comm_split_ag,
        tp_comm_atomic_ag=model.tp_comm_atomic_ag,
        symmetric_ar_type=model.symmetric_ar_type,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="hybrid" if precision == "fp8" else None,
        fp8_param=precision == "fp8",
        fp8_recipe=PERF_FP8_RECIPE,
        tensor_model_parallel_size=model.tensor_model_parallel_size,
        expert_model_parallel_size=model.expert_model_parallel_size,
        expert_tensor_parallel_size=model.expert_tensor_parallel_size,
        gated_linear_unit=model.gated_linear_unit or model.swiglu,
        add_bias_linear=model.add_bias_linear,
        add_qkv_bias=model.add_qkv_bias,
    )
    if model.swiglu:
        config_kwargs["activation_func"] = torch.nn.functional.silu
    if model.num_experts is not None:
        config_kwargs.update(
            num_moe_experts=model.num_experts,
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=model.moe_ffn_hidden_size,
            moe_router_topk=model.moe_router_topk,
        )
    return TransformerConfig(**config_kwargs)


def _extract_output_tensor(module_output) -> torch.Tensor:
    if isinstance(module_output, tuple):
        return module_output[0]
    return module_output


def _make_grouped_splits(
    num_tokens: int,
    num_gemms: int,
    skew: float = PERF_SPLIT_SKEW,
) -> List[int]:
    """Generate per-expert token splits for grouped GEMM benchmarks.

    When *skew* is 0 the splits are uniform (original behaviour).  A positive
    *skew* produces a Zipf-like distribution that better approximates the
    uneven token routing observed during real MoE training, where popular
    experts receive many more tokens than cold ones.
    """
    if num_tokens < num_gemms:
        raise ValueError(
            f"num_tokens ({num_tokens}) must be >= num_gemms ({num_gemms}) for grouped benchmarks."
        )

    if skew <= 0:
        # Uniform distribution (legacy path).
        tokens_per_gemm, remainder = divmod(num_tokens, num_gemms)
        return [tokens_per_gemm + (1 if index < remainder else 0) for index in range(num_gemms)]

    # Zipf-like weights: w_i = 1 / (i + 1)^skew
    weights = [1.0 / (i + 1) ** skew for i in range(num_gemms)]
    total_weight = sum(weights)

    # Distribute tokens proportionally, ensuring every expert gets at least 1.
    splits = [max(1, int(num_tokens * w / total_weight)) for w in weights]

    # Fix up rounding residual: distribute deficit across largest splits
    # to ensure no split goes below 1.
    residual = num_tokens - sum(splits)
    if residual < 0:
        # Sort indices by split size descending, trim the largest ones first.
        sorted_indices = sorted(range(num_gemms), key=lambda i: splits[i], reverse=True)
        for idx in sorted_indices:
            if residual >= 0:
                break
            reduce = min(splits[idx] - 1, -residual)
            splits[idx] -= reduce
            residual += reduce
    else:
        splits[0] += residual

    return splits


def _get_fp8_token_alignment() -> int:
    # TE FP8 kernels require the GEMM M dimension to be aligned by recipe.
    # For blockwise this is currently 16, while other recipes may differ.
    return get_fp8_align_size(Fp8Recipe(PERF_FP8_RECIPE))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _estimate_case_flops(case: ModuleCase, num_tokens: int, tp_size: int = 1) -> Tuple[float, float, float]:
    # Per-GPU GEMM: one dimension is divided by tp_size (column or row).
    gemm_flops = 2.0 * num_tokens * case.input_size * case.output_size / tp_size
    forward_flops = gemm_flops
    backward_flops = 2.0 * gemm_flops
    total_flops = forward_flops + backward_flops
    return forward_flops, backward_flops, total_flops


def _instantiate_case_module(case: ModuleCase, config: TransformerConfig) -> torch.nn.Module:
    cls = _MODULE_KIND_TO_CLASS.get(case.module_kind)
    if cls is None:
        raise ValueError(f"Unsupported module_kind: {case.module_kind}")

    kwargs = dict(
        input_size=case.input_size,
        output_size=case.output_size,
        config=config,
        init_method=config.init_method,
        bias=case.bias,
        skip_bias_add=case.skip_bias_add,
        is_expert=case.is_expert,
        tp_comm_buffer_name=case.tp_comm_buffer_name,
    )
    if case.module_kind in {"layernorm_column", "column"}:
        kwargs["gather_output"] = False
    if case.module_kind == "row":
        kwargs["input_is_parallel"] = case.input_is_parallel
    if case.module_kind in {"column_grouped", "row_grouped"}:
        kwargs["num_gemms"] = case.num_gemms

    return cls(**kwargs)


def _build_module_inputs(case: ModuleCase, precision: str) -> Tuple[Tuple, int]:
    input_shape = list(case.input_shape)
    effective_num_tokens = case.num_tokens
    is_grouped = case.module_kind in {"column_grouped", "row_grouped"}

    if precision == "fp8":
        alignment = _get_fp8_token_alignment()
        token_alignment = alignment
        if len(input_shape) == 3:
            # TE consumes [S, B, H] as a flattened token dimension S * B, so when we pad
            # FP8 shapes we must preserve batch size and ensure the padded token count is
            # divisible by both the recipe alignment and micro batch size.
            token_alignment = math.lcm(token_alignment, input_shape[1])

        if is_grouped and PERF_FP8_RECIPE == "blockwise":
            # Plan C: pad each per-expert split independently to the FP8
            # alignment boundary, mirroring TEGroupedMLP.fp8_padding in the
            # real training path.  This is more realistic than rounding only
            # the total token count, because each sub-GEMM's M dimension must
            # be aligned individually.
            raw_splits = _make_grouped_splits(case.num_tokens, case.num_gemms)
            aligned_splits = [_round_up_to_multiple(s, token_alignment) for s in raw_splits]
            effective_num_tokens = sum(aligned_splits)
        else:
            effective_num_tokens = _round_up_to_multiple(case.num_tokens, token_alignment)

        if len(input_shape) == 3:
            input_shape[0] = effective_num_tokens // input_shape[1]
        else:
            input_shape[0] = effective_num_tokens

    hidden_states = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    if is_grouped:
        if precision == "fp8" and PERF_FP8_RECIPE == "blockwise":
            # Use the already-aligned splits computed above.
            return (hidden_states, aligned_splits), effective_num_tokens
        return (hidden_states, _make_grouped_splits(effective_num_tokens, case.num_gemms)), effective_num_tokens
    return (hidden_states,), effective_num_tokens


def _prepare_main_grads(module: torch.nn.Module, config: TransformerConfig) -> None:
    if not config.gradient_accumulation_fusion:
        return
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if not hasattr(parameter, "main_grad"):
            parameter.main_grad = torch.zeros_like(parameter, requires_grad=False)
        if not hasattr(parameter, "grad_added_to_main_grad"):
            parameter.grad_added_to_main_grad = False


def _reset_module_grad_state(module: torch.nn.Module, config: TransformerConfig) -> None:
    module.zero_grad(set_to_none=True)
    if not config.gradient_accumulation_fusion:
        return
    for parameter in module.parameters():
        if hasattr(parameter, "main_grad"):
            parameter.main_grad.zero_()
        if hasattr(parameter, "grad_added_to_main_grad"):
            parameter.grad_added_to_main_grad = False


def _run_warmup(module, config, input_tensor, run_forward_fn) -> None:
    """Execute warmup iterations (not timed)."""
    for _ in range(PERF_WARMUP):
        _reset_module_grad_state(module, config)
        if input_tensor.grad is not None:
            input_tensor.grad = None
        output_tensor = run_forward_fn()
        output_tensor.float().sum().backward()
    torch.cuda.synchronize()


def _run_timed_iterations(
    module, config, input_tensor, run_forward_fn,
) -> Tuple[List[float], List[float], List[float], Tuple[int, ...]]:
    """Execute timed iterations and return per-iteration timing lists + output shape."""
    forward_times: List[float] = []
    backward_times: List[float] = []
    total_times: List[float] = []
    output_shape: Tuple[int, ...] = ()

    for _ in range(PERF_ITERS):
        _reset_module_grad_state(module, config)
        if input_tensor.grad is not None:
            input_tensor.grad = None

        start_event = torch.cuda.Event(enable_timing=True)
        forward_end_event = torch.cuda.Event(enable_timing=True)
        backward_start_event = torch.cuda.Event(enable_timing=True)
        backward_end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output_tensor = run_forward_fn()
        forward_end_event.record()

        backward_start_event.record()
        output_tensor.float().sum().backward()
        backward_end_event.record()
        torch.cuda.synchronize()

        output_shape = tuple(output_tensor.shape)
        forward_times.append(start_event.elapsed_time(forward_end_event))
        backward_times.append(backward_start_event.elapsed_time(backward_end_event))
        total_times.append(start_event.elapsed_time(backward_end_event))

    return forward_times, backward_times, total_times, output_shape


def _measure_case(case: ModuleCase, config: TransformerConfig, precision: str) -> BenchmarkResult:
    """Benchmark a single (case, precision) combination: instantiate, warmup, measure, return result."""
    with _get_quantization_context(config, is_init=True):
        module = _instantiate_case_module(case, config)
    module_inputs, effective_num_tokens = _build_module_inputs(case, precision)

    module.cuda()
    module.train()
    _prepare_main_grads(module, config)

    input_tensor = module_inputs[0]

    def _forward_func(*inputs):
        """Forward pass wrapped with FP8 context, suitable for checkpointing."""
        with _get_quantization_context(config):
            module_output = module(*inputs)
            return _extract_output_tensor(module_output)

    def _run_forward():
        """Execute forward, optionally with activation recomputation."""
        if PERF_RECOMPUTE:
            return torch.utils.checkpoint.checkpoint(
                _forward_func, *module_inputs, use_reentrant=False,
            )
        return _forward_func(*module_inputs)

    # Warmup
    _run_warmup(module, config, input_tensor, _run_forward)

    # Timed iterations
    forward_times, backward_times, total_times, output_shape = _run_timed_iterations(
        module, config, input_tensor, _run_forward,
    )

    # Aggregate results (median for robustness against measurement outliers).
    avg_forward_ms = statistics.median(forward_times)
    avg_backward_ms = statistics.median(backward_times)
    avg_total_ms = statistics.median(total_times)
    tokens_per_second = effective_num_tokens / (avg_total_ms / 1000.0)
    # Use expert_tensor_parallel_size for expert layers, tensor_model_parallel_size for dense.
    flops_tp = config.expert_tensor_parallel_size if case.is_expert else config.tensor_model_parallel_size
    forward_flops, backward_flops, total_flops = _estimate_case_flops(case, effective_num_tokens, flops_tp)
    achieved_tflops = total_flops / (avg_total_ms * 1.0e9)

    return BenchmarkResult(
        case_name=case.case_name,
        model_name=case.model_name,
        module_name=case.module_name,
        module_kind=case.module_kind,
        precision=precision,
        input_shape=tuple(input_tensor.shape),
        output_shape=output_shape,
        num_tokens=effective_num_tokens,
        forward_ms=avg_forward_ms,
        backward_ms=avg_backward_ms,
        total_ms=avg_total_ms,
        tokens_per_second=tokens_per_second,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
        total_flops=total_flops,
        achieved_tflops=achieved_tflops,
        tp_size=config.tensor_model_parallel_size,
        etp_size=config.expert_tensor_parallel_size,
        num_gemms=case.num_gemms,
        ub_name=case.tp_comm_buffer_name,
    )


# ============================================================================
# Section 7: FP8 policy generation & merging
# ============================================================================

def _analyze_fp8_thresholds(
    results: Sequence[BenchmarkResult], speedup_threshold: float = 1.0,
) -> dict:
    """Analyze benchmark results and compute per-module FP8 min_tokens thresholds.

    For dense module kinds (layernorm_column / column / row / duplicated), results
    are grouped by ``(module_kind, ub_name, tp_size)`` so that same-kind modules
    with different shapes (e.g. qkv vs fc1) receive distinct thresholds.  The
    output dict uses a nested ``{module_kind: {ub_name: [rules]}}`` layout.

    For MoE grouped kinds, results are grouped by ``(module_kind, etp_size,
    num_gemms)`` and the output keeps the legacy flat
    ``{module_kind: [rules]}`` layout (ub_name is not meaningful per-expert).

    Returns:
        dict mapping module_kind to either a ``{ub_name: [rules]}`` dict (dense)
        or a list of rule dicts (MoE).
    """
    # Group results by a key that uniquely identifies a (module, parallel config,
    # shape) combo. We need to pair bf16 and fp8 results for the same case.
    grouped = {}
    for r in results:
        is_moe = r.module_kind in _MOE_MODULE_KINDS
        parallel_key = (r.etp_size, r.num_gemms) if is_moe else (r.tp_size,)
        ub_name = r.ub_name
        group_key = (
            r.module_kind, ub_name, parallel_key, r.case_name, r.model_name, r.num_tokens,
        )
        grouped.setdefault(group_key, {})[r.precision] = r

    # For each (module_kind, ub_name, parallel_config), collect all
    # (num_tokens, speedup) pairs and find the min_tokens where speedup > threshold.
    threshold_candidates = {}
    for (module_kind, ub_name, parallel_key, _case, _model, num_tokens), precs in grouped.items():
        if "bf16" not in precs or "fp8" not in precs:
            continue
        speedup = precs["bf16"].total_ms / precs["fp8"].total_ms
        threshold_candidates.setdefault(
            (module_kind, ub_name, parallel_key), []
        ).append((num_tokens, speedup))

    # Build rules dict.  Dense -> nested by ub_name; MoE -> flat list.
    rules: dict = {}
    # Track per-(module_kind, ub_name, parallel_key) the most conservative rule
    # so repeat cases don't duplicate entries.
    dense_merged: dict = {}  # (module_kind, ub_name, parallel_key) -> rule
    moe_merged: dict = {}    # (module_kind, parallel_key) -> rule

    for (module_kind, ub_name, parallel_key), candidates in sorted(
        threshold_candidates.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        candidates.sort(key=lambda x: x[0])
        min_tokens = None
        measured_speedup = None
        for num_tokens, speedup in candidates:
            if speedup > speedup_threshold:
                min_tokens = num_tokens
                measured_speedup = round(speedup, 3)
                break
        if min_tokens is None:
            # FP8 never wins for this config -- skip (conservative: default to BF16).
            continue

        is_moe = module_kind in _MOE_MODULE_KINDS
        if is_moe:
            rule = {
                "etp": parallel_key[0],
                "num_gemms": parallel_key[1],
                "min_tokens": min_tokens,
                "measured_speedup": measured_speedup,
            }
            key = (module_kind, parallel_key)
            if key not in moe_merged or rule["min_tokens"] > moe_merged[key]["min_tokens"]:
                moe_merged[key] = rule
        else:
            rule = {
                "tp": parallel_key[0],
                "min_tokens": min_tokens,
                "measured_speedup": measured_speedup,
            }
            dkey = (module_kind, ub_name, parallel_key)
            if dkey not in dense_merged or rule["min_tokens"] > dense_merged[dkey]["min_tokens"]:
                dense_merged[dkey] = rule

    # Emit dense rules as nested dict.
    for (module_kind, ub_name, _pk), rule in dense_merged.items():
        rules.setdefault(module_kind, {}).setdefault(ub_name, []).append(rule)
    for module_kind, by_ub in rules.items():
        for ub_name, rule_list in by_ub.items():
            by_ub[ub_name] = sorted(rule_list, key=lambda r: r["tp"])

    # Emit MoE rules as flat list.
    for (module_kind, _pk), rule in moe_merged.items():
        rules.setdefault(module_kind, []).append(rule)
    for module_kind in list(rules):
        if module_kind in _MOE_MODULE_KINDS:
            rules[module_kind] = sorted(
                rules[module_kind], key=lambda r: (r["etp"], r["num_gemms"])
            )

    return rules


def _export_fp8_policy(
    results: Sequence[BenchmarkResult], path: str, speedup_threshold: float = 1.0,
) -> None:
    """Analyze benchmark results and export an FP8 dynamic policy JSON file.

    The exported file can be loaded by ``FP8DynamicPolicy`` in
    ``megatron.core.fp8_utils`` to drive selective FP8 decisions at training time.
    """
    rules = _analyze_fp8_thresholds(results, speedup_threshold)
    policy = {
        "version": 1,
        "speedup_threshold": speedup_threshold,
        "rules": rules,
    }
    policy_path = Path(path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2))


def merge_fp8_policy_reports(
    report_paths: Sequence[str], output_path: str, speedup_threshold: float = 1.0,
) -> dict:
    """Merge multiple benchmark report JSON files and export a unified FP8 policy.

    This allows combining results from separate runs with different TP/ETP sizes
    into a single policy file that covers all tested parallel configurations.

    Usage::

        python benchmark_te_parallel_layers.py merge-policy \\
            --reports tp1_report.json tp2_report.json tp4_report.json \\
            --output merged_policy.json \\
            --speedup-threshold 1.0

    Args:
        report_paths: List of paths to benchmark report JSON files.
        output_path: Path to write the merged policy JSON.
        speedup_threshold: Minimum FP8/BF16 speedup to consider FP8 beneficial.

    Returns:
        The merged policy dict.
    """
    all_results = []
    for path in report_paths:
        with open(path) as f:
            data = json.load(f)
        for item in data:
            all_results.append(BenchmarkResult(**{
                k: tuple(v) if k in ("input_shape", "output_shape") and isinstance(v, list) else v
                for k, v in item.items()
            }))
    rules = _analyze_fp8_thresholds(all_results, speedup_threshold)
    policy = {
        "version": 1,
        "speedup_threshold": speedup_threshold,
        "rules": rules,
    }
    policy_path = Path(output_path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2))
    return policy


# ============================================================================
# Section 8: Result formatting & reporting
# ============================================================================

def _compute_speedups(results: Sequence[BenchmarkResult]) -> List[Tuple[str, str, float]]:
    """Compute per-case FP8-over-BF16 speedup from paired results."""
    by_key = {}
    for r in results:
        by_key.setdefault((r.model_name, r.case_name), {})[r.precision] = r
    speedups = []
    for (model_name, case_name), precs in sorted(by_key.items()):
        if "bf16" in precs and "fp8" in precs:
            speedups.append((model_name, case_name, precs["bf16"].total_ms / precs["fp8"].total_ms))
    return speedups


def _format_winning_lines(speedups: Sequence[Tuple[str, str, float]], label: str = "winning modules") -> List[str]:
    winners = sorted((s for s in speedups if s[2] > 1.0), key=lambda s: s[2], reverse=True)
    if not winners:
        return [f"{label} => none"]
    return [label] + [f"{m} / {c} => {s:.3f}x" for m, c, s in winners]


def _append_group_summary(lines: List[str], group: Sequence[BenchmarkResult], title: str) -> None:
    if not group:
        return
    lines.extend(["", title])
    for model_name in sorted({r.model_name for r in group}):
        bf16_ms = sum(r.total_ms for r in group if r.model_name == model_name and r.precision == "bf16")
        fp8_ms = sum(r.total_ms for r in group if r.model_name == model_name and r.precision == "fp8")
        if fp8_ms > 0:
            lines.append(f"{model_name} => bf16={bf16_ms:.3f} ms, fp8={fp8_ms:.3f} ms, speedup={bf16_ms / fp8_ms:.3f}x")
        else:
            lines.append(f"{model_name} => bf16={bf16_ms:.3f} ms, fp8=N/A")
    lines.extend(_format_winning_lines(_compute_speedups(group)))


def _format_results(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        (
            "Transformer Engine perf report "
            f"(warmup={PERF_WARMUP}, iters={PERF_ITERS}, fp8_recipe={PERF_FP8_RECIPE}, "
            f"recompute={PERF_RECOMPUTE})"
        ),
        (
            "case | model | module | precision | input_shape | output_shape | "
            "total_ms | forward_ms | backward_ms | tokens/s | total_flops(T) | achieved_tflops"
        ),
    ]

    for r in results:
        lines.append(
            " | ".join([
                r.case_name, r.model_name, r.module_name, r.precision,
                str(r.input_shape), str(r.output_shape),
                f"{r.total_ms:.3f}", f"{r.forward_ms:.3f}", f"{r.backward_ms:.3f}",
                f"{r.tokens_per_second:.1f}", f"{r.total_flops / 1.0e12:.3f}", f"{r.achieved_tflops:.3f}",
            ])
        )

    if "bf16" in PERF_PRECISIONS and "fp8" in PERF_PRECISIONS:
        speedups = _compute_speedups(results)
        lines.extend(["", "FP8 speedup over BF16 (total_ms based)"])
        for model_name, case_name, speedup in speedups:
            lines.append(f"{model_name} / {case_name} => {speedup:.3f}x")
        lines.extend([""])
        lines.extend(_format_winning_lines(speedups, "FP8 winning shapes"))
        # Dynamically group results by base model name (strip the _s{seq}_b{batch} suffix).
        base_groups: dict = {}
        for r in results:
            base_name = _SWEEP_SUFFIX_RE.sub("", r.model_name)
            base_groups.setdefault(base_name, []).append(r)
        for base_name in sorted(base_groups):
            _append_group_summary(lines, base_groups[base_name], f"{base_name} summary")

    return "\n".join(lines)


# ============================================================================
# Section 9: Orchestration & entry points
# ============================================================================

def _is_rank_zero() -> bool:
    """Return True if this is rank 0 or if distributed is not initialized."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _save_report(results: List[BenchmarkResult]) -> None:
    """Write benchmark results to a JSON report file (if configured)."""
    if PERF_REPORT_PATH and _is_rank_zero():
        report_path = Path(PERF_REPORT_PATH)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps([asdict(item) for item in results], indent=2))


def _save_fp8_policy(results: List[BenchmarkResult], precisions: Sequence[str]) -> None:
    """Export FP8 dynamic policy JSON (if configured and both precisions were measured)."""
    if PERF_FP8_POLICY_PATH and "bf16" in precisions and "fp8" in precisions and _is_rank_zero():
        _export_fp8_policy(results, PERF_FP8_POLICY_PATH, PERF_SPEEDUP_THRESHOLD)


def run_te_parallel_layer_perf_benchmarks(
    cases: Sequence[Tuple[ModelSpec, ModuleCase]] = DEFAULT_CASES,
    precisions: Sequence[str] = PERF_PRECISIONS,
) -> List[BenchmarkResult]:
    """Run all benchmark cases and return a list of results.

    For each (model, case) combination, iterates over all precisions:
    1. Initialize model parallel
    2. Build TransformerConfig and instantiate the TE module
    3. Run warmup + timed iterations, measuring forward/backward latency
    4. Compute tokens/s and achieved TFLOPS

    If TE_LAYER_PERF_REPORT_PATH is set, results are also written to a JSON file.
    """
    _require_environment()
    results: List[BenchmarkResult] = []

    try:
        for model, case in cases:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=model.tensor_model_parallel_size,
                pipeline_model_parallel_size=model.pipeline_model_parallel_size,
                expert_model_parallel_size=model.expert_model_parallel_size,
                expert_tensor_parallel_size=model.expert_tensor_parallel_size,
            )
            try:
                for precision in precisions:
                    config = _make_config(model, precision)
                    try:
                        results.append(_measure_case(case, config, precision))
                    except (AssertionError, RuntimeError) as exc:
                        print(f"SKIP {case.case_name} [{precision}]: {exc}")
                    # Explicit cleanup to avoid CUDA memory fragmentation artifacts.
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                Utils.destroy_model_parallel()

        _save_report(results)
        _save_fp8_policy(results, precisions)

        return results
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.timeout(600)
def test_te_parallel_layer_perf_benchmark() -> None:
    """Pytest entry point for TE parallel layer performance benchmark.

    See module-level docstring for full usage examples and environment variables.
    """
    try:
        results = run_te_parallel_layer_perf_benchmarks()
    except RuntimeError as error:
        pytest.skip(str(error))

    expected_num_results = len(DEFAULT_CASES) * len(PERF_PRECISIONS)
    assert len(results) == expected_num_results
    assert all(result.total_ms > 0 for result in results)
    assert all(result.forward_ms > 0 for result in results)
    assert all(result.backward_ms >= 0 for result in results)
    assert all(result.total_flops > 0 for result in results)
    assert all(result.achieved_tflops > 0 for result in results)
    print("\n" + _format_results(results))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "merge-policy":
        # CLI: merge multiple report JSONs into a single policy.
        #   python benchmark_te_parallel_layers.py merge-policy \
        #       --reports r1.json r2.json r3.json \
        #       --output merged_policy.json \
        #       --speedup-threshold 1.0
        import argparse

        parser = argparse.ArgumentParser(
            prog=f"{sys.argv[0]} merge-policy",
            description="Merge multiple TE benchmark report JSONs into a unified FP8 policy.",
        )
        parser.add_argument(
            "--reports", nargs="+", required=True,
            help="Paths to benchmark report JSON files (from TE_LAYER_PERF_REPORT_PATH).",
        )
        parser.add_argument("--output", required=True, help="Output policy JSON path.")
        parser.add_argument(
            "--speedup-threshold", type=float, default=1.0,
            help="Minimum FP8/BF16 speedup to consider FP8 beneficial (default: 1.0).",
        )
        args = parser.parse_args(sys.argv[2:])
        policy = merge_fp8_policy_reports(args.reports, args.output, args.speedup_threshold)
        print(json.dumps(policy, indent=2))
    else:
        benchmark_results = run_te_parallel_layer_perf_benchmarks()
        if _is_rank_zero():
            print(_format_results(benchmark_results))
