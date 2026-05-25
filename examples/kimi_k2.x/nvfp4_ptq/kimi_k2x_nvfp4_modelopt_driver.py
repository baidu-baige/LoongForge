#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K2.x BF16 HF -> NVIDIA-style ModelOpt NVFP4 converter.

This wraps NVIDIA ModelOpt's HF PTQ example using a runtime ModelOpt checkout.
The NVIDIA Kimi NVFP4 recipe is read from a JSON file and translated into
the Python quant_cfg expected by ModelOpt.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import importlib
import json
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any


def record_stage(stage: str) -> None:
    """Log the current processing stage with a timestamp."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [ptq-stage] {stage}", flush=True)
    stage_file = os.environ.get("PTQ_STAGE_FILE")
    if stage_file:
        try:
            Path(stage_file).write_text(stage + "\n")
        except OSError as exc:
            print(f"[{now}] [ptq-stage] failed to write {stage_file}: {exc}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file into a dict."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_csv_ints(value: str) -> list[int]:
    """Split a comma-separated string into a list of integers."""
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value: str | None) -> list[str] | None:
    """Split a comma-separated string into a list of non-empty items, or return None for falsy input."""
    if not value:
        return None
    return [item for item in value.split(",") if item]


def checkpoint_num_layers(checkpoint_path: Path) -> int:
    """Read the number of hidden layers from a checkpoint config.json."""
    config = load_json(checkpoint_path / "config.json")
    text_config = config.get("text_config") or {}
    return int(text_config.get("num_hidden_layers") or config.get("num_hidden_layers"))


def expand_exclude_modules(patterns: list[str], num_layers: int) -> list[str]:
    """Expand wildcard ``.layers.*.`` patterns across all transformer layers."""
    expanded: list[str] = []
    for pattern in patterns:
        if ".layers.*." in pattern:
            expanded.extend(
                pattern.replace(".layers.*.", f".layers.{i}.")
                for i in range(num_layers)
            )
        else:
            expanded.append(pattern)

    deduped: list[str] = []
    for item in sorted(set(expanded)):
        covered_by_broader_pattern = any(
            other != item and "*" in other and fnmatch.fnmatchcase(item, other)
            for other in expanded
        )
        if not covered_by_broader_pattern:
            deduped.append(item)
    return deduped


def _quantizer_disable_patterns(exclude: str) -> list[str]:
    prefix = exclude[:-1] if exclude.endswith("*") else exclude
    prefixes = [prefix]
    if prefix.startswith("language_model."):
        prefixes.append(prefix[len("language_model.") :])

    patterns: list[str] = []
    for item in prefixes:
        patterns.append(f"{item}*")
        patterns.append(f"{item}*weight_quantizer")
        patterns.append(f"{item}*input_quantizer")
        patterns.append(f"{item}*output_quantizer")
        patterns.append(f"{item}*k_bmm_quantizer")
        patterns.append(f"{item}*v_bmm_quantizer")
    return patterns


def disable_quantizer(qcfg: Any, pattern: str) -> None:
    """Register a disable pattern in the given quantizer config."""
    if isinstance(qcfg, dict):
        qcfg[pattern] = {"enable": False}
        return
    if isinstance(qcfg, list):
        qcfg.append({"quantizer_name": pattern, "enable": False})
        return
    raise SystemExit(f"Unsupported ModelOpt quant_cfg type: {type(qcfg).__name__}")


def recipe_quantization(recipe: dict[str, Any]) -> dict[str, Any]:
    """Extract the quantization sub-dict from the official recipe."""
    return recipe["recipe_from_hf_quant_config"]["quantization"]


def patch_kimi_init_weights_for_modelopt(root: Path) -> None:
    """Patch Kimi remote-code ``_init_weights`` to be safe for ModelOpt quantized modules."""
    files = [
        root / "modeling_deepseek.py",
        root / "modeling_kimi_k25.py",
    ]
    if root.is_dir() and root.name != "transformers_modules":
        cache_root = root / "transformers_modules"
        if cache_root.is_dir():
            files.extend(cache_root.rglob("modeling_deepseek.py"))
            files.extend(cache_root.rglob("modeling_kimi_k25.py"))

    replacements = [
        (
            "        if isinstance(module, nn.Linear):\n"
            "            module.weight.data.normal_(mean=0.0, std=std)\n"
            "            if module.bias is not None:\n"
            "                module.bias.data.zero_()\n",
            "        if isinstance(module, nn.Linear):\n"
            "            if not hasattr(module, \"weight\"):\n"
            "                return\n"
            "            module.weight.data.normal_(mean=0.0, std=std)\n"
            "            if getattr(module, \"bias\", None) is not None:\n"
            "                module.bias.data.zero_()\n",
        ),
        (
            "        if isinstance(module, (nn.Linear, nn.Conv2d)):\n"
            "            module.weight.data.normal_(mean=0.0, std=std)\n"
            "            if module.bias is not None:\n"
            "                module.bias.data.zero_()\n",
            "        if isinstance(module, (nn.Linear, nn.Conv2d)):\n"
            "            if not hasattr(module, \"weight\"):\n"
            "                return\n"
            "            module.weight.data.normal_(mean=0.0, std=std)\n"
            "            if getattr(module, \"bias\", None) is not None:\n"
            "                module.bias.data.zero_()\n",
        ),
        (
            "        elif isinstance(module, nn.Embedding):\n"
            "            module.weight.data.normal_(mean=0.0, std=std)\n"
            "            if module.padding_idx is not None:\n"
            "                module.weight.data[module.padding_idx].zero_()\n",
            "        elif isinstance(module, nn.Embedding):\n"
            "            if not hasattr(module, \"weight\"):\n"
            "                return\n"
            "            module.weight.data.normal_(mean=0.0, std=std)\n"
            "            if module.padding_idx is not None:\n"
            "                module.weight.data[module.padding_idx].zero_()\n",
        ),
    ]

    patched: list[Path] = []
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        text = path.read_text()
        new_text = text
        for old, new in replacements:
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text)
            patched.append(path)

    if patched:
        for path in patched:
            print(f"Patched Kimi remote-code _init_weights for ModelOpt: {path}", flush=True)


def patch_trust_remote_code() -> None:
    """Monkey-patch Transformers to always allow trusted remote code."""
    import transformers.dynamic_module_utils as dynamic_module_utils
    import transformers.models.auto.auto_factory as auto_factory

    def resolve_trust_remote_code(*args, **kwargs):
        return True

    dynamic_module_utils.resolve_trust_remote_code = resolve_trust_remote_code
    auto_factory.resolve_trust_remote_code = resolve_trust_remote_code


def force_config_attention_implementation(config: Any, attn_implementation: str) -> None:
    """Recursively overwrite attention implementation fields in a config object graph."""
    seen: set[int] = set()
    attn_keys = {
        "_attn_implementation",
        "_attn_implementation_internal",
        "attn_implementation",
    }

    def visit(value: Any) -> None:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, dict):
            for key in attn_keys:
                if key in value:
                    value[key] = attn_implementation
            for item in value.values():
                visit(item)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return

        if hasattr(value, "__dict__"):
            for key in attn_keys:
                if hasattr(value, key):
                    try:
                        setattr(value, key, attn_implementation)
                    except Exception:
                        pass
            for item in vars(value).values():
                visit(item)

    visit(config)


def strip_input_quantization_config(config: Any) -> None:
    """Recursively remove ``quantization_config`` from a config object graph."""
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for item in value.values():
                visit(item)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return

        if hasattr(value, "__dict__"):
            if hasattr(value, "quantization_config"):
                try:
                    delattr(value, "quantization_config")
                except Exception:
                    try:
                        setattr(value, "quantization_config", None)
                    except Exception:
                        pass
            for item in vars(value).values():
                visit(item)

    visit(config)


def patch_transformers_attention_implementation(attn_implementation: str | None) -> None:
    """Force a specific attention implementation on all Transformers model/config loads."""
    if not attn_implementation:
        return

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.configuration_utils import PretrainedConfig

    original_auto_config_from_pretrained = AutoConfig.from_pretrained.__func__
    original_pretrained_config_from_pretrained = PretrainedConfig.from_pretrained.__func__
    original_auto_model_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

    def patched_auto_config_from_pretrained(cls, *args, **kwargs):
        if not kwargs.get("attn_implementation"):
            kwargs["attn_implementation"] = attn_implementation
        config = original_auto_config_from_pretrained(cls, *args, **kwargs)
        strip_input_quantization_config(config)
        force_config_attention_implementation(config, attn_implementation)
        return config

    def patched_pretrained_config_from_pretrained(cls, *args, **kwargs):
        if not kwargs.get("attn_implementation"):
            kwargs["attn_implementation"] = attn_implementation
        config = original_pretrained_config_from_pretrained(cls, *args, **kwargs)
        strip_input_quantization_config(config)
        force_config_attention_implementation(config, attn_implementation)
        return config

    def patched_auto_model_from_pretrained(cls, *args, **kwargs):
        kwargs.pop("attn_implementation", None)
        if kwargs.get("config") is not None:
            strip_input_quantization_config(kwargs["config"])
            force_config_attention_implementation(kwargs["config"], attn_implementation)
        return original_auto_model_from_pretrained(cls, *args, **kwargs)

    AutoConfig.from_pretrained = classmethod(patched_auto_config_from_pretrained)
    PretrainedConfig.from_pretrained = classmethod(patched_pretrained_config_from_pretrained)
    AutoModelForCausalLM.from_pretrained = classmethod(patched_auto_model_from_pretrained)
    transformers._kimi_nvfp4_attn_patch = attn_implementation
    print(
        f"Forced Transformers attention implementation for PTQ: {attn_implementation}",
        flush=True,
    )


def preload_transformers_for_modelopt_examples() -> None:
    """Initialize Transformers lazy exports before ModelOpt plugin imports run."""
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        import transformers
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_import_utils

        # This PTQ path only uses text calibration. Avoid importing a mismatched
        # torchvision binary while Transformers initializes AutoProcessor.
        transformers_import_utils._torchvision_available = False
        transformers_import_utils._torchvision_version = "N/A"

        # The Kimi remote code imports flash-attn when Transformers reports it
        # available. A mismatched flash-attn wheel can pass that metadata check
        # but fail while loading its CUDA extension, so force native attention.
        def flash_attn_unavailable(*_args, **_kwargs):
            return False

        transformers_import_utils.is_flash_attn_2_available = flash_attn_unavailable
        transformers_import_utils.is_flash_attn_3_available = flash_attn_unavailable
        transformers_import_utils.is_flash_attn_greater_or_equal = flash_attn_unavailable
        transformers_import_utils.is_flash_attn_greater_or_equal_2_10 = flash_attn_unavailable
        transformers_utils.is_flash_attn_2_available = flash_attn_unavailable
        transformers_utils.is_flash_attn_3_available = flash_attn_unavailable
        transformers_utils.is_flash_attn_greater_or_equal = flash_attn_unavailable
        transformers_utils.is_flash_attn_greater_or_equal_2_10 = flash_attn_unavailable

        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.auto.processing_auto import AutoProcessor
        from transformers.processing_utils import ProcessorMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        transformers.AutoConfig = AutoConfig
        transformers.AutoModel = AutoModel
        transformers.AutoModelForCausalLM = AutoModelForCausalLM
        transformers.AutoProcessor = AutoProcessor
        transformers.AutoTokenizer = AutoTokenizer
        transformers.PreTrainedTokenizerBase = PreTrainedTokenizerBase
        transformers.ProcessorMixin = ProcessorMixin
    except Exception as exc:
        import traceback

        version = "<unknown>"
        path = "<unknown>"
        try:
            import transformers

            version = getattr(transformers, "__version__", version)
            path = getattr(transformers, "__file__", path)
        except Exception:
            pass
        raise SystemExit(
            "Failed to import Transformers symbols required by ModelOpt llm_ptq "
            f"examples: {exc!r}. transformers_version={version}, transformers_path={path}. "
            f"traceback={''.join(traceback.format_exception(exc, value=exc, tb=exc.__traceback__)).strip()} "
            "Run the matching install_nvfp4_modelopt_deps.sh in this same Python environment."
        ) from exc

    print(
        "Transformers preflight passed: "
        f"version={getattr(transformers, '__version__', '<unknown>')} "
        f"path={getattr(transformers, '__file__', '<unknown>')} "
        "flash_attn_available=False",
        flush=True,
    )


def import_hf_ptq(modelopt_repo: Path):
    """Import ``hf_ptq`` from a local ModelOpt checkout after Transformers preflight."""
    llm_ptq_dir = modelopt_repo / "examples" / "llm_ptq"
    if not (llm_ptq_dir / "hf_ptq.py").is_file():
        raise SystemExit(f"Missing ModelOpt hf_ptq.py under {llm_ptq_dir}")
    preload_transformers_for_modelopt_examples()
    sys.path.insert(0, str(modelopt_repo))
    sys.path.insert(0, str(llm_ptq_dir))
    import hf_ptq  # type: ignore

    return hf_ptq


def patch_modelopt_tokenizer_deepcopy(modelopt_repo: Path) -> None:
    """Remove tokenizer deepcopy from ModelOpt dataset_utils to avoid large-memory copies."""
    dataset_utils_path = modelopt_repo / "modelopt" / "torch" / "utils" / "dataset_utils.py"
    if not dataset_utils_path.is_file():
        raise SystemExit(f"Missing ModelOpt dataset_utils.py under {dataset_utils_path}")

    text = dataset_utils_path.read_text()
    replacements = [
        (
            "    # Tokenizer encoding may modify the tokenizer in place, so we need to clone it.\n"
            "    tokenizer = copy.deepcopy(tokenizer)\n"
        ),
        (
            "    # batch_encode_plus will modify the tokenizer in place, so we need to clone it.\n"
            "    tokenizer = copy.deepcopy(tokenizer)\n"
        ),
        "    tokenizer = copy.deepcopy(tokenizer)\n",
    ]

    for needle in replacements:
        if needle in text:
            dataset_utils_path.write_text(text.replace(needle, "", 1))
            print(f"Removed tokenizer deepcopy from ModelOpt dataset_utils: {dataset_utils_path}")
            return

    print(f"ModelOpt dataset_utils already has no tokenizer deepcopy: {dataset_utils_path}")


def apply_official_recipe_to_hf_ptq(hf_ptq, args: argparse.Namespace, recipe: dict[str, Any]) -> list[str]:
    """Apply the official Kimi NVFP4 recipe to the hf_ptq module's quant config."""
    quant = recipe_quantization(recipe)
    runtime = recipe.get("runtime_defaults") or {}
    expected_qformat = runtime.get("qformat", "nvfp4_mlp_only")
    expected_kv = runtime.get("kv_cache_qformat", "fp8")

    if args.qformat != expected_qformat:
        raise SystemExit(f"Official recipe expects --qformat {expected_qformat}, got {args.qformat}")
    if args.kv_cache_qformat != expected_kv:
        raise SystemExit(
            f"Official recipe expects --kv_cache_qformat {expected_kv}, got {args.kv_cache_qformat}"
        )
    if quant.get("quant_algo") != "NVFP4" or quant.get("group_size") != 16:
        raise SystemExit(f"Unsupported Kimi NVFP4 recipe: {quant}")
    if quant.get("kv_cache_quant_algo") != "FP8":
        raise SystemExit(f"Unsupported Kimi KV cache recipe: {quant.get('kv_cache_quant_algo')}")

    raw_excludes = quant.get("exclude_modules") or []
    num_layers = checkpoint_num_layers(Path(args.pyt_ckpt_path))
    excludes = expand_exclude_modules(raw_excludes, num_layers)

    quant_cfg = copy.deepcopy(hf_ptq.QUANT_CFG_CHOICES[args.qformat])
    qcfg = quant_cfg.setdefault("quant_cfg", {})

    for exclude in raw_excludes:
        if exclude == "language_model.lm_head":
            disable_quantizer(qcfg, "language_model.lm_head*")
            disable_quantizer(qcfg, "lm_head*")
        elif exclude in ("mm_projector*", "vision_tower*"):
            disable_quantizer(qcfg, exclude)
        elif ".layers." in exclude:
            for pattern in _quantizer_disable_patterns(exclude):
                disable_quantizer(qcfg, pattern)
        else:
            disable_quantizer(qcfg, exclude)

    hf_ptq.QUANT_CFG_CHOICES[args.qformat] = quant_cfg
    model_name = recipe.get("model_name") or "NVIDIA Kimi NVFP4"
    print(f"Loaded {model_name} recipe with {len(excludes)} exclude_modules")
    return excludes


def skip_generation_if_requested(hf_ptq, skip_generate: bool) -> None:
    """Skip ModelOpt's pre-quantization generation preview while preserving export."""
    if not skip_generate:
        return

    def pre_quantize_noop(*args, **kwargs):
        return None, None

    hf_ptq.pre_quantize = pre_quantize_noop


def enable_kimi_moe_all_expert_warmup(model, max_tokens: int, every_forward: bool):
    """Patch every Kimi MoE module forward to warm up all experts before calibration."""
    import types

    import torch

    patched = []

    for module_name, module in model.named_modules():
        gate = getattr(module, "gate", None)
        experts = getattr(module, "experts", None)
        if gate is None or experts is None or not hasattr(module, "moe_infer"):
            continue

        topk_attr = None
        for candidate in ("top_k", "topk"):
            if hasattr(gate, candidate):
                topk_attr = candidate
                break
        if topk_attr is None:
            continue

        topk_group_attr = None
        for candidate in ("topk_group", "topk_groups"):
            if hasattr(gate, candidate):
                topk_group_attr = candidate
                break

        n_experts = getattr(gate, "n_routed_experts", None)
        if n_experts is None:
            n_experts = getattr(module, "n_routed_experts", None)
        if n_experts is None:
            n_experts = len(experts)

        n_groups = None
        for candidate in ("n_group", "n_groups"):
            if hasattr(gate, candidate):
                n_groups = getattr(gate, candidate)
                break
        if n_groups is None and topk_group_attr is not None:
            n_groups = getattr(gate, topk_group_attr)

        if not isinstance(n_experts, int) or n_experts <= 0:
            continue

        original_forward = module.forward
        warmed_attr = "_kimi_nvfp4_all_experts_warmed"

        def forward_with_all_expert_warmup(
            self,
            hidden_states,
            *args,
            _module_name=module_name,
            _original_forward=original_forward,
            _topk_attr=topk_attr,
            _topk_group_attr=topk_group_attr,
            _n_experts=n_experts,
            _n_groups=n_groups,
            **kwargs,
        ):
            if not every_forward and getattr(self, warmed_attr, False):
                return _original_forward(hidden_states, *args, **kwargs)

            warm_hidden_states = hidden_states
            if max_tokens > 0 and isinstance(hidden_states, torch.Tensor) and hidden_states.ndim >= 2:
                flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
                if flat_hidden_states.shape[0] > max_tokens:
                    flat_hidden_states = flat_hidden_states[:max_tokens]
                warm_hidden_states = flat_hidden_states.reshape(
                    1, flat_hidden_states.shape[0], flat_hidden_states.shape[-1]
                )

            gate = self.gate
            original_topk = getattr(gate, _topk_attr)
            original_topk_group = (
                getattr(gate, _topk_group_attr) if _topk_group_attr is not None else None
            )

            try:
                setattr(gate, _topk_attr, _n_experts)
                if _topk_group_attr is not None and _n_groups is not None:
                    setattr(gate, _topk_group_attr, _n_groups)
                with torch.no_grad():
                    _original_forward(warm_hidden_states, *args, **kwargs)
                setattr(self, warmed_attr, True)
            finally:
                setattr(gate, _topk_attr, original_topk)
                if _topk_group_attr is not None:
                    setattr(gate, _topk_group_attr, original_topk_group)

            return _original_forward(hidden_states, *args, **kwargs)

        module.forward = types.MethodType(forward_with_all_expert_warmup, module)
        patched.append((module, original_forward, module_name, n_experts))

    if patched:
        mode = "every forward" if every_forward else "once per MoE module"
        print(
            f"Enabled Kimi MoE all-expert calibration warmup for {len(patched)} module(s); "
            f"mode={mode}, max_tokens={max_tokens}.",
            flush=True,
        )
    else:
        print("Kimi MoE all-expert calibration warmup found no matching MoE modules.", flush=True)

    def restore() -> None:
        for module, original_forward, _, _ in patched:
            module.forward = original_forward
        if patched:
            print(
                f"Restored original Kimi MoE forward for {len(patched)} module(s).",
                flush=True,
            )

    return restore


def patch_kimi_moe_all_expert_calibration(
    hf_ptq,
    enabled: bool,
    max_tokens: int,
    every_forward: bool,
) -> None:
    """Wrap hf_ptq.mono_quantize to warm up all MoE experts before calibration."""
    if not enabled:
        return

    original = hf_ptq.mono_quantize

    def mono_quantize_with_kimi_moe_all_expert_warmup(
        args,
        quant_cfg,
        full_model,
        language_model,
        model_type,
        calibration_only,
        calib_dataloader,
        is_nemotron_vl_model,
    ):
        record_stage("kimi_moe_all_experts_warmup:patch_start")
        restore_moe_forward = enable_kimi_moe_all_expert_warmup(
            language_model,
            max_tokens=max_tokens,
            every_forward=every_forward,
        )
        record_stage("kimi_moe_all_experts_warmup:patch_done")
        try:
            return original(
                args,
                quant_cfg,
                full_model,
                language_model,
                model_type,
                calibration_only,
                calib_dataloader,
                is_nemotron_vl_model,
            )
        finally:
            restore_moe_forward()
            record_stage("kimi_moe_all_experts_warmup:restore_done")

    hf_ptq.mono_quantize = mono_quantize_with_kimi_moe_all_expert_warmup


def wrap_stage(hf_ptq, name: str) -> None:
    """Wrap an hf_ptq stage method with stage-recording and eval-mode guards."""
    if not hasattr(hf_ptq, name):
        return
    original = getattr(hf_ptq, name)

    def set_eval_if_model(model, label: str) -> None:
        """Set model to eval mode if it has an ``eval`` method."""
        if model is not None and hasattr(model, "eval"):
            model.eval()
            print(f"Set {label} to eval mode.", flush=True)

    def wrapped(*args, **kwargs):
        record_stage(f"{name}:start")
        if name == "quantize_main":
            if len(args) > 1:
                set_eval_if_model(args[1], "full_model")
            if len(args) > 2 and args[2] is not args[1]:
                set_eval_if_model(args[2], "language_model")
        result = original(*args, **kwargs)
        if name == "load_model" and isinstance(result, tuple):
            if len(result) > 0:
                set_eval_if_model(result[0], "full_model")
            if len(result) > 1 and result[1] is not result[0]:
                set_eval_if_model(result[1], "language_model")
        record_stage(f"{name}:done")
        return result

    setattr(hf_ptq, name, wrapped)


def ensure_nvfp4_weight_amax(module_name: str, module, weight_name: str = "weight") -> bool:
    """Ensure an NVFP4 weight quantizer has a non-empty amax value."""
    import torch

    def usable_amax(value) -> bool:
        return value is not None and not (
            isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)
        )

    quantizer_attr = "weight_quantizer" if weight_name == "weight" else f"{weight_name}_weight_quantizer"
    weight_quantizer = getattr(module, quantizer_attr, None)
    weight = getattr(module, weight_name, None)
    if weight_quantizer is None or weight is None:
        return False
    if not getattr(weight_quantizer, "is_enabled", False):
        return False

    existing_export_amax = getattr(weight_quantizer, "_amax", None)
    has_global_amax = hasattr(weight_quantizer, "global_amax")
    existing_global_amax = getattr(weight_quantizer, "global_amax", None) if has_global_amax else None
    if usable_amax(existing_export_amax) and (
        not has_global_amax or usable_amax(existing_global_amax)
    ):
        return False

    existing_amax = existing_export_amax
    if not usable_amax(existing_amax):
        existing_amax = getattr(weight_quantizer, "amax", None)

    if usable_amax(existing_amax):
        if isinstance(existing_amax, torch.Tensor):
            weight_quantizer._amax = existing_amax.detach().to(dtype=torch.float32)
        else:
            target_device = (
                weight.device
                if isinstance(weight, torch.Tensor) and not getattr(weight, "is_meta", False)
                else torch.device("cpu")
            )
            weight_quantizer._amax = torch.tensor(
                float(existing_amax), dtype=torch.float32, device=target_device
            )
        if has_global_amax and not usable_amax(existing_global_amax):
            weight_quantizer.global_amax = weight_quantizer._amax.detach().float().max()
        return True
    if not isinstance(weight, torch.Tensor) or getattr(weight, "is_meta", False):
        return False

    if has_global_amax:
        from modelopt.torch.quantization.utils import reduce_block_amax

        block_size = weight_quantizer.block_sizes[-1]
        per_block_amax = reduce_block_amax(weight.detach(), block_sizes={-1: block_size})
        per_block_amax = per_block_amax.to(dtype=torch.float32, device=weight.device)
        weight_quantizer._amax = per_block_amax
        weight_quantizer.global_amax = per_block_amax.max()
    else:
        amax = torch.max(torch.abs(weight.detach())).to(dtype=torch.float32)
        if amax.device != weight.device:
            amax = amax.to(weight.device)
        weight_quantizer.amax = amax
        weight_quantizer._amax = amax
    return True


def fill_missing_nvfp4_weight_amax(model) -> int:
    """Iterate all modules and fill any missing NVFP4 weight amax, returns patched count."""
    patched = 0
    for module_name, module in model.named_modules():
        if ensure_nvfp4_weight_amax(module_name, module):
            patched += 1

    if patched:
        print(f"Filled missing NVFP4 weight amax for {patched} quantized module(s).", flush=True)
    return patched


def _routed_expert_group_key(module_name: str) -> tuple[str, str] | None:
    """Extract (layer_prefix, param_name) for routed expert modules, or None."""
    marker = ".mlp.experts."
    if marker not in module_name:
        return None
    layer_prefix, expert_suffix = module_name.split(marker, 1)
    parts = expert_suffix.split(".")
    if len(parts) < 2:
        return None
    return layer_prefix, parts[1]


def _input_quantizer_for_weight(module, weight_name: str = "weight"):
    attr = "input_quantizer" if weight_name == "weight" else f"{weight_name}_input_quantizer"
    return getattr(module, attr, None)


def ensure_nvfp4_input_amax(
    module_name: str,
    module,
    weight_name: str = "weight",
    fallback_amax=None,
) -> bool:
    """Ensure an NVFP4 input quantizer has a non-empty amax value."""
    import torch

    def usable_amax(value) -> bool:
        return value is not None and not (
            isinstance(value, torch.Tensor) and getattr(value, "is_meta", False)
        )

    input_quantizer = _input_quantizer_for_weight(module, weight_name)
    if input_quantizer is None or not getattr(input_quantizer, "is_enabled", False):
        return False
    if usable_amax(getattr(input_quantizer, "amax", None)):
        return False

    weight = getattr(module, weight_name, None)
    target_device = None
    if isinstance(weight, torch.Tensor) and not getattr(weight, "is_meta", False):
        target_device = weight.device
    elif (
        fallback_amax is not None
        and isinstance(fallback_amax, torch.Tensor)
        and not getattr(fallback_amax, "is_meta", False)
    ):
        target_device = fallback_amax.device
    else:
        target_device = torch.device("cpu")

    if fallback_amax is None or (
        isinstance(fallback_amax, torch.Tensor) and getattr(fallback_amax, "is_meta", False)
    ):
        amax = torch.tensor(0.5, dtype=torch.float32, device=target_device)
        source = "fallback"
    elif isinstance(fallback_amax, torch.Tensor):
        amax = fallback_amax.detach().to(dtype=torch.float32, device=target_device)
        source = "routed expert group max"
    else:
        amax = torch.tensor(float(fallback_amax), dtype=torch.float32, device=target_device)
        source = "provided fallback"

    if torch.any(amax <= 0):
        amax = torch.clamp(amax, min=torch.finfo(torch.float32).tiny)
    input_quantizer.amax = amax
    print(
        f"Filled missing NVFP4 input amax for {module_name}.{weight_name} from {source}: "
        f"{amax.max().item():.6f}",
        flush=True,
    )
    return True


def fill_missing_routed_expert_input_amax(model) -> int:
    """Fill missing NVFP4 input amax for routed expert modules, returns patched count."""
    import torch

    grouped_existing_amax: dict[tuple[str, str], list[torch.Tensor]] = {}
    routed_modules: list[tuple[str, Any, tuple[str, str]]] = []

    for module_name, module in model.named_modules():
        group_key = _routed_expert_group_key(module_name)
        if group_key is None:
            continue

        weight_quantizer = getattr(module, "weight_quantizer", None)
        input_quantizer = getattr(module, "input_quantizer", None)
        if (
            weight_quantizer is None
            or input_quantizer is None
            or not getattr(weight_quantizer, "is_enabled", False)
            or not getattr(input_quantizer, "is_enabled", False)
        ):
            continue

        routed_modules.append((module_name, module, group_key))
        existing_amax = getattr(input_quantizer, "amax", None)
        if isinstance(existing_amax, torch.Tensor):
            if not getattr(existing_amax, "is_meta", False):
                grouped_existing_amax.setdefault(group_key, []).append(existing_amax.detach().float())
        elif existing_amax is not None:
            grouped_existing_amax.setdefault(group_key, []).append(
                torch.tensor(float(existing_amax), dtype=torch.float32)
            )

    group_max: dict[tuple[str, str], torch.Tensor] = {}
    for group_key, values in grouped_existing_amax.items():
        group_max[group_key] = torch.max(torch.stack([value.reshape(-1).max() for value in values]))

    patched = 0
    for module_name, module, group_key in routed_modules:
        fallback = group_max.get(group_key)
        if ensure_nvfp4_input_amax(module_name, module, fallback_amax=fallback):
            patched += 1

    if patched:
        print(
            f"Filled missing NVFP4 input amax for {patched} routed expert module(s).",
            flush=True,
        )
    return patched


def materialize_accelerate_offload_for_export(*models) -> None:
    """Materialize Accelerate-offloaded tensors before ModelOpt reads weights during export."""
    if os.environ.get("MATERIALIZE_ACCELERATE_OFFLOAD_FOR_EXPORT", "0") != "1":
        return

    from accelerate.hooks import remove_hook_from_module

    seen: set[int] = set()
    materialized = 0
    for model in models:
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        remove_hook_from_module(model, recurse=True)
        materialized += 1

    if materialized:
        print(
            f"Materialized Accelerate offload hooks before export for {materialized} model object(s).",
            flush=True,
        )


def patch_export_missing_nvfp4_weight_amax(hf_ptq) -> None:
    """Patch ModelOpt export to auto-fill missing NVFP4 weight/input amax."""
    unified_export_hf = importlib.import_module("modelopt.torch.export.unified_export_hf")
    original_export_weight = unified_export_hf._export_quantized_weight
    original_requantize_resmooth = getattr(
        unified_export_hf, "requantize_resmooth_fused_llm_layers", None
    )
    active_export_models: tuple[Any, ...] = ()

    def export_quantized_weight_with_kimi_amax_fix(*args, **kwargs):
        """Wrapper around _export_quantized_weight that ensures weight/input amax exist."""
        if args:
            sub_module = args[0]
            weight_name = kwargs.get("weight_name", args[2] if len(args) >= 3 else "weight")
            module_name = type(sub_module).__name__
            ensure_nvfp4_weight_amax(module_name, sub_module, weight_name)
            ensure_nvfp4_input_amax(module_name, sub_module, weight_name)
        return original_export_weight(*args, **kwargs)

    unified_export_hf._export_quantized_weight = export_quantized_weight_with_kimi_amax_fix

    if original_requantize_resmooth is not None:

        def requantize_resmooth_then_materialize(*args, **kwargs):
            """Keep Accelerate hooks for ModelOpt's dummy forward, then materialize export weights."""
            result = original_requantize_resmooth(*args, **kwargs)
            models = args[:1] + active_export_models
            materialize_accelerate_offload_for_export(*models)
            return result

        unified_export_hf.requantize_resmooth_fused_llm_layers = (
            requantize_resmooth_then_materialize
        )

    original = hf_ptq.export_quantized

    def export_quantized_with_kimi_amax_fix(*args, **kwargs):
        """Wrapper around hf_ptq.export_quantized that fills missing NVFP4 amax before export."""
        nonlocal active_export_models
        full_model = kwargs.get("full_model")
        if full_model is None and len(args) >= 2:
            full_model = args[1]
        language_model = kwargs.get("language_model")
        if language_model is None and len(args) >= 3:
            language_model = args[2]
        if full_model is not None:
            fill_missing_nvfp4_weight_amax(full_model)
            fill_missing_routed_expert_input_amax(full_model)
        active_export_models = (full_model, language_model)
        try:
            return original(*args, **kwargs)
        finally:
            active_export_models = ()

    hf_ptq.export_quantized = export_quantized_with_kimi_amax_fix


def enable_stage_wrappers(hf_ptq) -> None:
    """Apply stage-recording wrappers to all hf_ptq pipeline stages."""
    hf_ptq.record_ptq_stage = record_stage
    for name in ("load_model", "quantize_main", "mono_quantize", "export_quantized"):
        wrap_stage(hf_ptq, name)


def patch_accelerate_clear_device_cache() -> None:
    """Rate-limit Accelerate device-cache clears during low-memory dispatch."""
    interval = float(os.environ.get("ACCELERATE_CLEAR_CACHE_INTERVAL", "10"))
    if interval < 0:
        print("Accelerate clear_device_cache patch disabled.", flush=True)
        return

    import accelerate.utils.memory as accelerate_memory
    import accelerate.utils.modeling as accelerate_modeling

    original = accelerate_memory.clear_device_cache
    last_clear = 0.0

    def clear_device_cache_rate_limited(garbage_collection=False):
        nonlocal last_clear
        now = time.monotonic()
        if garbage_collection or now - last_clear >= interval:
            last_clear = now
            return original(garbage_collection=garbage_collection)
        return None

    accelerate_memory.clear_device_cache = clear_device_cache_rate_limited
    accelerate_modeling.clear_device_cache = clear_device_cache_rate_limited
    print(
        f"Rate-limited Accelerate clear_device_cache to every {interval:g}s.",
        flush=True,
    )


def official_hf_quant_config(recipe: dict[str, Any], excludes: list[str]) -> dict[str, Any]:
    """Build the ``hf_quant_config.json`` metadata from the official recipe."""
    producer = recipe["recipe_from_hf_quant_config"]["producer"]
    quant = recipe_quantization(recipe)
    return {
        "producer": producer,
        "quantization": {
            "quant_algo": quant["quant_algo"],
            "kv_cache_quant_algo": quant["kv_cache_quant_algo"],
            "group_size": quant["group_size"],
            "exclude_modules": excludes,
        },
    }


def official_config_quantization(recipe: dict[str, Any], excludes: list[str]) -> dict[str, Any]:
    """Build the ``quantization_config`` block for the top-level config.json."""
    producer = recipe["recipe_from_hf_quant_config"]["producer"]
    quant = recipe_quantization(recipe)
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": quant["group_size"],
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": quant["group_size"],
                },
                "targets": ["Linear"],
            }
        },
        "ignore": excludes,
        "quant_algo": quant["quant_algo"],
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "producer": producer,
        "quant_method": "modelopt",
    }


def normalize_output_metadata(export_path: Path, recipe: dict[str, Any], excludes: list[str]) -> None:
    """Rewrite export metadata to match the official Kimi-K2.5-NVFP4 config shape."""
    config_path = export_path / "config.json"
    hf_quant_path = export_path / "hf_quant_config.json"
    if not config_path.is_file() or not hf_quant_path.is_file():
        raise SystemExit(f"Missing ModelOpt export metadata under {export_path}")

    config = load_json(config_path)
    output_config = recipe.get("output_config") or {}

    config["quantization_config"] = official_config_quantization(recipe, excludes)
    config["dtype"] = "bfloat16"
    if output_config.get("remove_torch_dtype", True):
        config.pop("torch_dtype", None)

    if output_config.get("transformers_version"):
        config["transformers_version"] = output_config["transformers_version"]

    text_config = config.setdefault("text_config", {})
    text_config["dtype"] = "bfloat16"
    if output_config.get("text_model_type"):
        text_config["model_type"] = output_config["text_model_type"]
    if output_config.get("remove_torch_dtype", True):
        text_config.pop("torch_dtype", None)
    text_config.pop("transformers_version", None)

    hf_quant_path.write_text(json.dumps(official_hf_quant_config(recipe, excludes), indent=4) + "\n")
    config_path.write_text(json.dumps(config, indent=4) + "\n")
    print("Normalized output metadata to NVIDIA Kimi-K2.5-NVFP4 official config shape.")


def verify_output_metadata(export_path: Path, recipe: dict[str, Any], excludes: list[str]) -> None:
    """Assert that output config.json/hf_quant_config.json match the official recipe."""
    config = load_json(export_path / "config.json")
    hf_quant_config = load_json(export_path / "hf_quant_config.json")
    quant = recipe_quantization(recipe)
    producer = recipe["recipe_from_hf_quant_config"]["producer"]

    exported = hf_quant_config.get("quantization") or {}
    if hf_quant_config.get("producer") != producer:
        raise SystemExit(f"Unexpected producer metadata: {hf_quant_config.get('producer')}")
    if exported.get("quant_algo") != quant["quant_algo"]:
        raise SystemExit(f"Unexpected quant_algo: {exported.get('quant_algo')}")
    if exported.get("kv_cache_quant_algo") != quant["kv_cache_quant_algo"]:
        raise SystemExit(f"Unexpected kv_cache_quant_algo: {exported.get('kv_cache_quant_algo')}")
    if exported.get("group_size") != quant["group_size"]:
        raise SystemExit(f"Unexpected group_size: {exported.get('group_size')}")
    if sorted(exported.get("exclude_modules") or []) != excludes:
        raise SystemExit("hf_quant_config.json exclude_modules does not match official recipe.")

    config_quant = config.get("quantization_config") or {}
    if sorted(config_quant.get("ignore") or []) != excludes:
        raise SystemExit("config.json quantization_config.ignore does not match official recipe.")
    if config_quant.get("kv_cache_scheme") != {"dynamic": False, "num_bits": 8, "type": "float"}:
        raise SystemExit(f"Unexpected config.json kv_cache_scheme: {config_quant.get('kv_cache_scheme')}")

    output_config = recipe.get("output_config") or {}
    text_config = config.get("text_config") or {}
    if output_config.get("text_model_type") and text_config.get("model_type") != output_config["text_model_type"]:
        raise SystemExit(f"Unexpected text_config.model_type: {text_config.get('model_type')}")
    if output_config.get("remove_torch_dtype", True) and (
        "torch_dtype" in config or "torch_dtype" in text_config
    ):
        raise SystemExit("Output config still has torch_dtype metadata.")


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read a safetensors file header (excluding metadata)."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _output_tensor_headers(export_path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Extract dtype and shape for every tensor across all safetensor shards."""
    index = load_json(export_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map") or {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    headers: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard, keys in sorted(by_shard.items()):
        shard_header = _read_safetensors_header(export_path / shard)
        for key in keys:
            meta = shard_header.get(key)
            if meta is None:
                raise SystemExit(f"Index key {key} is missing from shard {shard}")
            headers[key] = (meta["dtype"], tuple(meta["shape"]))
    return headers


def verify_output_tensor_metadata(export_path: Path, excludes: list[str]) -> None:
    """Verify that all routed expert input_scale entries exist and excluded modules are unquantized."""
    headers = _output_tensor_headers(export_path)
    missing_input_scales: list[str] = []
    excluded_quantized: list[tuple[str, str, tuple[int, ...]]] = []

    for key, (dtype, shape) in headers.items():
        if ".mlp.experts." in key and key.endswith(".weight") and dtype == "U8":
            input_scale_key = key[: -len(".weight")] + ".input_scale"
            if input_scale_key not in headers:
                missing_input_scales.append(input_scale_key)

        if (
            key.endswith(".weight_scale")
            or key.endswith(".weight_scale_2")
            or key.endswith(".input_scale")
            or key.endswith(".k_scale")
            or key.endswith(".v_scale")
        ):
            for exclude in excludes:
                if fnmatch.fnmatch(key, exclude):
                    excluded_quantized.append((key, dtype, shape))
                    break

    if missing_input_scales:
        raise SystemExit(
            "Routed expert NVFP4 export is missing input_scale entries, examples: "
            + ", ".join(missing_input_scales[:10])
        )
    if excluded_quantized:
        raise SystemExit(
            "Excluded modules still have quantization tensors, examples: "
            + ", ".join(item[0] for item in excluded_quantized[:10])
        )

    print(
        "Verified output tensor metadata: routed expert input_scale is complete and official excludes are unquantized.",
        flush=True,
    )


def build_hf_ptq_args(args: argparse.Namespace, recipe: dict[str, Any]) -> argparse.Namespace:
    """Merge CLI arguments and recipe defaults into the hf_ptq argument namespace."""
    runtime = recipe.get("runtime_defaults") or {}
    dataset = args.dataset or runtime.get("dataset")
    calib_size = args.calib_size or str(runtime.get("calib_size", 512))
    if args.calib_dataset_jsonl:
        dataset = args.calib_dataset_jsonl
        calib_size = str(sum(parse_csv_ints(str(calib_size))))

    return argparse.Namespace(
        pyt_ckpt_path=args.pyt_ckpt_path,
        device=args.device,
        qformat=args.qformat or runtime.get("qformat", "nvfp4_mlp_only"),
        batch_size=args.batch_size,
        calib_size=parse_csv_ints(str(calib_size)),
        calib_seq=args.calib_seq or int(runtime.get("calib_seq", 512)),
        export_path=args.export_path,
        dataset=parse_csv_strings(dataset),
        inference_tensor_parallel=args.inference_tensor_parallel,
        inference_pipeline_parallel=args.inference_pipeline_parallel,
        awq_block_size=args.awq_block_size,
        sparsity_fmt="dense",
        auto_quantize_bits=None,
        recipe=None,
        kv_cache_qformat=args.kv_cache_qformat or runtime.get("kv_cache_qformat", "fp8"),
        export_fmt="hf",
        trust_remote_code=args.trust_remote_code,
        gpu_max_mem_percentage=args.gpu_max_mem_percentage,
        use_seq_device_map=args.use_seq_device_map,
        verbose=args.verbose,
        low_memory_mode=args.low_memory_mode,
        skip_generate=args.skip_generate,
        attn_implementation=None,
        auto_quantize_method="gradient",
        auto_quantize_score_size=128,
        auto_quantize_checkpoint=None,
        specdec_offline_dataset=None,
        calib_with_images=False,
        moe_calib_experts_ratio=None,
        vllm_fakequant_export=False,
        cast_mxfp4_to_nvfp4=False,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Kimi NVFP4 PTQ driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelopt_repo", required=True)
    parser.add_argument("--official_quant_config", required=True)
    parser.add_argument("--pyt_ckpt_path", required=True)
    parser.add_argument("--export_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qformat", default=None)
    parser.add_argument("--kv_cache_qformat", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--calib_size", default=None)
    parser.add_argument("--calib_dataset_jsonl", default=None)
    parser.add_argument("--calib_seq", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--gpu_max_mem_percentage", type=float, default=0.8)
    parser.add_argument("--inference_tensor_parallel", type=int, default=1)
    parser.add_argument("--inference_pipeline_parallel", type=int, default=1)
    parser.add_argument("--awq_block_size", type=int, default=0)
    parser.add_argument("--low_memory_mode", action="store_true")
    parser.add_argument("--use_seq_device_map", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--calibrate_all_moe_experts", action="store_true")
    parser.add_argument("--moe_all_experts_max_tokens", type=int, default=1)
    parser.add_argument("--moe_all_experts_every_forward", action="store_true")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def main() -> None:
    """Entry point: run the full Kimi K2.x BF16→NVFP4 PTQ pipeline."""
    args = parse_args()
    recipe = load_json(Path(args.official_quant_config))

    if args.calib_dataset_jsonl:
        calib_dataset_jsonl = Path(args.calib_dataset_jsonl)
        if not calib_dataset_jsonl.is_file():
            raise SystemExit(f"Calibration dataset JSONL does not exist: {calib_dataset_jsonl}")

    patch_kimi_init_weights_for_modelopt(Path(args.pyt_ckpt_path))
    hf_modules_cache = os.environ.get("HF_MODULES_CACHE")
    if hf_modules_cache:
        patch_kimi_init_weights_for_modelopt(Path(hf_modules_cache))
    patch_trust_remote_code()
    patch_modelopt_tokenizer_deepcopy(Path(args.modelopt_repo))
    hf_ptq = import_hf_ptq(Path(args.modelopt_repo))
    patch_accelerate_clear_device_cache()
    patch_transformers_attention_implementation(args.attn_implementation or "eager")
    patch_export_missing_nvfp4_weight_amax(hf_ptq)
    patch_kimi_moe_all_expert_calibration(
        hf_ptq,
        enabled=args.calibrate_all_moe_experts,
        max_tokens=args.moe_all_experts_max_tokens,
        every_forward=args.moe_all_experts_every_forward,
    )
    enable_stage_wrappers(hf_ptq)
    skip_generation_if_requested(hf_ptq, args.skip_generate)

    hf_args = build_hf_ptq_args(args, recipe)
    excludes = apply_official_recipe_to_hf_ptq(hf_ptq, hf_args, recipe)

    try:
        import faulthandler

        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception as exc:
        print(f"Could not enable SIGUSR1 Python stack dumps: {exc}", flush=True)

    record_stage("main:start")
    hf_ptq.main(hf_args)
    normalize_output_metadata(Path(args.export_path), recipe, excludes)
    verify_output_metadata(Path(args.export_path), recipe, excludes)
    verify_output_tensor_metadata(Path(args.export_path), excludes)
    record_stage("main:done")


if __name__ == "__main__":
    main()
