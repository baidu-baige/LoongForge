# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Typed config dataclasses for the LoongForge eval server.

These replace the argparse.Namespace pattern and mirror the training-side
convention of passing typed dataclasses between components.

``EvalServerArgs``  — runtime/infrastructure options (model: and server: YAML sections).
``EvalServerConfig`` — server network settings (server: YAML section).

Usage::

    args, server_cfg = parse_eval_server_config(config_path)
    model_spec = build_model_spec(args, raw_model_dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from omegaconf import MISSING


@dataclass
class EvalServerArgs:
    """Runtime options for the eval policy server.

    Mirrors ``training_args``-style separation: fields here are server/infra
    concerns, NOT model-structure fields (those live in each model's ModelConfig).

    Note: compile_model/compile_mode are model-structure fields and live in
    ModelConfig (e.g. Pi05ModelConfig), mirroring the training-side convention.
    """

    # Checkpoint and tokenizer paths
    ckpt_path: str = ""
    tokenizer_path: str = ""
    dataset_statistics_path: str = ""

    # Device / dtype
    device: str = "cuda"
    use_bf16: bool = True

    # Init mode
    random_init: bool = False

    # Server network
    port: int = 10093
    health_port: int = 10094

    # Working directory
    loongforge_root: str = ""

    # Model type (used for registry lookup). Required — no default; must be
    # provided via the ``model.model_type`` YAML field.
    model_type: str = MISSING

    # How many action steps from a model chunk to keep before replan (eval-side).
    # 0 = model-factory default (xvla uses 10 to match official clients).
    # Positive N = truncate to N. Negative = no truncation.
    chunk_execute_steps: int = 0

    # Embodiment tag for multi-embodiment models (e.g. GR00T-N1.6). Selects the
    # modality/state/action layout the model's predict_action decodes for. The
    # per-benchmark value (LIBERO -> ``libero_panda``) is set in the YAML
    # ``server:`` section; models that ignore it keep the default.
    embodiment_tag: str = "behavior_r1_pro"


def parse_eval_server_config(config_path: str) -> tuple[EvalServerArgs, Dict[str, Any]]:
    """Parse an eval YAML config and return (EvalServerArgs, raw_model_dict).

    The raw_model_dict is the verbatim ``model:`` section from the YAML, used
    by ``build_model_config`` to construct the typed ModelConfig via OmegaConf.

    Args:
        config_path: Path to the eval YAML config file.

    Returns:
        Tuple of (EvalServerArgs, raw_model_dict).
    """
    import yaml
    from omegaconf import OmegaConf

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = raw.get("model") or {}
    server = raw.get("server") or {}

    # Build a dict containing only fields declared in EvalServerArgs, plus model_type.
    # This avoids passing unknown server: keys (host, python, log, etc.) to OmegaConf structured merge.
    _EVAL_SERVER_FIELDS = {f.name for f in __import__("dataclasses").fields(EvalServerArgs)}
    server_known = {k: v for k, v in server.items() if k in _EVAL_SERVER_FIELDS}
    model_type = model.get("model_type")
    if not model_type:
        raise ValueError(
            f"'model.model_type' is required in the eval config but was missing: {config_path}"
        )
    server_with_extras = {
        **server_known,
        "model_type": str(model_type).lower(),
    }
    merged = OmegaConf.merge(OmegaConf.structured(EvalServerArgs), server_with_extras)
    server_args: EvalServerArgs = OmegaConf.to_object(merged)

    return server_args, dict(model)
