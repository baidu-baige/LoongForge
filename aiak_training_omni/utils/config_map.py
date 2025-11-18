"""config mapping"""
from pathlib import Path


# registry for model config
MODEL_CONFIG_REGISTRY = {
    "deepseek-v2-lite": {
        "config_path": "configs/models/deepseek",
        "config_name": "deepseek_v2_lite",
    },
    "qwen2_5-vl-7b": {
        "config_path": "configs/models/qwen2_5_vl",
        "config_name": "qwen2_5_vl_7b",
    },
}


def normalize_model_name(name: str) -> str:
    """Normalize input model name into canonical form."""
    # in case of adding other regular expressions
    return name.lower()


def get_config_from_model_name(model_name: str):
    """
    Lookup (config_path, config_name) from MODEL_CONFIG_REGISTRY,
    and automatically prepend absolute project_root to config_path.
    """
    name = normalize_model_name(model_name)

    if name not in MODEL_CONFIG_REGISTRY:
        raise KeyError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {list(MODEL_CONFIG_REGISTRY.keys())}"
        )
    entry = MODEL_CONFIG_REGISTRY[name]

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent   # AIAK root dir

    abs_config_path = str(project_root / entry["config_path"])

    return abs_config_path, entry["config_name"]
