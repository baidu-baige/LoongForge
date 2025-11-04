"""config augments"""
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict

@dataclass
class ConfigAugments:
    """Configuration augmentations for AIAK training LLM models."""
    def __init__(self):
        pass

    def flatten_with_priority(self) -> Dict[str, Any]:
        """flatten the config into a single dictionary, with priority given to child configs."""
        flat: Dict[str, Any] = {}

        def dfs(cfg: Any):
            if not isinstance(cfg, ConfigAugments):
                return

            for f in fields(cfg):
                value = getattr(cfg, f.name)

                if f.name == "defaults" or f.name == "_target_":
                    continue
                
                if isinstance(value, ConfigAugments):
                    dfs(value)
                elif is_dataclass(value):  
                    dfs(value)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        flat[k] = v
                else:
                    flat[f.name] = value

        dfs(self)
        return flat
