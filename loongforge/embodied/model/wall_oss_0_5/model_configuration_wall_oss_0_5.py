# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wall-OSS-0.5 ModelConfig."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WallOss05ModelConfig:
    """Model-structure config for the Wall-OSS-0.5 Qwen2.5 VLA path."""

    model_type: str = "wall_oss_0_5"

    # Wall-X runtime model knobs.
    backbone: str = "qwen2_5"
    attn_implementation: Optional[str] = "sdpa"
    attn_deterministic: Optional[bool] = True
    use_selective_recompute: bool = False
    disable_train_autocast: bool = False
    use_ema: bool = False
    flow_loss_weight: float = 1.0
    ar_loss_weight: float = 0.01

    # Optional tokenizer extension path for AR/token action variants.
    action_tokenizer_type: Optional[str] = None
    action_tokenizer_path: Optional[str] = None
    action_tokenizer_checkpoint_path: Optional[str] = None
    action_tokenizer_config_dir: Optional[str] = None
    new_special_tokens: Optional[List[str]] = None
    action_tokenizer: Dict[str, Any] = field(default_factory=dict)

    # Task dimensions. Defaults match the LIBERO subset smoke config.
    dof_config: Dict[str, int] = field(
        default_factory=lambda: {
            "master_right_ee_cartesian_pos": 3,
            "master_right_ee_rotation": 3,
            "master_right_gripper": 1,
            "action_padding": 19,
        }
    )
    ar_dof_config: Dict[str, int] = field(
        default_factory=lambda: {
            "master_right_ee_cartesian_pos": 3,
            "master_right_ee_rotation": 3,
            "master_right_gripper": 1,
            "action_padding": 19,
        }
    )
    agent_pos_config: Dict[str, int] = field(
        default_factory=lambda: {
            "follow_right_ee_cartesian_pos": 3,
            "follow_right_ee_rotation": 3,
            "follow_right_gripper": 2,
            "action_padding": 18,
        }
    )
    action_horizon: int = 10
    action_horizon_flow: int = 10
    use_state_string_representation: bool = False
    norm_forward_prefetch_distance: int = 0

    @property
    def action_dim(self) -> int:
        """Action dim."""
        return int(sum(self.dof_config.values()))

    @property
    def propri_dim(self) -> int:
        """Propri dim."""
        return int(sum(self.agent_pos_config.values()))
