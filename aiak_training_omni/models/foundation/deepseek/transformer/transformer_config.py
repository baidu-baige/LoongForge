"""custom transformer config for DeepSeek"""

from dataclasses import dataclass
from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class DeepSeekTransformerConfig(MLATransformerConfig):
    """DeepSeek transformer configuration."""

    # MTP hyperparameters
    num_nextn_predict_layers: int = 0
    """number of next-token prediction layers"""

    mtp_loss_coef: float = 0.1
    """mtp loss coefficient"""
