"""qwen model"""

from .qwen.qwen_model import QwenModel
from .qwen.qwen_config import QwenConfig
from .deepseek.deepseek_config import DeepseekConfig
from .deepseek.deepseek_model import DeepseekModelWithMTP
from transformers import AutoModel

AutoModel.register(QwenConfig, QwenModel)
AutoModel.register(DeepseekConfig, DeepseekModelWithMTP)
