"""qwen model"""
from .qwen.qwen_model import QwenModel
from .qwen.qwen_config import QwenConfig
from transformers import AutoModel

AutoModel.register(QwenConfig, QwenModel)