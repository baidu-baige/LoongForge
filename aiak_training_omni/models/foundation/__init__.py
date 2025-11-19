"""qwen model"""

from .qwen.qwen_model import QwenModel
from .qwen.qwen_config import QwenConfig
from .deepseek.deepseek_config import DeepseekConfig
from .deepseek.deepseek_model import DeepseekModelWithMTP
from .llama.llama_config import LLaMAConfig
from .llama.llama_model import LLaMAModel
from transformers import AutoModel

# The config name should not be the same as the huggingface config name
# or we can use exist_ok flag?
AutoModel.register(QwenConfig, QwenModel)
AutoModel.register(DeepseekConfig, DeepseekModelWithMTP)
AutoModel.register(LLaMAConfig, LLaMAModel)
