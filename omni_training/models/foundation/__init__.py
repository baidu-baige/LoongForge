"""qwen model"""

from .qwen2.qwen_model import Qwen2Model
from .qwen2.qwen_config import Qwen2Config
from .qwen3.qwen_model import Qwen3Model
from .qwen3.qwen_config import Qwen3Config
from .qwen3_next.qwen3_next_config import Qwen3NextConfig
from .qwen3_next.qwen3_next_model import Qwen3NextModel
from .deepseek.deepseek_config import DeepseekConfig
from .deepseek.deepseek_model import DeepseekModelWithMTP
from .mimo.mimo_config import MimoConfig
from .mimo.mimo_model import MimoModelWithMTP
from .minimax.minimax_config import MinimaxConfig
from .minimax.minimax_model import MinimaxModelWithMTP

from .llama.llama_config import LLaMAConfig
from .llama.llama_model import LLaMAModel
from transformers import AutoModel

from .internlm.internlm_model import InternLMModel
from .internlm.internlm_config import InternLMConfig

from .ernie4_5_vl.ernie4_5_vl_moe_model import ErnieMoeModel
from .ernie4_5_vl.ernie_config import ErnieMoeConfig
# The config name should not be the same as the huggingface config name
# or we can use exist_ok flag?
AutoModel.register(Qwen2Config, Qwen2Model, exist_ok=True)  # overwrite existing Qwen2Config
AutoModel.register(Qwen3Config, Qwen3Model, exist_ok=True)  # overwrite existing Qwen3Config
AutoModel.register(Qwen3NextConfig, Qwen3NextModel, exist_ok=True)
AutoModel.register(InternLMConfig, InternLMModel)
AutoModel.register(DeepseekConfig, DeepseekModelWithMTP)
AutoModel.register(MimoConfig, MimoModelWithMTP)
AutoModel.register(MinimaxConfig, MinimaxModelWithMTP)
AutoModel.register(LLaMAConfig, LLaMAModel)
AutoModel.register(ErnieMoeConfig, ErnieMoeModel)
