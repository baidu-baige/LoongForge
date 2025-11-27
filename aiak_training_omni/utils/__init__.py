"""common module"""

from . import xpu_init
from .utils import (
    build_transformer_config,
    print_rank_0,
    is_te_min_version,
    is_torch_min_version,
    get_device_arch_version,
)

from .global_vars import (
    get_tokenizer,
    get_args,
    get_chat_template,
    get_model_config,
    get_data_config,
)
