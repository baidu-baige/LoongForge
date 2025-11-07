"""SelfAttentio Without CP"""

from copy import deepcopy
from megatron.core.transformer.attention import SelfAttention, CrossAttention
from .ulysses_parallel import DistributedAttention
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.transformer.spec_utils import build_module


class UlyssesSelfAttention(SelfAttention):
    """Self-attention layer class"""

    def __init__(
        self, config, submodules, ulysses_gather_idx=0, ulysses_scatter_idx=2, **kwargs
    ):
        _submodules = deepcopy(submodules)
        _submodules.core_attention = lambda: None
        super().__init__(config, _submodules, **kwargs)

        _config = deepcopy(config)
        _config.context_parallel_size = 1
        _config.num_attention_heads //= config.context_parallel_size
        _config.num_query_groups //= config.context_parallel_size

        _core_attention = build_module(
            submodules.core_attention,
            config=_config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=kwargs.get("cp_comm_type"),
            softmax_scale=self.config.softmax_scale,
        )

        self.core_attention = DistributedAttention(
            _core_attention,
            get_context_parallel_group(check_initialized=False),
            (
                self.config.recompute_num_layers
                if self.config.recompute_num_layers is not None
                else 0
            ),
            gather_idx=ulysses_gather_idx,
            scatter_idx=ulysses_scatter_idx,
        )


class UlyssesCrossAttention(CrossAttention):
    """Cross-attention"""

    def __init__(self, config, submodules, **kwargs):
        _submodules = deepcopy(submodules)
        _submodules.core_attention = lambda: None
        super().__init__(config, _submodules, **kwargs)

        _config = deepcopy(config)
        _config.context_parallel_size = 1
        _config.num_attention_heads //= config.context_parallel_size
        _config.num_query_groups //= config.context_parallel_size

        _core_attention = build_module(
            submodules.core_attention,
            config=_config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=kwargs.get("cp_comm_type"),
            softmax_scale=self.config.softmax_scale,
        )

        self.core_attention = DistributedAttention(
            _core_attention,
            get_context_parallel_group(check_initialized=False),
            (
                self.config.recompute_num_layers
                if self.config.recompute_num_layers is not None
                else 0
            ),
        )
