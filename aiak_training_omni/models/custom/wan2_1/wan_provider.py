"""STDiT model provider"""

from megatron.core.transformer.spec_utils import import_module

from aiak_training_omni.utils import get_args, build_transformer_config, print_rank_0
from aiak_training_omni.utils.constants import VideoLanguageModelFamilies

from aiak_training_omni.models.factory import register_model_provider

from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)

from .wan_model import WanModel, convert_state_dict_from_hg_i2v_14b
from .wan_layer_spec import get_wan_layer_with_te_spec
from safetensors import safe_open
import torch


@register_model_provider(model_family=[VideoLanguageModelFamilies.WAN2_1_I2V])
def wan_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int = None,
    config=None,
) -> WanModel:
    """Builds the STDiT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        STDiTModel: The returned model
    """
    args = get_args()

    print_rank_0("building Wan2.1 model ...")

    if config is None:
        config = build_transformer_config(args, config_class=StditTransformerConfig)
    config.pipeline_dtype = torch.float32

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        assert args.transformer_impl == "transformer_engine"
        transformer_layer_spec = get_wan_layer_with_te_spec()

    # 1.3B T2V
    extra_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 16,
        "dim": 1536,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 12,
        "num_layers": 30,
        "eps": 1e-6,
    }

    # 14B I2V
    extra_config = {
        "has_image_input": True,
        "patch_size": [1, 2, 2],
        "in_dim": 36,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "eps": 1e-06,
    }
    model = WanModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=False,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        vp_stage=vp_stage,
        **extra_config,
    )

    return model
