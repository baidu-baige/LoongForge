"""Deepseek model provider"""
import inspect
from contextlib import nullcontext

from megatron.training import print_rank_0

from aiak_training_omni.utils import get_args, build_transformer_config
from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_provider

from aiak_training_omni.models.deepseek.transformer import DeepSeekTransformerConfig

from .deepseek_model import DeepseekModelWithMTP
from .deepseek_layer_spec import get_deepseek_decoder_block_and_mtp_spec


@register_model_provider(model_family=LanguageModelFamilies.DEEPSEEK)
def deekseek_model_provider(
    pre_process: bool = True, post_process: bool = True, parallel_output: bool = True,
) -> DeepseekModelWithMTP:
    """Builds the deepseek model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        DeepseekModelWithMTP: The returned model
    """
    args = get_args()

    print_rank_0('building Deepseek model ...')

    config = build_transformer_config(args, config_class=DeepSeekTransformerConfig)

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    assert args.spec is None, "Not support --spec option for Deepseek"
        
    decoder_block_spec, mtp_spec = get_deepseek_decoder_block_and_mtp_spec(config)

    model = DeepseekModelWithMTP(
        config=config,
        transformer_layer_spec=decoder_block_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        rope_scaling_factor=args.rope_scaling_factor,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        mtp_layer_spec=mtp_spec,
    )

    return model
