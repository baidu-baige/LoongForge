"""STDiT model provider"""

from megatron.core.transformer.spec_utils import import_module

from aiak_training_omni.utils import get_args, build_transformer_config, print_rank_0
from aiak_training_omni.utils.constants import VideoLanguageModelFamilies

from aiak_training_omni.models.factory import register_model_provider
from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)

from .stdit3_model import STDiT3Model
from .stdit3_layer_spec import get_stdit3_layer_with_te_spec


@register_model_provider(model_family=[VideoLanguageModelFamilies.STDIT3])
def stdit3_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int = None,
    config=None,
) -> STDiT3Model:
    """Builds the STDiT3 model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        STDiT3Model: The returned model
    """
    args = get_args()

    print_rank_0("building STDiT3 model ...")

    if config is None:
        config = build_transformer_config(args, config_class=StditTransformerConfig)

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.pipeline_model_parallel_size > 1:
        raise NotImplementedError("Pipeline parallelism is not supported yet.")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        assert args.transformer_impl == "transformer_engine"
        transformer_layer_spec = get_stdit3_layer_with_te_spec()

    model = STDiT3Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        vp_stage=vp_stage,
    )

    for param in model.y_embedder.parameters():
        param.requires_grad = False

    return model
