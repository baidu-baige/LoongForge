"""STDiT model provider"""

from megatron.core.transformer.spec_utils import import_module

from aiak_training_omni.utils import get_args, build_transformer_config, print_rank_0
from aiak_training_omni.utils.constants import VideoLanguageModelFamilies

from aiak_training_omni.models.factory import register_model_provider

from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)

from .stdit_model import STDiTModel
from .stdit_layer_spec import get_stdit_layer_with_te_spec


@register_model_provider(model_family=[VideoLanguageModelFamilies.STDIT])
def stdit_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int = None,
    config=None,
) -> STDiTModel:
    """Builds the STDiT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        STDiTModel: The returned model
    """
    args = get_args()

    print_rank_0("building STDiT model ...")

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
        transformer_layer_spec = get_stdit_layer_with_te_spec()

    f = config.num_latent_frames // config.latent_patch_size[0]
    h = config.max_latent_height // config.latent_patch_size[1]
    w = config.max_latent_width // config.latent_patch_size[2]

    if args.seq_length != f * h * w:
        raise ValueError(f"seq_length {args.seq_length} != {f} * {h} * {w}")

    if args.max_position_embeddings < f * h * w:
        raise ValueError(
            f"max_position_embeddings {args.max_position_embeddings} < {f} * {h} * {w}"
        )

    model = STDiTModel(
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

    return model
