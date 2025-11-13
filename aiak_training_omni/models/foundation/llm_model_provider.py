"""llm model provider"""
import inspect
from contextlib import nullcontext
from transformers import AutoModel

from megatron.core.transformer.spec_utils import import_module
from aiak_training_omni.utils import (
    get_args, get_model_config, print_rank_0
)
from aiak_training_omni.models.common import BaseMegatronLanuageModule


def llm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
):
    """Generic LLM model provider.
    
    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits. Defaults to True.
    
    Returns:
        The corresponding LLM model.
    """
    args = get_args()

    config = get_model_config()

    print_rank_0(f"Building {config.model_type} model...")

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")
    
    # TODO: This param should not be assigned like this, but in old version, we assign this in tokenizer
    config.padded_vocab_size = args.padded_vocab_size

    # copied from qwen model provider
    # TODO: remove or not?
    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError(
                "--fp8-param-gather requires `fp8_model_init` from TransformerEngine,but not found.")

    with build_model_context(**build_model_context_args):
        # TODO: For now, we extract the parameters that need to be passed in using args.
        model: BaseMegatronLanuageModule = AutoModel.from_config(
            config=config,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
        )

    return model
