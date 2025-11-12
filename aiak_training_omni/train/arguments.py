"""AIAK arguments"""

import os
import argparse
import importlib
from aiak_training_omni.models.common.vlm_model_config import VLMModelConfig
from omegaconf import OmegaConf
import torch.nn.functional as F

from megatron.core.transformer.enums import AttnBackend
from megatron.training.arguments import (
    add_megatron_arguments,
    validate_args as validate_megatron_args,
)

from aiak_training_omni.models import (
    get_support_model_family_and_archs,
    get_model_config,
    get_model_family,
    get_support_model_archs,
)
from aiak_training_omni.tokenizer import get_default_tokenizer
from aiak_training_omni.data import get_support_templates

from aiak_training_omni.utils import (
    constants,
    parse_arguments,
    print_rank_0,
    build_model_config,
)
from aiak_training_omni.utils.global_vars import set_model_config, set_hydra_config, get_hydra_config
from aiak_training_omni.utils.utils import get_default_sft_dataset_config
import importlib


def is_subclass_from_path(class_path: str, base_class: type):
    """Check whether the given class path belongs to the specified base class"""
    try:
        mod_path, cls_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return issubclass(cls, base_class)
    except (ValueError, ImportError, AttributeError, TypeError):
        return False


def parse_args_from_config(args):
    """parse args from config"""
    config = get_hydra_config()
    model_cfgs = build_model_config(args, config)    
    set_model_config(model_cfgs)


def parse_train_args(args_defaults={}):
    """parse arguments for training"""
    args, hydra_cfg = parse_arguments(
        extra_args_provider=aiak_extra_train_args_provider,
        validate_extra_args_provider=validate_aiak_extra_args,
        args_defaults=args_defaults,
        parse_unknown_args=True,
    )
    set_hydra_config(hydra_cfg)

    return args


def _add_log_tensor_args(parser):
    group = parser.add_argument_group(title="Arguments for Logging Tensor stats")

    group.add_argument(
        "--enable-log-tensor", action="store_true", help="trace debug info & tensors."
    )
    # default value means log all module's info
    group.add_argument(
        "--log-tensor-name-pattern",
        type=str,
        default=None,
        help="The module name pattern by which log tensor is applied",
    )
    group.add_argument(
        "--log-tensor-stage",
        type=str,
        default="forward",
        choices=["init", "forward", "backward"],
        help="log tensor at which stage",
    )
    group.add_argument(
        "--log-tensor-iter-pattern",
        type=str,
        default=None,
        help="for which iters to log tensors, value like 8,15,20",
    )
    group.add_argument(
        "--log-tensor-mbs-pattern",
        type=str,
        default=None,
        help="for which mbs to log tensors, value like 8,15,20",
    )
    group.add_argument(
        "--log-tensor-layer-pattern",
        type=str,
        default=None,
        help="for which layer to log tensors, value like 8,15,20",
    )
    group.add_argument(
        "--log-tensor-rank",
        type=str,
        default="0",
        help="for which rank to log tensors, value like 0,1,2,4",
    )

    # Save Tensor options
    group.add_argument(
        "--save-tensor", action="store_true", help="Save tensors to files."
    )
    group.add_argument(
        "--save-tensor-dir", type=str, default="", help="Save tensor to directory"
    )

    return parser


def aiak_extra_train_args_provider(parser: argparse.ArgumentParser):
    """Add AIAK arguments to parser"""
    parser.conflict_handler = "resolve"
    parser = _add_extra_model_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_extra_sft_args(parser)
    parser = _add_extra_video_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_extra_multimodal_args(parser)
    parser = _add_extra_parallel_args(parser)
    # add args for debug infos;
    parser = _add_log_tensor_args(parser)
    return parser


def validate_aiak_extra_args(args, config):
    """ "Validate AIAK extra arguments"""
    _validate_extra_model_args(args, config)
    _validate_extra_tokenizer_args(args)
    _validate_extra_training_args(args)
    _validate_extra_sft_args(args)
    _validata_extra_multimodal_args(args)
    _validata_extra_video_args(args)
    _validata_extra_parallel_args(args)

    # megatron one_logger is not supported in aiak
    args.enable_one_logger = False


def _add_extra_model_args(parser: argparse.ArgumentParser):
    """Add model arguments"""
    group = parser.add_argument_group(title="extra-model")
    group.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="The config file path for model configuration.",
    )
    group.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="The config file path for model configuration.",
    )

    # use for cogvlm2
    group.add_argument(
        "--no-rope-in-fp32",
        action="store_false",
        dest="rope_in_fp32",
        help="Disable Rope in FP32",
    )

    # use for baichuan2
    group.add_argument(
        "--use-normhead",
        action="store_true",
        help="use NormHead. https://arxiv.org/pdf/2309.10305.pdf. "
        "Note that this option is only valid for the model family baichuan2 now.",
    )

    # use for deepseek v3
    group.add_argument(
        "--mtp-loss-coef", type=float, default=0.1, help="The coefficient of MTP loss."
    )

    # use for mla
    group.add_argument(
        "--enable-fa-within-mla",
        action="store_true",
        help="Since qk_head_dim != v_head_dim in MLA, fa cannot be used by default. Enable "
        "this option, the head dimensions will be aligned by padding, so that fa can be used."
        "Deprecated: use --attention-backend=flash",
    )

     # # ============ Hydra config ============
    group.add_argument('--config-path', type=str, required=True,
                        help='Hydra path to config directory')
    group.add_argument('--config-name', type=str, required=True,
                        help='Hydra config file name (without .yaml suffix)')

    return parser


def _add_extra_tokenizer_args(parser: argparse.ArgumentParser):
    """Add data arguments"""
    group = parser.add_argument_group(title="extra-tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        default=None,
        choices=["NullTokenizer", "HFTokenizer"],  # recommended
        help="What type of tokenizer to use. Default: None, and aiak automatically determines "
        "the type of tokenizer required",
    )

    group.add_argument(
        "--hf-tokenizer-path",
        type=str,
        default=None,
        help="HuggingFace tokenizer path: "
        "1) A string, the *model id* of a predefined tokenizer hosted inside a model repo "
        "on huggingface.co"
        "2) A path to a *directory* containing vocabulary files required by the tokenizer",
    )

    group.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help="Whether or not to use the fast tokenizer when --tokenizer-type=HFTokenizer."
        "Default: False",
        dest="use_fast_tokenizer",
    )

    group.add_argument(
        "--split-special-tokens",
        action="store_true",
        help="Whether or not the special tokens should be split during the tokenization process "
        "when --tokenizer-type=HFTokenizer. Default: False",
    )

    group.add_argument(
        "--padding-side",
        default="right",
        choices=["left", "right"],
        help=f"The side on which the padding should be applied when --tokenizer-type=HFTokenizer. "
        "Default: right",
    )

    group.add_argument(
        "--additional-special-tokens",
        type=str,
        default=None,
        help="Additional special tokens to add to the tokenizer. Use commas to separate multiple tokens",
    )

    group.add_argument(
        "--vocab-size-in-config-file",
        type=int,
        default=None,
        help="Size of vocab from hf config file.",
    )

    group.add_argument(
        "--padded-vocab-size", type=int, default=None, help="Specify padded vocab size."
    )

    return parser


def _add_extra_sft_args(parser: argparse.ArgumentParser):
    """Add SFT arguments"""
    group = parser.add_argument_group(title="extra-sft")
    group.add_argument(
        "--chat-template",
        type=str,
        choices=get_support_templates(),
        default=None,
        help="The template to apply to instruction data.",
    )

    group.add_argument(
        "--sft-dataset-config",
        type=str,
        default=None,
        help="A json file that contains the dataset configuration."
        "default: configs/dataset_config.jsoin",
    )

    group.add_argument(
        "--sft-dataset",
        nargs="*",
        default=None,
        help="The name list for a set of dataset according to --data-path. Note that:"
        "(1) the dataset name should be defined in the dataset config file (--sft-dataset-config). "
        "(2) the accepted formats are: a single name or a list of names e.g. dataset1 dataset2. "
        "(3) if multiple dataset are required, the order of names should be consistent with"
        "--data-path. "
        "This argument is exclusive to the other independent --sft-*-dataset arguments.",
    )

    group.add_argument(
        "--sft-train-dataset",
        nargs="*",
        default=None,
        help="The name list for a set of independent train dataset according to --train-data-path. "
        "Follows the same pattern rules as --sft-dataset",
    )

    group.add_argument(
        "--sft-valid-dataset",
        nargs="*",
        default=None,
        help="The name list for a set of independent valid dataset according to --valid-data-path. "
        "Follows the same pattern rules as --sft-dataset",
    )

    group.add_argument(
        "--sft-test-dataset",
        nargs="*",
        default=None,
        help="The name list for a set of independent test dataset according to --test-data-path. "
        "Follows the same pattern rules as --sft-dataset",
    )

    group.add_argument(
        "--sft-sort-batch",
        action="store_true",
        help="Sort the entire dataset from smallest to largest; "
        "if the --packing-sft-data option is enabled, sort the data after packing. Default: False",
    )

    group.add_argument(
        "--sft-data-streaming",
        action="store_true",
        help="enable data streaming. Default: False",
    )

    group.add_argument(
        "--streaming-buffer-size",
        type=int,
        default=16384,
        help="The size of the buffer to randomly sample examples from in dataset streaming",
    )

    group.add_argument(
        "--sft-data-mix-strategy",
        type=str,
        choices=["concat", "interleave_under", "interleave_over"],
        default="concat",
        help="The strategy to mix the sft data. Default: concat",
    )

    group.add_argument(
        "--sft-num-preprocess-workers",
        type=int,
        default=None,
        help="The number of workers to use for data preprocessing. Only support non-streaming mode.",
    )

    group.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="Whether compute loss on prompt. Default: False",
    )

    group.add_argument(
        "--history-mask-loss",
        action="store_true",
        help="Only compute loss on last turn response, instead of full history. Default: False",
    )

    group.add_argument(
        "--is-tokenized-data",
        action="store_true",
        help="Whether the data is already tokenized. Default: False.",
    )

    group.add_argument(
        "--packing-sft-data",
        action="store_true",
        help="Whether to pack multiple sft data into one.",
    )

    group.add_argument(
        "--enable-discard-sample",
        action="store_true",
        help="Whether to discard sample when its length is greater than seq-length.",
    )

    group.add_argument(
        "--packing-batch-size",
        type=int,
        default=10000,
        help="Perform packing in batches, deciding how many samples each batch contains;"
        "if the --sft-sort-batch option is enabled, the samples will be sorted after packing.",
    )

    group.add_argument(
        f"--use-fixed-seq-lengths",
        action="store_true",
        help="If enabled, all input sequence lengths will be padding to --seq-length."
        "Only support language models now.",
    )

    group.add_argument(
        "--sample-type",
        type=str,
        default=None,
        help="Specify the default sample type for fallback cooker routing "
        "when no specific cooker matches. "
        "For example: 'multi_mix_vqa'. "
        "This allows the dataloader to apply the corresponding cooker "
        "to samples without explicit subflavors.",
    )
    return parser


def _add_extra_video_args(parser):
    group = parser.add_argument_group(title="extra-video")

    # use for stdit models
    group.add_argument(
        "--latent-in-channels", type=int, help="Number of channels in input latent data"
    )

    group.add_argument(
        "--latent-out-channels",
        type=int,
        help="Number of channels in output latent data",
    )

    group.add_argument(
        "--caption-channels", type=int, help="Number of channels in caption data"
    )

    group.add_argument(
        "--latent-patch-size",
        type=tuple,
        default=(1, 1, 1),
        help="Patch size for vision task",
    )

    group.add_argument(
        "--latent-space-scale",
        type=float,
        default=1.0,
        help="Space scale for vision task",
    )

    group.add_argument(
        "--latent-time-scale",
        type=float,
        default=1.0,
        help="Time scale for vision task",
    )

    group.add_argument(
        "--num-latent-frames", type=int, help="Number of frames in video"
    )

    group.add_argument("--max-latent-height", type=int, help="Maximum height of video")

    group.add_argument("--max-latent-width", type=int, help="Maximum width of video")

    group.add_argument(
        "--latent-frame-interval", type=int, default=1, help="Interval between frames"
    )

    group.add_argument("--max-text-length", type=int, help="Maximum text length")

    group.add_argument(
        "--max-video-length", type=int, default=32760, help="Maximum video length"
    )

    group.add_argument("--max-image-length", type=int, help="Maximum image length")

    group.add_argument(
        "--max-timestep-boundary",
        type=float,
        default=1,
        help="The maximum timestep boundary for dit, with a value range between 0 and 1.",
    )

    group.add_argument(
        "--min-timestep-boundary",
        type=float,
        default=0,
        help="The minimum timestep boundary for dit, with a value range between 0 and 1.",
    )

    group.add_argument("--stdit-bucket-config", type=str, help="bucket config file")

    group.add_argument(
        "--num-bucket-build-workers",
        type=int,
        default=1,
        help="Number of workers to build bucket",
    )

    # arguments for InternVL
    group.add_argument(
        "--loss-reduction-all-gather",
        action="store_true",
        help="Whether to gather all during loss reduction. Default is False.",
    )

    group.add_argument(
        "--conv-style",
        type=str,
        default="internvl2_5",
        help="Prompt style for a conversation.",
    )

    group.add_argument(
        "--force-image-size",
        type=int,
        default=448,
        help="Set the desired size for the image. Default is 448.",
    )

    group.add_argument(
        "--num-images-expected",
        type=int,
        default=48,
        help="The maximum number of images per packed sample. Default is 48.",
    )

    group.add_argument(
        "--pad2square",
        action="store_true",
        help="Pad the image to a square shape if set to True. Default is False.",
    )

    group.add_argument(
        "--use-data-resampling",
        action="store_true",
        help="Set to True to use data resampling. Default is False.",
    )

    group.add_argument(
        "--down-sample-ratio",
        type=float,
        default=0.5,
        help="Set the desired down-sampling ratio for the image. Default is 0.5.",
    )

    group.add_argument(
        "--max-buffer-size",
        type=int,
        default=20,
        help="The buffer size of the packed dataset. Default is 20.",
    )

    group.add_argument(
        "--max-packed-tokens",
        type=int,
        default=8192,
        help="The required token length of per packed sample. Default is 8192.",
    )

    group.add_argument(
        "--log_freq",
        type=int,
        default=1000,
        help="The log frequency of the packed dataset. Default is 1000.",
    )

    group.add_argument(
        "--strict-mode",
        action="store_true",
        help="Whether to pad the number of images to satisfy num_images_expected. Default is False.",
    )

    group.add_argument(
        "--replacement",
        action="store_true",
        help="Whether to restart the dataset after it is exhausted. Default is False.",
    )

    group.add_argument(
        "--allow-overflow",
        action="store_true",
        help="Whether to drop the sample over the specified max_packed_tokens. Default is False.",
    )

    group.add_argument(
        "--loss-reduction",
        type=str,
        default="square",
        help="Loss reduction method. Default is square.",
    )

    group.add_argument("--patch-size", type=int, default=14)

    group.add_argument("--group-by-length", action="store_true")

    group.add_argument(
        "--min-num-frame",
        type=int,
        default=8,
        help="The minimum number of frames for video data. Default is 8.",
    )

    group.add_argument(
        "--max-num-frame",
        type=int,
        default=32,
        help="The maximum number of frames for video data. Default is 32.",
    )

    group.add_argument(
        "--dynamic-image-size",
        action="store_true",
        help="Set to True to use dynamic high resolution strategy. Default is False.",
    )

    group.add_argument(
        "--min-dynamic-patch",
        type=int,
        default=1,
        help="The minimum number of dynamic patches. Default is 1.",
    )

    group.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=12,
        help="The maximum number of dynamic patches. Default is 12.",
    )

    group.add_argument(
        "--use_thumbnail",
        action="store_true",
        help="Set to True to add a thumbnail image. Default is False.",
    )

    group.add_argument(
        "--normalize_type",
        type=str,
        default="imagenet",
        help="The normalization type for the image. Default is imagenet.",
    )

    group.add_argument("--use-packed-ds", action="store_true")

    group.add_argument("--communicate-dataset", action="store_true")

    group.add_argument("--save-dataset-state", action="store_true")

    group.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="the value of the prefetch_factor parameter of the dataloader",
    )

    group.add_argument("--no-split-annotations", action="store_true")

    return parser


def _add_extra_training_args(parser: argparse.ArgumentParser):
    """Add training arguments"""
    group = parser.add_argument_group(title="extra-training")

    group.add_argument(
        "--training-phase",
        type=str,
        default=constants.TrainingPhase.PRETRAIN,
        choices=[constants.TrainingPhase.PRETRAIN, constants.TrainingPhase.SFT],
        help="Which phase to train. Default: pretrain",
    )

    group.add_argument(
        "--no-detail-log",
        action="store_false",
        help="If set, the detail-log-interval will no longer take effect.",
        dest="log_detail",
    )

    group.add_argument(
        "--detail-log-interval",
        type=int,
        default=20,
        help="Report timing interval."
        " detail-log-interval will only take effect when the"
        " timing-log-level is set to 0",
    )

    group.add_argument(
        "--variable-seq-lengths",
        action="store_true",
        help="DEPRECATED. This flag is ignored."
        "Support for variable sequence lengths across microbatches.",
    )

    group.add_argument(
        "--enable-ema",
        action="store_true",
        help="enable Model EMA (Exponential Moving Average)"
        " to maintain moving averages of the trained parameters",
    )

    group.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")

    group.add_argument(
        "--save-ema",
        type=str,
        default=None,
        help="Output directory to save ema checkpoints to, default to ${args.save}/ema",
    )

    group.add_argument(
        "--load-ema",
        type=str,
        default=None,
        help="Directory containing a ema checkpoint, default to ${args.load}/ema",
    )

    group.add_argument(
        "--ckpt-format",
        default="torch",
        choices=["torch", "torch_dist", "zarr"],
        help="Checkpoint format to use. Default: torch",
    )

    return parser


def _add_extra_multimodal_args(parser):
    """Add multimodal arguments"""
    # FIXME: Currently, multimodal implementation is based on cogvlm, and whether the newly added parameters
    # are universally applicable needs to be determined subsequently;

    group = parser.add_argument_group(title="extra-multimodal")
    group.add_argument(
        "--language-model-type",
        type=str,
        default=None,
        choices=get_support_model_archs(constants.LanguageModelFamilies.names()),
    )

    group.add_argument(
        "--trainable-modules",
        default=["all"],
        nargs="*",
        help="choices: all, language_model, adapter, vision_model, "
        "language_expert_linear, vision_expert_linear",
    ),

    group.add_argument(
        "--dataloader-save",
        type=str,
        default=None,
        help="Energon dataloader state save path",
    )

    group.add_argument(
        "--packing-pretrain-data",
        action="store_true",
        help="Whether to pack multiple pretrain data into one.",
    )

    group.add_argument(
        "--add-question-in-pretrain",
        action="store_true",
        help="Whether add question in pretrain VQASample",
    )

    # use for qwen2vl now
    group.add_argument(
        "--image-resolution", type=int, help="Resolution of image inputs"
    )

    group.add_argument(
        "--min-pixels", type=int, default=4 * 28 * 28, help="Minimum image pixels"
    )

    group.add_argument(
        "--max-pixels", type=int, default=16384 * 28 * 28, help="Maximum image pixels"
    )

    group.add_argument(
        "--frame-min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum frame pixels",
    )

    group.add_argument(
        "--frame-max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Maximum frame pixels",
    )

    group.add_argument(
        "--video-max-pixels",
        type=int,
        default=65536 * 28 * 28,
        help="Maximum video pixels",
    )

    group.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="The fps to extract frames for model inputs",
    )

    group.add_argument(
        "--fps-min-frames",
        type=int,
        default=4,
        help="The minimum number of frames of the video",
    )

    group.add_argument(
        "--fps-max-frames",
        type=int,
        default=768,
        help="The maximum number of frames of the video",
    )

    return parser


def _add_extra_parallel_args(parser):
    """Add parallel arguments"""
    group = parser.add_argument_group(title="extra-parallel")

    # NOTE：In order to be compatible with the old version of AIAK,
    # --context-parallel-ulysses-degree temporarily retained.
    group.add_argument(
        "--context-parallel-ulysses-degree",
        type=int,
        default=1,
        help="Degree of context parallelism in ulysses attention.",
    )
    group.add_argument(
        "--using-config-strategy",
        action="store_true",
        help="Use the parallel configuration in the model configuration file.",
    )
    return parser


def _validate_extra_model_args(args, config):
    """Setup model config based on the given model name."""
    if not hasattr(config, "model"):
        raise ValueError("Hydra config doesn't have model section.")
    args.model_family =  config.model.model_type
    model_config = None
    if "foundation" in config.model:
        model_config = config.model.foundation
    else:
        # compatibility for llm
        model_config = config.model

    if model_config is not None:
        # the structural configuration of model will be overwritten, such as num_layers, hidden_states..
        for key in model_config:
            if hasattr(args, key):
                setattr(args, key, model_config[key])
                print_rank_0(f"  {key} = {model_config[key]} ", args.rank)

        print_rank_0(
            "---------------- End of configuration ----------------", args.rank
        )

    if args.enable_fa_within_mla:
        args.attention_backend = AttnBackend.flash
        print_rank_0(
            f"--enable-fa-within-mla is enabled, setting attention backend to FlashAttention",
            args.rank,
        )


def _validate_extra_tokenizer_args(args):
    """Setup tokenizer based on the given model name."""
    if args.tokenizer_type is None:
        args.tokenizer_type = get_default_tokenizer(args.model_family)
        assert (
            args.tokenizer_type is not None
        ), "No default tokenizer found for the given model name, please set --tokenizer-type"

        print_rank_0(f"Configure tokenizer to {args.tokenizer_type}", args.rank)

    if args.additional_special_tokens is not None:
        args.additional_special_tokens = [
            token.strip() for token in args.additional_special_tokens.split(",")
        ]


def _validate_extra_sft_args(args):
    """Validate SFT arguments"""
    if args.training_phase != constants.TrainingPhase.SFT:
        return

    if args.tokenizer_type != "HFTokenizer":
        raise ValueError(
            "--tokenizer-type should be HFTokenizer when training phase is sft"
        )

    args.dataloader_type = "external"
    print_rank_0(
        f"INFO: Set dataloader type to external since --training-phase=SFT", args.rank
    )

    if args.chat_template is None:
        raise ValueError("--chat-template is required when training phase is sft")

    if args.train_on_prompt and args.history_mask_loss:
        raise ValueError(
            "--train-on-prompt and --history-mask-loss cannot both be True at the same time"
        )

    if args.sft_dataset_config is None:
        # set default sft-dataset-config
        default_config = get_default_sft_dataset_config()
        if default_config is not None:
            args.sft_dataset_config = default_config
            print_rank_0(
                f"WARNING: --sft-dataset-config is not specified, setup to default config ({default_config})",
                args.rank,
            )
        else:
            raise ValueError(
                "--sft-dataset-config is not specified, and "
                "the default config does not exist, please setup it"
            )
    if args.sft_data_streaming:
        assert (
            args.sft_sort_batch is None or not args.sft_sort_batch
        ), '--sft-sort-batch" cannot be used together with --sft-data-streaming'

    if args.use_fixed_seq_lengths:
        args.variable_seq_lengths = False
    else:
        # Defaults to True but enforced as fixed-length for specific features (e.g., tp-comm-overlap/ moe allgather)
        args.variable_seq_lengths = True
        if args.tp_comm_overlap:
            # tp_comm_overlap requires fixed-length
            args.variable_seq_lengths = False

        if (
            args.num_experts is not None
            and args.num_experts > 0
            and args.moe_token_dispatcher_type in ["allgather", "alltoall_seq"]
        ):
            # allgather or alltoall_seq requires fixed-length
            args.variable_seq_lengths = False

    if args.packing_sft_data:
        if args.micro_batch_size > 1:
            args.micro_batch_size = 1
            print_rank_0(
                "WARING: Setting args.micro_batch_size to 1 since packing_sft_data is enabled",
                args.rank,
            )

        if args.context_parallel_size > 1:
            if (
                args.context_parallel_ulysses_degree < args.context_parallel_size
                and args.cp_comm_type == "allgather"
            ):
                args.cp_comm_type = "p2p"
                print_rank_0(
                    "WARNING: Setting args.cp_comm_type to p2p since ring attention "
                    "does not support all gather while packing_sft_data is enabled",
                    args.rank,
                )

        # check if the model is supported
        if args.multi_latent_attention:
            if not args.enable_fa_within_mla:
                args.enable_fa_within_mla = True
                args.attention_backend = AttnBackend.flash
                print_rank_0(
                    "WARING: Setting args.enable_fa_within_mla to true since enable sft-packing with mla",
                    args.rank,
                )

    if args.padding_side == "left":
        args.padding_side = "right"
        print_rank_0(
            "WARING: Setting args.padding_side to right when run sft.", args.rank
        )


def _validate_extra_training_args(args):
    """Validate training arguments"""

    # check ema
    if args.enable_ema:
        assert args.model_family in [
            constants.VideoLanguageModelFamilies.STDIT,
            constants.VideoLanguageModelFamilies.STDIT3,
        ], f"EMA only supports STDIT models."

        if args.load_ema is None and args.load is not None:
            args.load_ema = os.path.join(args.load, "ema")

        if args.save_ema is None and args.save is not None:
            args.save_ema = os.path.join(args.save, "ema")


def _validata_extra_multimodal_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.VisionLanguageModelFamilies.names():
        return
    
    args.variable_seq_lengths = True
    if not (args.packing_pretrain_data or args.packing_sft_data):
        args.packing_batch_size = None


def _validata_extra_video_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.VideoLanguageModelFamilies.names():
        return

    if args.model_family == "wan2_1_i2v":
        return

    # make text length divisible by cp size
    if args.max_text_length is not None and args.context_parallel_size > 1:
        while (args.max_text_length % args.context_parallel_size) != 0:
            args.max_text_length += 1


def _validata_extra_parallel_args(args):
    """Validate parallel arguments"""
    # check cp, NOTE: maybe removed in the future
    if args.context_parallel_size > 1:
        if (
            args.context_parallel_ulysses_degree is None
            or args.context_parallel_ulysses_degree < 1
        ):
            # not set
            return

        assert (
            args.hierarchical_context_parallel_sizes is None
        ), "ERROR: Cannot specify both hierarchical_context_parallel_sizes and context_parallel_ulysses_degree"

        assert (
            args.context_parallel_ulysses_degree <= args.context_parallel_size
            and args.context_parallel_size % args.context_parallel_ulysses_degree == 0
        ), "ERROR: context_parallel_ulysses_degree must less than context_parallel_size and divisible by it"

        # only cp
        if args.context_parallel_ulysses_degree == 1:
            # just use cp
            assert (
                "a2a" not in args.cp_comm_type
            ), "p2p or allgather are allowed for non-ulysses context parallel"
        # only ulysses
        elif args.context_parallel_ulysses_degree == args.context_parallel_size:
            # just use all2all
            args.cp_comm_type = "a2a"
            print_rank_0(
                "Setting cp_comm_type to a2a because context_parallel_ulysses_degree equals "
                "to context_parallel_size",
                args.rank,
            )
        else:
            cp_degree = (
                args.context_parallel_size // args.context_parallel_ulysses_degree
            )
            args.cp_comm_type = "a2a+p2p"
            args.hierarchical_context_parallel_sizes = [
                args.context_parallel_ulysses_degree,
                cp_degree,
            ]

    # check tp overlap
    if args.tp_comm_overlap:
        if importlib.util.find_spec("torch_xmlir") is None and args.fp16:
            args.tp_comm_overlap = False
            print_rank_0(
                "Disabling tp comm overlap since fp16 is not supported on GPU",
                args.rank,
            )
