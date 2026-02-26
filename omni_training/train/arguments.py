"""AIAK arguments"""

import argparse

from omni_training.data import get_support_templates
from omni_training.models import get_support_model_archs
from omni_training.utils import constants


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
    # add args for rice vl 
    parser = _add_extra_training_rice_vl_args(parser)

    return parser


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

def _add_extra_training_rice_vl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Create a dedicated group for Rice-VL arguments for better organization in the help message."""
    group = parser.add_argument_group(
        title='Training Rice-VL',
        description='Arguments specific to the Rice-VL model training configuration.'
    )

    group.add_argument(
        '--training-rice-vl-max-answer-length',
        type=int,
        default=4096,
        help=(
            "The maximum number of characters allowed in an answer during training. "
            "Answers longer than this will be truncated."
        )
    )
    return parser

def _add_extra_model_args(parser: argparse.ArgumentParser):
    """Add model arguments"""
    group = parser.add_argument_group(title="extra-model")

    # only need to pass one argument
    parser.add_argument(
        "--config-file", 
        type=str, 
        required=False,
        help="Path to YAML config file. Example: /path/to/.../config_name.yaml")

    group.add_argument(
        "--model-name",
        type=str,
        default=None,
        required=False,
        help="The model name to use. This name should match the key in the model config registry.",
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

    # use for freeze parameters
    group.add_argument('--freeze-parameters',
                       type=str,
                       nargs="*",
                       default=[],
                       help='Prefixes of parameters to freeze (default: [])'
    )

    group.add_argument('--freeze-parameters-regex',
                       type=str,
                       default=None,
                       help='Regular expression pattern to match parameters to freeze (default: None)'
    )

    # adapter checkpoint control
    group.add_argument(
        "--allow-missing-adapter-checkpoint",
        action="store_true",
        help="Allow missing adapter checkpoint during model loading. Default is False (not allowed)."
    )

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

    group.add_argument(
        "--task-encoder",
        type=str,
        default=None,
        help="Task encoder class name for multimodal data pipeline "
        "(e.g., VLMTaskEncoder, InternVLTaskEncoder).",
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
        help="A yaml file that contains the dataset configuration."
        "default: configs/data/sft_dataset_config.yaml",
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

    group.add_argument('--dataset-metadata-path', type=str, help='dataset metadata path')

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
        choices=["torch", "torch_dist", "zarr", "fsdp_dtensor"],
        help="Checkpoint format to use. Default: torch",
    )
    
    group.add_argument(
        "--log-memory-stats",
        action="store_true",
        default=False,
        help="Log memory stats (allocated/peak) in training log output. Default: False.",
    )

    group.add_argument(
        "--legacy-reporting-loss-reduction",
        action="store_true",
        help="Use legacy reporting loss reduction method. Default is False.",
    )
    group.add_argument(
        "--force-all-weight-decay",
        action="store_false",
        default=None,
        help=(
            "Override whether to force every parameter into the weight-decay group. "
            "Not set keeps legacy behavior (force all). "
        ),
    )
    return parser


def _add_extra_multimodal_args(parser):
    """Add multimodal arguments"""

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

    group.add_argument('--energon-pack-algo', type=str, default="balanced",
        choices=["balanced", "sequential", "sequential_max_images"],
        help="Energon sample packing algorithm (default: balanced). Options: "
        "1) 'balanced': Greedy knapsack approach that sorts samples by length and"
            "packs them while balancing computational load across GPUs."
        "2) 'sequential': Fills buffers in order, minimizing training sequence disruption."
        "3) 'sequential_max_images': Like sequential but prioritizes maximizing images"
            "per buffer (may reorder samples).")
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
    # NOTE: --custom-pipeline-layers and --custom-virtual-pipeline-layers will be deprecated in the future.
    group.add_argument('--custom-pipeline-layers', type=str, default=None,
                       help='DEPRECATED. Use --pipeline-model-parallel-layout.'
                       'Add by aiak for pp layer imbalance.'
                       'For example 19,20,20,21. 19 for stage0 layers, 20 for stage1 layers...')
    group.add_argument('--custom-virtual-pipeline-layers', type=str, default=None,
                       help='DEPRECATED. Use --pipeline-model-parallel-layout.'
                       'Add by aiak for virtual pipeline layer imbalance.'
                       'For example 19,20,20,21. If we have two virtual chunks in one pp stage, '
                       '19 for stage0 virtual chunk0 layers, 20 for stage1 virtual chunk0 layers...')
    group.add_argument('--enable-encoder-hetero-dp', default=False, action="store_true")
    return parser