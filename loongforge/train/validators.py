# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate arguments"""

import importlib
import os
import warnings

import torch
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.transformer.enums import AttnBackend
from megatron.training.utils import warn_rank_0
from megatron.training.arguments import validate_args
from packaging.version import Version as PkgVersion

from loongforge.tokenizer import get_default_tokenizer
from loongforge.utils import (constants, get_device_arch_version,
                                      is_torch_min_version, print_rank_0, convert_custom_pipeline_to_layout)
from loongforge.utils.utils import get_default_sft_dataset_config, get_transformers_version


def validate_loongforge_extra_args(args, config):
    """ "Validate LoongForge extra arguments"""
    _validate_extra_model_args(args, config)
    _validate_extra_tokenizer_args(args)
    _validate_extra_sft_args(args)
    _validate_extra_training_args(args)
    _validate_extra_multimodal_args(args)
    _validata_extra_custom_args(args)
    _validata_extra_parallel_args(args)

    # megatron one_logger is not supported in loongforge
    args.enable_one_logger = False


def validate_megatron_args(args):
    """Validate Megatron arguments"""
    _validate_legacy_pipeline_args(args)
    validate_args(args)


def validate_custom_model_args(name, args):
    """Validate non foundational model arguments"""
    _validate_custom_model_args(name, args)


# YAML key → args attribute name, for cases where TransformerConfig field
# names diverge from argparse destinations (e.g. fp8_param vs fp8_param_gather).
_YAML_KEY_ALIASES = {
    "fp8_param": "fp8_param_gather",
}


def _validate_extra_model_args(args, config):
    """Setup model config based on the given model name."""
    if config is not None:
        for key in config:
            attr = _YAML_KEY_ALIASES.get(key, key)
            if hasattr(args, attr):
                setattr(args, attr, config[key])
                suffix = f" (from YAML '{key}')" if attr != key else ""
                print_rank_0(f"  {attr} = {config[key]}{suffix}", args.rank)

        print_rank_0(
            "---------------- End of configuration ----------------", args.rank
        )

    # Version guard for Qwen3-VL: transformers must be recent enough.
    if args.model_family == constants.VisionLanguageModelFamilies.QWEN3_VL:
        required = PkgVersion("4.57.1")
        current = get_transformers_version()
        assert current >= required, (
            f"transformers>={required} required for qwen3-vl, found {current}"
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

    if args.padding_side == "left":
        args.padding_side = "right"
        print_rank_0(
            "WARING: Setting args.padding_side to right when run sft.", args.rank
        )


def _validate_extra_training_args(args):
    """Validate training arguments"""

    if args.use_dsa_fused:
        current_variant = getattr(args, "experimental_attention_variant", None)
        if current_variant != "dsa":
            raise ValueError(
                "--use-dsa-fused requires experimental_attention_variant='dsa'. "
                f"Got experimental_attention_variant={current_variant!r}."
            )
        print_rank_0(
            "INFO: --use-dsa-fused is enabled; Omni fused DSA modules will be used.",
            args.rank,
        )
    if getattr(args, "use_dsa_sp_first", False):
        if not args.use_dsa_fused:
            raise ValueError(
                "--use-dsa-sp-first requires --use-dsa-fused to be enabled."
            )
        if not args.sequence_parallel:
            raise ValueError(
                "--use-dsa-sp-first requires --sequence-parallel to be enabled, "
                "since the scheme relies on S/TP partitioning."
            )
        print_rank_0(
            "INFO: --use-dsa-sp-first is enabled; All-to-All communication will be "
            "eliminated in the DSA fused path.",
            args.rank,
        )
    if args.num_experts is None and args.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']:
        args.moe_token_dispatcher_type = 'alltoall'
        warnings.warn(
            f"Since num_experts is {args.num_experts}, moe_token_dispatcher_type argument is not applicable. "
            f"Setting it to 'alltoall' to pass transformer config validation."
        )


def _validate_extra_multimodal_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.VisionLanguageModelFamilies.names():
        return

    args.variable_seq_lengths = True
    if not (args.packing_pretrain_data or args.packing_sft_data):
        args.packing_buffer_size = None


def _validata_extra_custom_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.CustomModelFamilies.names():
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


def _validate_legacy_pipeline_args(args):
    """Validate parallel arguments"""
    # Uneven virtual pipeline parallelism
    assert not (
        args.custom_pipeline_layers is not None
        and args.pipeline_model_parallel_layout is not None
    ), 'custom_pipeline_layers and pipeline_model_parallel_layout cannot be set at the same time'

    # convert custom_pipeline_layers to pipeline_model_parallel_layout
    if args.custom_pipeline_layers is not None:
        assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
            'The layer partition mode conflicts.'

        pp_splits = []
        if args.custom_pipeline_layers.find(',') != -1:
            pp_splits = [int(s) for s in args.custom_pipeline_layers.split(',')]

        assert len(pp_splits) == args.pipeline_model_parallel_size, (
            f"the number of elements in --custom-pipeline-layers must be equal to "
            f"pipeline size {args.pipeline_model_parallel_size}")

        assert args.num_layers == sum(pp_splits), \
            f"the sum of --custom-pipeline-layers must be equal to {args.num_layers}"

        if args.num_virtual_stages_per_pipeline_rank is not None:
            assert all(x >= args.num_virtual_stages_per_pipeline_rank for x in pp_splits), \
                f"when num_virtual_stages_per_pipeline_rank is {args.num_virtual_stages_per_pipeline_rank}, \
                each element in custom_pipeline_layers must be >= num_virtual_stages_per_pipeline_rank"

        args.pipeline_model_parallel_layout = convert_custom_pipeline_to_layout(
            custom_pipeline_layers=args.custom_pipeline_layers,
            num_virtual_stages_per_pipeline_rank=args.num_virtual_stages_per_pipeline_rank,
            mtp_num_layers=args.mtp_num_layers,
        )

        # To pass the megatron validation
        if args.num_virtual_stages_per_pipeline_rank is not None:
            args.num_virtual_stages_per_pipeline_rank = None

    # add loongforge for custom virtual pipeline layers check
    if args.custom_virtual_pipeline_layers is not None:
        assert args.pipeline_model_parallel_size > 1, \
            "custom_virtual_pipeline_layers is only supported when pipeline_model_parallel_size > 1"
        assert args.num_virtual_stages_per_pipeline_rank is not None, \
                "num_virtual_stages_per_pipeline_rank should be set when custom_virtual_pipeline_layers is set"
        assert args.custom_pipeline_layers is None, \
                "custom_pipeline_layers should not be set when custom_virtual_pipeline_layers is set"
        
        custom_vpp_splits = []
        if args.custom_virtual_pipeline_layers.find(',') != -1:
            custom_vpp_splits = [int(s) for s in args.custom_virtual_pipeline_layers.split(',') if s.strip()]
        
        assert sum(custom_vpp_splits) == args.num_layers, (
            f"the sum of --custom-virtual-pipeline-layers must be equal to {args.num_layers}"
        )
        
        assert len(custom_vpp_splits) == args.pipeline_model_parallel_size * \
            args.num_virtual_stages_per_pipeline_rank, (
            f"the number of elements in --custom-virtual-pipeline-layers must be equal to "
            f"pipeline size {args.pipeline_model_parallel_size} * num_virtual_stages_per_pipeline_rank "
            f"{args.num_virtual_stages_per_pipeline_rank}"
        )

        args.pipeline_model_parallel_layout = convert_custom_pipeline_to_layout(
            custom_virtual_pipeline_layers=args.custom_virtual_pipeline_layers,
            num_virtual_stages_per_pipeline_rank=args.num_virtual_stages_per_pipeline_rank,
            mtp_num_layers=args.mtp_num_layers,
        )

        # To pass the megatron validation
        if args.num_virtual_stages_per_pipeline_rank is not None:
            args.num_virtual_stages_per_pipeline_rank = None


def _check_arg_is_not_none(args, arg):
    """Check if an argument is not None."""
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


# Adapted from megatron/training/arguments.py
def _validate_custom_model_args(name, args, defaults={}):
    """Validate non foundational model arguments"""
    if args.custom_pipeline_recompute_layers is not None:
        assert args.recompute_granularity == "full", \
            "recompute-granularity should be full, when custom-pipeline-recompute-layers is set."

        pp_recompute_splits = []
        if args.custom_pipeline_recompute_layers.find(',') != -1:
            pp_recompute_splits = [int(s) for s in args.custom_pipeline_recompute_layers.split(',')]

        assert len(pp_recompute_splits) == args.pipeline_model_parallel_size, (
            f"the number of elements in --custom-pipeline-recompute-layers must be equal to "
            f"pipeline size {args.pipeline_model_parallel_size}")
        
        args.custom_pipeline_recompute_layers = pp_recompute_splits
    # Temporary model parallel size. Added by loongforge
    if args.pipeline_model_parallel_size > 1:
        warnings.warn(f"WARNING: Now for {name}, we only support pipeline model parallel size 1.")
        args.pipeline_model_parallel_size = 1

    if args.sequence_parallel:
        warnings.warn(f"WARNING: Now for {name}, we do not support sequence parallel.")
        args.sequence_parallel = False
    
    if args.expert_model_parallel_size > 1:
        warnings.warn(f"WARNING: Now for {name}, we only support expert model parallel size 1.")
        args.expert_model_parallel_size = 1

    if args.num_virtual_stages_per_pipeline_rank is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support num_virtual_stages_per_pipeline_rank.")
        args.num_virtual_stages_per_pipeline_rank = None

    if args.context_parallel_ulysses_degree is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support context_parallel_ulysses_degree.")
        args.context_parallel_ulysses_degree = 1

    if args.context_parallel_size > 1:
        warnings.warn(f"WARNING: Now for {name}, we only support context parallel size 1.")
        args.context_parallel_size = 1

    if args.tp_comm_overlap:
        warnings.warn(f"WARNING: Now for {name}, we do not support tp_comm_overlap.")
        args.tp_comm_overlap = False

    if getattr(args, 'pipeline_model_parallel_layout', None) is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support pipeline_model_parallel_layout.")
        args.pipeline_model_parallel_layout = None

    if args.num_query_groups is None:
        # To pass transformer config post init check
        warnings.warn(f"WARNING: Now for {name}, the num_query_groups is None, using default value"
                      " tensor_model_parallel_size to pass the transformer config check.")
        args.num_query_groups = args.tensor_model_parallel_size

    if args.num_attention_heads is None:
        # To pass transformer config post init check
        warnings.warn(f"WARNING: Now for {name}, the num_attention_heads is None, using default value"
                      " tensor_model_parallel_size to pass the transformer config check.")
        args.num_attention_heads = args.tensor_model_parallel_size

    if args.moe_router_topk is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support moe_router_topk.")
        args.moe_router_topk = None
    if args.moe_router_group_topk is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support moe_router_group_topk.")
        args.moe_router_group_topk = None
    
    # Temporary
    assert args.non_persistent_ckpt_type in ['global', 'local', None], \
        'Currently only global and local checkpoints are supported'
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                LocalCheckpointManager
        except ModuleNotFoundError as e:
            raise RuntimeError('nvidia_resiliency_ext is required for local checkpointing') from e

    # Set args.use_dist_ckpt from args.ckpt_format.
    if args.use_legacy_models:
        assert args.ckpt_format == "torch", \
            "legacy model format only supports the 'torch' checkpoint format."
    args.use_dist_ckpt = args.ckpt_format != "torch"


    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size

    # Total model size.
    assert args.world_size % total_model_size == 0, (
        f"world size ({args.world_size}) is not divisible by total_model_size ({total_model_size=})"
    )

    if args.attention_backend == AttnBackend.local:
        assert args.spec[0] == 'local' , '--attention-backend local is only supported with --spec local'

    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size

    args.data_parallel_size = args.world_size // total_model_size

    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {}, '
              'hierarchical context-parallel sizes: {}'
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {}, '.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.hierarchical_context_parallel_sizes,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size, flush=True))

    if args.hierarchical_context_parallel_sizes:
        from numpy import prod
        assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
    if "a2a+p2p" in args.cp_comm_type:
        assert args.hierarchical_context_parallel_sizes is not None, \
        "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm"

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    use_data_path = (args.data_path is not None) or (args.data_args_path is not None)
    if use_data_path:
        # Exactly one of the two has to be None if we use it.
        assert (args.data_path is None) or (args.data_args_path is None)
    use_per_split_data_path = any(
        elt is not None
        for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) or \
            args.per_split_data_args_path is not None
    if use_per_split_data_path:
         # Exactly one of the two has to be None if we use it.
        assert any(elt is not None
                   for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) is False \
                or args.per_split_data_args_path is None

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0

    # Uneven virtual pipeline parallelism
    assert args.num_layers_per_virtual_pipeline_stage is None or args.num_virtual_stages_per_pipeline_rank is None, \
        ('--num-layers-per-virtual-pipeline-stage and '
         '--num-virtual-stages-per-pipeline-rank cannot be set at the same time')

    if args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
        if args.overlap_p2p_comm:
            assert args.pipeline_model_parallel_size > 1, \
                'When interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1'
        else:
            assert args.pipeline_model_parallel_size > 2, \
                'When interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch'

        if args.num_virtual_stages_per_pipeline_rank is None:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
                ('please use --num-virtual-stages-per-pipeline-rank to specify virtual pipeline parallel '
                 'degree when enable uneven pipeline parallelism')
            if args.num_layers is not None:
                num_layers = args.num_layers
            else:
                num_layers = args.decoder_num_layers

            if args.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if args.account_for_loss_in_pipeline_split:
                num_layers += 1

            assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'number of layers of the model must be divisible pipeline model parallel size'
            num_layers_per_pipeline_stage = num_layers // args.transformer_pipeline_model_parallel_size

            assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
                'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
            args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                args.num_layers_per_virtual_pipeline_stage
        else:
            args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print('WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                  'since non-interleaved schedule does not support overlapping p2p communication '
                  'and aligned param AG')

        if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None \
            and args.custom_pipeline_layers is None:
            # Divisibility check not applicable for T5 models which specify encoder_num_layers
            # and decoder_num_layers.
            if args.num_layers is not None:
                num_layers = args.num_layers

                if args.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if args.account_for_loss_in_pipeline_split:
                    num_layers += 1

                assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                    'Number of layers should be divisible by the pipeline-model-parallel size'
    if args.rank == 0:
        print(f"Number of virtual stages per pipeline stage: {args.virtual_pipeline_model_parallel_size}")

    if args.data_parallel_sharding_strategy == "optim_grads_params":
        args.overlap_param_gather = True
        args.overlap_grad_reduce = True

    if args.data_parallel_sharding_strategy == "optim_grads":
        args.overlap_grad_reduce = True

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            'Must use --overlap-param-gather with --overlap-grad-reduce'
        assert not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models'

    if args.use_torch_fsdp2:
        assert is_torch_min_version("2.4.0"), \
            'FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.'
        assert args.pipeline_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with pipeline parallelism'
        assert args.expert_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with expert parallelism'
        assert not args.use_distributed_optimizer, \
            "--use-torch-fsdp2 is not supported with MCore's distributed optimizer"
        assert not args.gradient_accumulation_fusion, \
            '--use-torch-fsdp2 is not supported with gradient accumulation fusion'
        assert args.ckpt_format == 'torch_dist', \
            '--use-torch-fsdp2 requires --ckpt-format torch_dist'
        assert args.untie_embeddings_and_output_weights, \
            '--use-torch-fsdp2 requires --untie-embeddings-and-output-weights'
        assert not args.fp16, \
            '--use-torch-fsdp2 not supported with fp16 yet'
        assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one'

    if args.overlap_param_gather_with_optimizer_step:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather-with-optimizer-step only supported with distributed optimizer'
        assert args.overlap_param_gather, \
            'Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather'
        assert args.virtual_pipeline_model_parallel_size is not None, \
            '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism'
        assert not args.use_dist_ckpt, \
            '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet'

    dtype_map = {
        'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8': torch.uint8,
    }
    map_dtype = lambda d: d if isinstance(d, torch.dtype) else dtype_map[d]

    args.main_grads_dtype = map_dtype(args.main_grads_dtype)
    args.main_params_dtype = map_dtype(args.main_params_dtype)
    args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
    args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)

    if args.fp8_param_gather:
        assert args.use_distributed_optimizer or args.use_torch_fsdp2, \
            '--fp8-param-gather only supported with distributed optimizer or torch fsdp2'

    if args.use_megatron_fsdp:
        # NOTE: The flag `use_custom_fsdp` is deprecated and will be removed in future versions.
        #       Please use `use_megatron_fsdp` instead, as all functionality will be migrated there.
        #       Future updates will drop support for `use_custom_fsdp` to avoid confusion.
        args.use_custom_fsdp = True

        if args.data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"]:
            warn_rank_0(
                'Please make sure your TransformerEngine support FSDP + gradient accumulation fusion',
                args.rank,
            )

        if args.data_parallel_sharding_strategy == "optim_grads_params":
            assert args.check_weight_hash_across_dp_replicas_interval is None, \
                'check_weight_hash_across_dp_replicas_interval is not supported with optim_grads_params'

        assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one'

        assert args.ckpt_format == "fsdp_dtensor", \
            "Megatron FSDP only supports fsdp_dtensor checkpoint format"

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if args.accumulate_allreduce_grads_in_fp32:
            assert args.main_grads_dtype == torch.float32, \
                "--main-grads-dtype can only be fp32 when --accumulate-allreduce-grads-in-fp32 is set"

        if args.grad_reduce_in_bf16:
            args.accumulate_allreduce_grads_in_fp32 = False
        elif not args.accumulate_allreduce_grads_in_fp32 and args.main_grads_dtype == torch.float32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    assert args.num_dataset_builder_threads > 0

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.

    # Note: default false, but for sft, we should support variable sequence lengths
    # args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is None:
        warnings.warn("WARNING: For some components, like image projector, num_layers is None, using default value 1" \
        "to pass the validation check.")
        args.num_layers = 1

    if args.hidden_size is None:
        warnings.warn("WARNING: For some components, like image projector, hidden_size is None, using default value "
                      "num_attention_heads to pass the validation check.")
        args.hidden_size = args.num_attention_heads

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None and args.num_attention_heads is not None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None and args.context_parallel_size > 1:
        assert args.seq_length % (args.context_parallel_size * 2) == 0, \
            'seq-length should be a multiple of 2 * context-parallel-size ' \
            'if context-parallel-size > 1.'

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length, \
            f"max_position_embeddings ({args.max_position_embeddings}) must be greater than " \
            f"or equal to seq_length ({args.seq_length})."
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    # Persistent fused layer norm.
    if not is_torch_min_version("1.11.0a0"):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # MoE overlap handling: non-foundation components (e.g., VIT) do not participate
    # in MoE all2all overlap scheduling, so disable overlap flags for them.
    # VIT recompute is configured independently of foundation's MoE a2a settings.
    # To customize VIT recompute, use YAML override:
    #   +model.image_encoder.recompute_granularity=selective
    #   +model.image_encoder.recompute_modules=[core_attn,layernorm]
    orig_overlap_moe = getattr(args, 'overlap_moe_expert_parallel_comm', False)
    if orig_overlap_moe:
        warnings.warn(f"Warning: Now for {name}, we do not support overlap_moe_expert_parallel_comm and "
                      "delay_wgrad_compute.")
        args.overlap_moe_expert_parallel_comm = False
        args.delay_wgrad_compute = False

    # Filter out a2a_overlap modules from recompute_modules for non-foundation components,
    # since they don't participate in MoE all2all overlap scheduling.
    if args.recompute_modules:
        non_a2a_modules = [m for m in args.recompute_modules if not m.startswith('a2a')]
        if len(non_a2a_modules) < len(args.recompute_modules):
            warnings.warn(f"WARNING: Now for {name} model, a2a_overlap modules are not supported, "
                          "ignoring them.")
        args.recompute_modules = non_a2a_modules

    # When foundation uses selective recompute for MoE a2a overlap,
    # VIT inherits 'selective' but doesn't participate in MoE overlap.
    # Restore VIT to 'full' recompute by default — avoid requiring users to
    # mix a2a and non-a2a modules in CLI (which would pollute LLM's recompute config).
    # Users can independently configure VIT recompute via YAML override:
    #   +model.image_encoder.recompute_granularity=selective
    #   +model.image_encoder.recompute_modules=[core_attn,layernorm]
    if orig_overlap_moe and args.recompute_granularity == 'selective':
        warnings.warn(
            f"INFO: {name} does not participate in MoE all2all overlap scheduling. "
            f"Restoring recompute_granularity to 'full' for {name} to enable "
            f"full activation checkpointing. To customize {name} recompute independently, "
            f"use YAML override: +model.{name}.recompute_granularity=<granularity> "
            f"+model.{name}.recompute_modules=[<modules>]"
        )
        args.recompute_granularity = 'full'
        if args.recompute_method is None:
            args.recompute_method = 'uniform'
        if args.recompute_num_layers is None:
            args.recompute_num_layers = 1

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    if args.fine_grained_activation_offloading:
        warnings.warn(f"WARNING: Now for {name} model, fine_grained_activation_offloading is not supported.")
        args.fine_grained_activation_offloading = False

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        if args.sequence_parallel:
            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
        args.sequence_parallel = False

    if args.tp_comm_overlap:
        assert args.sequence_parallel, \
            'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.tensor_model_parallel_size > 1 or args.context_parallel_size > 1 and get_device_arch_version() < 10:
        # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
        if args.use_torch_fsdp2 or getattr(args, "use_custom_fsdp", False):
            fsdp_impl = "Torch-FSDP2" if args.use_torch_fsdp2 else "Custom-FSDP"
            warnings.warn(
                f"Using tensor model parallelism or context parallelism with {fsdp_impl} together. "
                "Try not to using them together since they require different CUDA_MAX_CONNECTIONS "
                "settings for best performance. sequence parallelism requires setting the "
                f"environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 while {fsdp_impl} "
                "requires not setting CUDA_DEVICE_MAX_CONNECTIONS=1 for better parallelization.")
    if args.preprocess_data_on_cpu is True:
        print("Skipping CUDA_DEVICE_MAX_CONNECTIONS checks because use megatron preprocess data")
    else:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1" and get_device_arch_version() < 10:
            # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
            if args.sequence_parallel:
                warnings.warn(
		    "Using sequence parallelism requires setting the environment variable "
                    "CUDA_DEVICE_MAX_CONNECTIONS to 1")
            if args.async_tensor_model_parallel_allreduce:
                warnings.warn(
                    "Using async gradient all reduce requires setting the environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Keep the 'add bias' args in sync; add_qkv_bias is more targeted.
    if args.add_bias_linear:
        args.add_qkv_bias = True

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')
    if args.position_embedding_type != 'rope':
        args.apply_rope_fusion = False

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type not in ['rope', 'alibi']:
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Relative position embeddings arguments
    if args.position_embedding_type == 'relative':
        assert args.transformer_impl == "transformer_engine", \
            'Local transformer implementation currently does not support attention bias-based position embeddings.'

    # MoE Spec check
    if args.num_experts is not None:
        warnings.warn("Warning: For those non foundation model, num_experts must be None.")
        args.num_experts = None

    # Context parallel
    if args.context_parallel_size > 1:
        assert not args.use_legacy_models, "Context parallelism is not supported in legacy models."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')

    # add loongforge for custom pipeline layers check
    if args.custom_pipeline_layers is not None:
        warnings.warn("Warning: For those non foundation model, custom_pipeline_layers must be None.")
        args.custom_pipeline_layers = None

    # # add loongforge for custom virtual layers in first pipeline stage check
    # if hasattr(args, 'custom_virtual_layers_first_pipeline') and \
    #         args.custom_virtual_layers_first_pipeline is not None:
    #     warnings.warn("Warning: For those non foundation model, custom_virtual_layers_first_pipeline must be None.")
    #     args.custom_virtual_layers_first_pipeline = None

    # add loongforge for custom virtual pipeline layers check
    if hasattr(args, 'custom_virtual_pipeline_layers') and args.custom_virtual_pipeline_layers is not None:
        warnings.warn("Warning: For those non foundation model, custom_virtual_pipeline_layers must be None.")
        args.custom_virtual_pipeline_layers = None

    # if args.custom_pipeline_recompute_layers is not None:
    #     warnings.warn("Warning: For those non foundation model, custom_pipeline_recompute_layers must be None.")
    #     args.custom_pipeline_recompute_layers = None

    # Data blend checks
    assert args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) <= 1, \
               "A single data source must be provided in training mode, else None"

    # Deterministic mode
    if args.deterministic_mode:
        assert not args.use_flash_attn, "Flash attention can not be used in deterministic mode."
        assert not args.cross_entropy_loss_fusion, "Cross Entropy Fusion is currently not deterministic."

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        assert os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}."

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    if args.result_rejected_tracker_filename is not None:
        # Append to passed-in args.iterations_to_skip.
        iterations_to_skip_from_file = RerunStateMachine.get_skipped_iterations_from_tracker_file(
            args.result_rejected_tracker_filename
        )
        args.iterations_to_skip.extend(iterations_to_skip_from_file)

    # Make sure all functionality that requires Gloo process groups is disabled.
    if not args.enable_gloo_process_groups:
        if args.use_distributed_optimizer:
            # If using distributed optimizer, must use distributed checkpointing.
            # Legacy checkpointing uses Gloo process groups to collect full distributed
            # optimizer state in the CPU memory of DP rank 0.
            assert args.use_dist_ckpt

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    if (
        args.use_dist_ckpt
        and not args.ckpt_fully_parallel_save
        and args.use_distributed_optimizer
        and args.rank == 0
    ):
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')

    # Inference args
    if args.inference_batch_times_seqlen_threshold > -1:
        assert args.pipeline_model_parallel_size > 1, \
            "--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1."

    # Optimizer CPU offload check
    if args.optimizer_cpu_offload:
        assert args.use_precision_aware_optimizer, (
            "The optimizer cpu offload must be used in conjunction with `--use-precision-aware-optimizer`, "
            "as the hybrid device optimizer reuses the code path of this flag."
        )

    if args.non_persistent_ckpt_type == "local":
        assert args.non_persistent_local_ckpt_dir is not None, \
            "Tried to use local checkpointing without specifying --local-ckpt-dir!"
    if args.replication:
        assert args.replication_jump is not None, "--replication requires the value of --replication-jump!"
        assert args.non_persistent_ckpt_type == "local", \
            f"--replication requires args.non_persistent_ckpt_type == 'local', but got: {args.non_persistent_ckpt_type}"
    elif args.replication_jump:
        print("Warning: --replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None
