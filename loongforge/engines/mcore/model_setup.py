# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import re
import logging
import dataclasses

import torch
from collections import OrderedDict
from dataclasses import asdict

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.core.transformer.module import Float16Module
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.distributed import (
    DistributedDataParallel as DDP,
)
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as megatron_FSDP,
)
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.transformer.moe import upcycling_utils
from megatron.training import (
    get_timers,
    print_rank_0,
)
from megatron.training.utils import (
    unwrap_model,
    update_use_dist_ckpt,
    to_empty_if_meta_device,
)
from megatron.training.training import (
    get_optimizer_param_scheduler,
    preprocess_common_state_dict,
    print_module_param_dtypes,
)

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from loongforge.engines.mcore import get_args, global_vars
from loongforge.engines.mcore.checkpointing import (
    load_checkpoint,
    _load_checkpoint_from_path,
    save_checkpoint,
    checkpoint_exists,
)
from loongforge.models.common.peft.lora import LoRA, VLMLoRA
from loongforge.models.common.peft.utils import apply_peft_transformation


def is_hf_checkpoint(load_path):
    """Check if the checkpoint is in HuggingFace format."""
    if load_path is None:
        return False
    safe_index_path = os.path.join(load_path, "model.safetensors.index.json")
    safe_path = os.path.join(load_path, "model.safetensors")
    bin_index_path = os.path.join(load_path, "pytorch_model.bin.index.json")
    bin_path = os.path.join(load_path, "pytorch_model.bin")
    return os.path.exists(safe_index_path) or os.path.exists(safe_path) or \
            os.path.exists(bin_index_path) or os.path.exists(bin_path)


def _resolve_convert_pp_layout(model_config):
    """Resolve ``pipeline_model_parallel_layout`` for the Bridge converter.

    For pure-LLM configs the layout lives on ``model_config`` itself. For VLM
    (OmniCombinationModel / VLMModelConfig) configs it lives on the foundation
    sub-config, because VLMModelConfig does not carry the field at the top
    level. Returning None here makes the converter fall back to a balanced VPP
    partition, which disagrees with the model's custom layout and breaks
    ``load_state_dict(strict=True)`` during bridge online loading.
    """
    layout = getattr(model_config, 'pipeline_model_parallel_layout', None)
    if layout is None:
        foundation = getattr(model_config, 'foundation', None)
        if foundation is not None:
            layout = getattr(foundation, 'pipeline_model_parallel_layout', None)
    if layout is None:
        return None
    # Parser._build_param_dict requires convert_pp_layout.layout to be populated.
    if getattr(layout, 'layout', None) is None:
        return None
    return layout
# PLACEHOLDER_MODEL_SETUP_CONTINUE


@torch.no_grad()
def update_ema(ema_model, model, rate=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(rate).add_(param.data, alpha=1 - rate)


def freeze_parameters(model, freeze_parameters, freeze_parameters_regex):
    """Freezes model parameters based on exact name matches or regex patterns."""
    for model_module in model:
        if freeze_parameters:
            logging.info(f"freeze_parameters: {freeze_parameters}")
            for n, p in model_module.named_parameters():
                for freeze_p in freeze_parameters:
                    if n.startswith(freeze_p):
                        p.requires_grad = False

        if freeze_parameters_regex is not None:
            logging.info(f"freeze_parameters_regex: {freeze_parameters_regex}")
            try:
                pattern = re.compile(freeze_parameters_regex)
            except re.error as e:
                logging.info(
                    f"Invalid freeze_parameters_regex '{freeze_parameters_regex}': {e}"
                )
                raise

            for n, p in model_module.named_parameters():
                if pattern.search(n):
                    p.requires_grad = False

    # Only log checking info if freezing enable
    if freeze_parameters or freeze_parameters_regex:
        frozen_params = sorted(
            f"FROZEN: {n}"
            for m in model
            for n, p in m.named_parameters()
            if not p.requires_grad
        )
        trainable_params = sorted(
            f"TRAINABLE: {n}"
            for m in model
            for n, p in m.named_parameters()
            if p.requires_grad
        )
        logging.info(
            "<Freezing model parameters> \n"
            + "\n".join(frozen_params)
            + "\n</Freezing model parameters>"
            + "\n\n"
            + "<Trainable model parameters> \n"
            + "\n".join(trainable_params)
            + "\n</Trainable model parameters>"
        )
# PLACEHOLDER_MODEL_SETUP_CONTINUE2


def check_vlm_peft_config(model_config):
    """Check whether the VLM PEFT configuration is compatible with the current model architecture."""
    if not hasattr(model_config, 'peft_config') or model_config.peft_config is None:
        return
    peft_config = model_config.peft_config
    if (
        model_config.image_encoder is not None
        and model_config.image_encoder.freeze
        and peft_config.apply_to_image_encoder
    ):
        raise ValueError("Cannot freeze image encoder when using PEFT.")
    if (
        model_config.image_projector is not None
        and model_config.image_projector.freeze
        and peft_config.apply_to_image_projector
    ):
        raise ValueError("Cannot freeze image projector when using PEFT.")
    if (
        model_config.foundation is not None
        and model_config.foundation.freeze
        and peft_config.apply_to_foundation
    ):
        raise ValueError("Cannot freeze foundation model when using PEFT.")
    if (
        model_config.video_encoder is not None
        and model_config.video_encoder.freeze
        and peft_config.apply_to_video_encoder
    ):
        raise ValueError("Cannot freeze video encoder when using PEFT.")
    if (
        model_config.video_projector is not None
        and model_config.video_projector.freeze
        and peft_config.apply_to_video_projector
    ):
        raise ValueError("Cannot freeze video projector when using PEFT.")
    target_prefix = []
    if peft_config.apply_to_foundation:
        target_prefix.append("foundation")
    if peft_config.apply_to_image_encoder:
        target_prefix.append("image_encoder")
    if peft_config.apply_to_image_projector:
        target_prefix.append("image_projector")
    if peft_config.apply_to_video_encoder:
        target_prefix.append("video_encoder")
    if peft_config.apply_to_video_projector:
        target_prefix.append("video_projector")
    if len(target_prefix) == 1:
        target_prefix = f"*{target_prefix[0]}*"
    else:
        combined = "|".join(target_prefix)
        target_prefix = f"*({combined})*"
    target_modules = [x for x in peft_config.target_modules]
    for i in range(len(target_modules)):
        target = target_modules[i]
        if "*" not in target:
            target_modules[i] = target_prefix + target
    peft_config.target_modules = target_modules
    return peft_config
# PLACEHOLDER_MODEL_SETUP_CONTINUE3


def add_hooks(model, args, prefix):
    """add hooks for model, only enable hooks when open the switch: enable-log-tensor"""
    from inspector.hooks import register_hooks

    print_rank_0(
        f"Set up Log Tensor Hook:\n" f"  name pattern: {args.log_tensor_name_pattern}\n"
    )
    rank = torch.distributed.get_rank()
    log_fn = lambda string: print(f"[Rank {rank}] {string}")
    matched_modules = register_hooks(model, args, rank, log_fn, prefix)
    if len(matched_modules) > 0:
        print_rank_0(
            f"For log tensor name pattern: {args.log_tensor_name_pattern}, find the following layers:"
        )
        for l in matched_modules:
            print_rank_0(f"  {l}")
    else:
        print_rank_0(
            f"No layers found for the log tensor name pattern: {args.log_tensor_name_pattern}"
        )


def _p2p_embedding_weights_for_mtp(unwrapped_model, args):
    """Copy embedding.word_embeddings.weight from the first PP stage to the last PP stage via P2P.

    When PP >= 2 and MTP is enabled, the last PP stage creates its own embedding
    (for multi-token prediction) but its weights are not loaded from the checkpoint.
    This function uses point-to-point send/recv to copy embedding weights directly.
    """
    pp_world_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_world_size < 2 or not getattr(args, 'mtp_num_layers', 0):
        return

    first_rank = mpu.get_pipeline_model_parallel_first_rank()
    last_rank = mpu.get_pipeline_model_parallel_last_rank()

    # unwrapped_model is a list (one per virtual PP chunk); take the first element
    model = unwrapped_model[0]

    if mpu.is_pipeline_first_stage():
        embedding_weight = model.embedding.word_embeddings.weight.data
        torch.distributed.send(embedding_weight, dst=last_rank, group=mpu.get_pipeline_model_parallel_group())
        print(f"[MTP] Sent embedding.word_embeddings.weight "
              f"to last PP stage (rank {last_rank}), shape={embedding_weight.shape}")

    elif mpu.is_pipeline_last_stage():
        embedding_weight = model.embedding.word_embeddings.weight.data
        torch.distributed.recv(embedding_weight, src=first_rank, group=mpu.get_pipeline_model_parallel_group())
        print(f"[MTP] Received embedding.word_embeddings.weight "
              f"from first PP stage (rank {first_rank}), shape={embedding_weight.shape}")


# Add project root to Python path for tools imports
import sys
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_tools_path = os.path.join(_project_root, "tools")
if _tools_path not in sys.path:
    sys.path.insert(0, _tools_path)

from mcore_bridge.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    model_config=None,
):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process, vp_stage=i
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
            model.model_type = model_type
        return model

    if args.init_model_with_meta_device:
        with torch.device('meta'):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]
# PLACEHOLDER_GET_MODEL_CONTINUE

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    peft_class = model_config.peft_config if hasattr(model_config, 'peft_config') else None

    def peft_pre_wrap_hook(args, model, peft_class):
        """Pre-wrap hook that handles PEFT transformation.

        Args:
            model: List of base model modules before distributed wrapping

        Returns:
            List of potentially PEFT-transformed model modules
        """
        # Pre hook for peft
        if peft_class is None:
            return model
        print_rank_0("Applying PEFT pre-wrap hook...")

        # Load pretrained checkpoint if available
        # Support both HF format and mcore format
        if args.pretrained_checkpoint is None or (
            not checkpoint_exists(args.pretrained_checkpoint)
            and not is_hf_checkpoint(args.pretrained_checkpoint)
        ):
            raise ValueError(
                f"Invalid pretrained checkpoint directory found: {args.pretrained_checkpoint}"
            )

        # Explicitly set finetune to avoid loading optimizer and RNG states
        args.finetune = True

        # Check if it's HF format
        if is_hf_checkpoint(args.pretrained_checkpoint):
            # HF checkpoint: use online loading
            print_rank_0(f"Loading base model weights from HF chekckpoint: {args.pretrained_checkpoint}")

            from tools.mcore_bridge.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online

            # Temporarily set args.load for load_hf_checkpoint_online
            orig_load = args.load
            args.load = args.pretrained_checkpoint
            _pp_layout = _resolve_convert_pp_layout(model_config)
            if _pp_layout is not None:
                args.convert_pp_layout = _pp_layout
# PLACEHOLDER_GET_MODEL_PEFT

            # Load HF checkpoint online
            iteration, num_fp_ops = load_hf_checkpoint_online(
                model,
                None,  # optimizer
                None,  # opt_param_scheduler
                args
            )
            print_rank_0(f"HF checkpoint loaded successfully, iteration={iteration}")

            # Restore original args.load
            args.load = orig_load
        else:
            # Mcore checkpoint: use standard loading
            print_rank_0(f"Loading base model weights from: {args.pretrained_checkpoint}")

            # Directly call load_checkpoint_from path in order to avoid
            # the load directory overriding the pretrained checkpoint path
            # This is needed to initialize the base model weights first,
            # and then conditionally load adapter states after
            _load_checkpoint_from_path(
                load_dir=args.pretrained_checkpoint,
                args=args,
                load_arg='load',
                ddp_model=model,
                optimizer=None,  # Don't load optimizer - will be created after PEFT
                opt_param_scheduler=None,  # Don't load scheduler - will be created after PEFT
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
                ignore_ckpt_step=True,  # ckpt_step applies only to adapter checkpoints, not pretrained base model
            )

        if "VLM" in type(model_config.peft_config).__name__:
            peft_config = check_vlm_peft_config(model_config)
            peft_class = VLMLoRA(**asdict(peft_config))
        else:
            peft_class = LoRA(**asdict(model_config.peft_config))
        transformed_model = apply_peft_transformation(peft_class, model)
        return transformed_model, peft_class

    if peft_class is not None:
        print_rank_0("Applying PEFT pre-wrap hook...")
        # Use pre wrap hook to handle PEFT transformation
        model, peft_class = peft_pre_wrap_hook(args, model, peft_class)

    # Set tensor model parallel attributes if not set
    # In case pre_wrap_hook augmented the model (e.g. adding PEFT adapters)
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
# PLACEHOLDER_GET_MODEL_PARAMS
    # Print number of parameters.
    num_parameters = sum(
        [sum([p.nelement() for p in model_module.parameters()]) for model_module in model]
    )
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        print(
            ' > number of parameters on (tensor, pipeline) '
            'model parallel rank ({}, {}): {}'.format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            ),
            flush=True,
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not (args.use_torch_fsdp2 and args.use_cpu_initialization)
        and not args.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        param_pattern = args.use_fp32_dtype_for_param_pattern
        if param_pattern and not isinstance(param_pattern, list):
            param_pattern = [param_pattern]

        config = get_model_config(model[0])

        model = [Float16Module(config, model_module) for model_module in model]
        fp32_training_weights = param_pattern
        #covert fp32
        if fp32_training_weights:
            for module in zip(model):
                if not isinstance(module, list):
                    module = module[0]
                for name, buf in module.module.named_parameters():
                    if any(fp32_weight in name for fp32_weight in fp32_training_weights):
                        buf.data = buf.data.to(dtype=torch.float32)
                        print(f'check update param precison {name}')

                for name, buf in module.module.named_buffers():
                    if any(fp32_weight in name for fp32_weight in fp32_training_weights):
                        buf.data = buf.data.to(dtype=torch.float32)
                        print(f'check update buffer precison {name}')
# PLACEHOLDER_GET_MODEL_DDP

        if param_pattern:
            print_rank_0("> model param_dtypes:")
            print_module_param_dtypes(model[0])

    # Materialize tensors on meta device (GPU allocation) if not using FSDP2 and not using Megatron FSDP.
    if args.init_model_with_meta_device and not args.use_torch_fsdp2 and not args.use_megatron_fsdp:
        # for model_module in model:
        model = [
            to_empty_if_meta_device(model_module, device=torch.device("cuda"))
            for model_module in model
        ]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif args.use_megatron_fsdp:
            DP = megatron_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        if getattr(args, "use_torch_fsdp2", False):
            reshard_after_forward = getattr(args, "torch_fsdp2_reshard_after_forward", True)
            ddp_config = TorchFullyShardedDataParallelConfig(
                reshard_after_forward=reshard_after_forward
            )
        else:
            kwargs = {}
            for f in dataclasses.fields(DistributedDataParallelConfig):
                if hasattr(args, f.name):
                    kwargs[f.name] = getattr(args, f.name)
            kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
            kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
            kwargs['check_for_large_grads'] = args.check_for_large_grads
            if args.ddp_num_buckets is not None:
                assert (
                    args.ddp_bucket_size is None
                ), "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
                assert args.ddp_num_buckets > 0, "--ddp-num-buckets must be greater than 0"
                kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
            else:
                kwargs['bucket_size'] = args.ddp_bucket_size
# PLACEHOLDER_GET_MODEL_DDP2
            kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
            kwargs['average_in_collective'] = args.ddp_average_in_collective
            if args.use_megatron_fsdp and args.use_precision_aware_optimizer:
                kwargs["preserve_fp32_weights"] = False

            kwargs["force_turn_on_bucketing"] = args.force_turn_on_bucketing
            ddp_config = DistributedDataParallelConfig(**kwargs)

            # In the Megatron FSDP and DDP use path, we need to initialize the bucket size.
            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        with torch.cuda.stream(torch.cuda.Stream()):
            model = [
                DP(
                    config=config,
                    ddp_config=ddp_config,
                    module=model_chunk,
                    # Turn off bucketing for model_chunk 2 onwards, since communication for these
                    # model chunks is overlapped with compute anyway.
                    disable_bucketing=(model_chunk_idx > 0)
                    or args.overlap_param_gather_with_optimizer_step,
                )
                for (model_chunk_idx, model_chunk) in enumerate(model)
            ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model, peft_class
# PLACEHOLDER_SETUP_MODEL


def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    no_wd_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
    checkpointing_context=None,
):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    model_config = global_vars.get_model_config()

    def provider_with_freeze(*p_args, **p_kwargs):
        m = model_provider_func(*p_args, **p_kwargs)

        # m can be a Module or list/tuple of Modules depending on PP/VPP.
        mods = m if isinstance(m, (list, tuple)) else [m]
        freeze_parameters(mods, args.freeze_parameters, args.freeze_parameters_regex)
        return m

    model, peft_class = get_model(
        provider_with_freeze, model_type, model_config=model_config
    )
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers

    # If the user is asking for a non-zero embedding init std, skip weight decay for embeddings
    # to avoid embeddings from shrinking to zero as recommended in https://arxiv.org/abs/2312.16903
    # default_skip_embedding_weight_decay=args.embedding_init_method_std is not None,

    # Control whether to force every parameter into the weight-decay group.
    # Legacy default (flag unset) keeps the old behavior: force everything.
    # When --force-all-weight-decay true/false is provided, we respect that choice.
    if getattr(args, "force_all_weight_decay", None):
        no_wd_decay_cond = (False,)

    optimizer = get_megatron_optimizer(
        config,
        model,
        no_wd_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_gloo_process_groups=args.enable_gloo_process_groups,
        # If the user is asking for a non-zero embedding init std, skip weight decay for embeddings
        #  to avoid embeddings from shrinking to zero as recommended in https://arxiv.org/abs/2312.16903
        default_skip_embedding_weight_decay=args.embedding_init_method_std is not None,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
# PLACEHOLDER_SETUP_UPCYCLE

    # moe upcycling
    if args.moe_use_upcycling:
        torch.distributed.barrier()
        assert not checkpoint_exists(args.save), (
            "The upcycling destination directory already exists. "
            "Please check if --moe-use-upcycling is mistakenly enabled. "
            "Upcycling should only be set for the first run when converting the dense model. "
            "All subsequent runs should remove this flag. "
        )
        # before changing moe related global args, save them in local variables
        num_experts = args.num_experts
        expert_model_parallel_size = args.expert_model_parallel_size
        moe_ffn_hidden_size = args.ffn_hidden_size

        # set dense model related args in to global args before getting dense model
        args.num_experts = None
        args.expert_model_parallel_size = 1
        args.ffn_hidden_size = moe_ffn_hidden_size * args.moe_upcycling_granularity

        # get dense model
        dense_model_for_upcycling = get_model(model_provider_func, model_type)

        # recover moe upcycling related args in global args before executing upcycling
        args.num_experts = num_experts
        args.expert_model_parallel_size = expert_model_parallel_size
        args.ffn_hidden_size = moe_ffn_hidden_size

        # execute upcycling
        _, args.num_floating_point_operations_so_far = (
            upcycling_utils.load_and_upcycle_model(
                load_checkpoint,
                unwrapped_model,
                dense_model_for_upcycling,
                load_kwargs={
                    "model": dense_model_for_upcycling,
                    "optimizer": None,
                    "opt_param_scheduler": None,
                },
            )
        )
        args.iteration = 1
        save_checkpoint(
            args.iteration, model, None, None, args.num_floating_point_operations_so_far
        )
        torch.distributed.barrier()
        del dense_model_for_upcycling
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        print_rank_0(f"Upcycled checkpoint saved to {args.save}")
# PLACEHOLDER_SETUP_LOAD

    if hasattr(model_config, "peft_config") and model_config.peft_config is not None:
        # For LoRA training, must have base model checkpoint (mcore or HF format)
        has_base_ckpt = args.pretrained_checkpoint is not None and (
            checkpoint_exists(args.pretrained_checkpoint) or is_hf_checkpoint(args.pretrained_checkpoint)
        )
        assert has_base_ckpt, (
            "Use LoRA must setup base-model pretrain checkpoint (mcore or HF format). "
            f"args.pretrained_checkpoint={args.pretrained_checkpoint}"
        )

    # For PEFT, the pretrained checkpoint is loaded in get_model()
    if peft_class is not None:
        should_load_checkpoint = args.load is not None and checkpoint_exists(args.load)
        if should_load_checkpoint:
            # The finetune toggle is explicitly set to True in order to avoid loading optimizer and RNG states
            # This is switched off here in order to load these states from the checkpoint
            args.finetune = False
    else:
        should_load_checkpoint = (
            args.load is not None and checkpoint_exists(args.load)
        ) or (
            args.pretrained_checkpoint is not None
            and checkpoint_exists(args.pretrained_checkpoint)
        )

    if should_load_checkpoint and not args.moe_use_upcycling:
        timers("load-checkpoint", log_level=0).start(barrier=True)
        # Offline checkpoint loading (pre-converted sharded checkpoint)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model,
            optimizer,
            opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and getattr(args, "use_torch_fsdp2", False)
            and args.ckpt_format == "torch_dist",
            peft_class=peft_class,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

        #For models such as GLM-5, the model structure is similar to DeepSeek,
        #but the weights are different from DeepSeek.
        #MTP does not have separate embedding weights, and in pipeline scenarios,
        #weights need to be copied from the first PP stage.
        if args.should_get_embedding_weights_for_mtp:
            _p2p_embedding_weights_for_mtp(unwrapped_model, args)
# PLACEHOLDER_SETUP_HF

    elif is_hf_checkpoint(args.load) and not args.moe_use_upcycling:
        # Online HF checkpoint loading
        assert (not args.use_megatron_fsdp), "Megatron FSDP and HF checkpoint loading cannot be used together. " \
            "Please set --use-megatron-fsdp to False."
        timers("load-checkpoint", log_level=0).start(barrier=True)
        _pp_layout = _resolve_convert_pp_layout(model_config)
        if _pp_layout is not None:
            args.convert_pp_layout = _pp_layout
            print_rank_0("[bridge] convert_pp_layout resolved (foundation-aware for VLM); "
                         "converter will use the model's custom VPP layer layout.")
        else:
            print_rank_0("[bridge] WARNING: no pipeline_model_parallel_layout on model_config or its "
                         "foundation; converter falls back to balanced VPP (likely wrong for custom layouts).")
        args.iteration, args.num_floating_point_operations_so_far = load_hf_checkpoint_online(
            model,
            optimizer,
            opt_param_scheduler,
            args
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    if args.enable_ema:
        ema = get_model(model_provider_func, model_type)
        if args.iteration == 0:
            for e, m in zip(ema, model):
                update_ema(e, m, rate=0)
        else:
            load_checkpoint(ema, None, None, load_arg="load_ema")
    else:
        ema = None

    # get model without FP16 and/or DDP wrappers
    if (
        args.iteration == 0
        and len(unwrapped_model) == 1
        and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
    ):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
        update_use_dist_ckpt(args)
# PLACEHOLDER_SETUP_CONVERT

        save_checkpoint(
            args.iteration,
            model,
            optimizer,
            opt_param_scheduler,
            args.num_floating_point_operations_so_far,
            preprocess_common_state_dict_fn=preprocess_common_state_dict,
            peft_class=peft_class,
        )

        print_rank_0(
            "> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format)
        )
        torch.distributed.barrier()
        exit()

    return model, ema, optimizer, opt_param_scheduler, peft_class

