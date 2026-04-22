# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""LoongForge initialization."""

import os
import logging
import argparse

import torch

from megatron.core import mpu, tensor_parallel

from megatron.training.arguments import (
    parse_args,
    validate_args as validate_megatron_args,
    add_megatron_arguments,
)

from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.global_vars import (
    set_global_variables as set_megatron_global_variables,
)
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)

from megatron.training.initialize import (
    _initialize_distributed,
    _set_random_seed,
    _init_autoresume,
    _compile_dependencies,
    _initialize_tp_communicators,
)
from megatron.core.parallel_state import initialize_model_parallel, is_initialized

from loongforge.utils.global_vars import set_loongforge_extra_global_vars
from loongforge.utils import get_model_config

import inspect

_ParallelStatesDict = {}
_CurrentParallelStateModel = "defaults"
_DecoderTensorParallelSize = 1
_ImageEncoderDataParallelSize = 1
_VideoEncoderDataParallelSize = 1
_AudioEncoderDataParallelSize = 1
_NumMicroBatchesPerDecoderDP = 1
_NumRealMicroBatchesPerDecoderDP = 1
_NumEncodeRounds = 1
_ModelSize = 1

def get_model_size():
    """Return the model parallel size."""
    return _ModelSize

def get_num_micro_batches_per_decoder_dp():
    """Return the number of micro-batches per decoder DP group
    and the number of encode rounds."""
    return _NumMicroBatchesPerDecoderDP, _NumEncodeRounds

def get_num_real_micro_batches_per_decoder_dp():
    """Return the number of real (non-mock) micro-batches per decoder DP group."""
    return _NumRealMicroBatchesPerDecoderDP

def is_mock_microbatch(microbatch_index: int) -> bool:
    """Return True if the given microbatch index corresponds to a mock (padding) microbatch."""
    return microbatch_index >= _NumRealMicroBatchesPerDecoderDP


def get_encoder_dp_size(name):
    """
    Get the data parallel size of the encoder.
    """
    if name == 'image_encoder':
        return _ImageEncoderDataParallelSize
    elif name == 'video_encoder':
        return _VideoEncoderDataParallelSize
    elif name == 'audio_encoder':
        return _AudioEncoderDataParallelSize
    else:
        raise ValueError(f'Unknown encoder type: {name}')

def destroy_model_parallel_group():
    """Set the groups to none."""
    for k, v in vars((mpu)).items():
        if k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v):
            setattr(mpu, k, None)

def change_parallel_state(module_name):
    """
    Change the parallel state of the model to the state saved in _ParallelStatesDict
    """
    global _CurrentParallelStateModel
    if module_name == _CurrentParallelStateModel:
        return
    target_globals = vars(mpu)
    source_globals = _ParallelStatesDict[module_name]
    if _CurrentParallelStateModel in _ParallelStatesDict:
        current_globals = _ParallelStatesDict[_CurrentParallelStateModel]
        for k in current_globals:
            if k in target_globals:
                current_globals[k] = target_globals[k]
    for k, v in source_globals.items():
        if k in target_globals:
            target_globals[k] = v
    _CurrentParallelStateModel = module_name

def save_parallel_state(module_name):
    """
    Save the current parallel state of the model
    """
    state_snapshot = {
        k: v for k, v in vars((mpu)).items()
        if k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v)
    }
    
    # The gloo communication groups of image_encoder, video_encoder, and audio_encoder
    # are kept consistent with text_decoder.
    if module_name in ['image_encoder', 'video_encoder', 'audio_encoder']:
        for k in ["_DATA_PARALLEL_GROUP_GLOO", 
            "_EXPERT_DATA_PARALLEL_GROUP_GLOO", 
            "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO",
            "_DATA_PARALLEL_GROUP_WITH_CP_GLOO",
            "_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO"]:
                state_snapshot[k] = _ParallelStatesDict["text_decoder"][k]

    _ParallelStatesDict.setdefault(module_name, {}).update(state_snapshot)

def create_parallel_state(
        module_name, 
        tp_size=0, 
        enable_encoder_hetero_dp=False,
        enable_full_hetero_dp=False):
    """
    Create the parallel state of the model and save it
    """
    if tp_size == 0:
        tp_size = _DecoderTensorParallelSize
    assert tp_size <= _DecoderTensorParallelSize and _DecoderTensorParallelSize % tp_size == 0
    destroy_model_parallel_group()

    if enable_encoder_hetero_dp:
        assert tp_size == 1, f"encoder_tp_size must be 1 when enable_encoder_hetero_dp is True, but got {tp_size}"
        if module_name == "image_encoder":
            global _ImageEncoderDataParallelSize
            _ImageEncoderDataParallelSize = _DecoderTensorParallelSize // tp_size
        elif module_name == "video_encoder":
            global _VideoEncoderDataParallelSize
            _VideoEncoderDataParallelSize = _DecoderTensorParallelSize // tp_size
        elif module_name == "audio_encoder":
            global _AudioEncoderDataParallelSize
            _AudioEncoderDataParallelSize = _DecoderTensorParallelSize // tp_size

    if enable_full_hetero_dp:
        assert tp_size == 1, f"encoder_tp_size must be 1 when enable_full_hetero_dp is True, but got {tp_size}"

    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        use_sharp=False,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        nccl_communicator_config_path=None,
        distributed_timeout_minutes=30,
        order="tp-cp-ep-dp-pp")
    save_parallel_state(module_name)

logger = logging.getLogger(__name__)


def initialize_loongforge_megatron(
    args,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    store=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker()

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_megatron_global_variables(args, build_tokenizer=False)

    # set loongforge extra global args
    set_loongforge_extra_global_vars(args, build_tokenizer=True)

    # set logging level
    setup_logging(args)

    # init rerun state
    def state_save_func():
        return {
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states()
        }

    def state_restore_func(state_dict):
        if state_dict["rng_tracker_states"]:
            tensor_parallel.get_cuda_rng_tracker().set_states(
                state_dict["rng_tracker_states"]
            )

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(args.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=args.error_injection_rate,
            error_injection_type=RerunDiagnostic(args.error_injection_type),
        ),
        result_rejected_tracker_filename=args.result_rejected_tracker_filename,
    )

    # torch.distributed initialization
    def finish_mpu_init():
        """torch.distributed initialization"""

        from .parser import parse_args_from_config
        # set model config from args and hydra config (must be before _initialize_distributed,
        # because get_embedding_ranks may depend on get_model_config())
        parse_args_from_config(args)

        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store)

        save_parallel_state('text_decoder')
        global _DecoderTensorParallelSize
        _DecoderTensorParallelSize = mpu.get_tensor_model_parallel_world_size()

        if args.enable_full_hetero_dp:
            assert args.context_parallel_size == 1, (
                "Full Heterogeneous DP does not support Context Parallelism. "
                f"context_parallel_size must be 1, but got {args.context_parallel_size}"
            )
            world_size: int = torch.distributed.get_world_size()
            global _ModelSize
            _ModelSize = (
                args.tensor_model_parallel_size 
                * args.pipeline_model_parallel_size 
                * args.context_parallel_size
            )
            if world_size % _ModelSize != 0:
                raise RuntimeError(f"world_size ({world_size}) is not divisible by {_ModelSize}")
            data_parallel_size: int = world_size // _ModelSize
            global _NumMicroBatchesPerDecoderDP, _NumRealMicroBatchesPerDecoderDP, _NumEncodeRounds
            _NumRealMicroBatchesPerDecoderDP = args.global_batch_size // data_parallel_size
            assert _NumRealMicroBatchesPerDecoderDP >= 1, (
                f"_NumRealMicroBatchesPerDecoderDP ({_NumRealMicroBatchesPerDecoderDP}) "
                f"must be at least 1"
            )
            if _NumRealMicroBatchesPerDecoderDP < _ModelSize:
                _NumMicroBatchesPerDecoderDP = _ModelSize
            elif _NumRealMicroBatchesPerDecoderDP % _ModelSize != 0:
                _NumMicroBatchesPerDecoderDP = (
                    (_NumRealMicroBatchesPerDecoderDP + _ModelSize - 1) // _ModelSize * _ModelSize
                )
            else:
                _NumMicroBatchesPerDecoderDP = _NumRealMicroBatchesPerDecoderDP
            _NumEncodeRounds = _NumMicroBatchesPerDecoderDP // _ModelSize

        model_config = get_model_config()
        from megatron.training import print_rank_0
        print_rank_0(f"model_config: {model_config}")
        if hasattr(model_config, "image_encoder") and model_config.image_encoder is not None:
            create_parallel_state(
                'image_encoder', 
                model_config.image_encoder.tensor_model_parallel_size, 
                args.enable_encoder_hetero_dp,
                args.enable_full_hetero_dp
            )
        if hasattr(model_config, "video_encoder") and model_config.video_encoder is not None:
            create_parallel_state(
                'video_encoder', 
                model_config.video_encoder.tensor_model_parallel_size, 
                args.enable_encoder_hetero_dp,
                args.enable_full_hetero_dp
            )
        if hasattr(model_config, "audio_encoder") and model_config.audio_encoder is not None:
            create_parallel_state(
                'audio_encoder', 
                model_config.audio_encoder.tensor_model_parallel_size, 
                args.enable_encoder_hetero_dp,
                args.enable_full_hetero_dp
            )

        change_parallel_state('text_decoder')

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
            use_cudagraphable_rng=args.cuda_graph_impl != "none",
        )

        # Setup MoE aux loss scale value.
        if args.num_experts is not None:
            from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler

            MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    if skip_mpu_initialization:
        return None

    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            # TODO: Should this be activated with just decoder-tp-comm-overlap too?
            _initialize_tp_communicators()

        # No continuation function
        return None


def setup_logging(args) -> None:
    """Sets the default logging level based on cmdline args and env vars.

    Precedence:
    1. Command line argument `--logging-level`
    2. Env var `MEGATRON_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    logging_level = None
    env_logging_level = os.getenv("MEGATRON_LOGGING_LEVEL", None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if args.logging_level is not None:
        logging_level = args.logging_level

    if logging_level is not None:
        logger.info(f"Setting logging level to {logging_level}")
        logging.getLogger().setLevel(logging_level)
