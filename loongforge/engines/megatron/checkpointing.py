# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing.

Modified copy of ``megatron/training/checkpointing.py`` (Loong-Megatron). Functions identical to upstream are
imported from it; every local copy documents its divergence in its own docstring. Upstream keeps these bodies
inside ``load_checkpoint`` or exposes no hook where the local additions fit, so they cannot be imported.
"""

import contextlib
import os
import random
import shutil
import sys
import threading
from logging import getLogger
from time import time

import numpy as np
import torch
from torch.distributed.checkpoint import FileSystemReader, default_planner

from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.msc_utils import open_file
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.training import ft_integration, wandb_utils
from megatron.training.async_utils import is_empty_async_queue, schedule_async_save
from megatron.training.global_vars import get_args
from megatron.training.one_logger_utils import on_save_checkpoint_start, on_save_checkpoint_success
from megatron.training.utils import append_to_progress_log, is_last_rank, print_rank_0, unwrap_model
from megatron.training.checkpointing import (
    get_checkpoint_version,
    set_checkpoint_version,
    isfile,
    ensure_directory_exists,
    get_load_checkpoint_path_by_args,
    get_distributed_optimizer_checkpoint_name,
    get_checkpoint_tracker_filename,
    checkpoint_exists,
    CheckpointType,
    _get_checkpoint_format,
    _to_dtensor,
    cleanup_old_non_persistent_checkpoint,
    _get_non_persistent_iteration,
    _load_global_dist_base_checkpoint,
    # Identical to upstream, so imported.
    check_checkpoint_args,
    get_checkpoint_name,
    find_checkpoint_rank_0,
    read_metadata,
    get_rng_state,
    _build_sharded_state_dict_metadata,
    maybe_save_dataloader_state,
    generate_state_dict,
    fix_query_key_value_ordering,
    _load_non_persistent_base_checkpoint,
)

from loongforge.models.common.peft.utils import apply_peft_adapter_filter_to_state_dict
try:
    # The uneven_dtensor import only checks that Megatron-FSDP is available.
    from megatron.core.distributed.fsdp.src.megatron_fsdp import uneven_dtensor  # noqa: F401
    from megatron.core.transformer.fsdp_dtensor_checkpoint import print_diff_in_state_dicts

    HAVE_MEGATRON_FSDP = True
except ImportError:
    HAVE_MEGATRON_FSDP = False


# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import save_modelopt_state, save_sharded_modelopt_state

    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

logger = getLogger(__name__)
_NON_PERSISTENT_CKPT_SUBDIR = 'non_persistent'


def _load_model_state_dict(module, state_dict, strict: bool):
    """Load state while retaining the TransformerEngine compatibility fallback.

    Upstream defines this as the nested ``load_model_state_dict`` inside ``load_checkpoint``; the behavior is
    the same and it is a module function here so it can be tested on its own.
    """
    try:
        module.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        print(f"[load_state_dict strict={strict}] exception:\n{e}")
        if not strict:
            raise
        load_return = module.load_state_dict(state_dict, strict=False)
        print(f"[load_state_dict strict=False] return: {load_return}")


def load_dataloader_state(args):
    """Read the per-DP-rank dataloader state saved by ``maybe_save_dataloader_state``.

    Returns the state dict, or ``None`` when no state is configured, loading is off,
    or the file does not exist. Kept next to the save hook so the file layout has a
    single owner.
    """
    if args.load is None or not getattr(args, "dataloader_save", None):
        return None
    dp_rank = mpu.get_data_parallel_rank()
    data_save_name = get_checkpoint_name(
        args.dataloader_save,
        args.iteration,
        pipeline_rank=0,  # Only the first pipeline parallel rank stores the dataloader checkpoint.
        basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
    )
    if not os.path.exists(data_save_name):
        print(f"dataset state {data_save_name} does not exist")
        return None
    try:
        dataset_state_dict = torch.load(data_save_name, map_location="cpu")
        print(f"loading dataset state from {data_save_name}")
        return dataset_state_dict["dataloader_state_dict"]
    except Exception as e:
        print("loading dataset state failed. Skipping. " + str(e))
        return None


def save_checkpoint(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context=None,
    pipeline_rank=None,
    expert_rank=None,
    tensor_rank=None,
    pipeline_parallel=None,
    expert_parallel=None,
    non_persistent_ckpt=False,
    train_data_iterator=None,
    preprocess_common_state_dict_fn=None,
    release=False,
    save_arg='save',
    peft_class=None,
):
    """Save a model, optimizer and optionally dataloader checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).

    Differs from upstream: the ``peft_class`` argument drops non-adapter weights from the state dict before
    writing, so a PEFT run stores only the adapter.

    If non_persistent_ckpt is True,
    the checkpoint will be saved with special functionality for removing old checkpoints.
    There are several types of non-persistent checkpoints:
    "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed.
    "local" - Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk).

    Dataloader checkpoint is only saved if the dataloader supports it. Currently this applies only
    to the Megatron Energon dataloader (multimodal) and not the built-in Megatron dataloader (text-only).
    """
    start_ckpt = time()
    args = get_args()

    if args.async_save and not is_empty_async_queue():
        print_rank_0(
            'WARNING: Starting a checkpoint save before previous has finished. Consider increasing the checkpoint interval.'
        )

    # Prepare E2E metrics at start of save checkpoint
    productive_metrics = on_save_checkpoint_start(args.async_save)

    # Monitor for the checkpointing timeout (no-op if FT is not enabled)
    ft_integration.on_checkpointing_start()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    # Handle non_persistent_ckpt flag. Besides overwriting `args.save` and
    # `args.use_dist_ckpt`, non-persistent global ckpt requires no additional logic
    ckpt_type = CheckpointType.GLOBAL if args.use_dist_ckpt else CheckpointType.LEGACY
    save_dir = getattr(args, save_arg)
    if non_persistent_ckpt:
        if args.non_persistent_ckpt_type == 'global':
            ckpt_type = CheckpointType.GLOBAL
            save_dir = (
                args.non_persistent_global_ckpt_dir
                if args.non_persistent_global_ckpt_dir
                else os.path.join(save_dir, _NON_PERSISTENT_CKPT_SUBDIR)
            )
            # TODO Can we ensure the previous checkpoint is saved? We don't want to allow two saves in parallel.
            cleanup_old_non_persistent_checkpoint(
                save_dir, leave_ckpt_num=1, do_async=args.async_save
            )
        elif args.non_persistent_ckpt_type == 'local':
            ckpt_type = CheckpointType.LOCAL
            save_dir = checkpointing_context['local_checkpoint_manager'].local_ckpt_dir
        else:
            raise NotImplementedError(
                f"Please use local or global non-persistent checkpoints (got: {args.non_persistent_ckpt_type})"
            )

    ckpt_format = args.ckpt_format if ckpt_type == CheckpointType.GLOBAL else 'torch'
    print_rank_0(
        'saving checkpoint at iteration {:7d} to {} in {} format'.format(
            iteration, save_dir, ckpt_format
        )
    )

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.ckpt_format)

    # Collect rerun state across all ranks
    rerun_state_machine = get_rerun_state_machine()
    rerun_state = rerun_state_machine.state_dict(
        data_iterator=train_data_iterator, ckpt_format=args.ckpt_format
    )

    # Checkpoint name.
    return_base_dir = ckpt_type != CheckpointType.LEGACY
    checkpoint_name = get_checkpoint_name(
        save_dir,
        iteration,
        release=release,
        pipeline_parallel=pipeline_parallel,
        tensor_rank=tensor_rank,
        pipeline_rank=pipeline_rank,
        expert_parallel=expert_parallel,
        expert_rank=expert_rank,
        return_base_dir=return_base_dir,
    )

    # Save dataloader state if the dataloader supports it (currently only Megatron Energon).
    maybe_save_dataloader_state(
        train_data_iterator, iteration, getattr(args, "dataloader_save", None)
    )

    # Save distributed optimizer's custom parameter state.
    if (
        args.use_distributed_optimizer
        and not args.no_save_optim
        and optimizer is not None
        and ckpt_type == CheckpointType.LEGACY
    ):
        optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        if not optimizer.is_stub_optimizer:
            optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if args.async_save:
        if ckpt_type == CheckpointType.LEGACY:
            raise NotImplementedError(
                'Async checkpoint save not implemented for legacy checkpoints'
            )
        elif ckpt_type == CheckpointType.GLOBAL and args.ckpt_format != 'torch_dist':
            raise NotImplementedError(
                f'Async checkpoint save not implemented for {args.ckpt_format} distributed checkpoint format'
            )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Collect args, model, RNG.
    if (
        not torch.distributed.is_initialized()
        or mpu.get_expert_data_parallel_rank() == 0
        or ckpt_type != CheckpointType.LEGACY
    ):
        if ckpt_type != CheckpointType.LEGACY:
            sharded_sd_metadata = _build_sharded_state_dict_metadata(args)
            if args.use_distributed_optimizer:
                print_rank_0(
                    f'Storing distributed optimizer sharded state of type'
                    f' {sharded_sd_metadata["distrib_optim_sharding_type"]}'
                )
        else:
            sharded_sd_metadata = None
        state_dict = generate_state_dict(
            args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=dict(metadata=sharded_sd_metadata),
            model_sd_kwargs=dict(metadata=sharded_sd_metadata),
            rerun_state=rerun_state,
        )

        if peft_class is not None:
            state_dict = apply_peft_adapter_filter_to_state_dict(state_dict, peft_class)

        state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
        if ckpt_type == CheckpointType.GLOBAL and ckpt_format == "torch_dist":
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                # TODO Handle non-empty directories (e.g., after a crash during saving).
                ensure_directory_exists(checkpoint_name, check_parent=False)
            if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                save_strategy = checkpointing_context['save_strategy']
                # Already saved once before - don't need to rerun sharding validation
                validate_sharding_integrity = not args.ckpt_assume_constant_structure
            else:
                validate_sharding_integrity = True
                save_strategy = get_default_save_sharded_strategy(args.ckpt_format)
                if args.ckpt_assume_constant_structure and args.ckpt_format == 'torch_dist':
                    save_strategy.use_cached_ckpt_structure = args.ckpt_assume_constant_structure
                    if (
                        checkpointing_context is not None
                        and 'load_strategy' in checkpointing_context
                    ):
                        cached_global_metadata = getattr(
                            checkpointing_context['load_strategy'], 'cached_global_metadata', None
                        )
                        if cached_global_metadata is not None:
                            logger.debug("Plugging in the read metadata from the load strategy...")
                            save_strategy.cached_global_metadata = cached_global_metadata
                        else:
                            logger.debug(
                                "Failed to plug in the read metadata from the load strategy..."
                            )

                if args.ckpt_fully_parallel_save:
                    save_strategy = FullyParallelSaveStrategyWrapper(
                        save_strategy,
                        mpu.get_data_parallel_group(with_context_parallel=True),
                        args.ckpt_assume_constant_structure,
                    )
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context['save_strategy'] = save_strategy
            end_ckpt = time()
            logger.debug(
                f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt "
            )
            async_save_request = dist_checkpointing.save(
                state_dict,
                checkpoint_name,
                save_strategy,
                async_sharded_save=args.async_save,
                validate_access_integrity=validate_sharding_integrity,
                preprocess_common_before_consistancy_check=preprocess_common_state_dict_fn,
                content_metadata=sharded_sd_metadata,
            )
            # [ModelOpt]: save sharded modelopt_state
            if has_nvidia_modelopt:
                save_sharded_modelopt_state(model, checkpoint_name, (args.ckpt_format, 1))
        elif ckpt_type == CheckpointType.GLOBAL and ckpt_format in ["torch_dcp", "fsdp_dtensor"]:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                # TODO Handle non-empty directories (e.g., after a crash during saving).
                ensure_directory_exists(checkpoint_name, check_parent=False)

            fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(checkpoint_name)
            torch.distributed.checkpoint.save(
                state_dict=state_dict, storage_writer=fs_storage_writer
            )
        else:
            # [ModelOpt]: Inject modelopt_state into state_dict
            if has_nvidia_modelopt:
                if ckpt_type == CheckpointType.LOCAL:
                    print_rank_0('WARNING: Local checkpointing does not support nvidia_modelopt.')
                else:
                    save_modelopt_state(model, state_dict)

            end_ckpt = time()
            logger.debug(
                f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt "
            )
            if ckpt_type == CheckpointType.LOCAL:
                try:
                    from megatron.core.dist_checkpointing.tensor_aware_state_dict import (
                        MCoreTensorAwareStateDict,
                    )
                except ModuleNotFoundError:
                    raise RuntimeError(
                        "The 'nvidia_resiliency_ext' module is required for local "
                        "checkpointing but was not found. Please ensure it is installed."
                    )
                if (sharded_sd_metadata or {}).get('distrib_optim_sharding_type') in [
                    'fully_reshardable',
                    'dp_zero_gather_scatter',
                ]:
                    # Note: Currently full reshardabilty is not supported when local checkpoints are used.
                    raise RuntimeError(
                        f"Local checkpointing does not support optimizer sharding type '{sharded_sd_metadata['distrib_optim_sharding_type']}'. "
                        "Don't use '--dist-ckpt-optim-fully-reshardable' when saving local checkpoints."
                    )
                algo = args.non_persistent_local_ckpt_algo
                cached_metadata = None
                if (
                    args.ckpt_assume_constant_structure
                    and 'local_checkpoint_cache' in checkpointing_context
                ):
                    cached_metadata = checkpointing_context['local_checkpoint_cache']
                state_dict_for_save, cacheable_metadata = MCoreTensorAwareStateDict.from_state_dict(
                    state_dict,
                    algo=algo,
                    cached_metadata=cached_metadata,
                    parallelization_group=mpu.get_data_parallel_group(with_context_parallel=True),
                )
                async_save_request = checkpointing_context['local_checkpoint_manager'].save(
                    state_dict_for_save, iteration, is_async=bool(args.async_save)
                )
                checkpointing_context['local_checkpoint_cache'] = cacheable_metadata
            else:
                assert ckpt_type == CheckpointType.LEGACY
                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)
    start_misc = time()
    if ckpt_type != CheckpointType.LOCAL:
        if not args.async_save:
            assert async_save_request is None
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    # And update the latest iteration
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(save_dir)

        if ckpt_type == CheckpointType.LOCAL:

            def iter_finalize_fn():
                print_rank_0(
                    '  successfully saved local checkpoint from iteration {:7d}'.format(iteration)
                )
                if args.log_progress and args.async_save:
                    append_to_progress_log(
                        f'Saved async local checkpoint\tIteration: {iteration}', barrier=False
                    )

        else:

            def iter_finalize_fn():
                prev_iteration = 0
                save_retain_interval = getattr(
                    args, 'save_retain_interval', None
                )  # For backwards compatibility of tests.
                if save_retain_interval is not None:
                    if os.path.exists(
                        tracker_filename
                    ):  # TODO: Make this work with MSC remote paths?
                        with open_file(tracker_filename, 'r') as f:
                            prev_iteration = int(f.read().strip())
                with open_file(tracker_filename, 'w') as f:
                    f.write("release" if release else str(iteration))
                tensor_rank_to_print = (
                    tensor_rank if tensor_rank is not None else mpu.get_tensor_model_parallel_rank()
                ) + 1
                pipeline_rank_to_print = (
                    pipeline_rank
                    if pipeline_rank is not None
                    else mpu.get_pipeline_model_parallel_rank()
                ) + 1
                print_rank_0(
                    f'  successfully saved checkpoint from iteration {int(iteration):7d} to {save_dir} '
                    f'[ t {tensor_rank_to_print}/{mpu.get_tensor_model_parallel_world_size()}, '
                    f'p {pipeline_rank_to_print}/{mpu.get_pipeline_model_parallel_world_size()} ]'
                )
                if args.log_progress and args.async_save:
                    append_to_progress_log(
                        f'Saved async checkpoint\tIteration: {iteration}', barrier=False
                    )

                def delete_checkpoint(args, iteration_to_delete):
                    checkpoint_name = get_checkpoint_name(
                        save_dir, iteration=iteration_to_delete, return_base_dir=True
                    )
                    try:
                        shutil.rmtree(
                            checkpoint_name
                        )  # TODO: Make this work with MSC remote paths?
                        print_rank_0(
                            f'  successfully deleted checkpoint from iteration {iteration_to_delete:7d} '
                            f'at {save_dir}'
                        )
                        if args.log_progress:
                            append_to_progress_log(
                                f'Deleted checkpoint\tIteration: {iteration_to_delete}',
                                barrier=False,
                            )
                    except Exception as e:
                        print_rank_0(
                            f'  encountered exception "{e}" when trying to delete checkpoint from '
                            f'iteration {iteration_to_delete:7d} at {save_dir}'
                        )
                        # Any exception encountered in checkpoint deletion can be ignored and is not fatal.
                        pass

                if save_retain_interval is not None:
                    if (
                        prev_iteration > 0
                        and prev_iteration != iteration
                        and prev_iteration % save_retain_interval != 0
                    ):
                        checkpoint_name = get_checkpoint_name(
                            save_dir, iteration=prev_iteration, return_base_dir=True
                        )
                        # Don't delete if `checkpoint_name` is a symbolic link.
                        if os.path.islink(
                            checkpoint_name
                        ):  # TODO: Make this work with MSC remote paths?
                            print_rank_0(
                                f'  skipping deleting checkpoint from iteration {prev_iteration:7d} '
                                f'at {save_dir} since it is a symbolic link'
                            )
                        else:
                            # Asynchronous version of delete_checkpoint(args, iteration_to_delete=prev_iteration).
                            threading.Thread(
                                target=delete_checkpoint, args=(args, prev_iteration)
                            ).start()

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    # Additional callback for one_logger (last rank)
    if not torch.distributed.is_initialized() or is_last_rank():

        def onelogger_finalize_fn():
            on_save_checkpoint_success(productive_metrics, args.async_save)

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(onelogger_finalize_fn)
        else:
            onelogger_finalize_fn()

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() or is_last_rank():

        def wandb_finalize_fn():
            wandb_utils.on_save_checkpoint_success(
                checkpoint_name, get_checkpoint_tracker_filename(save_dir), save_dir, iteration
            )

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(wandb_finalize_fn)
        else:
            wandb_finalize_fn()

    if args.async_save:
        schedule_async_save(async_save_request)
        print_rank_0(
            '  scheduled an async checkpoint save at iteration {:7d} to {}'.format(
                iteration, save_dir
            )
        )

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    end_misc = time()
    logger.debug(f"rank: {rank}, takes {end_misc - start_misc} to finalize ckpt save ")

    ft_integration.on_checkpointing_end(is_async_finalization=False)


def _load_base_checkpoint(
    load_dir,
    args,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
    ignore_ckpt_step: bool = False,
):
    """Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.

    Differs from upstream: ``ignore_ckpt_step`` disables the ``args.ckpt_step`` override, and the two legacy
    ``torch.load`` calls pass ``weights_only=False`` to read older checkpoint payloads.
    """
    # Try to load non-persistent checkpoint first
    non_persistent_global_dir = (
        args.non_persistent_global_ckpt_dir
        if args.non_persistent_global_ckpt_dir or load_dir is None
        else os.path.join(load_dir, _NON_PERSISTENT_CKPT_SUBDIR)
    )
    non_persistent_iteration = _get_non_persistent_iteration(
        non_persistent_global_dir, args, checkpointing_context
    )
    iteration, release = -1, False
    tracker_filename = 'because load directory is not defined'
    if load_dir is not None:
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)

    # Allow user to specify the loaded iteration.
    if getattr(args, "ckpt_step", None) and not ignore_ckpt_step:
        iteration = args.ckpt_step

    if non_persistent_iteration != -1:  # there is a non-persistent checkpoint
        if non_persistent_iteration >= iteration:
            return _load_non_persistent_base_checkpoint(
                non_persistent_global_dir,
                args,
                rank0,
                sharded_state_dict,
                non_persistent_iteration,
                checkpointing_context,
            )
        else:
            print_rank_0('WARNING: non-persistent checkpoints are older than persistent checkpoint')

    # Otherwise we are dealing with global checkpoints
    # If no tracker file, return nothing
    if iteration == -1:
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {}'.format(tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from random')
        # Conditionally exit if checkpoint not found.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False, None

    # Determine the type of the checkpoint on disk.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    ckpt_format = _get_checkpoint_format(checkpoint_name, args)

    if not rank0:
        dist_infix = "distributed " if ckpt_format == "torch_dist" else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(
                f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}'
            )

    ckpt_type = None

    # Handle global distributed checkpoint
    if ckpt_format == "torch_dist":
        return _load_global_dist_base_checkpoint(
            load_dir,
            args,
            rank0,
            sharded_state_dict,
            iteration,
            release,
            checkpointing_context=checkpointing_context,
        )
    elif ckpt_format == "torch":
        ckpt_type = CheckpointType.LEGACY
        # Handle global legacy checkpoint
        if rank0:
            checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        else:
            checkpoint_name = get_checkpoint_name(
                load_dir, iteration, release, return_base_dir=False
            )
        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
        except ModuleNotFoundError:
            from megatron.legacy.fp16_deprecated import loss_scaler  # noqa: F401 (registers module in sys.modules)

            # For backward compatibility.
            if not rank0:
                print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler'
            ]
            sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
                'megatron.legacy.fp16_deprecated.loss_scaler'
            ]
            sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
            state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
            sys.modules.pop('fp16.loss_scaler', None)
            sys.modules.pop('megatron.fp16.loss_scaler', None)
            sys.modules.pop('megatron.model', None)
        except Exception as e:
            print('could not load the checkpoint')
            print(e)
            sys.exit()
    elif ckpt_format == "torch_dcp":
        ckpt_type = CheckpointType.TORCH_DCP

        if rank0:
            # _load_base_checkpoint is called from load_args_from_checkpoint. torch.distributed is not initialized.
            # Load only metadata.
            state_dict = {"args": None, "iteration": None}
            torch.distributed.checkpoint.load(state_dict=state_dict, checkpoint_id=checkpoint_name)
        else:
            # _load_base_checkpoint is called from load_checkpoint with a proper state dict.
            state_dict = sharded_state_dict

            fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(checkpoint_name)

            torch.distributed.checkpoint.load_state_dict(
                state_dict=state_dict, storage_reader=fs_storage_reader
            )
    elif ckpt_format == "fsdp_dtensor":
        assert HAVE_MEGATRON_FSDP, "Should not be called if Megatron-FSDP is not available."
        if rank0:
            return {}, checkpoint_name, release, CheckpointType.FSDP_DTENSOR

        ckpt_type = CheckpointType.FSDP_DTENSOR
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(checkpoint_name)
        allow_partial_load = not getattr(args, 'strict_fsdp_dtensor_load', False)
        if allow_partial_load:
            state_dict_metadata = fs_storage_reader.read_metadata().state_dict_metadata
            rank = torch.distributed.get_rank()
            import time as _time

            _time.sleep(rank * 0.001)  # Make that logs of different ranks do not overlap
            print_diff_in_state_dicts(state_dict_metadata, sharded_state_dict)

        planner = default_planner.DefaultLoadPlanner(allow_partial_load=allow_partial_load)
        torch.distributed.checkpoint.load_state_dict(
            state_dict=sharded_state_dict, storage_reader=fs_storage_reader, planner=planner
        )
        state_dict = sharded_state_dict
    else:
        raise NotImplementedError(f"checkpoint format {ckpt_format} not supported")

    return state_dict, checkpoint_name, release, ckpt_type


def load_checkpoint(
    ddp_model,
    optimizer,
    opt_param_scheduler,
    load_arg='load',
    strict=True,
    checkpointing_context=None,
    skip_load_to_model_and_opt=False,
    peft_class=None,
):
    """Load a model checkpoint and return the iteration.

    Differs from upstream: the ``peft_class`` argument is forwarded to ``_load_checkpoint_from_path``, and the
    body lives there instead of here.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    skip_load_to_model_and_opt (bool): whether to call `load_state_dict`
        for :attr:`model` and :attr:`optimizer`. In case of running FSDP2 with mcore distributed
        checkpointing, the tensors are already loaded in-place by `_load_base_checkpoint`.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    # Finetuning directories
    pretrained_dir = getattr(args, 'pretrained_checkpoint', None)
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(
            f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}'
        )
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        args.finetune = True
    return _load_checkpoint_from_path(
        args,
        load_dir,
        ddp_model,
        optimizer,
        opt_param_scheduler,
        load_arg,
        strict,
        checkpointing_context,
        peft_class=peft_class,
    )


def _load_checkpoint_from_path(
    args,
    load_dir,
    ddp_model,
    optimizer,
    opt_param_scheduler,
    load_arg='load',
    strict: bool = True,
    checkpointing_context=None,
    skip_load_to_model_and_opt: bool = False,
    ignore_ckpt_step: bool = False,
    peft_class=None,
) -> tuple[int, int]:
    """Load a checkpoint from a given path.

    Upstream has no separate function: this is the body of ``load_checkpoint`` with the ``get_args()`` call and
    the pretrained-directory switch removed. It adds ``ignore_ckpt_step`` on both base loads, the PEFT resume
    filter of the sharded state dict, and ``strict=False`` on the model load when resuming a PEFT run.

    Args:
        load_dir: The directory containing the checkpoint.
        ddp_model: The model module(s) to load state into.
        optimizer: The optimizer instance to load state into.
        opt_param_scheduler: The scheduler instance to load state into.
        strict: Whether to enforce strict loading (see torch.nn.Module.load_state_dict).
        checkpointing_context: Dictionary to store context across loads (e.g., strategies).
        skip_load_to_model_and_opt: If True, only loads metadata (iteration, rng) but
                                      skips loading state into model and optimizer modules.
        ignore_ckpt_step: If True, ignores the ckpt_step config and loads latest checkpoint.
                          Used when loading pretrained checkpoints in PEFT scenarios.

    Returns:
        A tuple containing:
        - iteration: The training iteration number.
        - num_floating_point_operations_so_far: The total FLOPs computed so far.
    """

    model = unwrap_model(ddp_model)

    ckpt_format = args.ckpt_format
    if args.auto_detect_ckpt_format or ckpt_format == "torch_dist":
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir,
            args,
            rank0=True,
            checkpointing_context=checkpointing_context,
            ignore_ckpt_step=ignore_ckpt_step,
        )

        ckpt_format = None
        if ckpt_type == CheckpointType.TORCH_DCP:
            ckpt_format = "torch_dcp"
        elif ckpt_type == CheckpointType.FSDP_DTENSOR:
            ckpt_format = "fsdp_dtensor"
        elif ckpt_type == CheckpointType.LEGACY:
            ckpt_format = "torch"
        elif ckpt_type in [CheckpointType.LOCAL, CheckpointType.GLOBAL]:
            ckpt_format = "torch_dist"
        elif ckpt_type is None:
            pass  # Not loaded.
        else:
            raise NotImplementedError(f"checkpoint format {ckpt_format} not supported")

    load_kwargs = {}
    ignore_rng_state = False
    ignore_rerun_state = True
    if ckpt_format == "torch_dist":
        ckpt_tp_pp = (
            state_dict['args'].tensor_model_parallel_size,
            state_dict['args'].pipeline_model_parallel_size,
        )
        run_tp_pp = (args.tensor_model_parallel_size, args.pipeline_model_parallel_size)

        ckpt_world_size = getattr(state_dict['args'], 'world_size', 0)
        run_world_size = getattr(args, 'world_size', 0)
        ckpt_dp = getattr(state_dict['args'], 'data_parallel_size', 0)
        run_dp = getattr(args, 'data_parallel_size', 0)
        mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(
            run_tp_pp, ckpt_tp_pp
        )

        # Determine if RNG state will be loaded
        if (
            ckpt_tp_pp == run_tp_pp
            and not release
            and not args.finetune
            and not args.no_load_rng
            and not getattr(state_dict['args'], 'no_save_rng', False)
        ):
            gen_sd_rng_state = get_rng_state(args.ckpt_format)  # we can load the rng state
        else:
            ignore_rng_state = True
            gen_sd_rng_state = None
            if ckpt_tp_pp != run_tp_pp:
                print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

        if ckpt_type == CheckpointType.LOCAL:
            sharded_sd_metadata = _build_sharded_state_dict_metadata(args)
        else:
            sharded_sd_metadata = dist_checkpointing.load_content_metadata(
                preloaded_state_dict=state_dict
            )
        print_rank_0(
            f'sharded_state_dict metadata loaded from the checkpoint: {sharded_sd_metadata}'
        )
        # Determine if optimizer state will be loaded
        if (
            not release
            and not args.finetune
            and not args.no_load_optim
            and not getattr(state_dict['args'], 'no_save_optim', False)
        ):
            gen_sd_optim = optimizer
            gen_sd_opt_param_scheduler = opt_param_scheduler

            if args.use_distributed_optimizer:
                if sharded_sd_metadata is None:
                    # Backward-compatibility with old checkpoints which don't have content versioning
                    # Can be removed after ending support for MLM optimizer checkpoints with MCore < v0.13
                    # (for MCore v0.13+ checkpoints `sharded_sd_metadata is not None`)
                    sharded_sd_metadata = {
                        'distrib_optim_sharding_type': (
                            'fully_sharded_model_space'
                            if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                            else 'dp_zero_gather_scatter'
                        )
                    }
                if (
                    ckpt_tp_pp != run_tp_pp
                    and sharded_sd_metadata['distrib_optim_sharding_type']
                    not in DistributedOptimizer.checkpoint_fully_reshardable_formats
                ):
                    raise RuntimeError(
                        f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type"
                        f" {sharded_sd_metadata['distrib_optim_sharding_type']}."
                        f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving."
                    )

                # Check if fully parallel load is compatible with sharding type
                if (
                    args.ckpt_fully_parallel_load
                    and sharded_sd_metadata['distrib_optim_sharding_type']
                    == 'dp_zero_gather_scatter'
                ):
                    raise RuntimeError(
                        "Fully parallel load is not supported for dp_zero_gather_scatter checkpoints. "
                        "Please remove --ckpt-fully-parallel-load flag"
                    )
        else:
            gen_sd_optim = None
            gen_sd_opt_param_scheduler = None

        optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
        model_sd_kwargs = dict(metadata=sharded_sd_metadata)

        # Determine if rerun state will be loaded
        gen_sd_rerun_state = None
        if (
            ckpt_world_size == run_world_size
            and ckpt_tp_pp == run_tp_pp
            and ckpt_dp == run_dp
            and not release
            and not args.finetune
            and 'rerun_state_machine' in state_dict
        ):
            rerun_state_machine = get_rerun_state_machine()
            if rerun_state_machine.validate_state_dict(state_dict['rerun_state_machine']):
                gen_sd_rerun_state = rerun_state_machine.state_dict(
                    data_iterator=None, ckpt_format=ckpt_format, force=True
                )
                ignore_rerun_state = False
        if ckpt_world_size != run_world_size or ckpt_tp_pp != run_tp_pp or ckpt_dp != run_dp:
            print_rank_0("Job sharding has changed: Rerun state will be ignored")

        # [ModelOpt]: Initial loading from non-resume sharded checkpoint to a Distillation Model
        # will result in key mismatch with loss modules potentially containing parameters, since
        # it requires generating a state_dict before loading. Here we hide those modules if present.
        with (
            contextlib.ExitStack() as stack
        ):  # Allows multiple context managers for each model shard
            if args.finetune and hasattr(model[0], "hide_loss_modules"):
                for m in model:
                    stack.enter_context(m.hide_loss_modules())
            load_kwargs['sharded_state_dict'] = generate_state_dict(
                args,
                model,
                gen_sd_optim,
                gen_sd_opt_param_scheduler,
                gen_sd_rng_state,
                optim_sd_kwargs=optim_sd_kwargs,
                model_sd_kwargs=model_sd_kwargs,
                rerun_state=gen_sd_rerun_state,
            )
    elif args.ckpt_format == "torch_dcp":
        model_sd = model[0].state_dict()
        optimizer_sd = optimizer.state_dict(is_loading=True)
        sharded_state_dict = {
            "model": model_sd,
            "optimizer": optimizer_sd,
            "args": None,
            "iteration": 1,
            "rng_state": get_rng_state(args.ckpt_format),
            "checkpoint_version": None,
            "opt_param_scheduler": opt_param_scheduler.state_dict(),
            "num_floating_point_operations_so_far": 0,
        }
        load_kwargs["sharded_state_dict"] = sharded_state_dict
    elif args.ckpt_format == "fsdp_dtensor":
        reader = FileSystemReader(get_load_checkpoint_path_by_args(args))
        try:
            state_dict_metadata = reader.read_metadata().state_dict_metadata
        except FileNotFoundError:
            state_dict_metadata = {}

        gen_sd_rerun_state = None
        gen_sd_opt_param_scheduler = None
        gen_sd_rng_state = None
        gen_sd_optim = None
        if not args.finetune:
            if "rerun_state_machine" in state_dict_metadata:
                gen_sd_rerun_state = get_rerun_state_machine().state_dict(
                    data_iterator=None, ckpt_format=ckpt_format, force=True
                )
            if not args.no_load_rng:
                gen_sd_rng_state = get_rng_state(args.ckpt_format)
            if not args.no_load_optim:
                gen_sd_optim = optimizer
                gen_sd_opt_param_scheduler = opt_param_scheduler

        optim_sd_kwargs = dict(metadata=_build_sharded_state_dict_metadata(args), is_loading=True)

        load_kwargs["sharded_state_dict"] = generate_state_dict(
            args,
            model=model,
            optimizer=gen_sd_optim,
            opt_param_scheduler=gen_sd_opt_param_scheduler,
            rng_state=gen_sd_rng_state,
            optim_sd_kwargs=optim_sd_kwargs,
            rerun_state=gen_sd_rerun_state,
            iteration=1,
        )
    # Apply PEFT resume filtering (common across all checkpoint formats)
    is_peft_resume = (
        peft_class is not None
        and args.load is not None
        and load_dir == args.load
        and load_dir != args.pretrained_checkpoint
        and not args.finetune
    )
    if is_peft_resume and "sharded_state_dict" in load_kwargs:
        load_kwargs["sharded_state_dict"] = apply_peft_adapter_filter_to_state_dict(
            load_kwargs["sharded_state_dict"], peft_class
        )
    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir,
        args,
        rank0=False,
        checkpointing_context=checkpointing_context,
        ignore_ckpt_step=ignore_ckpt_step,
        **load_kwargs,
    )

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Convert to regular torch tensor to DTensor.
    if ckpt_type == CheckpointType.LEGACY and args.ckpt_format == "torch_dcp":
        dtensor_state_dict = _to_dtensor(ddp_model, state_dict["model"])
        state_dict["model"] = dtensor_state_dict

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0(
                    'A metadata file exists but unable to load '
                    'iteration from checkpoint {}, exiting'.format(checkpoint_name)
                )
                sys.exit()
    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    if load_arg == 'load':
        if 'args' in state_dict and not args.finetune:
            checkpoint_args = state_dict['args']
            check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args, 'consumed_train_samples', 0)
            args.skipped_train_samples = getattr(checkpoint_args, 'skipped_train_samples', 0)
            update_num_microbatches(consumed_samples=args.consumed_train_samples, verbose=True)
            args.consumed_valid_samples = getattr(checkpoint_args, 'consumed_valid_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    strict = False if args.retro_add_retriever else strict
    if not skip_load_to_model_and_opt:
        load_strict = False if is_peft_resume else strict

        if len(ddp_model) == 1:
            _load_model_state_dict(ddp_model[0], state_dict['model'], load_strict)
        else:
            for i in range(len(ddp_model)):
                # If there is no corresponding model, this is an empty pipeline stage.
                model_key = 'model%d' % i
                if model_key not in state_dict:
                    continue
                _load_model_state_dict(ddp_model[i], state_dict[model_key], load_strict)
    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if (
                not skip_load_to_model_and_opt
                and optimizer is not None
                and not optimizer.is_stub_optimizer
            ):
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            # For distributed checkpoint it's already loaded in load_state_dict above
            is_torch_dist = ckpt_format == "torch_dist"
            if (
                args.use_distributed_optimizer
                and not is_torch_dist
                and ckpt_format not in ["torch_dcp", "fsdp_dtensor"]
                and optimizer is not None
            ):
                # NOTE: this is a manual read of the tracker file.
                # This code should not be reached when reading from a non_persistent checkpoint
                assert not is_torch_dist
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(
                    model_checkpoint_name
                )
                optimizer.load_parameter_state(
                    optim_checkpoint_name,
                    update_legacy_format=args.ckpt_convert_update_legacy_dist_opt_format,
                )

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError as e:
            print_rank_0(
                'Unable to load optimizer from checkpoint {}. '
                'Specify --no-load-optim or --finetune to prevent '
                'attempting to load the optimizer state, '
                'exiting ...'.format(checkpoint_name)
            )
            raise e
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            if args.load_main_params_from_ckpt:
                optimizer.reload_model_params(state_dict=state_dict)
            else:
                optimizer.reload_model_params()

    # rerun state
    if not ignore_rerun_state:
        try:
            if 'rerun_state_machine' in state_dict:
                get_rerun_state_machine().load_state_dict(state_dict['rerun_state_machine'])
        except Exception as e:
            print(f"Unable to restore RerunMachine from checkpoint: {e}. Skipping.")

    # rng states.
    if not release and not args.finetune and not args.no_load_rng and not ignore_rng_state:
        try:
            if 'rng_state' in state_dict:
                if args.ckpt_format == "fsdp_dtensor":
                    # FSDP DTensor checkpoints store rng_state in a different format.
                    tp_rank = mpu.get_tensor_model_parallel_rank()
                    pp_rank = mpu.get_pipeline_model_parallel_rank()
                    if f"({pp_rank}, {tp_rank})" in state_dict['rng_state']:
                        rng_state = state_dict['rng_state'][f"({pp_rank}, {tp_rank})"]
                    else:
                        print("WARNING: RNG state not found for current TP/PP rank")
                        rng_state = next(iter(state_dict['rng_state'].values()))
                else:
                    rng_state = state_dict['rng_state']

                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = rng_state[mpu.get_data_parallel_rank()]
                else:
                    rng_state = rng_state[0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0(
                'Unable to load rng state from checkpoint {}. '
                'Specify --no-load-rng or --finetune to prevent '
                'attempting to load the rng state, '
                'exiting ...'.format(checkpoint_name)
            )
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f'  successfully loaded checkpoint from {load_dir} '
        f'[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, '
        f'p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] '
        f'at iteration {iteration}'
    )

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() or is_last_rank():
        wandb_utils.on_load_checkpoint_success(checkpoint_name, load_dir)

    torch.cuda.empty_cache()

    if iteration > 0:
        # Notify FT that a checkpoint was loaded.
        is_local_chkpt = ckpt_type == CheckpointType.LOCAL
        ft_integration.on_checkpoint_loaded(is_local_chkpt=is_local_chkpt)

    return iteration, num_floating_point_operations_so_far
