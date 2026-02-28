"""
Pretrain utilities.
Modified from Megatron-LM/megatron/training.py, https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training.py
"""
import os
import dataclasses
import gc
from datetime import datetime, timedelta
import logging
import sys
import re

try:
    from nvidia_resiliency_ext.inprocess import CallWrapper
except ImportError:
    CallWrapper = type(None)

from megatron.training.log_handler import CustomHandler

from typing import Optional
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)

import time

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from collections import OrderedDict

from megatron.core import mpu
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
)
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
    update_num_microbatches,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
)


from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.distributed import (
    DistributedDataParallel as DDP,
    finalize_model_grads,
)
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as megatron_FSDP
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.training.global_vars import get_energy_monitor

from megatron.core.parallel_state import (
    update_pg_timeout
)

from megatron.training import (
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    print_rank_0,
    print_rank_last,
    ft_integration,
)
from megatron.training.initialize import (
    write_args_to_tensorboard,
    set_jit_fusion_options,
)
from megatron.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    checkpoint_exists,
)
from megatron.training.utils import (
    calc_params_l2_norm,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.training import (
    append_to_progress_log,
    print_datetime,
    build_train_valid_test_data_iterators,
    evaluate_and_print_results,
    num_floating_point_operations,
    get_start_time_from_progress_log,
    get_model,
    get_optimizer_param_scheduler,
    preprocess_common_state_dict,
    should_disable_forward_pre_hook,
    disable_forward_pre_hook,
    enable_forward_pre_hook,
    dummy_train_step,
    post_training_step_callbacks,
    checkpoint_and_decide_exit,
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

from omni_training.utils import get_args, constants
from .initialize import initialize_aiak_megatron
from omni_training.data.dp_balance.wrapper.dp_balance.training_wrapper import (
    train_step_decorator,
    train_log_decorator
)


try:
    from inspector.hooks import register_hooks
    HAS_INSPECTOR = True
except ImportError:
    HAS_INSPECTOR = False

stimer = StragglerDetector()


@torch.no_grad()
def update_ema(ema_model, model, rate=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(rate).add_(param.data, alpha=1 - rate)


def enable_memory_history_record(memory_snapshot_path):
    """Enable memory history record"""
    torch.cuda.memory._record_memory_history(
        True,
        # keep 100,000 alloc/free events from before the snapshot
        trace_alloc_max_entries=100000,
        # record stack information for the trace events
        trace_alloc_record_context=True,
    )

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print("saving allocated state during OOM")
        snapshot = torch.cuda.memory._snapshot()
        from pickle import dump

        dump(
            snapshot,
            open(
                f"oom_rank-{torch.distributed.get_rank()}_{memory_snapshot_path}", "wb"
            ),
        )

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


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
                logging.info(f"Invalid freeze_parameters_regex '{freeze_parameters_regex}': {e}")
                raise

            for n, p in model_module.named_parameters():
                if pattern.search(n):
                    p.requires_grad = False

    # Only log checking info if freezing enable
    if freeze_parameters or freeze_parameters_regex:
        logging.info(
            "<Freezing model parameters> \n"
            + [sorted(f"FROZEN: {n}" for m in model for n, p in m.named_parameters() if not p.requires_grad)]
            + "\n</Freezing model parameters>"
            + "\n\n"
            + "<Trainable model parmeters> \n"
            + [sorted(f"TRAINABLE: {n}" for m in model for n, p in m.named_parameters() if p.requires_grad)]
            + "\n</Trainable model parmeters>"
        )


def add_hooks(model, args, prefix):
    """add hooks for model, only enable hooks when open the switch: enable-log-tensor"""
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


def pretrain(
    train_args,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper: Optional[CallWrapper] = None,
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
        store: an optional instance of torch.distributed.Store, to be used by
            torch.distributed.init_process_group
        inprocess_call_wrapper: an optional instance of inprocess.CallWrapper,
            it is automatically injected when in-process restart is in use
    """

    if inprocess_call_wrapper is not None:
        iteration = inprocess_call_wrapper.iteration
        store = torch.distributed.PrefixStore(str(iteration), store)

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_aiak_megatron(
        args=train_args,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=store,
    )

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Initialize fault tolerance
    # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
    if args.enable_ft_package:
        ft_integration.setup(args)
        ft_integration.maybe_setup_simulated_fault()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor(
        [_TRAIN_START_TIME], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)

    _TRAIN_START_TIME = start_time_tensor.item()

    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    # enable memory histroy record
    if hasattr(args, "record_memory_history") and args.record_memory_history:
        enable_memory_history_record(args.memory_snapshot_path)

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == "local":
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
                LocalCheckpointManager,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import (
                parse_group_sequence,
                GroupWrapper,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
                CliqueReplicationStrategy,
            )
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvidia_resiliency_ext' module is required for local "
                "checkpointing but was not found. Please ensure it is installed."
            )

        if args.replication:
            repl_strategy = CliqueReplicationStrategy.from_replication_params(
                args.replication_jump, args.replication_factor
            )
        else:
            repl_strategy = None

        checkpointing_context = {
            "local_checkpoint_manager": LocalCheckpointManager(
                args.non_persistent_local_ckpt_dir, repl_strategy=repl_strategy
            )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model, ema, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context
    )

    if args.enable_log_tensor and HAS_INSPECTOR:
        # following only for trace tesnors
        # debug infos
        unwrap_models = unwrap_model(model)
        index = 0
        for tmp_model in unwrap_models:
            prefix = "chunk" + str(index)
            add_hooks(tmp_model, args, prefix)
            index += 1

    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate scheduler are built")
    config = get_model_config(model[0])

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )

    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(
        ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
        barrier=True,
    )

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        # Add job name to the wandb config to make it easier to run more singleton dependency jobs.
        wandb_writer.config.update({'slurm_job_name': os.getenv("SLURM_JOB_NAME", "N/A")})

    if not args.skip_train:
        print_rank_0("training ...")

        if args.dataloader_type == "cyclic" and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func=forward_step_func,
                model=model,
                ema=ema,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                train_data_iterator=train_data_iterator,
                valid_data_iterator=valid_data_iterator,
                process_non_loss_data_func=process_non_loss_data_func,
                config=config,
                checkpointing_context=checkpointing_context,
                non_loss_data_func=non_loss_data_func,
            )

        print_datetime("after training is done")

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                num_floating_point_operations_so_far=num_floating_point_operations_so_far,
                checkpointing_context=checkpointing_context,
                train_data_iterator=train_data_iterator,
                preprocess_common_state_dict_fn=preprocess_common_state_dict,
            )

            if args.enable_ema and ema is not None:
                save_checkpoint(
                    iteration=iteration,
                    model=ema,
                    optimizer=None,
                    opt_param_scheduler=None,
                    num_floating_point_operations_so_far=num_floating_point_operations_so_far,
                    save_arg="save_ema",
                )

    else:
        print_rank_0("skipping training (--skip-train is on) ...")

        iteration = args.iteration

    if args.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    if args.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    ft_integration.shutdown()


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

    def provider_with_freeze(*p_args, **p_kwargs):
        m = model_provider_func(*p_args, **p_kwargs)

        # m can be a Module or list/tuple of Modules depending on PP/VPP.
        mods = m if isinstance(m, (list, tuple)) else [m]
        freeze_parameters(mods, args.freeze_parameters, args.freeze_parameters_regex)
        return m

    model = get_model(provider_with_freeze, model_type)
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

    # Control whether to force every parameter into the weight‑decay group.
    # Legacy default (flag unset) keeps the old behavior: force everything.
    # When --force-all-weight-decay true/false is provided, we respect that choice.
    if getattr(args, "force_all_weight_decay", None):
        no_wd_decay_cond = False,

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

    if (
        args.load is not None or args.pretrained_checkpoint is not None
    ) and not args.moe_use_upcycling:
        timers("load-checkpoint", log_level=0).start(barrier=True)

        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model,
            optimizer,
            opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and getattr(args, "use_torch_fsdp2", False)
            and args.ckpt_format == "torch_dist",
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

        save_checkpoint(
            args.iteration,
            model,
            optimizer,
            opt_param_scheduler,
            args.num_floating_point_operations_so_far,
            preprocess_common_state_dict_fn=preprocess_common_state_dict,
        )

        print_rank_0(
            "> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format)
        )
        torch.distributed.barrier()
        exit()

    return model, ema, optimizer, opt_param_scheduler


def compute_throughputs_and_append_to_progress_log(
    iteration, num_floating_point_operations_so_far
):
    """Compute throughputs and append to progress log."""
    args = get_args()
    if args.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    global _TRAIN_START_TIME
    job_throughput = (
        num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
    ) / ((time.time() - _TRAIN_START_TIME) * 10**12 * args.world_size)

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = (
        num_floating_point_operations_so_far - start_num_floating_point_operations
    ) / (elapsed_time * 10**12 * args.world_size)

    tokens_so_far = args.consumed_train_samples * args.seq_length
    saved_ckpt_prefix = (
        "Saving async checkpoint" if args.async_save else "Saved checkpoint"
    )
    append_to_progress_log(
        f"{saved_ckpt_prefix}\tIteration: {iteration}\t"
        f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
        f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
        f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
        f"Tokens (in billions): {tokens_so_far / 10**9:.2f}"
    )


def save_checkpoint_and_time(
    iteration,
    model,
    ema,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context,
    non_persistent_ckpt=False,
    train_data_iterator=None,
):
    """Save checkpoint and time."""
    args = get_args()
    timers = get_timers()
    energy_monitor = get_energy_monitor()

    # Stop timer to get accurate train interval time and exclude checkpointing duration
    timers("interval-time").stop()
    energy_monitor.pause()
    # Extra barrier is added to make sure all ranks report the max time.
    timer_key = (
        "save-checkpoint-non-persistent" if non_persistent_ckpt else "save-checkpoint"
    )
    timers(timer_key, log_level=0).start(barrier=True)

    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    save_checkpoint(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        num_floating_point_operations_so_far=num_floating_point_operations_so_far,
        checkpointing_context=checkpointing_context,
        non_persistent_ckpt=non_persistent_ckpt,
        train_data_iterator=train_data_iterator,
        preprocess_common_state_dict_fn=preprocess_common_state_dict,
    )
    if args.fp8:
        # Run garbage collection after checkpoint saving to free memory from
        # dequantized bf16 tensors that were temporarily created during fp8
        # model checkpoint saving.
        gc.collect()
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)

    if args.enable_ema and ema is not None:
        save_checkpoint(
            iteration=iteration,
            model=ema,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=num_floating_point_operations_so_far,
            save_arg="save_ema",
        )

    timers(timer_key).stop(barrier=True)
    timers.log([timer_key])

    if args.log_progress and not non_persistent_ckpt:
        compute_throughputs_and_append_to_progress_log(
            iteration, num_floating_point_operations_so_far
        )

    # Recover timing
    energy_monitor.resume()
    timers("interval-time", log_level=0).start(barrier=True)


@train_step_decorator
def train_step(
    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        adjust_tensor_shapes_fn = None
        # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
        # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
        # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            for optim_instance in optimizer.chained_optimizers:
                if isinstance(optim_instance, DistributedOptimizer):
                    optim_instance._copy_main_params_to_param_buffer()

        # Forward pass.
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )

    should_checkpoint, should_exit, exit_code = (
        rerun_state_machine.should_checkpoint_and_exit()
    )
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
            num_zeros_in_grad
        )

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            # Special handling for total_inputs which may be int type
            if key == "total_inputs":
                total = sum(x[key] for x in losses_reduced)
                loss_reduced[key] = total
                continue
            
            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:
                if args.training_phase == constants.TrainingPhase.SFT and not args.legacy_reporting_loss_reduction:
                    # in mcore the normalization happens on micro batch instead of global
                    val = torch.vstack(val)
                    val = val[:, 0] / val[:, 1]
                    val = val.mean()
                    torch.distributed.all_reduce(
                        val,
                        group=mpu.get_data_parallel_group(with_context_parallel=True)
                    )
                    val /= torch.distributed.get_world_size(
                        group=mpu.get_data_parallel_group(with_context_parallel=True)
                    )
                    loss_reduced[key] = val
                else:
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    val = torch.vstack(val).sum(dim=0)
                    torch.distributed.all_reduce(
                        val,
                        group=mpu.get_data_parallel_group(with_context_parallel=True)
                    )
                    loss_reduced[key] = val[0] / val[1]
            elif val[0].numel() == 1:
                # legacy behavior, we average over the number of microbatches
                val = torch.cat(val).mean()
                # since we remove the dpcp allreduce in loss func
                torch.distributed.all_reduce(
                    val,
                    group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                val /= torch.distributed.get_world_size(
                    group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                loss_reduced[key] = val
            else:
                raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
    return (
        {},
        skipped_iter,
        should_checkpoint,
        should_exit,
        exit_code,
        grad_norm,
        num_zeros_in_grad,
    )


@train_log_decorator
def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    decoupled_learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    energy_monitor = get_energy_monitor()
    # total inputs
    total_inputs = loss_dict.pop("total_inputs", None)
    total_inputs = total_inputs.item() if total_inputs is not None else None

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0

    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )

    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(
                    key, torch.tensor([0.0], dtype=torch.float, device="cuda")
                )
                + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan

    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )

    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "all-grads-sync",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
        "update-ema",
    ]

    # Calculate batch size.
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )

    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log(
                {"samples vs steps": args.consumed_train_samples}, iteration
            )

        writer.add_scalar("learning-rate", learning_rate, iteration)
        writer.add_scalar(
            "learning-rate vs samples", learning_rate, args.consumed_train_samples
        )
        if wandb_writer:
            wandb_writer.log({"learning-rate": learning_rate}, iteration)
        if args.decoupled_lr is not None:
            writer.add_scalar(
                "decoupled-learning-rate", decoupled_learning_rate, iteration
            )
        if args.skipped_train_samples > 0:
            writer.add_scalar(
                "skipped-train-samples", args.skipped_train_samples, iteration
            )
            if wandb_writer:
                wandb_writer.log(
                    {"skipped-train-samples": args.skipped_train_samples}, iteration
                )
        writer.add_scalar("batch-size", batch_size, iteration)
        writer.add_scalar(
            "batch-size vs samples", batch_size, args.consumed_train_samples
        )
        if wandb_writer:
            wandb_writer.log({"batch-size": batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(
                key + " vs samples", loss_dict[key], args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)

        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar(
                "loss-scale vs samples", loss_scale, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"loss-scale": loss_scale}, iteration)

        if args.log_world_size_to_tensorboard:
            writer.add_scalar("world-size", args.world_size, iteration)
            writer.add_scalar(
                "world-size vs samples", args.world_size, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"world-size": args.world_size}, iteration)

        if grad_norm is not None:
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "grad-norm vs samples", grad_norm, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"grad-norm": grad_norm}, iteration)

        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros vs samples", num_zeros_in_grad, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"num-zeros": num_zeros_in_grad}, iteration)

        if params_norm is not None:
            writer.add_scalar("params-norm", params_norm, iteration)
            writer.add_scalar(
                "params-norm vs samples", params_norm, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"params-norm": params_norm}, iteration)

        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_names = []
        if "aux_loss" in args.moe_router_load_balancing_type:
            track_names.append("load_balancing_loss")
        if "seq_aux_loss" in args.moe_router_load_balancing_type:
            track_names.append("seq_load_balancing_loss")
        if "global_aux_loss" in args.moe_router_load_balancing_type:
            track_names.append("global_load_balancing_loss")
        if args.moe_z_loss_coeff is not None:
            track_names.append("z_loss")
        track_moe_metrics(
            loss_scale=moe_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
            per_layer_logging=args.moe_per_layer_logging,
            force_initialize=True,
            track_names=track_names,
            num_layers=args.num_layers,
            moe_layer_freq=args.moe_layer_freq,
            mtp_num_layers=args.mtp_num_layers,
        )
    if args.mtp_num_layers is not None:
        mtp_loss_scale = 1 / get_num_microbatches()
        MTPLossLoggingHelper.track_mtp_metrics(
            mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict
        )
    # Track sparse attention indexer loss
    if args.dsa_indexer_loss_coeff is not None and args.dsa_indexer_loss_coeff > 0:
        indexer_loss_scale = 1 / get_num_microbatches()
        from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=indexer_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )
    if iteration % args.log_interval == 0:
        if args.record_memory_history and is_last_rank():
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            with open(args.memory_snapshot_path, "wb") as f:
                dump(snapshot, f)

        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        if total_inputs is None:
            token_per_sec = (
                int(args.seq_length)
                * int(args.global_batch_size)
                / elapsed_time_per_iteration
            )
        else:
            token_per_sec = total_inputs / elapsed_time_per_iteration
        token_throughput = token_per_sec / args.world_size

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size
        )

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar(
                    "iteration-time", elapsed_time_per_iteration, iteration
                )
            if wandb_writer:
                wandb_writer.log(
                    {"iteration-time": elapsed_time_per_iteration}, iteration
                )

        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
        log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)

        if args.skipped_train_samples > 0:
            log_string += " skipped samples: {:12d} |".format(
                args.skipped_train_samples
            )

        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time_per_iteration * 1000.0
        )
        log_string += f" throughput (token/sec/GPU): {token_throughput:.1f} |"

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar(
                    "Token throughput (per-sec-per-GPU)", token_throughput, iteration
                )
            if wandb_writer:
                wandb_writer.log(
                    {"Token throughput (per-sec-per-GPU)": token_throughput}, iteration
                )

        if args.log_throughput:
            log_string += f" flop throughput (TFLOP/sec/GPU): {throughput:.1f} |"
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar(
                        "TFLOP throughput (per-sec-per-GPU)", throughput, iteration
                    )
                if wandb_writer:
                    wandb_writer.log(
                        {"TFLOP throughput (per-sec-per-GPU)": throughput}, iteration
                    )

        if args.log_energy:
            energy = (energy_monitor.lap() / total_iterations) / args.world_size
            power = energy / elapsed_time_per_iteration
            log_string += f' energy per GPU (J/iter/GPU): {energy:.1f} |'
            log_string += f' power per GPU (W/GPU): {power:.1f} |'
            if writer:
                writer.add_scalar('iter-energy/gpu', energy, iteration)
                writer.add_scalar('power/gpu', power, iteration)
            if wandb_writer:
                wandb_writer.log({'iter-energy/gpu': energy}, iteration)
                wandb_writer.log({'power/gpu': power}, iteration)
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += f" learning rate: {learning_rate:.6E} |"
        if args.decoupled_lr is not None and (
            mpu.is_pipeline_first_stage(ignore_virtual=True)
            or mpu.is_pipeline_last_stage(ignore_virtual=True)
        ):
            assert decoupled_learning_rate is not None
            log_string += f" decoupled learning rate: {decoupled_learning_rate:.6E} |"
        else:
            assert decoupled_learning_rate is None

        log_string += f" global batch size: {batch_size:5d} |"
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.tensor(
                    [0.0], dtype=torch.float, device="cuda"
                )

        log_string += f" loss scale: {loss_scale:.1f} |"
        if grad_norm is not None:
            log_string += f" grad norm: {grad_norm:.6f} |"
        if num_zeros_in_grad is not None:
            log_string += f" num zeros: {num_zeros_in_grad} |"
        if params_norm is not None:
            log_string += f" params norm: {params_norm:.3f} |"

        if hasattr(args, "log_memory_stats") and args.log_memory_stats:
            mem_stats = torch.cuda.memory_stats()
            world_size = torch.distributed.get_world_size()
            allocated_bytes = torch.tensor(
                [mem_stats['allocated_bytes.all.current']],
                dtype=torch.float,
                device='cuda',
            )
            max_allocated_bytes = torch.tensor(
                [mem_stats['allocated_bytes.all.peak']],
                dtype=torch.float,
                device='cuda',
            )
            # sum across all ranks
            torch.distributed.all_reduce(allocated_bytes, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(max_allocated_bytes, op=torch.distributed.ReduceOp.SUM)
            avg_allocated_mb = allocated_bytes.item() / world_size / 1024 / 1024
            avg_max_allocated_mb = max_allocated_bytes.item() / world_size / 1024 / 1024
            log_string += f" mem-allocated-bytes-avg(MB): {avg_allocated_mb:.2f} |"
            log_string += f" mem-max-allocated-bytes-avg(MB): {avg_max_allocated_mb:.2f} |"

        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[nan_iters_key]
        )

        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        
        print_rank_last(log_string)

        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(
                    args, num_microbatches=num_microbatches, verbose=True
                )
            report_memory(f"(after {iteration} iterations)")
            report_memory_flag = False

    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and iteration % args.tensorboard_log_interval == 0:
        timers.write(timers_to_log, writer, iteration, reset=False, normalizer=total_iterations)
        timers.write(timers_to_log, wandb_writer, iteration, normalizer=args.log_interval, reset=False)
    
    if args.timing_log_level < 1 and iteration % args.detail_log_interval == 0:
        # Only the time for one iteration is recorded, so the normalizer is set to 1.
        timers.log(timers_to_log, None, normalizer=1)
    elif iteration % args.log_interval == 0:
        timers.log(
            timers_to_log, None, normalizer=args.log_interval
        )

    return report_memory_flag


def train(
    forward_step_func,
    model,
    ema,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
    checkpointing_context,
    non_loss_data_func,
):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    energy_monitor = get_energy_monitor()

    if args.run_workload_inspector_server:
        try:
            from workload_inspector.utils.webserver import run_server
            import threading

            threading.Thread(
                target=run_server, daemon=True, args=(torch.distributed.get_rank(),)
            ).start()
        except ModuleNotFoundError:
            print_rank_0("workload inspector module not found.")

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    if args.enable_ema and ema is not None:
        for model_module in ema:
            model_module.eval()

    # Tracking loss.
    total_loss_dict = {}
    # Iterations.
    iteration = args.iteration

    # Make sure rerun_state_machine has the right iteration loaded from checkpoint.
    rerun_state_machine = get_rerun_state_machine()
    if rerun_state_machine.current_iteration != iteration:
        print_rank_0(f"Overwriting rerun_state_machine.current_iteration from "
                     f"{rerun_state_machine.current_iteration} to {iteration}...")
        rerun_state_machine.current_iteration = iteration

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params.
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], (megatron_FSDP, DDP)) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [
                model_chunk.start_grad_sync for model_chunk in model
            ]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    if args.log_energy:
        energy_monitor.setup()
        energy_monitor.resume()

    timers("interval-time", log_level=0).start(barrier=True)
    print_datetime("before the start of training step")
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert (
            args.manual_gc_interval >= 0
        ), "Manual garbage collection interval should be laerger than or equal to 0."
        gc.disable()
        gc.collect()

    # Singleton initialization of straggler detector.
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(
            world,
            rank,
            mmcnt=mmcnt,
            enabled=not args.disable_straggler_on_startup,
            port=args.straggler_ctrlr_port,
        )
    num_floating_point_operations_since_last_log_event = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0
    # Wrap forward_backward_func for Full iteration CUDA graph
    forward_backward_func = get_forward_backward_func()
    if args.cuda_graph_impl == "local" and args.cuda_graph_scope == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(forward_backward_func, 
                                                     cuda_graph_warmup_steps=args.cuda_graph_warmup_steps)

    prof = None
    if (
        args.profile
        and torch.distributed.get_rank() in args.profile_ranks
        and args.use_pytorch_profiler
    ):
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.tensorboard_dir
            ),
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if args.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(
            model, cross_check=True
        ), "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")

    # Initialize CUDA Graphs helper.
    if args.cuda_graph_impl == "transformer_engine":
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=config,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            optimizers=[optimizer],
        )
    while iteration < args.train_iters:
        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
            if args.use_pytorch_profiler:
                prof.step()
            elif iteration == args.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True)
        # Update the timeout for all process groups after initialization
        # We update the timeout after the first successful iteration,
        # which takes longer than others usually
        if args.distributed_timeout_seconds_after_init is not None and iteration == start_iteration + 1:
            # TODO: some dynamic timeout setting is required
            # based on the iteration time considering interval-based steps (e.g. eval, checkpoint)
            # e.g. timeout for normal iterations vs timeout for iterations with checkpoint
            # this timeout is triggered when there's no collective communication
            # for the duration of timeout
            update_pg_timeout(timedelta(seconds=args.distributed_timeout_seconds_after_init))
        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        # Standard microbatch update (sequence packing overrides this in rl_utils.py)
        update_num_microbatches(
            args.consumed_train_samples, consistency_check=False, verbose=True
        )
        # Skip automatic checkpoint on microbatch changes when sequence packing is active
        # as it intentionally reconfigures microbatches
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, (
                f"Number of microbatches should be increasing due to batch size rampup; "
                f"instead going from {num_microbatches} to {get_num_microbatches()}"
            )
            if args.save is not None:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    ema,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(
            args.consumed_train_samples, consistency_check=True, verbose=True
        )

        # Capture CUDA Graphs.
        if (
            args.cuda_graph_impl == "transformer_engine"
            and iteration == args.cuda_graph_warmup_steps
        ):
            if iteration > start_iteration and should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model, param_sync=False)
            cuda_graph_helper.create_cudagraphs()
            if iteration > start_iteration and should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                cuda_graph_helper.cuda_graph_set_manual_hooks()

        # Completely skip iteration if needed.
        if iteration in args.iterations_to_skip:
            # Dummy train_step to fast forward train_data_iterator.
            dummy_train_step(train_data_iterator)
            if iteration == start_iteration:
                start_iteration = iteration + 1
            iteration += 1
            batch_size = (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * get_num_microbatches()
            )
            args.consumed_train_samples += batch_size
            args.skipped_train_samples += batch_size
            continue

        if (
            args.log_detail
            and args.timing_log_level < 1
            and (iteration + 1) % args.detail_log_interval == 0
        ):
            timers.set_show_detail_log(True)

        args.curr_iteration = iteration
        ft_integration.on_training_step_start()
        (
            loss_dict,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        ) = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
        )
        ft_integration.on_training_step_end()

        if should_checkpoint:
            save_checkpoint_and_time(
                iteration,
                model,
                ema,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(model)
                    config.param_sync_func = param_sync_func
                    pre_hook_enabled = True
                    # Set the manual hooks here since it's not set right after the capturing.
                    if (
                        args.cuda_graph_impl == "transformer_engine"
                        and iteration == args.cuda_graph_warmup_steps
                    ):
                        cuda_graph_helper.cuda_graph_set_manual_hooks()

        iteration += 1
        batch_size = (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )
        args.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = (
            get_current_global_batch_size() - get_current_running_global_batch_size()
        )
        if args.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        args.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = num_floating_point_operations(
            args, batch_size
        )
        num_floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += (
            num_floating_point_operations_in_batch
        )

        # update ema
        if args.enable_ema and ema is not None:
            timers("update-ema", log_level=1).start(barrier=True)
            for e, m in zip(ema, model):
                update_ema(e, m, rate=args.ema_decay)
            timers("update-ema").stop()

        # Logging.
        if not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if len(param_group["params"]) == 0:
                continue
            if param_group["is_decoupled_lr"]:
                decoupled_learning_rate = param_group["lr"]
            else:
                learning_rate = param_group["lr"]

        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )

        # Evaluation.
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            if args.log_energy:
                energy_monitor.pause()
            timers("interval-time").stop()
            if should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model)
                pre_hook_enabled = False

            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f"iteration {iteration}"
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                verbose=False,
                write_to_tensorboard=True,
                non_loss_data_func=non_loss_data_func,
            )

            eval_duration += timers("eval-time").elapsed()
            eval_iterations += args.eval_iters
            timers("eval-time").stop()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)

            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                pre_hook_enabled = True

            timers("interval-time", log_level=0).start(barrier=True)
            if args.log_energy:
                energy_monitor.resume()

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
            prof,
            num_floating_point_operations_since_last_log_event,
        )

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
        )
        if should_exit:
            break

        # The timer will call cuda.sync, causing the asynchronous stream to be ineffective
        # and leading to a decrease in performance. Therefore, considering performance,
        # only the time of a single iteration is recorded to minimize the impact on performance.
        timers.set_show_detail_log(False)

    # Flush TensorBoard, WandB writers and one-logger.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    ft_integration.on_checkpointing_start()
    # This will finalize all unfinalized async request and terminate
    # a persistent async worker if persistent ckpt worker is enabled
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)
    if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
        ft_integration.get_rank_monitor_client().shutdown_workload_monitoring()

    if args.log_energy:
        energy_monitor.lap()
        total_energy = energy_monitor.get_total()
        print_rank_0(f"Total training energy (GPU): {total_energy / 1e6} MJ")
        energy_monitor.shutdown()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()

        ft_integration.shutdown()
        sys.exit(exit_code)

    return iteration, num_floating_point_operations_so_far
