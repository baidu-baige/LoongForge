import datetime
import functools
import json
import math
import os
import shutil
import time
from types import MethodType
from typing import Any

import accelerate
import torch
from accelerate import Accelerator, DistributedType, skip_first_batches
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    TERecipeKwargs,
    TorchDynamoPlugin,
    release_memory,
    set_seed,
)
from accelerate.utils.dataclasses import FP8BackendType
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader

from .. import utils
from ..configs import load_config
from ..modules import ModuleDict
from ..optimizers import build_optimizer
from ..samplers import ParallelBatchSampler, build_sampler
from ..schedulers import build_scheduler
from ..strategies import EMAModel, module_auto_wrap_policy
from ..transforms import build_transform


class Trainer:
    """High-level training loop manager built on top of Accelerate.

    Coordinates dataloaders, models, optimizers, schedulers, checkpointing, logging, mixed precision, and optional EMA.
    """

    def __init__(
        self,
        project_dir: str,
        max_epochs: int = 0,
        max_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str | None = None,
        dynamo_config: dict | None = None,
        ddp_config: dict | None = None,
        fp8_config: dict | None = None,
        data_parallel_size: int = 1,
        log_with: str | list[str] | None = None,
        log_interval: int = 1,
        log_cpu_memory: bool = True,
        checkpoint_interval: int | float = 1,
        checkpoint_total_limit: int = -1,
        checkpoint_keeps: list[int] | None = None,
        checkpoint_save_optimizer: bool = True,
        checkpoint_safe_serialization: bool = False,
        checkpoint_strict: bool = True,
        activation_checkpointing: bool = False,
        activation_class_names: list[str] | None = None,
        with_ema: bool = False,
        allow_tf32: bool = True,
        seed: int = 6666,
        logging_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer and underlying Accelerator.

        Args:
            project_dir: Project working directory; checkpoints are saved under
                ``project_dir/models`` and logs under ``logging_dir``.
            max_epochs: Max number of epochs (mutually exclusive with ``max_steps``).
            max_steps: Max number of steps (mutually exclusive with ``max_epochs``).
            gradient_accumulation_steps: Micro-steps per optimizer step.
            mixed_precision: Precision mode, e.g. ``'fp16'``, ``'bf16'``, or ``'fp8'``.
            dynamo_config: Extra kwargs for how torch dynamo should be handled.
            ddp_config: Extra kwargs for DDP via Accelerate handlers.
            fp8_config: FP8 backend/recipe config when ``mixed_precision='fp8'``.
            data_parallel_size: In-process data parallel replication factor for batch sampler.
            log_with: Logger backends for Accelerate trackers.
            log_interval: Logging interval in steps.
            log_cpu_memory: Whether to log CPU memory usage.
            checkpoint_interval: Interval for saving checkpoints (int steps or float epochs if by-epoch).
            checkpoint_total_limit: Max number of checkpoints to retain; older are pruned.
            checkpoint_keeps: Specific epoch/step identifiers to keep when pruning.
            checkpoint_save_optimizer: Save optimizer states only for the latest checkpoint when False.
            checkpoint_safe_serialization: Use safetensors for model weights when True.
            checkpoint_strict: Strict key matching when loading model weights.
            activation_checkpointing: Enable activation checkpoint wrapping for selected classes.
            activation_class_names: Target class names to wrap when activation checkpointing is enabled.
            with_ema: Maintain EMA copies of model weights.
            allow_tf32: Enable TF32 on CUDA backends.
            seed: Random seed (> 0).
            logging_dir: Directory for Accelerate tracker logs (TensorBoard events);
                defaults to the non-empty ``AIHC_TENSORBOARD_LOG_PATH`` when set,
                otherwise ``project_dir/logs``. The text ``.log`` is also written here.
            **kwargs: Extra options such as grad clipping configuration.
        """
        assert seed > 0
        set_seed(seed)
        if project_dir.endswith('/'):
            project_dir = project_dir[:-1]
        project_name = os.path.basename(project_dir)
        if logging_dir is None:
            logging_dir = os.environ.get('AIHC_TENSORBOARD_LOG_PATH')
        if not logging_dir:
            logging_dir = os.path.join(project_dir, 'logs')
        else:
            logging_dir = os.path.expanduser(os.path.expandvars(logging_dir))
        project_config = ProjectConfiguration(
            project_dir=project_dir,
            logging_dir=logging_dir,
        )
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
        )
        if dynamo_config is not None:
            dynamo_plugin = TorchDynamoPlugin(**dynamo_config)
        else:
            dynamo_plugin = None
        kwargs_handlers = []
        if ddp_config is not None:
            ddp_handler = DistributedDataParallelKwargs(**ddp_config)
            kwargs_handlers.append(ddp_handler)
        if mixed_precision == 'fp8':
            if fp8_config is None:
                fp8_config = dict(backend='te', recipe_kwargs=dict())
            fp8_backend = fp8_config['backend']
            if fp8_backend == 'te':
                reset_accelerate_te(**fp8_config['recipe_kwargs'])
                fp8_recipe_handler = TERecipeKwargs()
                kwargs_handlers.append(fp8_recipe_handler)
            else:
                assert False
        # Build the core Accelerator with logging/dataloader settings and optional handlers
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=project_config,
            dataloader_config=dataloader_config,
            dynamo_plugin=dynamo_plugin,
            kwargs_handlers=kwargs_handlers,
        )
        self.accelerator.init_trackers(project_name)
        if allow_tf32:
            # Enable TF32 to improve throughput on Ampere+ GPUs while keeping numerical stability
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            log_name = 'train_{}.log'.format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.mixed_precision = mixed_precision
        self.data_parallel_size = data_parallel_size
        self.log_interval = log_interval
        self.log_cpu_memory = log_cpu_memory
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_total_limit = checkpoint_total_limit
        self.checkpoint_keeps = checkpoint_keeps
        self.checkpoint_save_optimizer = checkpoint_save_optimizer
        self.checkpoint_safe_serialization = checkpoint_safe_serialization
        self.checkpoint_strict = checkpoint_strict
        self.activation_checkpointing = activation_checkpointing
        self.activation_class_names = activation_class_names
        self.seed = seed
        self.kwargs = kwargs

        self.with_ema = with_ema
        self.ema_models = []

        self._dataloaders = []
        self._models = []
        self._optimizers = []
        self._schedulers = []

        if max_epochs > 0:
            assert max_steps == 0
            by_epoch = True
        else:
            assert max_steps > 0
            by_epoch = False
        self._by_epoch = by_epoch
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._cur_step = 0
        self._skip_batches = 0

        self._start_tic = None
        self._epoch_tic = None
        self._step_tic = None
        self._outputs = dict()
        self._loss_nan_count = 0

        if self.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.state.deepspeed_plugin.deepspeed_config['zero_force_ds_cpu_optimizer'] = False
        # Register trainer state for checkpoint save/load hooks
        self.accelerator.register_for_checkpointing(self)
        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)

    @property
    def project_dir(self) -> str:
        return self.accelerator.project_dir

    @property
    def logging_dir(self) -> str:
        return self.accelerator.logging_dir

    @property
    def model_dir(self) -> str:
        return os.path.join(self.project_dir, 'models')

    @property
    def distributed_type(self) -> DistributedType:
        return self.accelerator.distributed_type

    @property
    def num_processes(self) -> int:
        return self.accelerator.num_processes

    @property
    def process_index(self) -> int:
        return self.accelerator.process_index

    @property
    def local_process_index(self) -> int:
        return self.accelerator.local_process_index

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self) -> bool:
        return self.accelerator.is_local_main_process

    @property
    def is_last_process(self) -> bool:
        return self.accelerator.is_last_process

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def dtype(self) -> torch.dtype:
        if self.mixed_precision == 'fp16':
            return torch.float16
        if self.mixed_precision in ('bf16', 'fp8'):
            return torch.bfloat16
        else:
            return torch.float32

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.accelerator.gradient_accumulation_steps

    @property
    def dataloaders(self) -> list[DataLoader]:
        return self._dataloaders

    @property
    def dataloader(self) -> DataLoader:
        return self._dataloaders[0]

    @property
    def models(self) -> list[Module]:
        return self._models

    @property
    def model(self) -> Module:
        return self._models[0]

    @property
    def optimizers(self) -> list[Optimizer]:
        return self._optimizers

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizers[0]

    @property
    def schedulers(self) -> list[Any]:
        return self._schedulers

    @property
    def scheduler(self) -> Any:
        return self._schedulers[0]

    @property
    def data_size(self) -> int:
        return len(self.dataloader.dataset)

    @property
    def batch_size(self) -> int:
        if self.dataloader.batch_sampler is not None:
            batch_sampler = self.dataloader.batch_sampler
        else:
            batch_sampler = self.dataloader.sampler
        while True:
            if hasattr(batch_sampler, 'batch_sampler'):
                batch_sampler = batch_sampler.batch_sampler
            else:
                break
        if hasattr(batch_sampler, 'batch_size'):
            batch_size = batch_sampler.batch_size
        elif hasattr(batch_sampler, 'batch_sizes'):
            batch_size = min(batch_sampler.batch_sizes)
        else:
            assert False
        return batch_size * self.num_processes * self.gradient_accumulation_steps // self.data_parallel_size

    @property
    def epoch_size(self) -> int:
        return int(math.ceil((len(self.dataloader) + self._skip_batches) / self.gradient_accumulation_steps))

    @property
    def max_epochs(self) -> int:
        if self._max_epochs > 0:
            return self._max_epochs
        else:
            return int(math.ceil(self._max_steps / self.epoch_size))

    @property
    def max_steps(self) -> int:
        if self._max_steps > 0:
            return self._max_steps
        else:
            return self._max_epochs * self.epoch_size

    @property
    def cur_epoch(self) -> int:
        return int(math.ceil(self.cur_step / self.epoch_size))

    @property
    def cur_step(self) -> int:
        return self._cur_step

    def print(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    def state_dict(self) -> dict[str, int]:
        return {'step': self._cur_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cur_step = state_dict['step']

    @classmethod
    def load(cls, config_or_path: Any):
        config = load_config(config_or_path).copy()
        trainer = cls(project_dir=config.project_dir, **config.train)
        trainer.prepare(
            dataloaders=config.dataloaders.train,
            models=config.models.train if hasattr(config.models, 'train') else config.models,
            optimizers=config.optimizers,
            schedulers=config.schedulers,
        )
        return trainer

    def save_config(self, config: Any) -> None:
        if not self.is_main_process:
            return
        config = load_config(config)
        config_path = os.path.join(self.project_dir, 'config.json')
        config.save(config_path)

    def load_checkpoint(self, checkpoint: str | list[str] | None, models: Module | list[Module], strict: bool = True) -> None:
        if checkpoint is None:
            return
        checkpoint = self.get_checkpoint(checkpoint)
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]
        if not isinstance(models, list):
            models = [models]
        for i in range(len(checkpoint)):
            config_path = os.path.join(checkpoint[i], 'config.json')
            config = json.load(open(config_path, 'r'))
            class_name = config['_class_name']
            self.logger.info(f'Load {class_name} from {checkpoint[i]}')
            state_dict = utils.load_state_dict(checkpoint[i])
            flag = False
            for model in models:
                if model.__class__.__name__ == class_name:
                    mes = model.load_state_dict(state_dict, strict=strict)
                    if self.is_main_process and not strict:
                        self.logger.info(mes)
                    flag = True
                    break
            if not flag:
                raise ValueError(f'No model loaded by {checkpoint[i]}')

    def get_checkpoint(self, checkpoint: str | list[str] | None = None) -> str | list[str] | None:
        if checkpoint is None:
            checkpoints = os.listdir(self.model_dir)
            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
            if len(checkpoints) > 0:
                checkpoint = os.path.join(self.model_dir, checkpoints[-1])
            else:
                return None
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]
        for i in range(len(checkpoint)):
            if checkpoint[i].startswith('checkpoint'):
                checkpoint[i] = os.path.join(self.model_dir, checkpoint[i])
            assert os.path.exists(checkpoint[i])
        return checkpoint if len(checkpoint) > 1 else checkpoint[0]

    def remove_checkpoint(self, total_limit: int | None = None) -> None:
        if not self.is_main_process:
            return
        total_limit = total_limit or self.checkpoint_total_limit
        checkpoints = os.listdir(self.model_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
        if self.checkpoint_keeps is not None:
            # Keep only checkpoints not explicitly marked to preserve
            new_checkpoints = []
            for checkpoint in checkpoints:
                if self._by_epoch:
                    checkpoint_id = int(checkpoint.split('_')[-3])
                else:
                    checkpoint_id = int(checkpoint.split('_')[-1])
                if checkpoint_id not in self.checkpoint_keeps:
                    new_checkpoints.append(checkpoint)
            checkpoints = new_checkpoints
        if len(checkpoints) >= total_limit > 0:
            # Trim oldest checkpoints to satisfy total_limit
            num_to_remove = len(checkpoints) - self.checkpoint_total_limit + 1
            for checkpoint in checkpoints[:num_to_remove]:
                checkpoint = os.path.join(self.model_dir, checkpoint)
                self.logger.info('Remove checkpoint {}'.format(checkpoint))
                shutil.rmtree(checkpoint)
            checkpoints = checkpoints[num_to_remove:]
        if not self.checkpoint_save_optimizer:
            # Optionally drop optimizer states to save disk usage
            if len(checkpoints) > 0:
                checkpoints = checkpoints[:-1]
            for checkpoint in checkpoints:
                if self.distributed_type == DistributedType.DEEPSPEED:
                    checkpoint = os.path.join(self.model_dir, checkpoint, 'pytorch_model')
                    if os.path.exists(checkpoint):
                        shutil.rmtree(checkpoint)
                elif self.distributed_type == DistributedType.FSDP:
                    for i in range(len(self.optimizers)):
                        optimizer_name = 'optimizer.bin' if i == 0 else f'optimizer_{i}.bin'
                        checkpoint_i = os.path.join(self.model_dir, checkpoint, optimizer_name)
                        if os.path.exists(checkpoint_i):
                            os.remove(checkpoint_i)
                        checkpoint_i = os.path.join(self.model_dir, checkpoint, f'optimizer_{i}')
                        if os.path.exists(checkpoint_i):
                            shutil.rmtree(checkpoint_i)
                else:
                    for i in range(len(self.optimizers)):
                        optimizer_name = 'optimizer.bin' if i == 0 else f'optimizer_{i}.bin'
                        checkpoint_i = os.path.join(self.model_dir, checkpoint, optimizer_name)
                        if os.path.exists(checkpoint_i):
                            os.remove(checkpoint_i)

    def resume(self, checkpoint: str | list[str] | None = None) -> None:
        checkpoint = self.get_checkpoint(checkpoint)
        if checkpoint is None:
            return
        if self.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.load_state(checkpoint, load_module_strict=self.checkpoint_strict)
        else:
            self.accelerator.load_state(checkpoint, strict=self.checkpoint_strict)
        # Walk through nested samplers/batch_samplers to find the base sampler.
        if self.dataloader.batch_sampler is not None:
            sampler = self.dataloader.batch_sampler
        else:
            sampler = self.dataloader.sampler
        while True:
            if hasattr(sampler, 'batch_sampler'):
                sampler = sampler.batch_sampler
            elif hasattr(sampler, 'sampler'):
                sampler = sampler.sampler
            else:
                break
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(int(math.floor(self.cur_step / self.epoch_size)))
            # When sampler supports set_epoch, skip only the remainder batches in current epoch.
            skip_batches = (self.cur_step % self.epoch_size) * self.gradient_accumulation_steps
        else:
            # Otherwise skip all processed batches so far.
            skip_batches = self.cur_step * self.gradient_accumulation_steps
        if skip_batches > 0:
            for i in range(len(self._dataloaders)):
                self._dataloaders[i] = skip_first_batches(self._dataloaders[i], skip_batches)
        self._skip_batches = skip_batches

    def get_dataloaders(self, data_config: Any) -> DataLoader:
        from giga_datasets import DefaultCollator, load_dataset

        # Global batch size = per-GPU batch size * number of processes * grad accumulation
        batch_size_per_gpu = data_config.get('batch_size_per_gpu', 1)
        batch_size = batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        dataset = load_dataset(data_config.data_or_config)
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform = build_transform(data_config.transform)
        dataset.set_transform(transform)
        if 'batch_sampler' in data_config:
            # Use custom batch sampler path
            batch_sampler_cfg = data_config.batch_sampler
            batch_sampler = build_sampler(
                batch_sampler_cfg,
                dataset=dataset,
                batch_size_per_gpu=batch_size_per_gpu,
                batch_size=batch_size,
            )
        else:
            # Fallback: build sampler + wrap with vanilla BatchSampler
            sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
            sampler = build_sampler(sampler_cfg, dataset=dataset, batch_size=batch_size)
            batch_sampler = BatchSampler(sampler, batch_size=batch_size_per_gpu, drop_last=False)
        if self.data_parallel_size > 1:
            # Replicate each batch for in-process data parallel
            batch_sampler = ParallelBatchSampler(batch_sampler, data_parallel_size=self.data_parallel_size)
        collator = data_config.get('collator', {})
        collator = DefaultCollator(**collator)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=data_config.num_workers,
        )
        if self.distributed_type == DistributedType.DEEPSPEED:
            # Configure DeepSpeed micro batch size per GPU
            if getattr(batch_sampler, 'batch_size', None) is not None:
                batch_size = batch_sampler.batch_size
            elif getattr(batch_sampler, 'batch_sizes', None) is not None:
                batch_size = min(batch_sampler.batch_sizes)
            else:
                assert False
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
        if getattr(batch_sampler, 'batch_size', None) is None:
            # Allow uneven batches when batch size is dynamic (e.g., buckets)
            self.accelerator.even_batches = False
        return dataloader

    def get_models(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_optimizers(self, optimizers: Any) -> list[Optimizer] | Any:
        optimizers = utils.as_list(optimizers)
        for i in range(len(optimizers)):
            if isinstance(optimizers[i], dict):
                if len(optimizers) == 1 and len(self.models) > 1:
                    params = []
                    for model in self.models:
                        params += list(model.parameters())
                elif len(optimizers) == len(self.models):
                    params = self.models[i].parameters()
                else:
                    assert False
                params = list(filter(lambda p: p.requires_grad, params))
                optimizers[i] = build_optimizer(optimizers[i], params=params)
        return optimizers

    def get_schedulers(self, schedulers: Any) -> list[Any]:
        schedulers = utils.as_list(schedulers)
        assert len(schedulers) == len(self.optimizers)
        for i in range(len(schedulers)):
            if isinstance(schedulers[i], dict):
                schedulers[i] = build_scheduler(
                    schedulers[i],
                    optimizer=self.optimizers[i],
                    epoch_size=self.epoch_size,
                    max_epochs=self.max_epochs,
                    max_steps=self.max_steps,
                )
        return schedulers

    def set_ema_models(self) -> None:
        if self.with_ema:
            for model in self.models:
                state_dict = model.state_dict()
                ema_model = EMAModel(rank=self.process_index, world_size=self.num_processes)
                ema_model.load_state_dict(state_dict, device=self.device)
                self.ema_models.append(ema_model)

    def apply_activation_checkpointing(self, models: list[Module] | None = None) -> None:
        if self.activation_checkpointing and self.activation_class_names is not None:
            models = models or self.models
            for model in models:
                auto_wrap_policy = module_auto_wrap_policy(self.activation_class_names)
                if self.mixed_precision == 'fp8' and self.accelerator.fp8_backend == FP8BackendType.TE:
                    from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

                    checkpoint_wrapper_fn = functools.partial(
                        checkpoint_wrapper,
                        checkpoint_fn=te_checkpoint,
                    )
                    apply_activation_checkpointing(
                        model,
                        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
                        auto_wrap_policy=auto_wrap_policy,
                    )
                else:
                    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)

    def apply_fp8(self, models: list[Module] | None = None) -> None:
        if self.mixed_precision == 'fp8':
            models = models or self.models
            if self.distributed_type == DistributedType.FSDP and self.accelerator.fp8_backend == FP8BackendType.TE:
                from accelerate.utils import apply_fp8_autowrap

                for i in range(len(models)):
                    models[i] = apply_fp8_autowrap(models[i], self.accelerator.fp8_recipe_handler)

    def prepare(self, dataloaders: Any, models: Any, optimizers: Any, schedulers: Any) -> None:
        """Build and prepare dataloaders, models, optimizers and schedulers.

        This wraps objects with Accelerate and applies optional EMA and activation checkpointing. It also sets DeepSpeed's micro batch size if
        applicable.
        """
        self._dataloaders = utils.as_list(self.get_dataloaders(dataloaders))
        self._models = utils.as_list(self.get_models(models))
        self.set_ema_models()
        self.apply_activation_checkpointing()
        self.apply_fp8()
        self._optimizers = utils.as_list(self.get_optimizers(optimizers))
        self._schedulers = utils.as_list(self.get_schedulers(schedulers))
        # Flatten objects into a single list, wrap by Accelerator, then split back
        objects = [self._dataloaders, self._models, self._optimizers, self._schedulers]
        inputs = functools.reduce(lambda x, y: x + y, objects)
        outputs = utils.as_list(self.accelerator.prepare(*inputs))
        start_idx = 0
        for obj in objects:
            end_idx = start_idx + len(obj)
            obj[:] = outputs[start_idx:end_idx]
            start_idx = end_idx
        for scheduler in self.schedulers:
            if isinstance(scheduler, accelerate.scheduler.AcceleratedScheduler):
                # Make scheduler step per accumulation step rather than per batch item
                scheduler.split_batches = True

    def train(self) -> None:
        """Run the main training loop over the configured number of steps.

        Steps are grouped by gradient accumulation; checkpointing and logging are performed at configured intervals.
        """
        # release_memory()
        self.print_before_train()
        dataloader_iter = iter(self.dataloader)
        for self._cur_step in range(self._cur_step, self.max_steps):
            self._cur_step += 1
            # Accumulate gradients across multiple micro-steps
            for _ in range(self.gradient_accumulation_steps):
                batch_dict = next(dataloader_iter)
                with self.accelerator.accumulate(*self.models):
                    losses = self.forward_step(batch_dict)
                    loss = self.parse_losses(losses)
                    self.backward_step(loss)
            # Periodic logging and checkpointing
            self.print_step()
            self.save_checkpoint_step()
        self.print_after_train()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def forward_step(self, batch_dict: dict[str, Any]) -> Any:
        return self.model(batch_dict)

    def backward_step(self, loss: Tensor) -> None:
        """Backward pass, optional grad clipping, optimizer and scheduler
        steps.

        - NaN losses are detected and skipped to avoid corrupting model state.
        - Gradient clipping is applied only when gradients are synchronized.
        - EMA is updated after each full optimizer step when enabled.
        """
        if torch.isnan(loss).any():
            if self.is_main_process:
                self.logger.info('loss is NAN, ignore backward')
            return
        self.accelerator.backward(loss)
        max_grad_norm = self.kwargs.get('max_grad_norm', None)
        grad_norm_type = self.kwargs.get('grad_norm_type', 2)
        if self.accelerator.sync_gradients and max_grad_norm is not None:
            params = []
            for model in self.models:
                params += list(model.parameters())
            self.accelerator.clip_grad_norm_(params, max_grad_norm, grad_norm_type)
        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if self.accelerator.sync_gradients and self.with_ema:
            for model, ema_model in zip(self.models, self.ema_models):
                if self.distributed_type == DistributedType.DEEPSPEED:
                    if self.accelerator.deepspeed_config['zero_optimization']['stage'] == 3:
                        state_dict = self.accelerator.get_state_dict(model)
                    else:
                        state_dict = self.accelerator.unwrap_model(model, keep_torch_compile=False).state_dict()
                elif self.accelerator.is_fsdp2:
                    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=False)
                    state_dict = get_model_state_dict(model, options=options)
                else:
                    state_dict = self.accelerator.unwrap_model(model, keep_torch_compile=False).state_dict()
                ema_model.step(state_dict)

    def save_checkpoint_step(self) -> None:
        if self._by_epoch and self.checkpoint_interval < self.max_epochs:
            checkpoint_interval = int(self.checkpoint_interval * self.epoch_size)
        else:
            checkpoint_interval = int(self.checkpoint_interval)
        if self.cur_step % checkpoint_interval == 0 or self.cur_step == self.max_steps:
            output_name = 'checkpoint_epoch_{}_step_{}'.format(self.cur_epoch, self.cur_step)
            output_dir = os.path.join(self.model_dir, output_name)
            if self.is_main_process:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                self.remove_checkpoint()
            # Release cached GPU memory before heavy save operations to reduce OOM risk.
            release_memory()
            self.accelerator.wait_for_everyone()
            if self.distributed_type == DistributedType.DEEPSPEED:
                self.accelerator.save_state(output_dir, exclude_frozen_parameters=True)
            else:
                self.accelerator.save_state(output_dir)

    def save_model_hook(self, models: list[Module], weights: list[dict[str, Tensor]], output_dir: str) -> None:
        """Customize how model weights are saved by Accelerate.

        When the model is a ``ModuleDict``, save each submodule into its own
        subdirectory. If EMA is enabled, only the EMA copy is written in this
        diffusers-style layout; the non-EMA model remains in the backend
        checkpoint state (for example ``pytorch_model_fsdp.bin``).
        """
        assert len(models) == 1
        model = self.accelerator.unwrap_model(models[0], keep_torch_compile=False)
        weights_name = SAFETENSORS_WEIGHTS_NAME if self.checkpoint_safe_serialization else WEIGHTS_NAME
        if len(weights) > 1:
            assert False
        if not self.with_ema:
            if len(weights) == 0:
                state_dict = self.accelerator.get_state_dict(models[0])
            else:
                state_dict = weights.pop()
            if self.is_main_process:
                if isinstance(model, ModuleDict):
                    model_names = list(model.keys())
                    for model_name in model_names:
                        model_output_dir = os.path.join(output_dir, model_name)
                        os.makedirs(model_output_dir, exist_ok=True)
                        if hasattr(model[model_name], 'save_config'):
                            model[model_name].save_config(model_output_dir)
                        sub_state_dict = {k[len(model_name) + 1 :]: v for k, v in state_dict.items() if k.startswith(model_name)}
                        output_path = os.path.join(model_output_dir, weights_name)
                        self.logger.info(f'Save {model_name} to {output_path}')
                        utils.save_state_dict(sub_state_dict, output_path)
                else:
                    model_name = getattr(self, 'model_name', 'model')
                    model_output_dir = os.path.join(output_dir, model_name)
                    os.makedirs(model_output_dir, exist_ok=True)
                    if hasattr(model, 'save_config'):
                        model.save_config(model_output_dir)
                    output_path = os.path.join(model_output_dir, weights_name)
                    self.logger.info(f'Save {model_name} to {output_path}')
                    utils.save_state_dict(state_dict, output_path)
        if self.with_ema:
            ema_model = self.ema_models[0]
            state_dict = ema_model.state_dict()
            if self.is_main_process:
                save_config_path = os.path.join(output_dir, 'ema_config.json')
                ema_model.save_config(save_config_path)
                if isinstance(model, ModuleDict):
                    model_names = list(model.keys())
                    for model_name in model_names:
                        model_output_dir = os.path.join(output_dir, model_name + '_ema')
                        os.makedirs(model_output_dir, exist_ok=True)
                        if hasattr(model[model_name], 'save_config'):
                            model[model_name].save_config(model_output_dir)
                        sub_state_dict = {k[len(model_name) + 1 :]: v for k, v in state_dict.items() if k.startswith(model_name)}
                        output_path = os.path.join(model_output_dir, weights_name)
                        self.logger.info(f'Save {model_name}_ema to {output_path}')
                        utils.save_state_dict(sub_state_dict, output_path)
                else:
                    model_name = getattr(self, 'model_name', 'model')
                    model_output_dir = os.path.join(output_dir, model_name + '_ema')
                    os.makedirs(model_output_dir, exist_ok=True)
                    if hasattr(model, 'save_config'):
                        model.save_config(model_output_dir)
                    output_path = os.path.join(model_output_dir, weights_name)
                    self.logger.info(f'Save {model_name}_ema to {output_path}')
                    utils.save_state_dict(state_dict, output_path)

    def load_model_hook(self, models: list[Module], input_dir: str) -> None:
        """Customize how model weights are loaded by Accelerate.

        Supports loading EMA weights and ``ModuleDict`` structures.
        """
        assert len(models) == 0 or len(models) == 1
        weights_name = SAFETENSORS_WEIGHTS_NAME if self.checkpoint_safe_serialization else WEIGHTS_NAME
        if self.with_ema:
            model = self.models[0] if len(models) == 0 else models[0]
            model = self.accelerator.unwrap_model(model, keep_torch_compile=False)
            ema_model = self.ema_models[0]
            config_path = os.path.join(input_dir, 'ema_config.json')
            ema_model.load_config(config_path)
            if isinstance(model, ModuleDict):
                model_names = list(model.keys())
                state_dict = dict()
                for model_name in model_names:
                    input_path = os.path.join(input_dir, model_name + '_ema', weights_name)
                    self.logger.info(f'Load {model_name}_ema from {input_path}')
                    sub_state_dict = utils.load_state_dict(input_path)
                    sub_state_dict = {model_name + '.' + k: v for k, v in sub_state_dict.items()}
                    state_dict.update(sub_state_dict)
            else:
                model_name = getattr(self, 'model_name', 'model')
                input_path = os.path.join(input_dir, model_name + '_ema', weights_name)
                self.logger.info(f'Load {model_name}_ema from {input_path}')
                state_dict = utils.load_state_dict(input_path)
            ema_model.load_state_dict(state_dict, device=self.device)
        if len(models) == 0:
            return
        model = models[-1]
        if isinstance(model, ModuleDict):
            model_names = list(model.keys())
            model_paths = [os.path.join(input_dir, model_name, weights_name) for model_name in model_names]
        else:
            model_name = getattr(self, 'model_name', 'model')
            model_paths = [os.path.join(input_dir, model_name, weights_name)]
        if not all(os.path.exists(input_path) for input_path in model_paths):
            self.logger.info(
                'Skip loading non-EMA model from %s; falling back to the backend checkpoint state',
                input_dir,
            )
            return
        model = models.pop()
        if isinstance(model, ModuleDict):
            model_names = list(model.keys())
            state_dict = dict()
            for model_name in model_names:
                input_path = os.path.join(input_dir, model_name, weights_name)
                self.logger.info(f'Load {model_name} from {input_path}')
                sub_state_dict = utils.load_state_dict(input_path)
                sub_state_dict = {model_name + '.' + k: v for k, v in sub_state_dict.items()}
                state_dict.update(sub_state_dict)
        else:
            model_name = getattr(self, 'model_name', 'model')
            input_path = os.path.join(input_dir, model_name, weights_name)
            self.logger.info(f'Load {model_name} from {input_path}')
            state_dict = utils.load_state_dict(input_path)
        model.load_state_dict(state_dict, strict=self.checkpoint_strict)

    def print_before_train(self) -> None:
        if not self.is_main_process:
            return
        for model in self.models:
            self.logger.info(model)
        msg = 'num_processes: {}'.format(self.num_processes)
        msg += ', process_index: {}'.format(self.process_index)
        msg += ', data_size: {}'.format(self.data_size)
        msg += ', batch_size: {}'.format(self.batch_size)
        msg += ', epoch_size: {}'.format(self.epoch_size)
        self.logger.info(msg)
        self._epoch_tic = self._step_tic = self._start_tic = time.time()

    def print_step(self) -> None:
        if not self.is_main_process:
            return
        if self.cur_step % self.log_interval == 0:
            outputs = dict()
            for key, val in self._outputs.items():
                val = val['sum'] / val['num'] if val['num'] > 0 else float('nan')
                outputs[key] = val
            self._outputs.clear()
            self.accelerator.log(outputs, self.cur_step)
            time_cost = time.time() - self._step_tic
            self._step_tic = time.time()
            speed = self.log_interval * self.batch_size / time_cost
            eta_sec = max(0, time_cost / self.log_interval * (self.max_steps - self.cur_step))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            lr = self.scheduler.get_last_lr()[0]
            if self._by_epoch:
                inner_step = (self.cur_step - 1) % self.epoch_size + 1
                msg = 'Epoch[%d/%d][%d/%d]' % (self.cur_epoch, self.max_epochs, inner_step, self.epoch_size)
            else:
                msg = 'Step[%d/%d]' % (self.cur_step, self.max_steps)
            msg += ' eta: %s, time: %.3f, speed: %.3f, lr: %.3e' % (eta_str, time_cost, speed, lr)
            if self.log_cpu_memory:
                msg += ', cpu_mem: %s' % utils.get_cpu_memory()
            if self.mixed_precision == 'fp16':
                if self.accelerator.scaler is not None and self.accelerator.scaler.is_enabled():
                    grad_scale = self.accelerator.scaler.get_scale()
                elif self.distributed_type == DistributedType.DEEPSPEED:
                    optimizer = self.optimizer
                    if hasattr(optimizer, 'optimizer'):
                        optimizer = optimizer.optimizer
                    if hasattr(optimizer, 'loss_scaler'):
                        grad_scale = optimizer.loss_scaler.cur_scale
                    elif hasattr(optimizer, 'cur_scale'):
                        grad_scale = optimizer.cur_scale
                    else:
                        assert False
                else:
                    grad_scale = None
            else:
                grad_scale = None
            if grad_scale is not None:
                msg += ', grad_scale: %.4f' % grad_scale
            for key, val in outputs.items():
                msg += ', %s: %.4f' % (key, val)
            self.logger.info(msg)
        if self._by_epoch and self.cur_step % self.epoch_size == 0:
            time_cost = time.time() - self._epoch_tic
            time_cost = str(datetime.timedelta(seconds=int(time_cost)))
            self._epoch_tic = time.time()
            self.logger.info('Total_time: %s' % time_cost)

    def print_after_train(self) -> None:
        if not self.is_main_process:
            return
        time_cost = time.time() - self._start_tic
        time_cost = str(datetime.timedelta(seconds=int(time_cost)))
        self.logger.info('Total_time: %s' % time_cost)

    def parse_losses(self, losses: dict[str, Tensor] | Tensor) -> Tensor:
        """Reduce per-device losses and compute total loss for
        logging/training.

        If a dict of losses is provided, each entry is averaged and then summed. If a tensor is provided, it is treated as total loss.
        """
        outputs = {}
        if isinstance(losses, dict):
            assert 'total_loss' not in losses
            for key, val in losses.items():
                losses[key] = val.mean()
            loss = sum(losses.values())
            for key, val in losses.items():
                outputs[key] = self.accelerator.gather(val).mean()
            total_loss = sum(outputs.values())
            total_loss = self.accelerator.gather(total_loss).mean()
            outputs['total_loss'] = total_loss
        elif isinstance(losses, torch.Tensor):
            loss = losses.mean()
            total_loss = self.accelerator.gather(loss).mean()
            outputs['total_loss'] = total_loss
        else:
            assert False
        if torch.isnan(total_loss).any():
            loss = torch.tensor(float('nan'))
        loss_nan_total_limit = self.kwargs.get('loss_nan_total_limit', 100)
        # Stop training if NaN loss persists for too long to avoid wasting compute
        if loss_nan_total_limit > 0 and torch.isnan(loss).any():
            self._loss_nan_count += 1
            if self._loss_nan_count > loss_nan_total_limit:
                exit(-1)
        else:
            self._loss_nan_count = 0
        for key, val in outputs.items():
            if key not in self._outputs:
                self._outputs[key] = {'sum': 0.0, 'num': 0}
            self._outputs[key]['sum'] += val.item()
            self._outputs[key]['num'] += 1
        return loss


def reset_accelerate_te(**recipe_kwargs: Any) -> None:
    def _convert_model(model, *args, **kwargs):
        pass

    setattr(accelerate.accelerator, 'convert_model', _convert_model)
    setattr(accelerate.utils, 'convert_model', _convert_model)
    setattr(accelerate.utils.other, 'convert_model', _convert_model)
    setattr(accelerate.utils.transformer_engine, 'convert_model', _convert_model)

    def _apply_fp8_autowrap(model, *args, **kwargs):
        from accelerate.utils import contextual_fp8_autocast
        from transformer_engine.common import recipe as te_recipe

        if 'fp8_format' in recipe_kwargs:
            recipe_kwargs['fp8_format'] = getattr(te_recipe.Format, recipe_kwargs['fp8_format'])
        use_during_eval = recipe_kwargs.pop('use_autocast_during_eval', False)
        fp8_recipe = te_recipe.DelayedScaling(**recipe_kwargs)
        new_forward = contextual_fp8_autocast(model.forward, fp8_recipe, use_during_eval)

        if hasattr(model.forward, '__func__'):
            model.forward = MethodType(new_forward, model)
        else:
            model.forward = new_forward

        return model

    setattr(accelerate.accelerator, 'apply_fp8_autowrap', _apply_fp8_autowrap)
    setattr(accelerate.utils, 'apply_fp8_autowrap', _apply_fp8_autowrap)
    setattr(accelerate.utils.transformer_engine, 'apply_fp8_autowrap', _apply_fp8_autowrap)
