import datetime
import os
import time
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration, set_seed

from .. import utils
from ..configs import load_config
from ..transforms import build_transform


class Tester:
    """Evaluation runner built on Accelerate.

    Handles building dataloaders and models, and provides helper logging utilities for evaluation loops implemented by subclasses.
    """

    def __init__(
        self,
        project_dir: str,
        mixed_precision: str | None = None,
        log_interval: int = 100,
        seed: int = 6666,
        logging_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize evaluation runner.

        Args:
            project_dir: Project working directory to store logs.
            mixed_precision: Precision mode like ``'fp16'`` or ``'bf16'``.
            log_interval: Logging interval in steps.
            seed: Random seed (> 0).
            logging_dir: Directory for logs; defaults to the non-empty
                ``AIHC_TENSORBOARD_LOG_PATH`` when set, otherwise ``project_dir/logs``.
            **kwargs: Extra options for subclasses.
        """
        assert seed > 0
        set_seed(seed)
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
            even_batches=False,
        )
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            project_config=project_config,
            dataloader_config=dataloader_config,
        )
        os.makedirs(self.logging_dir, exist_ok=True)
        if self.is_main_process:
            log_name = 'test_{}.log'.format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.log_interval = log_interval
        self.seed = seed
        self.kwargs = kwargs

        self._dataloaders = []
        self._models = []

        self._cur_step = 0
        self._start_tic = None
        self._step_tic = None

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
    def distributed_type(self):
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
    def mixed_precision(self):
        return self.accelerator.mixed_precision

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.mixed_precision == 'fp16' else torch.float32

    @property
    def dataloaders(self):
        return self._dataloaders

    @property
    def dataloader(self):
        return self._dataloaders[0]

    @property
    def models(self):
        return self._models

    @property
    def model(self):
        return self._models[0]

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
        batch_size = batch_sampler.batch_size
        return batch_size * self.num_processes

    @property
    def epoch_size(self) -> int:
        return len(self.dataloader)

    @property
    def cur_step(self) -> int:
        return self._cur_step

    def print(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    @classmethod
    def load(cls, config_or_path: Any):
        """Factory that builds a tester from a config path or object.

        Args:
            config_or_path: Directory path, JSON path, or Config instance.

        Returns:
            Tester: Initialized tester with prepared dataloader and model.
        """
        config = load_config(config_or_path).copy()
        tester = cls(project_dir=config.project_dir, **config.test)
        tester.prepare(
            dataloaders=config.dataloaders.test,
            models=config.models.test if hasattr(config.models, 'test') else config.models,
        )
        return tester

    def get_checkpoint(self, checkpoint: str | list[str] | None = None):
        """Resolve a checkpoint path from name or latest in model dir.

        Args:
            checkpoint: Name, path, or list of paths.

        Returns:
            str | list[str] | None: Resolved path(s) or None if not found.
        """
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

    def get_dataloaders(self, data_config: Any):
        """Build evaluation dataloader from a data config.

        Args:
            data_config: Configuration containing dataset, transform, and loader args.

        Returns:
            torch.utils.data.DataLoader: The prepared dataloader.
        """
        from giga_datasets import DefaultCollator, load_dataset

        dataset = load_dataset(data_config.data_or_config)
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform = build_transform(data_config.transform)
        dataset.set_transform(transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=DefaultCollator(),
            batch_size=data_config.batch_size_per_gpu,
            num_workers=data_config.num_workers,
        )
        return dataloader

    def get_models(self, *args: Any, **kwargs: Any):
        """Subclasses must implement to construct and return model(s)."""
        raise NotImplementedError

    def prepare(self, dataloaders: Any, models: Any) -> None:
        """Build dataloaders and models, then wrap dataloaders with Accelerate.

        Args:
            dataloaders: Data config or dataloader(s).
            models: Model config or model instance(s).
        """
        self._dataloaders = utils.as_list(self.get_dataloaders(dataloaders))
        self._models = utils.as_list(self.get_models(models))
        self._dataloaders = utils.as_list(self.accelerator.prepare(*self._dataloaders))

    def test(self) -> None:
        """Subclasses must implement the evaluation loop."""
        raise NotImplementedError

    def print_before_test(self) -> None:
        """Log dataset and batch statistics at the test start."""
        msg = 'num_processes: {}'.format(self.num_processes)
        msg += ', process_index: {}'.format(self.process_index)
        msg += ', data_size: {}'.format(self.data_size)
        msg += ', batch_size: {}'.format(self.batch_size)
        msg += ', epoch_size: {}'.format(self.epoch_size)
        self.logger.info(msg)
        self._step_tic = self._start_tic = time.time()

    def print_step(self) -> None:
        """Periodic progress logging during evaluation."""
        if self.cur_step % self.log_interval == 0:
            time_cost = time.time() - self._step_tic
            self._step_tic = time.time()
            speed = self.log_interval * self.batch_size / time_cost
            eta_sec = max(0, time_cost / self.log_interval * (self.epoch_size - self.cur_step))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            msg = 'Node[%d] Step[%d/%d]' % (self.process_index, self.cur_step, self.epoch_size)
            msg += ' eta: %s, time: %.3f, speed: %.3f' % (eta_str, time_cost, speed)
            self.logger.info(msg)

    def print_after_test(self) -> None:
        """Log total evaluation time at the end."""
        time_cost = time.time() - self._start_tic
        time_cost = str(datetime.timedelta(seconds=int(time_cost)))
        self.logger.info('Node[%d] Total_time: %s' % (self.process_index, time_cost))
