# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied training entry."""

from loongforge.engines.torch.parser import parse_train_args
from loongforge.training.registry import resolve_torch_trainer
from loongforge.training.torch_runner import TorchRunner


def main():
    """Parse configs, build the runner, and start the training loop."""
    training_args, model_cfg, data_cfg = parse_train_args()
    method, build_optimization = resolve_torch_trainer(training_args.trainer_type)
    TorchRunner(training_args, model_cfg, data_cfg, method, build_optimization(training_args)).train()


if __name__ == "__main__":
    main()
