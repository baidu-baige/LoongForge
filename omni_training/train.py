# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training Entry"""

from omni_training.train import parse_train_args
from omni_training.train import build_model_trainer


def main():
    """train cmd"""

    # parse args and config
    args = parse_train_args()

    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == "__main__":
    main()
