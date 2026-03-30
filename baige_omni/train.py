# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training Entry"""

import logging
logging.basicConfig(level=logging.WARNING)
from baige_omni.train import parse_train_args
from baige_omni.train import build_model_trainer


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
