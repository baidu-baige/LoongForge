"""Training Entry"""

from aiak_training_omni.train.arguments import parse_args_from_config
from aiak_training_omni.train import parse_train_args
from aiak_training_omni.train import build_model_trainer
import hydra
import argparse
from omegaconf import OmegaConf


def main():
    """train cmd"""
    # print(OmegaConf.to_yaml(config))

    # parse args
    args = parse_train_args()

    parse_args_from_config(args)
    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == "__main__":
    main()
