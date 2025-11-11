"""Training Entry"""

#from aiak_training_omni.utils import register_custom_resolvers
from aiak_training_omni.train import parse_train_args
from aiak_training_omni.train import build_model_trainer
import hydra
import argparse
from omegaconf import OmegaConf


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
