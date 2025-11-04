"""Training Entry"""

from aiak_training_omni.train import parse_args_from_cfg
from aiak_training_omni.train import build_model_trainer
import hydra
import argparse
from omegaconf import OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    """train cmd"""
    print(OmegaConf.to_yaml(cfg))

    # parse args
    args = parse_args_from_cfg(cfg)

    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == '__main__':
    main()
