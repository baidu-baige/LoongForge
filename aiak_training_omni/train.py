"""Training Entry"""

from aiak_training_omni.train import parse_train_args
from aiak_training_omni.train import build_model_trainer


def main():
    """train cmd"""

    # parse args
    args = parse_train_args()

    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == '__main__':
    main()
