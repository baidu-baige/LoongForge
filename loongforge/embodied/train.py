# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied training entry."""

def main():
    """Parse configs, build the trainer, and start the training loop."""
    # Imported inside main() so spawn DataLoader workers, which import this
    # module as ``__mp_main__``, do not pay for the full training stack
    # (albumentations alone costs ~16s) before they can start decoding.
    from loongforge.embodied.train.parser import parse_train_args
    from loongforge.embodied.train.trainers import build_model_trainer

    training_args, model_cfg, data_cfg = parse_train_args()
    trainer = build_model_trainer(training_args, model_cfg, data_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
