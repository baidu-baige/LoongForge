"""LoongForge entry point for GigaBrain-0.7 training (vendored giga stack).

Thin wrapper around the vendored giga_train.launch_from_config. The giga code
was copied verbatim into LoongForge:

  loongforge/embodied/model/giga_brain07/
      giga_models/        (architecture: GigaBrain07Policy etc.)
      giga_brain_0/       (GigaBrain07Trainer / Transform / Loss)
      giga_train/         (training engine: Trainer, launch, samplers, ...)
      gigabrain07.py      (runner export used by the config's `runners`)
      _shims/             (permissive stubs for optional eval-only deps)
  loongforge/embodied/data/datasets/giga_brain07/
      giga_datasets/      (LeRobotDataset + transforms + collators)

Runs on the container-default python with NO extra pip installs: the heavy
stack (torch/transformers/lerobot/accelerate/tyro) is already in the image, and
the optional deps (lmdb/paramiko/gdown/...) are satisfied by _shims. Video is
decoded with the torchcodec backend because the NV torchvision build has no
io.VideoReader.

Usage:
    python -m loongforge.embodied.model.giga_brain07.train_giga_brain07 \
        --config /mnt/cfs/pxy/LoongForge/configs/models/embodied/giga_brain07.yaml
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))                       # .../model/giga_brain07
_DATA = os.path.abspath(os.path.join(
    _HERE, "..", "..", "data", "datasets", "giga_brain07"))              # .../data/datasets/giga_brain07
_SHIMS = os.path.join(_HERE, "_shims")

# Make the vendored packages importable: shims first (optional-dep stubs), then
# the model dir (giga_models / giga_brain_0 / giga_train / gigabrain07), then the
# data dir (giga_datasets). Mirror into PYTHONPATH so accelerate subprocesses
# (multi-GPU launch) inherit the same paths.
for p in (_SHIMS, _HERE, _DATA):
    if p not in sys.path:
        sys.path.insert(0, p)
_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([_SHIMS, _HERE, _DATA, *_pp]))

# Offline: all weights are local.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoongForge GigaBrain-0.7 trainer")
    parser.add_argument("--config", required=True, help="path to giga_brain07 yaml/json config")
    args = parser.parse_args()

    from giga_train import launch_from_config, setup_environment

    setup_environment()
    # Optional quick-run override (used by the smoke/validate + bs-sweep paths).
    # The multi-GPU launch path re-spawns accelerate subprocesses and passes the
    # --config argument through to run_task.py as a FILE PATH, so we must hand
    # launch_from_config a path string (not a loaded Config object). When an
    # override is requested, rewrite the yaml in place first, then pass the path.
    _max_steps = os.environ.get("GB0_MAX_STEPS")
    config_path = args.config
    if _max_steps:
        import yaml
        with open(config_path) as _f:
            _cfg = yaml.safe_load(_f)
        _cfg.setdefault("train", {})["max_steps"] = int(_max_steps)
        with open(config_path, "w") as _f:
            yaml.safe_dump(_cfg, _f, sort_keys=False)
    launch_from_config(config_path)


if __name__ == "__main__":
    main()
