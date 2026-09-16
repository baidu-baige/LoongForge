# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resolve recipes and dispatch training to the selected engine."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from loongforge.contracts import TrainSpec
from loongforge.models.catalog import get_model_spec

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="loongforge", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", allow_abbrev=False, help="run a training recipe")
    train.add_argument("--engine", choices=("mcore", "torch"))
    train.add_argument("--model", help="model name in the catalog")
    train.add_argument("--recipe", type=Path)
    train.add_argument("--dry-run", action="store_true", help="print resolved inputs without importing an engine")
    args, backend_args = parser.parse_known_args(argv)
    try:
        spec = resolve_train(args.engine, args.model, args.recipe, backend_args)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(asdict(spec), indent=2))
        return
    from loongforge.engine.dispatch import run_train

    run_train(spec)


def resolve_train(engine, model, recipe_path, backend_args):
    recipe = _load_recipe(recipe_path) if recipe_path is not None else {}
    for key, value in (("engine", engine), ("model", model)):
        if value is not None and recipe.get(key) is not None and value != recipe[key]:
            raise ValueError(f"--{key} conflicts with recipe {key}")
    engine = engine or recipe.get("engine")
    model = model or recipe.get("model")
    recipe_args = recipe.get("args", [])
    identity = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    identity.add_argument("--model-name")
    identity.add_argument("--config-file", type=Path)
    selected, _ = identity.parse_known_args(recipe_args + list(backend_args))
    if model and selected.model_name and model != selected.model_name:
        raise ValueError("--model-name conflicts with the selected model")
    model = model or selected.model_name
    if model:
        entry = get_model_spec(model, engine)
        engine = entry.engine
        config_file = selected.config_file or entry.config_file
    else:
        if selected.config_file is None or engine is None:
            raise ValueError("Specify --model, or --engine mcore with --config-file")
        if engine == "torch":
            raise ValueError("Torch requires --model to select model and data schemas")
        config_file = selected.config_file
    if not config_file.is_file():
        raise ValueError(f"Model config does not exist: {config_file}")
    if model and selected.config_file is not None and engine == "mcore":
        raise ValueError("Select either a catalog model or --config-file for MCore")

    defaults = _load_recipe(_CONFIGS / "engines" / f"{engine}.yaml") if recipe else {}
    if defaults and defaults.get("engine") != engine:
        raise ValueError("Engine defaults do not match the selected engine")
    forwarded = defaults.get("args", []) + recipe_args + list(backend_args)
    if model and selected.model_name is None:
        forwarded = ["--model-name", model] + forwarded
    return TrainSpec(engine, model, str(config_file.resolve()), tuple(forwarded))


def _load_recipe(path):
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Recipe must be YAML: {path}")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or set(recipe) - {"engine", "model", "args"}:
        raise ValueError(f"Recipe must be a mapping of engine, model, args: {path}")
    if recipe.get("engine") not in (None, "mcore", "torch"):
        raise ValueError(f"Unknown recipe engine: {recipe['engine']}")
    if "model" in recipe and (not isinstance(recipe["model"], str) or not recipe["model"].strip()):
        raise ValueError("Recipe model must be a nonempty string")
    args = recipe.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("Recipe args must be a list of backend argument strings")
    return recipe


if __name__ == "__main__":
    main()
