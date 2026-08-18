#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compose GLM-5.2 and filtered Kimi-K2.6 HF shards."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


KIMI_PREFIX = "vision_tower."


def _load_weight_map(checkpoint: Path) -> dict[str, str]:
    """Load the tensor-to-shard map from a Hugging Face checkpoint."""
    index_path = checkpoint / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as index_file:
        return json.load(index_file)["weight_map"]


def _link_weight_map(
    weight_map: dict[str, str],
    source: Path,
    output: Path,
    label: str,
    materialize: bool = False,
    filter_prefix: str | None = None,
) -> dict[str, str]:
    """Link/copy shards, or rewrite them with only a selected key prefix."""
    shard_names = sorted(set(weight_map.values()))
    output_names = {name: f"{label}-{name}" for name in shard_names}
    for shard_name, output_name in output_names.items():
        source_shard = (source / shard_name).resolve(strict=True)
        destination = output / output_name
        if filter_prefix is not None:
            keys = [key for key in weight_map if weight_map[key] == shard_name]
            with safe_open(str(source_shard), framework="pt", device="cpu") as shard:
                tensors = {
                    key: shard.get_tensor(key)
                    for key in keys
                    if key.startswith(filter_prefix)
                }
            if not tensors:
                raise ValueError(
                    f"No {filter_prefix!r} tensors found in Kimi shard {source_shard}"
                )
            save_file(tensors, str(destination))
        elif materialize:
            shutil.copyfile(source_shard, destination)
        else:
            destination.symlink_to(source_shard)
    return {key: output_names[shard] for key, shard in weight_map.items()}


def compose(
    glm_hf: Path, kimi_hf: Path, output: Path, materialize: bool = False
) -> None:
    """Create an atomic HF checkpoint composition."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    glm_map = _load_weight_map(glm_hf)
    kimi_map = {
        key: shard
        for key, shard in _load_weight_map(kimi_hf).items()
        if key.startswith(KIMI_PREFIX)
    }
    if not kimi_map:
        raise KeyError("Kimi checkpoint is missing vision_tower tensors")
    conflict = set(glm_map).intersection(kimi_map)
    if conflict:
        raise ValueError(
            "Tensor name conflict between GLM and Kimi checkpoints: "
            f"{sorted(conflict)[:8]}"
        )

    temp_path = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        composed_map = _link_weight_map(
            glm_map, glm_hf, temp_path, "glm", materialize=materialize
        )
        composed_map.update(
            _link_weight_map(
                kimi_map,
                kimi_hf,
                temp_path,
                "kimi",
                filter_prefix=KIMI_PREFIX,
            )
        )
        total_size = sum(
            (temp_path / shard_name).stat().st_size
            for shard_name in set(composed_map.values())
        )
        index = {"metadata": {"total_size": total_size}, "weight_map": composed_map}
        (temp_path / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config_source = (glm_hf / "config.json").resolve(strict=True)
        if materialize:
            shutil.copyfile(config_source, temp_path / "config.json")
        else:
            (temp_path / "config.json").symlink_to(config_source)
        manifest = {
            "glm_hf": str(glm_hf.resolve()),
            "glm_tensor_count": len(glm_map),
            "kimi_hf": str(kimi_hf.resolve()),
            "kimi_vision_tensor_count": len(kimi_map),
            "tensor_files_rewritten": len(set(kimi_map.values())),
        }
        (temp_path / "composition.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_path.rename(output)
    except BaseException:
        shutil.rmtree(temp_path)
        raise


def main() -> None:
    """Parse command-line paths and compose the requested checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-hf", type=Path, required=True)
    parser.add_argument("--kimi-hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--materialize",
        "--copy",
        action="store_true",
        help="Copy shards and config into the composed checkpoint instead of symlinking.",
    )
    args = parser.parse_args()
    compose(args.glm_hf, args.kimi_hf, args.output, materialize=args.materialize)
    print(f"Composed checkpoint: {args.output}")


if __name__ == "__main__":
    main()
