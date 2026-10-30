# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Export a LingBot-VA training checkpoint into an evaluation checkpoint root.

Training saves DCP shards (``$OUTPUT_DIR/checkpoints/steps_<N>``); the LingBot-VA
policy server loads one root holding ``transformer/`` (sharded safetensors with an
index) plus the frozen ``vae/``, ``text_encoder/`` and ``tokenizer/``. This script
consolidates the model tensors, strips the training wrapper prefix, writes the
shards, index and the transformer ``config.json`` of the checkpoint the run started
from (switched to ``attn_mode=torch``), and links the frozen components from an
official LingBot-VA checkpoint.

Usage:
    python examples/embodied/lingbot_va/export_eval_checkpoint.py \\
        --ckpt   /path/to/outputs/lingbot_va_robotwin/checkpoints/steps_50000 \\
        --base   /path/to/lingbot-va-base \\
        --codec  /path/to/lingbot-va-posttrain-robotwin \\
        --out    /path/to/eval_ckpt/lingbot_va_robotwin_steps_50000

Use ``lingbot-va-posttrain-libero-long`` as ``--codec`` for LIBERO checkpoints.
Tensor dtypes are kept (FP32 master weights); the server casts to BF16 when
``server.use_bf16`` is true.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_COMPONENTS = ("vae", "text_encoder", "tokenizer")


def _load_dcp_model_tensors(ckpt: str) -> dict:
    """Consolidate the ``model.*`` tensors with the repo's single-process DCP reader."""
    path = REPO_ROOT / "tools" / "torch_dcp_convert" / "dcp_to_safetensors.py"
    spec = importlib.util.spec_from_file_location("dcp_to_safetensors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_full_state_dict(module._resolve_dcp_dir(ckpt))


def _parse_size(value: str) -> int:
    units = {"gib": 1024**3, "mib": 1024**2, "gb": 1000**3, "mb": 1000**2}
    value = value.strip().lower()
    for suffix, scale in units.items():
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * scale)
    return int(value)


def _inference_tensors(state_dict: dict) -> dict:
    """Map training state-dict keys onto the transformer's own parameter names."""
    tensors = {}
    for name, tensor in state_dict.items():
        # TE modules carry empty ``_extra_state`` entries that inference does not load.
        if name.endswith("._extra_state"):
            continue
        # The trainer wraps the transformer as ``LingBotVAEmbodiedModel.model``.
        key = name[len("model."):] if name.startswith("model.") else name
        if key in tensors:
            raise ValueError(f"duplicate tensor name after prefix stripping: {key}")
        tensors[key] = tensor.detach().cpu().contiguous()
    return tensors


def _write_transformer(tensors: dict, out_dir: Path, max_shard_bytes: int) -> int:
    shards, current, current_bytes = [], {}, 0
    for name in sorted(tensors):
        size = tensors[name].numel() * tensors[name].element_size()
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = tensors[name]
        current_bytes += size
    if current:
        shards.append(current)

    weight_map, total = {}, 0
    for number, shard in enumerate(shards, start=1):
        filename = f"diffusion_pytorch_model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out_dir / filename), metadata={"format": "pt"})
        for name, tensor in shard.items():
            weight_map[name] = filename
            total += tensor.numel() * tensor.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(shards)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="Training checkpoint: a steps_N directory or its dcp/.")
    parser.add_argument("--base", required=True,
                        help="Checkpoint the run was initialized from (its transformer/config.json "
                             "describes the trained architecture), e.g. lingbot-va-base.")
    parser.add_argument("--codec", required=True,
                        help="Official LingBot-VA checkpoint providing vae/, text_encoder/ and tokenizer/.")
    parser.add_argument("--out", required=True, help="Evaluation checkpoint root to create.")
    parser.add_argument("--max-shard-size", default="4GiB")
    args = parser.parse_args(argv)

    base, codec, out = Path(args.base), Path(args.codec), Path(args.out)
    if not (base / "transformer" / "config.json").is_file():
        raise FileNotFoundError(f"{base} has no transformer/config.json")
    missing = [p for p in FROZEN_COMPONENTS if not (codec / p).exists()]
    if missing:
        raise FileNotFoundError(f"{codec} is missing {missing}")
    transformer_dir = out / "transformer"
    if transformer_dir.exists() and any(transformer_dir.iterdir()):
        raise FileExistsError(f"{transformer_dir} is not empty")
    transformer_dir.mkdir(parents=True, exist_ok=True)

    tensors = _inference_tensors(_load_dcp_model_tensors(args.ckpt))
    shard_count = _write_transformer(tensors, transformer_dir, _parse_size(args.max_shard_size))

    # Training runs FlexAttention; the policy server runs the torch attention path.
    config = json.loads((base / "transformer" / "config.json").read_text(encoding="utf-8"))
    config["attn_mode"] = "torch"
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for name in FROZEN_COMPONENTS:
        link = out / name
        if not link.exists():
            os.symlink((codec / name).resolve(), link)
    print(f"[export_eval_checkpoint] {len(tensors)} tensors in {shard_count} shards -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
