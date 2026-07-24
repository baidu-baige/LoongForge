# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from FastWAM (https://github.com/yuantianyuan01/FastWAM).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precompute FastWAM text embeddings for LoongForge datasets.

The FastWAM collator looks up cached prompt embeddings by filename:

    {sha256(prompt)}.t5_len{context_len}.{enc_id}.pt

where ``prompt`` is the fully formatted FastWAM prompt. This script scans task
strings from LeRobot-style metadata or explicit prompt files, encodes them with
the same Wan T5 tokenizer/text encoder used by LoongForge FastWAM, and writes
cache files consumable by ``FastWAMPreprocessor``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"

logger = logging.getLogger("precompute_fastwam_text_embeds")


def _model_id_to_enc_id(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe encoder identifier."""
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _read_jsonl(path: Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a newline-delimited JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _tasks_from_lerobot_root(root: Path) -> list[str]:
    """Collect task strings from a LeRobot dataset root's meta/tasks.jsonl."""
    task_paths = []
    direct_tasks_path = root / "meta" / "tasks.jsonl"
    if direct_tasks_path.exists():
        task_paths.append(direct_tasks_path)
    else:
        task_paths.extend(sorted(root.glob("*/meta/tasks.jsonl")))

    if not task_paths:
        return []

    tasks = []
    for tasks_path in task_paths:
        for item in _read_jsonl(tasks_path):
            task = item.get("task") or item.get("instruction") or item.get("prompt")
            if task is not None:
                tasks.append(str(task))
    return tasks


def _tasks_from_file(path: Path) -> list[str]:
    """Load task strings from a plain text or JSONL prompt file."""
    if path.suffix == ".jsonl":
        tasks = []
        for item in _read_jsonl(path):
            if isinstance(item, dict):
                task = item.get("task") or item.get("instruction") or item.get("prompt") or item.get("text")
            else:
                task = item
            if task is not None:
                tasks.append(str(task))
        return tasks

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _collect_tasks(dataset_roots: list[Path], prompt_files: list[Path], prompts: list[str]) -> list[str]:
    """Aggregate and deduplicate tasks from dataset roots, prompt files, and inline prompts."""
    tasks = []
    for root in dataset_roots:
        root_tasks = _tasks_from_lerobot_root(root)
        if not root_tasks:
            logger.warning("No tasks found under %s/meta/tasks.jsonl or %s/*/meta/tasks.jsonl", root, root)
        tasks.extend(root_tasks)
    for path in prompt_files:
        tasks.extend(_tasks_from_file(path))
    tasks.extend(prompts)

    seen = set()
    unique = []
    for task in tasks:
        if task not in seen:
            seen.add(task)
            unique.append(task)
    return unique


def _format_prompt(task: str) -> str:
    """Apply the prompt template to a task string, skipping already-formatted prompts."""
    if task.startswith("A video recorded"):
        return task
    return DEFAULT_PROMPT_TEMPLATE.format(task=task)


def _resolve_dtype(dtype: str):
    """Map a dtype string (bf16/fp16/fp32 and aliases) to a torch.dtype."""
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _load_text_encoder(args: argparse.Namespace, device: str, dtype: torch.dtype):
    """Load the Wan T5 text encoder and tokenizer from the configured model paths."""
    from loongforge.embodied.model.fastwam.wan.loader import _load_registered_model, _resolve_configs
    from loongforge.embodied.model.fastwam.wan.text_encoder import HuggingfaceTokenizer

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=args.context_len,
        clean="whitespace",
    )
    return text_encoder, tokenizer


def _encode_batch(text_encoder, tokenizer, prompts: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of prompts and zero-pad positions beyond each sequence's true length."""
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device, dtype=torch.bool)
    context = text_encoder(ids, mask)
    seq_lens = mask.gt(0).sum(dim=1).long()
    for index, seq_len in enumerate(seq_lens):
        context[index, seq_len:] = 0
    return context.cpu(), mask.cpu()


def _cache_path(cache_dir: Path, prompt: str, context_len: int, enc_id: str) -> Path:
    """Return the cache file path for a given prompt, context length, and encoder id."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.t5_len{context_len}.{enc_id}.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the text embedding precompute script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", action="append", default=[],
        help="LeRobot dataset root with meta/tasks.jsonl. Can be repeated.",
    )
    parser.add_argument(
        "--prompt-file", action="append", default=[],
        help="Text/jsonl prompt file. Can be repeated.",
    )
    parser.add_argument(
        "--prompt", action="append", default=[],
        help="Single raw task/prompt string. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for FastWAM text embedding cache files.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point: collect prompts, encode them, and write cache files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    dataset_roots = [Path(path).expanduser() for path in args.dataset_root]
    prompt_files = [Path(path).expanduser() for path in args.prompt_file]
    tasks = _collect_tasks(dataset_roots, prompt_files, args.prompt)
    prompts = [_format_prompt(task) for task in tasks]
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise ValueError("No prompts found. Provide --dataset-root, --prompt-file, or --prompt.")

    cache_dir = Path(args.output_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    enc_id = _model_id_to_enc_id(args.model_id)
    pending = [
        prompt for prompt in prompts
        if args.overwrite or not _cache_path(cache_dir, prompt, args.context_len, enc_id).exists()
    ]

    logger.info("Collected %d unique prompt(s); %d need encoding", len(prompts), len(pending))
    logger.info("Cache directory: %s", cache_dir)
    logger.info("Filename encoder id: %s", enc_id)
    if not pending:
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available; falling back to CPU")
        args.device = "cpu"
    dtype = _resolve_dtype(args.dtype)
    text_encoder, tokenizer = _load_text_encoder(args, args.device, dtype)

    for start in range(0, len(pending), args.batch_size):
        batch_prompts = pending[start:start + args.batch_size]
        context, mask = _encode_batch(text_encoder, tokenizer, batch_prompts, args.device)
        for prompt, prompt_context, prompt_mask in zip(batch_prompts, context, mask, strict=True):
            path = _cache_path(cache_dir, prompt, args.context_len, enc_id)
            torch.save(
                {
                    "context": prompt_context.to(torch.float32),
                    "mask": prompt_mask.to(torch.bool),
                    "prompt": prompt,
                    "model_id": args.model_id,
                    "tokenizer_model_id": args.tokenizer_model_id,
                    "context_len": args.context_len,
                },
                path,
            )
        logger.info("Encoded %d/%d prompt(s)", min(start + len(batch_prompts), len(pending)), len(pending))


if __name__ == "__main__":
    main()
