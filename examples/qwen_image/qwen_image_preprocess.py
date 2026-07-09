# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen-Image-Edit-2511 offline preprocessing script.

Encodes each (image, edit_image, prompt) sample into training tensors and
saves them as .pth files in the AIAK flat-dict format consumed by
``loongforge/train/diffusion/pretrain_qwen_image.py``.

Only the DiffSynth text encoder (Qwen2.5-VL) + VAE + tokenizer/processor are
loaded; the DiT itself is not required for this stage.

Output keys per sample:
  input_latents, edit_latents, prompt_emb, prompt_emb_mask,
  noise, timestep_id, timestep, latents, training_target, scale,
  height, width

Fixed-seed noise/timestep generation matches the DiffSynth reference
``prepare_dit_cache.py`` so the resulting cache is numerically equivalent to
the DiffSynth benchmark cache (up to bf16 hardware non-determinism).

Dependencies:
  pip install diffsynth==2.0.6 --no-deps
  pip install accelerate Pillow pandas tqdm

Metadata file (json / jsonl / csv), one row per sample, columns:
  image        target image path (relative to --dataset_base_path)
  edit_image   edit image path, or a list of paths (multi-edit)
  prompt       text prompt

Example:
  accelerate launch qwen_image_preprocess.py \\
    --dataset_base_path ./data/samples \\
    --dataset_metadata_path ./data/samples/metadata.json \\
    --model_root /path/to/Qwen-Image-Edit-2511 \\
    --output_path ./data/qwen_image_edit2511_cache \\
    --seed 1234 --timestep_id 321
"""

import argparse
import glob
import importlib.machinery
import json
import os
import shutil
import sys
import types
from typing import List, Union

# Avoid failing on the top-level torchaudio import in diffsynth.core.data.operators.
# The torchaudio C extension can be ABI-incompatible with custom torch builds,
# and this image preprocessing path does not use audio features.
if "torchaudio" not in sys.modules:
    _stub = types.ModuleType("torchaudio")
    _stub.__version__ = "0.0.0"
    _stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    _stub.load = lambda *a, **kw: (_ for _ in ()).throw(
        NotImplementedError("torchaudio not available"))
    sys.modules["torchaudio"] = _stub

import pandas
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from diffsynth.core.data.operators import ImageCropAndResize, LoadImage
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.pipelines.qwen_image import (
    ModelConfig,
    QwenImagePipeline,
    QwenImageUnit_EditImageEmbedder,
    QwenImageUnit_InputImageEmbedder,
    QwenImageUnit_PromptEmbedder,
)


def load_metadata(metadata_path: str) -> List[dict]:
    """Load a .csv / .json / .jsonl metadata file into a list of dicts."""
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r") as f:
            return json.load(f)
    if metadata_path.endswith(".jsonl"):
        with open(metadata_path, "r") as f:
            return [json.loads(line.strip()) for line in f]
    df = pandas.read_csv(metadata_path)
    return [df.iloc[i].to_dict() for i in range(len(df))]


def load_image(path: str, base: str) -> Image.Image:
    """Load a PIL image, resolving relative paths against ``base``.

    Uses ``diffsynth.core.data.operators.LoadImage`` so behavior matches the
    reference DiffSynth data pipeline byte-for-byte.
    """
    full = path if os.path.isabs(path) else os.path.join(base, path)
    return LoadImage(convert_RGB=True)(full)


def load_edit_images(entry, base: str) -> Union[Image.Image, List[Image.Image]]:
    """Return a single edit image or list of edit images (for multi-edit)."""
    if entry is None:
        return None
    if isinstance(entry, list):
        return [load_image(p, base) for p in entry]
    return load_image(entry, base)


def normalize_latent(x):
    """Squeeze leading batch dim off VAE latents (matches AIAK flat layout)."""
    if isinstance(x, list):
        return [normalize_latent(t) for t in x]
    if torch.is_tensor(x) and x.dim() == 5 and x.shape[0] == 1:
        return x.squeeze(0)
    return x


def to_cpu(x):
    """Move a tensor / list-of-tensors to CPU, preserving structure."""
    if isinstance(x, list):
        return [to_cpu(t) for t in x]
    if torch.is_tensor(x):
        return x.cpu()
    return x


@torch.no_grad()
def encode_sample(
    pipe: QwenImagePipeline,
    prompt: str,
    image: Image.Image,
    edit_image: Union[Image.Image, List[Image.Image], None],
    prompt_unit: QwenImageUnit_PromptEmbedder,
    input_unit: QwenImageUnit_InputImageEmbedder,
    edit_unit: QwenImageUnit_EditImageEmbedder,
    image_op: ImageCropAndResize,
    tiled: bool,
) -> dict:
    """Run DiffSynth's per-sample pipeline units to produce AIAK-format tensors.

    Image geometry mirrors ``UnifiedDataset.default_image_operator`` so the
    output is bit-identical to the DiffSynth ``sft:data_process`` cache:

      * ``image`` is passed through ``ImageCropAndResize`` (16-div, max_pixels,
        BILINEAR resize + center_crop) before VAE encoding.
      * ``edit_image`` (single or list) is passed through the same operator
        first, then handed to ``QwenImageUnit_EditImageEmbedder`` which
        performs its own 32-div ``edit_image_auto_resize``.
    """
    # Apply the same 16-div resize+crop DiffSynth's dataset would have done.
    image = image_op(image)
    if edit_image is not None:
        if isinstance(edit_image, list):
            edit_image = [image_op(e) for e in edit_image]
        else:
            edit_image = image_op(edit_image)

    # Text encoding uses the Qwen2.5-VL text encoder + processor.
    prompt_out = prompt_unit.process(pipe, prompt=prompt, edit_image=edit_image)
    prompt_emb = prompt_out["prompt_emb"]
    prompt_emb_mask = prompt_out["prompt_emb_mask"]

    width, height = image.size

    # VAE encode target image -> input_latents. ``latents`` from the training
    # branch is unused in the AIAK cache (we compute a deterministic noisy
    # latent below).
    noise_placeholder = torch.zeros(1)  # ignored in training mode
    pipe.scheduler.set_timesteps(1000, training=True)
    input_out = input_unit.process(
        pipe,
        input_image=image,
        noise=noise_placeholder,
        tiled=tiled,
        tile_size=64,
        tile_stride=32,
    )
    input_latents = input_out["input_latents"]

    # VAE encode the edit image(s) -> edit_latents.
    edit_out = edit_unit.process(
        pipe,
        edit_image=edit_image,
        tiled=tiled,
        tile_size=64,
        tile_stride=32,
        edit_image_auto_resize=True,
    )
    edit_latents = edit_out.get("edit_latents")

    return {
        "prompt_emb": prompt_emb.cpu().float(),
        "prompt_emb_mask": prompt_emb_mask.cpu(),
        "input_latents": to_cpu(normalize_latent(input_latents)).float(),
        "edit_latents": [t.float() for t in to_cpu(normalize_latent(edit_latents))]
        if edit_latents is not None and isinstance(edit_latents, list)
        else (to_cpu(normalize_latent(edit_latents)).float()
              if edit_latents is not None else None),
        "height": int(height),
        "width": int(width),
    }


def build_fixed_targets(
    encoded: dict,
    seed: int,
    idx: int,
    timestep_id: int,
) -> dict:
    """Add deterministic noise / timestep / latents / training_target / scale."""
    input_latents = encoded["input_latents"]  # already float32
    gen = torch.Generator("cpu").manual_seed(seed + idx)
    noise = torch.randn(input_latents.shape, generator=gen, dtype=torch.float32)

    scheduler = FlowMatchScheduler("Qwen-Image")
    scheduler.set_timesteps(1000, training=True)
    ts_id = torch.tensor([timestep_id], dtype=torch.long)
    timestep = scheduler.timesteps[ts_id].to(dtype=torch.float32).view(1)
    latents = scheduler.add_noise(input_latents, noise, timestep)
    training_target = noise - input_latents
    scale = scheduler.training_weight(timestep).to(dtype=torch.float32).view(1)

    encoded.update({
        "noise": noise,
        "timestep_id": ts_id,
        "timestep": timestep,
        "latents": latents,
        "training_target": training_target,
        "scale": scale,
    })
    return encoded


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_base_path", type=str, required=True,
                   help="Base directory for image / edit_image paths in metadata.")
    p.add_argument("--dataset_metadata_path", type=str, required=True,
                   help="Metadata file (.json / .jsonl / .csv).")
    p.add_argument("--model_root", type=str, required=True,
                   help="Qwen-Image-Edit-2511 model root (contains transformer/, "
                        "text_encoder/, vae/, tokenizer/, processor/).")
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen-Image",
                   help="HF model_id for the tokenizer (uses model_root/tokenizer/ by default).")
    p.add_argument("--processor_id", type=str, default="Qwen/Qwen-Image-Edit",
                   help="HF model_id for the processor (uses model_root/processor/ by default).")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output cache directory; each sample is written as <idx>.pth.")
    p.add_argument("--max_pixels", type=int, default=1024 * 1024,
                   help="Max pixel budget for the target image (default 1MP).")
    p.add_argument("--seed", type=int, default=1234,
                   help="Base seed for deterministic per-sample noise generation.")
    p.add_argument("--timestep_id", type=int, default=321,
                   help="Fixed timestep index (0..999) for the pre-noised latents.")
    p.add_argument("--tiled_vae", action="store_true",
                   help="Enable tiled VAE encoding (for large images at low VRAM).")
    p.add_argument("--torch_dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return p


def main():
    """Entry point."""
    args = build_parser().parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    model_root = os.path.abspath(args.model_root)

    def _resolve_weight_files(subdir: str) -> Union[str, List[str]]:
        """Return safetensors file list under ``<model_root>/<subdir>``.

        DiffSynth's ``ModelConfig.path`` must point at file(s), not a directory;
        passing a directory raises ``IsADirectoryError`` when the loader opens
        it. Expand the safetensors glob here so ``download_if_necessary`` sees
        a concrete file list.
        """
        base = os.path.join(model_root, subdir)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"{base} is not a directory")
        files = sorted(glob.glob(os.path.join(base, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors files under {base}")
        return files[0] if len(files) == 1 else files

    model_configs = [
        ModelConfig(path=_resolve_weight_files("text_encoder")),
        ModelConfig(path=_resolve_weight_files("vae")),
    ]
    tokenizer_config = ModelConfig(
        path=os.path.join(model_root, "tokenizer"),
        model_id=args.tokenizer_id,
        origin_file_pattern="tokenizer/",
    )
    processor_config = ModelConfig(
        path=os.path.join(model_root, "processor"),
        model_id=args.processor_id,
        origin_file_pattern="processor/",
    )

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=str(device),
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        processor_config=processor_config,
    )
    # DiT / ControlNet / image2lora modules are not used for preprocessing;
    # free the memory if from_pretrained happened to load them.
    for attr in ("dit", "blockwise_controlnet", "siglip2_image_encoder",
                 "dinov3_image_encoder", "image2lora_style",
                 "image2lora_coarse", "image2lora_fine"):
        if getattr(pipe, attr, None) is not None:
            setattr(pipe, attr, None)

    prompt_unit = QwenImageUnit_PromptEmbedder()
    input_unit = QwenImageUnit_InputImageEmbedder()
    edit_unit = QwenImageUnit_EditImageEmbedder()
    # Match DiffSynth's ``UnifiedDataset.default_image_operator`` exactly so
    # ``image`` and ``edit_image`` receive the same resize+crop treatment as
    # the reference ``sft:data_process`` cache.
    image_op = ImageCropAndResize(
        height=None,
        width=None,
        max_pixels=args.max_pixels,
        height_division_factor=16,
        width_division_factor=16,
    )

    metadata = load_metadata(args.dataset_metadata_path)
    proc_idx = accelerator.process_index
    num_procs = accelerator.num_processes
    local_ids = list(range(proc_idx, len(metadata), num_procs))

    out_dir = os.path.join(args.output_path, str(proc_idx))
    os.makedirs(out_dir, exist_ok=True)

    for data_id in tqdm(local_ids, desc=f"proc {proc_idx}"):
        save_path = os.path.join(out_dir, f"{data_id:05d}.pth")
        if os.path.exists(save_path):
            continue
        row = metadata[data_id]
        prompt = str(row.get("prompt", ""))
        image_field = row.get("image")
        edit_field = row.get("edit_image")
        if image_field is None:
            print(f"[WARN] sample {data_id} has no 'image' field, skipping")
            continue
        try:
            image = load_image(image_field, args.dataset_base_path)
            edit_image = load_edit_images(edit_field, args.dataset_base_path)
            encoded = encode_sample(
                pipe=pipe,
                prompt=prompt,
                image=image,
                edit_image=edit_image,
                prompt_unit=prompt_unit,
                input_unit=input_unit,
                edit_unit=edit_unit,
                image_op=image_op,
                tiled=args.tiled_vae,
            )
            build_fixed_targets(encoded, args.seed, data_id, args.timestep_id)
            torch.save(encoded, save_path)
        except Exception as exc:  # noqa: BLE001 — log-and-continue on per-sample errors
            print(f"[WARN] skipping sample {data_id} ({image_field}): {exc}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Flatten per-process subdirectories into the output root.
        for proc_dir in sorted(os.listdir(args.output_path)):
            subdir = os.path.join(args.output_path, proc_dir)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if not fname.endswith(".pth"):
                    continue
                src = os.path.join(subdir, fname)
                dst = os.path.join(args.output_path, fname)
                if os.path.exists(dst):
                    print(f"[WARN] conflict: {dst} already exists, skipping {src}")
                    continue
                shutil.move(src, dst)
            shutil.rmtree(subdir)
        print(f"Preprocessing complete. Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
