# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Wan2.2-I2V preprocessing script.

Encodes each (video, text) sample into training tensors and saves them as .pth files.
Only the T5 text encoder and VAE are required; the DiT is not loaded.

Output keys per sample:
  context, input_latents, y, height, width, num_frames,
  max_timestep_boundary, min_timestep_boundary

Dependencies:
  pip install diffsynth==1.1.8
  pip install "huggingface_hub[cli]"

Download required model files:
  huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan-AI/Wan2.2-I2V-A14B

Example:
MODEL_T5=/ssd1/models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth
MODEL_VAE=/ssd1/models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth
accelerate launch wan_preprocess.py \
  --dataset_base_path fake_dataset \
  --dataset_metadata_path fake_dataset/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --model_paths "${MODEL_T5},${MODEL_VAE}" \
  --tokenizer_local_path "/ssd1/models/Wan-AI/Wan2.2-I2V-A14B/google/umt5-xxl" \
  --output_path ./data/preprocessed \
  --max_timestep_boundary 0.358 --min_timestep_boundary 0

"""

import os, json, argparse, shutil
import numpy as np
import pandas
import imageio
import imageio.v3 as iio
import torch
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.prompters.wan_prompter import WanPrompter


def preprocess_image(image: Image.Image, device, torch_dtype) -> torch.Tensor:
    """PIL.Image -> Tensor [1, C, H, W], range [-1, 1]."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    t = torch.from_numpy(arr).to(device=device, dtype=torch_dtype)
    t = t * (2.0 / 255.0) - 1.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t


def preprocess_video(frames: list, device, torch_dtype) -> torch.Tensor:
    """List[PIL.Image] -> Tensor [1, C, T, H, W], range [-1, 1]."""
    tensors = [preprocess_image(f, device, torch_dtype) for f in frames]
    return torch.stack([t.squeeze(0) for t in tensors], dim=1).unsqueeze(0)


def load_metadata(metadata_path: str) -> list:
    """Load a .csv / .json / .jsonl metadata file into a list of dicts."""
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r") as f:
            return json.load(f)
    elif metadata_path.endswith(".jsonl"):
        with open(metadata_path, "r") as f:
            return [json.loads(line.strip()) for line in f]
    else:
        df = pandas.read_csv(metadata_path)
        return [df.iloc[i].to_dict() for i in range(len(df))]


def load_video_frames(
    path: str,
    height: int,
    width: int,
    num_frames: int,
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
) -> list:
    """Load video frames, center-crop/resize to (height, width).

    The frame count is clipped to satisfy
    ``len(frames) % time_division_factor == time_division_remainder``,
    which is required by the VAE temporal downsampling.
    """
    ext = path.rsplit(".", 1)[-1].lower()

    def crop_resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = max(width / w, height / h)
        nw, nh = round(w * scale), round(h * scale)
        import torchvision.transforms.functional as TF
        img = TF.resize(img, (nh, nw), interpolation=TF.InterpolationMode.BILINEAR)
        return TF.center_crop(img, (height, width))

    def clip_num_frames(total: int) -> int:
        n = min(total, num_frames)
        while n > 1 and n % time_division_factor != time_division_remainder:
            n -= 1
        return n

    if ext in ("jpg", "jpeg", "png", "webp"):
        return [crop_resize(Image.open(path).convert("RGB"))]

    elif ext == "gif":
        images = iio.imread(path, mode="RGB")
        n = clip_num_frames(len(images))
        frames = []
        for arr in images:
            frames.append(crop_resize(Image.fromarray(arr)))
            if len(frames) >= n:
                break
        return frames

    else:  # mp4, avi, mov, ...
        reader = imageio.get_reader(path)
        n = clip_num_frames(int(reader.count_frames()))
        frames = [crop_resize(Image.fromarray(reader.get_data(i))) for i in range(n)]
        reader.close()
        return frames


@torch.no_grad()
def preprocess(
    frames: list,
    prompt: str,
    pipe: WanVideoPipeline,
    torch_dtype: torch.dtype,
    max_timestep_boundary: float,
    min_timestep_boundary: float,
) -> dict:
    """Encode one (video, text) sample into training tensors.

    Returns a CPU-resident dict ready for torch.save.
    Noise is excluded and sampled fresh on each training step.
    """
    device = pipe.device
    height, width, num_frames = frames[0].size[1], frames[0].size[0], len(frames)

    # T5 text encoding -> context [1, seq_len, dim]
    pipe.load_models_to_device(["text_encoder"])
    ids, mask = pipe.prompter.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids, mask = ids.to(device), mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[:, v:] = 0  # zero-pad beyond actual sequence length
    context = context.to(dtype=torch_dtype, device=device)

    # VAE encode video -> input_latents [1, C_z, T_l, H//8, W//8]
    pipe.load_models_to_device(["vae"])
    input_latents = pipe.vae.encode(
        preprocess_video(frames, device, torch_dtype), device=device, tiled=False
    ).to(dtype=torch_dtype, device=device)

    # VAE encode first frame -> y [1, C_z+4, T_l, H//8, W//8]
    # y = visibility mask (frame 0 = 1, others = 0) concatenated with first-frame latent.
    image = preprocess_image(frames[0].resize((width, height)), device, torch_dtype)
    msk = torch.zeros(1, num_frames, height // 8, width // 8, device=device, dtype=torch_dtype)
    msk[:, 0] = 1.0
    # Expand the first-frame mask slot by the VAE temporal downsampling factor (4).
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8).transpose(1, 2)[0]
    vae_input = torch.cat([
        image.squeeze(0).unsqueeze(1),  # [C, 1, H, W]
        torch.zeros(3, num_frames - 1, height, width, device=device, dtype=torch_dtype),
    ], dim=1)
    y = pipe.vae.encode([vae_input], device=device, tiled=False)[0].to(dtype=torch_dtype, device=device)
    y = torch.cat([msk, y]).unsqueeze(0)  # [1, C_z+4, T_l, H//8, W//8]

    return {
        "context":               context.cpu(),
        "input_latents":         input_latents.cpu(),
        "y":                     y.cpu(),
        "height":                height,
        "width":                 width,
        "num_frames":            num_frames,
        "max_timestep_boundary": max_timestep_boundary,
        "min_timestep_boundary": min_timestep_boundary,
    }


def build_parser() -> argparse.ArgumentParser:
    """build parser"""
    p = argparse.ArgumentParser(
        description="Wan2.2-I2V offline preprocessing: encode dataset samples into .pth cache files."
    )
    p.add_argument("--dataset_base_path",     type=str, required=True)
    p.add_argument("--dataset_metadata_path", type=str, default=None,
                   help="Path to metadata file (.csv / .json / .jsonl).")
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--width",      type=int, default=832)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--max_pixels", type=int, default=1280 * 720)
    p.add_argument("--model_paths", type=str, required=True,
                   help="Comma-separated local paths to T5 and VAE weight files.")
    p.add_argument("--tokenizer_local_path", type=str, required=True,
                   help="Local directory of the UMT5 tokenizer.")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output root; per-process subdirs are created automatically.")
    p.add_argument("--max_timestep_boundary", type=float, default=1.0)
    p.add_argument("--min_timestep_boundary", type=float, default=0.0)
    p.add_argument("--dataset_num_workers",   type=int,   default=0)
    return p


def main():
    """main"""
    args = build_parser().parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    torch_dtype = torch.bfloat16

    model_configs = [ModelConfig(path=p.strip()) for p in args.model_paths.split(",")]
    # abspath prevents AutoTokenizer from misinterpreting a relative path as a repo ID.
    tokenizer_config = ModelConfig(path=os.path.abspath(args.tokenizer_local_path))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=str(device),
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    for param in pipe.parameters() if hasattr(pipe, "parameters") else []:
        param.requires_grad_(False)

    assert args.dataset_metadata_path is not None, \
        "--dataset_metadata_path is required."
    metadata = load_metadata(args.dataset_metadata_path)

    proc_idx  = accelerator.process_index
    num_procs = accelerator.num_processes
    local_ids = list(range(proc_idx, len(metadata), num_procs))

    out_dir = os.path.join(args.output_path, str(proc_idx))
    os.makedirs(out_dir, exist_ok=True)

    for data_id in tqdm(local_ids, desc=f"proc {proc_idx}"):
        save_path = os.path.join(out_dir, f"{data_id}.pth")
        if os.path.exists(save_path):
            continue

        row = metadata[data_id]
        video_file = os.path.join(args.dataset_base_path, row["video"])
        prompt     = str(row.get("prompt", ""))

        try:
            frames = load_video_frames(video_file, args.height, args.width, args.num_frames)
            result = preprocess(
                frames=frames, prompt=prompt, pipe=pipe, torch_dtype=torch_dtype,
                max_timestep_boundary=args.max_timestep_boundary,
                min_timestep_boundary=args.min_timestep_boundary,
            )
            model_input_keys = ["input_latents", "context", "y"]
            data = {key: result[key] for key in model_input_keys if key in result}
            torch.save(data, save_path)
            # torch.save(result, save_path)
        except Exception as e:
            print(f"[WARN] Skipping sample {data_id} ({video_file}): {e}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Merge per-process subdirectories into the output root and remove them.
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
                    print(f"[WARN] Conflict: {dst} already exists, skipping {src}")
                    continue
                shutil.move(src, dst)
            shutil.rmtree(subdir)
        print(f"Preprocessing complete. Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
