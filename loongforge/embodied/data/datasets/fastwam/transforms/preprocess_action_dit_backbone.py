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

"""Preprocess ActionDiT backbone weights from WanVideoDiT and save as a .pt payload.

This script extracts shared backbone weights from the Wan2.2 video DiT and
initialises the ActionDiT checkpoint consumed by ``ActionDiT.from_pretrained``.
The output payload format is:

    {
        "policy": {...},
        "backbone_state_dict": {key: tensor, ...},
        "meta": {hidden_dim, ffn_dim, num_layers, num_heads, attn_head_dim,
                 text_dim, freq_dim, eps},
    }

Architecture defaults match ``FastWAMModelConfig`` (Wan2.2-TI2V-5B backbone,
ActionDiT hidden_dim=1024). Override ``--action-hidden-dim`` when experimenting
with different action expert sizes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F


logger = logging.getLogger("preprocess_action_dit_backbone")

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

# ActionDiT defaults (must match FastWAMModelConfig.action_dit_config)
DEFAULT_ACTION_HIDDEN_DIM = 1024
DEFAULT_ACTION_FFN_DIM = 4096
DEFAULT_ACTION_DIM = 7
DEFAULT_NUM_HEADS = 24
DEFAULT_ATTN_HEAD_DIM = 128
DEFAULT_NUM_LAYERS = 30
DEFAULT_TEXT_DIM = 4096
DEFAULT_FREQ_DIM = 256
DEFAULT_EPS = 1e-6


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Convert a dtype string alias (bf16/fp16/fp32 etc.) to ``torch.dtype``."""
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """1-D linear interpolation along the last dimension of *tensor*."""
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Resize src tensor to target_shape via sequential 1D linear interpolation."""
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the backbone preprocessing script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True,
        help="Output .pt path for the preprocessed ActionDiT backbone.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument(
        "--device", default="cpu",
        help="Device for loading and preprocessing (cpu recommended).",
    )
    parser.add_argument(
        "--dtype", default="float32",
        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
    )
    parser.add_argument("--action-hidden-dim", type=int, default=DEFAULT_ACTION_HIDDEN_DIM)
    parser.add_argument("--action-ffn-dim", type=int, default=DEFAULT_ACTION_FFN_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument(
        "--no-alpha-scaling", action="store_true",
        help="Disable alpha=sqrt(dv/da) scaling when last dim is resized.",
    )
    parser.add_argument(
        "--local-model-path", default=None,
        help="Local root directory for model artifacts (equivalent to DIFFSYNTH_MODEL_BASE_PATH). "
             "Model files are expected at <local-model-path>/<model-id>/.",
    )
    return parser.parse_args()


def main() -> None:
    """Extract and resize the ActionDiT backbone from a pretrained WAN model, then save to disk."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    from loongforge.embodied.model.fastwam.action.dit import ActionDiT
    from loongforge.embodied.model.fastwam.wan.loader import _load_registered_model, _resolve_configs

    apply_alpha_scaling = not args.no_alpha_scaling
    torch_dtype = _resolve_dtype(args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    action_cfg = {
        "action_dim": args.action_dim,
        "hidden_dim": args.action_hidden_dim,
        "ffn_dim": args.action_ffn_dim,
        "num_heads": DEFAULT_NUM_HEADS,
        "attn_head_dim": DEFAULT_ATTN_HEAD_DIM,
        "num_layers": DEFAULT_NUM_LAYERS,
        "text_dim": DEFAULT_TEXT_DIM,
        "freq_dim": DEFAULT_FREQ_DIM,
        "eps": DEFAULT_EPS,
    }

    # video_dit_config mirrors FastWAMModelConfig.video_dit_config (Wan2.2-5B)
    video_dit_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48, "out_dim": 48,
        "hidden_dim": 3072, "ffn_dim": 14336,
        "freq_dim": DEFAULT_FREQ_DIM, "text_dim": DEFAULT_TEXT_DIM,
        "num_heads": DEFAULT_NUM_HEADS, "attn_head_dim": DEFAULT_ATTN_HEAD_DIM,
        "num_layers": DEFAULT_NUM_LAYERS,
        "eps": DEFAULT_EPS, "seperated_timestep": True,
        "require_clip_embedding": False, "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": args.action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
    }

    logger.info(
        "Preprocessing ActionDiT backbone: model_id=%s device=%s dtype=%s "
        "action_hidden_dim=%d action_ffn_dim=%d apply_alpha_scaling=%s",
        args.model_id, args.device, torch_dtype,
        args.action_hidden_dim, args.action_ffn_dim, apply_alpha_scaling,
    )

    dit_config, _, _, _ = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=True,
    )
    if args.local_model_path:
        dit_config.local_model_path = args.local_model_path
    dit_config.download_if_necessary()

    video_expert = _load_registered_model(
        dit_config.path,
        "wan_video_dit",
        torch_dtype=torch_dtype,
        device=args.device,
        model_kwargs_override=video_dit_config,
    ).eval()

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)

    if action_cfg["num_heads"] != video_expert.num_heads:
        raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
    if action_cfg["attn_head_dim"] != video_expert.attn_head_dim:
        raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if action_cfg["num_layers"] != len(video_expert.blocks):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": action_cfg["hidden_dim"],
            "ffn_dim": action_cfg["ffn_dim"],
            "num_layers": action_cfg["num_layers"],
            "num_heads": action_cfg["num_heads"],
            "attn_head_dim": action_cfg["attn_head_dim"],
            "text_dim": action_cfg["text_dim"],
            "freq_dim": action_cfg["freq_dim"],
            "eps": float(action_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    logger.info(
        "Saved ActionDiT backbone payload to %s (copied=%d, interpolated=%d, skipped=%d).",
        output_path, copied, interpolated, skipped,
    )


if __name__ == "__main__":
    main()
