"""Vendor module implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
import einops
from PIL import Image, ImageDraw

_VISION_MODELS_DIR = Path(__file__).resolve().parent
for _DEPTH_PACKAGE_DIR in (
    _VISION_MODELS_DIR / "lingbot-depth",
    _VISION_MODELS_DIR / "MoGe",
):
    if _DEPTH_PACKAGE_DIR.exists():
        sys.path.insert(0, str(_DEPTH_PACKAGE_DIR))

try:
    from mdm.model.v2 import MDMModel as v2_morgbd
    from moge.model.v2 import MoGeModel as v2
    from moge.utils.vis import colorize_depth
except:
    print('Load MoGe Module Failed!!')


def make_grid(images, pil_images):
    # Assuming each image is the same size
    """Vendor interface."""
    new_images = []
    new_captions = []
    for image, pil_image in zip(images, pil_images):
        new_images.append(image)
        pil_image = pil_image.resize((image.size[0], image.size[1]))
        new_images.append(pil_image)
        new_captions.append("Predicted")
        new_captions.append("GT")

    images = new_images
    captions = new_captions

    width, height = images[0].size
    font_size = 14
    caption_height = font_size + 10

    # Calculate the size of the final image
    images_per_row = min(len(images), 16)  # Round up for odd number of images
    row_count = (len(images) + 1) // images_per_row
    total_width = width * images_per_row
    total_height = (height + caption_height) * row_count

    # Create a new blank image
    new_image = Image.new("RGB", (total_width, total_height), "white")

    draw = ImageDraw.Draw(new_image)

    for i, (image, caption) in enumerate(zip(images, captions)):
        row = i // images_per_row
        col = i % images_per_row
        x_offset = col * width
        y_offset = row * (height + caption_height)

        new_image.paste(image, (x_offset, y_offset))
        text_position = (x_offset + 10, y_offset + height)
        draw.text(text_position, caption, fill="red", font_size=font_size)

    return new_image


def build_depth_model(config):
    """Vendor interface."""
    model_type = config['depth']['model_type']
    if model_type != 'MoRGBD':
        raise ValueError(f"Only Lingbot-Depth distillation is supported, got {model_type!r}.")

    moge_model = v2.from_pretrained(config['depth']['moge_path'])
    for p in moge_model.parameters():
        p.requires_grad = False
    moge_model.cuda()
    moge_model.eval()

    morgbd_model = v2_morgbd.from_pretrained(config['depth']['morgbd_path'])
    for p in morgbd_model.parameters():
        p.requires_grad = False
    morgbd_model.cuda()
    morgbd_model.eval()
    return moge_model, morgbd_model


def _rgb_range(images):
    """Vendor interface."""
    if images.numel() == 0:
        return 0.0, 0.0
    rgb_min = float(images.detach().amin().cpu())
    rgb_max = float(images.detach().amax().cpu())
    if rgb_min < -1e-6 or rgb_max > 255.0 + 1e-6:
        raise ValueError(f"Unexpected RGB range [{rgb_min}, {rgb_max}]. Expected [0,1] or [0,255].")
    return rgb_min, rgb_max


def _rgb_to_unit_float(images):
    # is_uint8 = images.dtype == torch.uint8
    """Vendor interface."""
    images = images.float()
    # _, rgb_max = _rgb_range(images)
    # if is_uint8 or rgb_max > 1.0 + 1e-6:
    images = images / 255.0
    return images


def _video_config(config):
    """Vendor interface."""
    if config is None:
        return {}
    if hasattr(config, "get"):
        return config.get("video", config)
    return getattr(config, "video", config)


def _cfg_get(config, key, default=None):
    """Vendor interface."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _tokens_grid_hw(num_tokens):
    """Vendor interface."""
    side = int(num_tokens ** 0.5)
    for height in range(side, 0, -1):
        if num_tokens % height == 0:
            return height, num_tokens // height
    return 1, num_tokens


def _resize_feature_image(image, height, width, config):
    """Vendor interface."""
    log_scale = int(_cfg_get(config, "log_scale", 0) or 0)
    if log_scale <= 0:
        log_scale = max(1, 256 // max(height, width))
    if log_scale == 1:
        return image
    resample = getattr(Image, "Resampling", Image).NEAREST
    return image.resize((width * log_scale, height * log_scale), resample=resample)


def _feature_pair_to_images(pred_tokens, target_tokens, config):
    """Vendor interface."""
    height, width = _tokens_grid_hw(target_tokens.shape[0])
    pair = torch.cat([target_tokens, pred_tokens], dim=0).float()
    pair = torch.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)
    centered = pair - pair.mean(dim=0, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[: min(3, vh.shape[0])].T
    except RuntimeError:
        projected = centered[:, : min(3, centered.shape[-1])]

    if projected.shape[-1] < 3:
        pad = torch.zeros(projected.shape[0], 3 - projected.shape[-1], dtype=projected.dtype)
        projected = torch.cat([projected, pad], dim=-1)

    min_val = projected.amin(dim=0, keepdim=True)
    max_val = projected.amax(dim=0, keepdim=True)
    rgb = (projected - min_val) / (max_val - min_val).clamp_min(1e-6)
    rgb = (rgb * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

    target_rgb = rgb[: target_tokens.shape[0]].reshape(height, width, 3)
    pred_rgb = rgb[target_tokens.shape[0] :].reshape(height, width, 3)

    diff = (pred_tokens.float() - target_tokens.float()).abs().mean(dim=-1)
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    diff = (diff - diff.min()) / (diff.max() - diff.min()).clamp_min(1e-6)
    cmap = matplotlib.colormaps.get_cmap("magma")
    diff_rgb = (cmap(diff.reshape(height, width).cpu().numpy())[:, :, :3] * 255).astype(np.uint8)

    target_img = Image.fromarray(target_rgb)
    pred_img = Image.fromarray(pred_rgb)
    diff_img = Image.fromarray(diff_rgb)
    return (
        _resize_feature_image(target_img, height, width, config),
        _resize_feature_image(pred_img, height, width, config),
        _resize_feature_image(diff_img, height, width, config),
    )


def _feature_tokens_to_image(tokens, config):
    """Vendor interface."""
    height, width = _tokens_grid_hw(tokens.shape[0])
    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    centered = tokens - tokens.mean(dim=0, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[: min(3, vh.shape[0])].T
    except RuntimeError:
        projected = centered[:, : min(3, centered.shape[-1])]

    if projected.shape[-1] < 3:
        pad = torch.zeros(projected.shape[0], 3 - projected.shape[-1], dtype=projected.dtype)
        projected = torch.cat([projected, pad], dim=-1)

    min_val = projected.amin(dim=0, keepdim=True)
    max_val = projected.amax(dim=0, keepdim=True)
    rgb = (projected - min_val) / (max_val - min_val).clamp_min(1e-6)
    rgb = (rgb * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(rgb.reshape(height, width, 3))
    return _resize_feature_image(image, height, width, config)


def _tensor_rgb_to_pil(rgb_tensor):
    """Vendor interface."""
    tensor = rgb_tensor.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"RGB image tensor must have 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in (1, 3):
        raise ValueError(f"RGB image tensor must be CHW or HWC, got {tuple(rgb_tensor.shape)}")
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)

    if tensor.is_floating_point():
        tensor = tensor.float()
        if tensor.numel() > 0 and tensor.min() >= 0 and tensor.max() <= 1:
            tensor = tensor * 255.0
        array = tensor.clamp(0, 255).to(torch.uint8).numpy()
    else:
        array = tensor.clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(array).convert("RGB")


def _select_rgb_image(rgb_images, sample_idx):
    """Vendor interface."""
    if rgb_images is None:
        return None
    if isinstance(rgb_images, Image.Image):
        return rgb_images.convert("RGB")
    if isinstance(rgb_images, (list, tuple)):
        if sample_idx >= len(rgb_images):
            return None
        item = rgb_images[sample_idx]
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if torch.is_tensor(item):
            return _tensor_rgb_to_pil(item)
        return Image.fromarray(np.asarray(item)).convert("RGB")
    if torch.is_tensor(rgb_images):
        if rgb_images.ndim == 5:
            if sample_idx >= rgb_images.shape[0]:
                return None
            return _tensor_rgb_to_pil(rgb_images[sample_idx, 0])
        if rgb_images.ndim == 4:
            if sample_idx >= rgb_images.shape[0]:
                return None
            return _tensor_rgb_to_pil(rgb_images[sample_idx])
        if rgb_images.ndim == 3:
            return _tensor_rgb_to_pil(rgb_images)
    return Image.fromarray(np.asarray(rgb_images)).convert("RGB")


def _resize_rgb_for_panel(image, target_size):
    """Vendor interface."""
    if image is None:
        return None
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize(target_size, resample=resample)


def _make_video_feature_panel(
    target_img,
    pred_img,
    diff_img,
    current_rgb_img=None,
    target_rgb_img=None,
    current_feat_img=None,
):
    """Vendor interface."""
    target_size = target_img.size
    images = []
    captions = []
    current_rgb_img = _resize_rgb_for_panel(current_rgb_img, target_size)
    target_rgb_img = _resize_rgb_for_panel(target_rgb_img, target_size)
    current_feat_img = _resize_rgb_for_panel(current_feat_img, target_size)
    if current_rgb_img is not None:
        images.append(current_rgb_img)
        captions.append("Current RGB")
    if target_rgb_img is not None:
        images.append(target_rgb_img)
        captions.append("Target RGB")
    if current_feat_img is not None:
        images.append(current_feat_img)
        captions.append("Current feat")
    images.extend([target_img, pred_img, diff_img])
    captions.extend(["Target feat", "Pred feat", "Abs diff"])
    width, height = images[0].size
    caption_height = 20
    panel = Image.new("RGB", (width * len(images), height + caption_height), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (image, caption) in enumerate(zip(images, captions)):
        x_offset = idx * width
        panel.paste(image, (x_offset, 0))
        draw.text((x_offset + 6, height + 3), caption, fill="red")
    return panel


def _panel_image_or_blank(image, target_size):
    """Vendor interface."""
    if image is None:
        return Image.new("RGB", target_size, "white")
    return _resize_rgb_for_panel(image, target_size)


def _make_video_feature_grid_panel(
    current_rgb_img,
    current_target_img,
    current_pred_img,
    current_diff_img,
    future_rgb_img,
    future_target_img,
    future_pred_img,
    future_diff_img,
):
    """Vendor interface."""
    target_size = future_target_img.size
    rows = [
        (
            [
                _panel_image_or_blank(current_rgb_img, target_size),
                _panel_image_or_blank(current_target_img, target_size),
                _panel_image_or_blank(current_pred_img, target_size),
                _panel_image_or_blank(current_diff_img, target_size),
            ],
            ["Current RGB", "Current feat GT", "Current feat Pred", "Abs diff"],
        ),
        (
            [
                _panel_image_or_blank(future_rgb_img, target_size),
                _panel_image_or_blank(future_target_img, target_size),
                _panel_image_or_blank(future_pred_img, target_size),
                _panel_image_or_blank(future_diff_img, target_size),
            ],
            ["Future RGB", "Future feat GT", "Future feat Pred", "Abs diff"],
        ),
    ]

    width, height = target_size
    caption_height = 20
    row_height = height + caption_height
    panel = Image.new("RGB", (width * 4, row_height * 2), "white")
    draw = ImageDraw.Draw(panel)
    for row_idx, (images, captions) in enumerate(rows):
        y_offset = row_idx * row_height
        for col_idx, (image, caption) in enumerate(zip(images, captions)):
            x_offset = col_idx * width
            panel.paste(image, (x_offset, y_offset))
            draw.text((x_offset + 6, y_offset + height + 3), caption, fill="red")
    return panel


def build_video_model(config):
    """Vendor interface."""
    video_cfg = _video_config(config)
    try:
        from loongforge.embodied.model.lingbot_vla_v2.vendor.vision_models.dino_video import build_dino_video_teacher
    except Exception as exc:
        raise ImportError(
            "Failed to import the self-contained DINO video teacher. "
            "Check lingbotvla/models/vla/vision_models/dino_video/."
        ) from exc
    return build_dino_video_teacher(video_cfg)


def get_video_target(video_teacher, pil_images, future_pil_images, config, effective_fps=None):
    """Vendor interface."""
    video_cfg = _video_config(config)
    use_patch = bool(_cfg_get(video_cfg, "use_patch_loss", True))
    use_cls = bool(_cfg_get(video_cfg, "use_cls_loss", False))
    if not use_patch and not use_cls:
        raise ValueError("future-video alignment requires use_patch_loss or use_cls_loss to be enabled.")
    if pil_images is None or future_pil_images is None:
        raise ValueError("future-video distillation requires pil_images and future_pil_images.")
    input_size = int(_cfg_get(video_cfg, "input_size", 256))
    use_warmup_frame = bool(_cfg_get(video_cfg, "use_warmup_frame", False))
    num_future_frames = int(_cfg_get(video_cfg, "num_future_frames", 1))
    device = pil_images.device

    f_data = future_pil_images.shape[1]
    if f_data < num_future_frames or (num_future_frames > 1 and f_data != num_future_frames):
        raise ValueError(
            f"future-video frame count mismatch: data F={f_data} vs config num_future_frames={num_future_frames}"
        )

    current = pil_images[:, :1, :, :, :]
    futures = future_pil_images[:, :num_future_frames, :, :, :]
    current = _rgb_to_unit_float(
        einops.rearrange(current, "b n c h w -> (b n) c h w", n=1).contiguous()
    )
    futures = _rgb_to_unit_float(
        einops.rearrange(futures, "b n c h w -> (b n) c h w", n=num_future_frames).contiguous()
    )

    if current.shape[-2:] != (input_size, input_size):
        current = F.interpolate(current, size=(input_size, input_size), mode="bilinear", align_corners=False)
    if futures.shape[-2:] != (input_size, input_size):
        futures = F.interpolate(futures, size=(input_size, input_size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=current.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=current.dtype).view(1, 3, 1, 1)
    current = (current - mean) / std
    futures = (futures - mean) / std
    futures = einops.rearrange(futures, "(b n) c h w -> b n c h w", n=num_future_frames)
    future_frames = [futures[:, i] for i in range(num_future_frames)]

    if use_warmup_frame:
        frames = [current.clone(), current, *future_frames]
        current_index = 1
    else:
        frames = [current, *future_frames]
        current_index = 0
    video = torch.stack(frames, dim=2).contiguous()

    return_cls = use_cls
    return_current = bool(_cfg_get(video_cfg, "use_current_patch_loss", False))
    kwargs = dict(return_cls=return_cls)
    if return_current:
        kwargs["return_current"] = True
    if current_index != 0:
        kwargs["current_index"] = current_index
    if effective_fps is not None:
        kwargs["fps"] = effective_fps
    with torch.no_grad():
        video_target = video_teacher.get_future_feature(video, **kwargs)
    if return_cls:
        if not return_current:
            patch_target, cls_target = video_target
            return (
                patch_target.detach().to(dtype=torch.bfloat16),
                cls_target.detach().to(dtype=torch.bfloat16),
            )
        patch_target, cls_target, current_patch, _current_cls = video_target
        return {
            "patch": patch_target.detach().to(dtype=torch.bfloat16),
            "cls": cls_target.detach().to(dtype=torch.bfloat16),
            "current_patch": current_patch.detach().to(dtype=torch.bfloat16),
        }
    if not return_current:
        return video_target.detach().to(dtype=torch.bfloat16)
    patch_target, current_patch = video_target
    return {
        "patch": patch_target.detach().to(dtype=torch.bfloat16),
        "cls": None,
        "current_patch": current_patch.detach().to(dtype=torch.bfloat16),
    }


def get_depth_target(model_type, depth_model, pil_images):
    """Vendor interface."""
    if model_type != 'MoRGBD':
        raise ValueError(f"Only Lingbot-Depth depth targets are supported, got {model_type!r}.")
    pil_images = pil_images[:, :1, :, :, :]
    images = einops.rearrange(pil_images, 'b n c h w -> (b n) c h w', n=1).contiguous().float()

    input_images = _rgb_to_unit_float(images)
    moge_model, morgbd_model = depth_model
    output_moge = moge_model.infer(input_images, resolution_level=3, num_tokens=256, apply_mask=False)
    depth_pred = output_moge['depth'].squeeze().detach().clone() # moge2
    depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
    depth_pred *= 1
    depth_down_scale = 1
    depth_target, cls_token = morgbd_model.infer_feat(input_images, depth_pred,
                                            depth_down_scale=depth_down_scale,
                                            resolution_level=3,
                                            num_tokens=256,
                                            enable_depth_mask=False)
    depth_target = depth_target.permute(0, 2, 3, 1)
    depth_target = depth_target.view(depth_target.shape[0], -1, depth_target.shape[-1])

    return depth_target.to(dtype=torch.bfloat16), cls_token


def log_depth(
    vis_head,
    depth_pred_feats,
    depth_target_feats=None,
    steps=0,
    config=None,
    cls_token=None,
    is_future=False,
):
    """Vendor interface."""
    model_type = config['depth']['model_type']
    if model_type != 'MoRGBD':
        raise ValueError(f"Only Lingbot-Depth depth visualization is supported, got {model_type!r}.")
    if config.get('mode') != "query":
        raise ValueError(f"Only query depth alignment is supported, got {config.get('mode')!r}.")
    depth_token_size = config['depth']['token_size']
    visual_dir = config['visual_dir']

    depth_pred_feats = depth_pred_feats.view(
        depth_pred_feats.shape[0],
        depth_token_size,
        depth_token_size,
        depth_pred_feats.shape[-1],
    )
    depth_pred_feats = depth_pred_feats.permute(0, 3, 1, 2)

    import cv2
    morgbd_model = vis_head
    depth_target_feats = depth_target_feats.view(
        depth_target_feats.shape[0],
        depth_token_size,
        depth_token_size,
        depth_target_feats.shape[-1],
    )
    depth_target_feats = depth_target_feats.permute(0, 3, 1, 2)

    output_morgbd_preds = morgbd_model.dec_depth(
        depth_pred_feats,
        cls_token,
        num_tokens=256,
        resolution_level=3,
        img_h=224,
        img_w=224,
    )
    output_morgbd_targets = morgbd_model.dec_depth(
        depth_target_feats,
        cls_token,
        num_tokens=256,
        resolution_level=3,
        img_h=224,
        img_w=224,
    )

    output_morgbd_preds = output_morgbd_preds['depth_reg'].squeeze().cpu().numpy()
    output_morgbd_targets = output_morgbd_targets['depth_reg'].squeeze().cpu().numpy()

    for idx, (output_morgbd_target, output_morgbd_pred) in enumerate(zip(output_morgbd_targets, output_morgbd_preds)):

        depth_list = [output_morgbd_target, output_morgbd_pred]
        depth_color_list = [cv2.cvtColor(colorize_depth(depth_raw), cv2.COLOR_RGB2BGR) for depth_raw in depth_list]

        depth_concat = np.concatenate(depth_color_list, axis=1)

        if not is_future:
            dst_path = os.path.join(visual_dir, f"depth_morgbd_{steps}_{idx}.png")
        else:
            dst_path = os.path.join(visual_dir, f"depth_morgbd_{steps}_{idx}_future.png")
        cv2.imwrite(dst_path, depth_concat)


def log_video(
    depth_pred_feats,
    depth_target_feats=None,
    steps=0,
    config=None,
    is_future=True,
    current_rgb_images=None,
    target_rgb_images=None,
    current_target_feats=None,
    current_pred_feats=None,
):
    """Vendor interface."""
    if depth_pred_feats is None or depth_target_feats is None or config is None:
        return None

    visual_dir = _cfg_get(config, "visual_dir", None)
    if visual_dir is None:
        return None
    os.makedirs(visual_dir, exist_ok=True)

    video_cfg = _video_config(config)
    pred_feats = depth_pred_feats.detach().float().cpu()
    target_feats = depth_target_feats.detach().float().cpu()
    if pred_feats.ndim != 3 or target_feats.ndim != 3:
        raise ValueError("log_video expects [batch, tokens, channels] feature tensors.")
    if pred_feats.shape != target_feats.shape:
        raise ValueError(
            f"log_video shape mismatch: pred={tuple(pred_feats.shape)}, "
            f"target={tuple(target_feats.shape)}"
        )

    max_samples = int(_cfg_get(video_cfg, "log_max_samples", 8))
    num_samples = min(pred_feats.shape[0], max_samples)
    current_feats = None
    if current_target_feats is not None:
        current_feats = current_target_feats.detach().float().cpu()
        if current_feats.shape != target_feats.shape:
            raise ValueError(
                "log_video current_target_feats shape mismatch: "
                f"current={tuple(current_feats.shape)}, target={tuple(target_feats.shape)}"
            )
    current_pred_feats_cpu = None
    if current_pred_feats is not None:
        current_pred_feats_cpu = current_pred_feats.detach().float().cpu()
        if current_pred_feats_cpu.shape != target_feats.shape:
            raise ValueError(
                "log_video current_pred_feats shape mismatch: "
                f"current_pred={tuple(current_pred_feats_cpu.shape)}, target={tuple(target_feats.shape)}"
            )

    suffix = "_future" if is_future else ""
    metrics_path = os.path.join(visual_dir, f"video_dino_{steps}{suffix}_metrics.jsonl")

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for idx in range(num_samples):
            pred_tokens = pred_feats[idx]
            target_tokens = target_feats[idx]
            current_tokens = current_feats[idx] if current_feats is not None else None
            current_pred_tokens = (
                current_pred_feats_cpu[idx] if current_pred_feats_cpu is not None else None
            )
            target_img, pred_img, diff_img = _feature_pair_to_images(pred_tokens, target_tokens, video_cfg)
            if current_tokens is not None and current_pred_tokens is not None:
                current_target_img, current_pred_img, current_diff_img = _feature_pair_to_images(
                    current_pred_tokens,
                    current_tokens,
                    video_cfg,
                )
                panel = _make_video_feature_grid_panel(
                    _select_rgb_image(current_rgb_images, idx),
                    current_target_img,
                    current_pred_img,
                    current_diff_img,
                    _select_rgb_image(target_rgb_images, idx),
                    target_img,
                    pred_img,
                    diff_img,
                )
            else:
                current_img = (
                    _feature_tokens_to_image(current_tokens, video_cfg)
                    if current_tokens is not None
                    else None
                )
                panel = _make_video_feature_panel(
                    target_img,
                    pred_img,
                    diff_img,
                    current_rgb_img=_select_rgb_image(current_rgb_images, idx),
                    target_rgb_img=_select_rgb_image(target_rgb_images, idx),
                    current_feat_img=current_img,
                )
            panel.save(os.path.join(visual_dir, f"video_dino_{steps}_{idx}{suffix}.png"))

            mse = F.mse_loss(pred_tokens, target_tokens).item()
            metrics = {
                "sample_idx": idx,
                "mse": float(mse),
            }
            cosine_distance = (1.0 - F.cosine_similarity(pred_tokens, target_tokens, dim=-1)).mean().item()
            metrics["cosine_distance"] = float(cosine_distance)
            metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
    return None
