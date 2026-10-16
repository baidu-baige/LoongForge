# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Tokenization and Aloha conversion follow VLDA/openpi.

"""VLDA sample transforms shared by training and inference."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from loongforge.embodied.data.datasets.transforms.registry import register_transform_builder
from loongforge.embodied.data.datasets.transforms.utils.normalizer import Normalizer

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
JOINT_FLIP = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
JOINT_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False])


def encode_state(state, adapt_to_pi):
    state = np.asarray(state).copy()
    if adapt_to_pi:
        state = state * JOINT_FLIP
        position = state[..., [6, 13]] * (0.05800 - 0.01844) + 0.01844
        ratio = (0.022**2 + position**2 - 0.036**2) / (2 * 0.022 * position)
        state[..., [6, 13]] = (np.arcsin(np.clip(ratio, -1, 1)) - 0.5476) / (1.6296 - 0.5476)
    return state


def encode_actions(actions, state, data_cfg):
    actions = np.asarray(actions).copy()
    if data_cfg.adapt_to_pi:
        actions = actions * JOINT_FLIP
        actions[..., [6, 13]] = actions[..., [6, 13]] * (1.4910 + 0.6213) - 0.6213 - 0.5476
    if data_cfg.delta_joint_actions:
        actions[..., JOINT_MASK] -= state[..., JOINT_MASK]
    return actions


def decode_actions(actions, state, data_cfg):
    actions = np.asarray(actions).copy()
    if data_cfg.delta_joint_actions:
        actions[..., JOINT_MASK] += state[..., JOINT_MASK]
    if data_cfg.adapt_to_pi:
        actions = actions * JOINT_FLIP
        actions[..., [6, 13]] = (actions[..., [6, 13]] + 0.5476 + 0.6213) / (1.4910 + 0.6213)
    return actions


def resize_image(image, size, backend="loongforge"):
    """Antialiased linear resize and black padding, before uint8 normalization."""
    image = torch.as_tensor(np.asarray(image).copy())
    if image.shape[-1] != 3:
        image = image.permute(1, 2, 0)
    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)
    height, width = image.shape[:2]
    ratio = max(height / size, width / size)
    new_height, new_width = int(height / ratio), int(width / ratio)
    if backend == "native":
        resized = Image.fromarray(image.numpy()).resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size))
        canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
        return raw_image_tensor(np.asarray(canvas))
    image = (
        F.interpolate(
            image.permute(2, 0, 1)[None].float(),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)[0]
    )
    top, left = (size - new_height) // 2, (size - new_width) // 2
    return F.pad(image, (left, size - new_width - left, top, size - new_height - top)).float() / 127.5 - 1


def raw_image_tensor(image):
    image = torch.as_tensor(np.asarray(image).copy())
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    return image.float() / 127.5 - 1


class VLDATokenizer:
    """SentencePiece tokenization, including openpi's separate newline token."""

    def __init__(self, path, max_length):
        import sentencepiece

        path = Path(path)
        if path.is_dir():
            candidates = [path / "paligemma_tokenizer.model", path / "tokenizer.model"]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.max_length = max_length

    def __call__(self, prompt, state=None):
        prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discrete = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
            prompt = f"Task: {prompt}, State: {' '.join(map(str, discrete))};\nAction: "
            tokens = self.processor.encode(prompt, add_bos=True)
        else:
            tokens = self.processor.encode(prompt, add_bos=True) + self.processor.encode("\n")
        length = min(len(tokens), self.max_length)
        ids = torch.zeros(self.max_length, dtype=torch.long)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        ids[:length] = torch.tensor(tokens[:length])
        mask[:length] = True
        return ids, mask


class NativeQuantileNormalizer:
    """Match native VLDA's NumPy statistics precision and additive epsilon."""

    def __init__(self, statistics):
        self.q01 = np.asarray(statistics["q01"])
        self.q99 = np.asarray(statistics["q99"])

    def __call__(self, value):
        normalized = (np.asarray(value) - self.q01) / (self.q99 - self.q01 + 1e-6) * 2 - 1
        return torch.from_numpy(normalized)

    def inverse(self, value):
        unnormalized = (np.asarray(value) + 1) / 2 * (self.q99 - self.q01 + 1e-6) + self.q01
        return torch.from_numpy(unnormalized)


class VLDATransform:
    """Build VLDA observations using LF normalization and state-conditioned prompts."""

    def __init__(self, model_cfg, data_cfg, stats, tokenizer_path):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.stats = {
            name: stats[key if key in stats else name]
            for name, key in (("state", data_cfg.state_key), ("actions", data_cfg.action_key))
        }
        if data_cfg.normalization_mode == "q99":
            for name, values in self.stats.items():
                if not {"q01", "q99"}.issubset(values):
                    raise ValueError(
                        f"VLDA q99 normalization requires q01/q99 for {name}. "
                        "Add quantile statistics to dataset/meta/stats.json or set data.norm_stats_path "
                        "to a statistics file containing q01/q99."
                    )
        if data_cfg.preprocessing_backend == "native":
            self.state_normalizer = NativeQuantileNormalizer(self.stats["state"])
            self.action_normalizer = NativeQuantileNormalizer(self.stats["actions"])
        else:
            self.state_normalizer = Normalizer(data_cfg.normalization_mode, self.stats["state"])
            self.action_normalizer = Normalizer(data_cfg.normalization_mode, self.stats["actions"])
        self.tokenizer = VLDATokenizer(tokenizer_path, data_cfg.max_token_len)

    def __call__(self, sample):
        cfg, data = self.model_cfg, self.data_cfg
        raw_state = encode_state(sample["state"], data.adapt_to_pi)
        state = self.state_normalizer(torch.as_tensor(raw_state, dtype=torch.float32))
        tokens, mask = self.tokenizer(sample["prompt"], state.numpy() if cfg.discrete_state_input else None)
        images, image_masks = {}, {}
        for key, image in zip(IMAGE_KEYS, sample["images"], strict=True):
            image_masks[key] = torch.tensor(image is not None)
            if image is None:
                image = np.zeros_like(sample["images"][0])
            images[key] = resize_image(image, data.image_size, data.preprocessing_backend)
        output = {
            "images": images,
            "image_masks": image_masks,
            "state": F.pad(state.float(), (0, cfg.max_state_dim - state.shape[-1])),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
        }
        if "actions" in sample:
            actions = encode_actions(sample["actions"], raw_state, data)
            actions = self.action_normalizer(torch.as_tensor(actions, dtype=torch.float32))
            output["actions"] = F.pad(actions.float(), (0, cfg.max_action_dim - actions.shape[-1]))
        if "neg_actions" in sample:
            neg_states = encode_state(sample["neg_states"], data.adapt_to_pi)
            negatives = encode_actions(sample["neg_actions"], neg_states, data)
            negatives = self.action_normalizer(torch.as_tensor(negatives, dtype=torch.float32))
            output["neg_actions"] = F.pad(negatives.float(), (0, cfg.max_action_dim - negatives.shape[-1]))
        for key in ("vjepa_image", "vjepa_future_image"):
            if key in sample:
                output[key] = {IMAGE_KEYS[0]: raw_image_tensor(sample[key])}
        return output


@register_transform_builder("vlda")
def build_vlda_transforms(context):
    import os

    path = context.training_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
    return [VLDATransform(context.model_cfg, context.data_cfg, context.dataset_stats, path)]
