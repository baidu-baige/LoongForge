# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero-AgiBot on RoboDojo: eval-side protocol of the official XPolicyLab adapter.

Mirrors ``XPolicyLab/policy/DreamZero`` (``model.py`` + the vendored ``groot`` eval
transforms of the DreamZero-AgiBot checkpoint, ``experiment_cfg/conf.yaml``
``transform_agibot``):

- Three env-native 640x480 views (head, left wrist, right wrist), RGB. Every env step's
  frame is buffered; each inference sends the last ``video_history`` (4) frames per view,
  left-padded with the oldest frame (``_stack_recent_frames``).
- Per view: uint8 -> float [0, 1], CenterCrop to 0.95 of the metadata resolution
  (456x608), bilinear antialiased resize to 176x320, back to uint8 by truncation
  (``VideoToNumpy``); then the 2x2 grid [head | right wrist; left wrist | black]
  (``DreamTransform._prepare_video``) -> 352x640.
- State: the 14-D RoboDojo joints [left 6 + gripper, right 6 + gripper] mapped into the
  20-D AgiBot layout (6-DOF arm -> AgiBot slots [0, 1, 3, 4, 5, 6], J3 / head / waist 0),
  q99-normalized per key (clamped to [-1, 1]), concatenated and zero-padded to 64.
- Action: the (48, 32) chunk split per AgiBot action entry and denormalized with that
  entry's q99 statistics, arm / head / waist made absolute by adding the raw observed state
  (``relative_action_keys``), the first ``action_horizon`` (24) steps converted back to
  14-D joints (J3 dropped).
- Prompt: the AgiBot four-view template with the instruction lowercased, whitespace-cleaned.
"""

from __future__ import annotations

import html
import json
import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# (key, dim) in the checkpoint's state / action concat order (transform_agibot ConcatTransform).
AGIBOT_STATE_KEYS: Tuple[Tuple[str, int], ...] = (
    ("left_arm_joint_position", 7),
    ("right_arm_joint_position", 7),
    ("left_effector_position", 1),
    ("right_effector_position", 1),
    ("head_position", 2),
    ("waist_pitch", 1),
    ("waist_lift", 1),
)
AGIBOT_ACTION_KEYS: Tuple[Tuple[str, int], ...] = AGIBOT_STATE_KEYS + (("robot_velocity", 2),)
# Checkpoint conf.yaml ``relative_action_keys`` (effectors stay absolute).
AGIBOT_RELATIVE_ACTION_KEYS = frozenset(
    {"left_arm_joint_position", "right_arm_joint_position", "head_position", "waist_pitch", "waist_lift"}
)
# 6-DOF ARX X5 arm -> AgiBot 7-slot arm, J3 (slot 2) locked at 0 (official model.py).
ARX_TO_AGIBOT_ARM_SLOTS = (0, 1, 3, 4, 5, 6)
AGIBOT_STATE_DIM = sum(dim for _, dim in AGIBOT_STATE_KEYS)  # 20

# DreamTransform AgiBot branch (dreamzero_cotrain.py collate), with ``{i}`` = lowercased instruction.
_AGIBOT_PROMPT_TEMPLATE = (
    "A multi-view video shows that a robot {i} The video is split into four views: The top-left view "
    "shows the camera view from the robot's head, the top-right view shows the camera view from the "
    "right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right "
    "view is a black screen (inactive view). The robot {i}"
)


def joint14_to_agibot_state(joint14: Sequence[float]) -> np.ndarray:
    """RoboDojo 14-D joints -> AgiBot 20-D state (``_xpolicylab_packed_to_agibot_state``)."""
    j = np.asarray(joint14, dtype=np.float32).reshape(-1)
    if j.size != 14:
        raise ValueError(f"expected 14-D RoboDojo joints, got {j.size}")
    out = np.zeros(AGIBOT_STATE_DIM, dtype=np.float32)
    for arm_idx, offset in enumerate((0, 7)):
        for src, slot in enumerate(ARX_TO_AGIBOT_ARM_SLOTS):
            out[arm_idx * 7 + slot] = j[offset + src]
        out[14 + arm_idx] = j[offset + 6]
    return out


def agibot_to_joint14(left_arm: np.ndarray, right_arm: np.ndarray, left_ee: np.ndarray, right_ee: np.ndarray) -> np.ndarray:
    """Inverse of :func:`joint14_to_agibot_state` for one step (``_agibot_to_xpolicylab_packed``)."""
    parts = []
    for arm, ee in ((left_arm, left_ee), (right_arm, right_ee)):
        arm = np.asarray(arm, dtype=np.float32).reshape(-1)
        parts.append(arm[list(ARX_TO_AGIBOT_ARM_SLOTS)])
        parts.append(np.asarray(ee, dtype=np.float32).reshape(-1)[:1])
    return np.concatenate(parts).astype(np.float32)


def split_keys(vector: np.ndarray, keys: Sequence[Tuple[str, int]]) -> Dict[str, np.ndarray]:
    """Split the last axis of ``vector`` into ``{key: slice}`` following ``keys``."""
    out, start = {}, 0
    for key, dim in keys:
        out[key] = vector[..., start:start + dim]
        start += dim
    return out


def stack_recent_frames(frames: Sequence[np.ndarray], history: int) -> np.ndarray:
    """Last ``history`` frames, left-padded with the oldest one (official ``_stack_recent_frames``)."""
    padded = list(frames)
    while len(padded) < history:
        padded.insert(0, padded[0])
    return np.stack(padded[-history:], axis=0)


def agibot_view_transform(
    frames: np.ndarray, crop_scale: float = 0.95, size: Tuple[int, int] = (176, 320), metadata_hw: Tuple[int, int] = (480, 640)
) -> np.ndarray:
    """Eval video transform of one view, (T, H, W, 3) uint8 -> (T, 176, 320, 3) uint8.

    VideoToTensor -> VideoCrop (eval CenterCrop, size from the metadata resolution) ->
    VideoResize (bilinear, antialias) -> VideoToNumpy (``(x * 255).to(uint8)``, truncating).
    """
    import torch
    from torchvision.transforms.v2 import functional as tvf

    frames = np.asarray(frames)
    if frames.shape[1:3] != tuple(metadata_hw):
        raise ValueError(f"DreamZero-AgiBot expects {metadata_hw} views (metadata resolution), got {frames.shape[1:3]}")
    x = torch.from_numpy(frames).to(torch.float32) / 255.0
    x = x.permute(0, 3, 1, 2)
    x = tvf.center_crop(x, [int(metadata_hw[0] * crop_scale), int(metadata_hw[1] * crop_scale)])
    x = tvf.resize(x, list(size), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
    return (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy()


def agibot_grid(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """2x2 grid (T, 2H, 2W, 3): head top-left, right wrist top-right, left wrist bottom-left, black."""
    t, h, w, c = head.shape
    grid = np.zeros((t, 2 * h, 2 * w, c), dtype=head.dtype)
    grid[:, :h, :w] = head
    grid[:, h:, :w] = left
    grid[:, :h, w:] = right
    return grid


def agibot_prompt(instruction: str) -> str:
    """AgiBot four-view prompt, cleaned as ``HuggingfaceTokenizer(clean='whitespace')`` (ASCII path)."""
    text = str(instruction).lower()
    prompt = _AGIBOT_PROMPT_TEMPLATE.format(i=text)
    try:
        import ftfy

        prompt = ftfy.fix_text(prompt)
    except ImportError:  # pragma: no cover - ftfy only changes mojibake / non-ASCII text
        pass
    prompt = html.unescape(html.unescape(prompt)).strip()
    return re.sub(r"\s+", " ", prompt).strip()


class AgibotQ99:
    """Per-key q99 statistics from the checkpoint's ``experiment_cfg/metadata.json`` (``agibot``)."""

    def __init__(self, stats: Dict[str, Dict[str, Dict[str, Sequence[float]]]]):
        self.state = {k: (np.asarray(v["q01"], np.float32), np.asarray(v["q99"], np.float32)) for k, v in stats["state"].items()}
        self.action = {k: (np.asarray(v["q01"], np.float32), np.asarray(v["q99"], np.float32)) for k, v in stats["action"].items()}

    @classmethod
    def from_metadata(cls, path: str, embodiment_tag: str = "agibot") -> "AgibotQ99":
        """Load the ``embodiment_tag`` statistics block of a DreamZero ``metadata.json``."""
        with open(path, "r") as f:
            meta = json.load(f)
        if embodiment_tag not in meta:
            raise ValueError(f"metadata.json at {path} has no {embodiment_tag!r} entry (keys: {sorted(meta)})")
        return cls(meta[embodiment_tag]["statistics"])

    def normalize_state(self, state20: np.ndarray) -> np.ndarray:
        """Official ``Normalizer.forward`` q99: q01 == q99 dims pass through, then clamp to [-1, 1]."""
        import torch

        parts = []
        for key, x in split_keys(np.asarray(state20, np.float32), AGIBOT_STATE_KEYS).items():
            q01, q99 = (torch.from_numpy(a) for a in self.state[key])
            x = torch.from_numpy(np.ascontiguousarray(x))
            mask = q01 != q99
            out = torch.zeros_like(x)
            out[..., mask] = (x[..., mask] - q01[mask]) / (q99[mask] - q01[mask])
            out[..., mask] = 2 * out[..., mask] - 1
            out[..., ~mask] = x[..., ~mask]
            parts.append(torch.clamp(out, -1, 1).numpy())
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def unnormalize_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Official ``Normalizer.inverse`` q99 per action key, ``(x + 1) / 2 * (q99 - q01) + q01``."""
        import torch

        out = {}
        for key, x in split_keys(np.asarray(action, np.float32), AGIBOT_ACTION_KEYS).items():
            q01, q99 = (torch.from_numpy(a) for a in self.action[key])
            x = torch.from_numpy(np.ascontiguousarray(x))
            out[key] = ((x + 1) / 2 * (q99 - q01) + q01).numpy()
        return out


def agibot_chunk_to_joint14(action_dict: Dict[str, np.ndarray], raw_state20: np.ndarray, horizon: int) -> np.ndarray:
    """Relative keys + raw observed state, first ``horizon`` steps -> (horizon, 14) joints."""
    raw = split_keys(np.asarray(raw_state20, np.float32), AGIBOT_STATE_KEYS)
    absolute = {
        key: value + raw[key][None, :] if key in AGIBOT_RELATIVE_ACTION_KEYS else value
        for key, value in action_dict.items()
    }
    steps = min(int(horizon), absolute["left_arm_joint_position"].shape[0])
    return np.stack(
        [
            agibot_to_joint14(
                absolute["left_arm_joint_position"][i],
                absolute["right_arm_joint_position"][i],
                absolute["left_effector_position"][i],
                absolute["right_effector_position"][i],
            )
            for i in range(steps)
        ]
    )


class DreamZeroRoboDojoEvalModel:
    """Stateful ``predict_action`` adapter: one call per env step, one 14-D joint action out.

    ``reset_session_on_new_episode=False`` (default) follows the official adapter, whose
    ``Model.reset()`` clears only its frame buffers: the action head keeps its KV session
    across episodes until its own triggers (new language, single frame, local attention
    window full) restart it.
    """

    def __init__(
        self,
        infer_model: Any,
        stats: AgibotQ99,
        negative_prompt: str,
        action_horizon: int = 24,
        video_history: int = 4,
        max_state_dim: int = 64,
        reset_session_on_new_episode: bool = False,
    ):
        self._infer_model = infer_model
        self._stats = stats
        self._negative_prompt = negative_prompt
        self._action_horizon = int(action_horizon)
        self._video_history = int(video_history)
        self._max_state_dim = int(max_state_dim)
        self._reset_session = bool(reset_session_on_new_episode)
        self._episode_id: Optional[str] = None
        self._frames: List[deque] = []
        self._queue: deque = deque()

    def reset(self) -> None:
        """Drop all session state, including the model's KV / language caches."""
        self._episode_id = None
        self._frames = []
        self._queue = deque()
        self._infer_model.reset()

    def _start_episode(self, episode_id: str) -> None:
        if self._episode_id is not None and self._reset_session:
            self._infer_model.reset()
        self._episode_id = episode_id
        self._frames = [deque(maxlen=self._video_history) for _ in range(3)]
        self._queue = deque()

    def build_inputs(self, instruction: str, state14: np.ndarray) -> Tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        """Model inputs from the buffered frames: (video grid, prompt, padded normalized state, raw state20)."""
        views = [agibot_view_transform(stack_recent_frames(list(buf), self._video_history)) for buf in self._frames]
        video = agibot_grid(*views)
        raw_state20 = joint14_to_agibot_state(state14)
        state = self._stats.normalize_state(raw_state20)
        padded = np.zeros(self._max_state_dim, dtype=np.float32)
        padded[: state.size] = state
        return video, agibot_prompt(instruction), padded, raw_state20

    def predict_action(
        self,
        images: list,
        instructions: list,
        state: Optional[np.ndarray] = None,
        episode_id: str = "default",
        episode_step: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        """One env step: buffer the three views, refill the 24-step chunk if empty, pop one action."""
        if state is None:
            raise ValueError("DreamZero RoboDojo eval requires the 14-D joint state")
        views = images[0] if images and isinstance(images[0], list) else images
        if len(views) != 3:
            raise ValueError(f"DreamZero RoboDojo eval expects [head, left wrist, right wrist], got {len(views)} views")
        if str(episode_id) != self._episode_id:
            self._start_episode(str(episode_id))
        for buf, view in zip(self._frames, views):
            image = np.asarray(view)
            if image.ndim != 3 or image.shape[-1] not in (3, 4):
                raise ValueError(f"expected HWC RGB view, got {image.shape}")
            buf.append(image[..., :3].astype(np.uint8))
        if not self._queue:
            video, prompt, state_norm, raw_state20 = self.build_inputs(str(instructions[0]), state)
            chunk = self._infer_model.predict(
                video=video,
                prompt=prompt,
                negative_prompt=self._negative_prompt,
                state=state_norm,
                apply_eval_image_transform=False,
            )
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1, np.asarray(chunk).shape[-1])
            actions = self._stats.unnormalize_action(chunk[:, : sum(d for _, d in AGIBOT_ACTION_KEYS)])
            self._queue.extend(agibot_chunk_to_joint14(actions, raw_state20, self._action_horizon))
            if not self._queue:
                raise RuntimeError("DreamZero predict returned an empty action chunk")
        return self._queue.popleft()[None, :]
