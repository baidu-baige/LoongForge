# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero evaluation factory (LIBERO).

Server-side eval boundary for DreamZero on LIBERO:

- Builds the DreamZero policy via ``build_model`` / ``dreamzero_model_provider``
  (checkpoint loading happens inside the provider through
  ``dit_init_checkpoint_path`` — the full DreamZero checkpoint layout with
  ``action_head.model.*`` keys, e.g. RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT).
- Wraps it in ``DreamZeroInferenceModel`` (tokenizer + action-head batch) plus
  an eval-side stateful adapter (``DreamZeroLiberoEvalModel``) that owns the
  LIBERO protocol: 33-frame two-view history window, the official ``libero_sim``
  prompt template, q99 state normalization and q99 action denormalization from
  the checkpoint's ``metadata.json`` statistics.

Official LIBERO eval parameters (RLinf ``libero_spatial_dreamzero_eval.yaml``
and the DreamZero release): embodiment projector id 21, action_horizon 16
executed fully open-loop per chunk, num_frames 33 (= 8*max_chunk_size + 1),
target video 160x320 (per-view 256x256 resized model-side), exterior-left +
wrist-right grid, max_episode_steps 480, bf16.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.dreamzero.model_configuration_dreamzero import DreamZeroConfig
from loongforge.embodied.model.dreamzero.modeling_dreamzero_infer import DreamZeroInferenceModel
from loongforge.embodied.model.registry import build_model

logger = logging.getLogger(__name__)

# Official DreamZero CFG negative prompt (dreamzero_collator.py text_negative).
_DREAMZERO_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, "
    "painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, "
    "ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, "
    "fused fingers, stagnant image, cluttered background, three legs, many people in the "
    "background, walking backwards."
)

# Official libero_sim prompt template (dreamzero_collator.py LIBERO_SIM branch).
_LIBERO_PROMPT_TEMPLATE = (
    "A multi-view video shows that a robot {instruction} "
    "The video is split into two horizontal views: the left view shows the "
    "exterior camera and the right view shows the wrist camera. The robot {instruction}"
)

# ``libero_sim`` -> WAN action-head projector id in the official multi-
# embodiment mapping (datasets/dreamzero/dataset/modality_configs.py). The
# released RLinf LIBERO-SFT checkpoints, however, train a SINGLE-category
# projector (``state_encoder.layer1.W`` is ``(1, 64, 1024)``), so the runtime
# ``embodiment_id`` tensor must be 0 — the model indexes W[cat_ids] directly.
LIBERO_SIM_EMBODIMENT_ID = 21
DREAMZERO_RUNTIME_EMBODIMENT_ID = 0


@dataclass
class DreamZeroEvalConfig(DreamZeroConfig):
    """DreamZero model config extended with eval-only LIBERO semantics."""

    # Eval-side scheduler length; consumed by build_action_head_config.
    eval_num_inference_timesteps: int = 16
    # LIBERO geometry (official libero_spatial_dreamzero_eval.yaml).
    num_frames: int = 33
    action_horizon: int = 16
    n_action_steps: int = 16
    num_frame_per_block: int = 2
    num_action_per_block: int = 16
    num_state_per_block: int = 1
    max_chunk_size: int = 4
    max_state_dim: int = 64
    max_action_dim: int = 32
    dit_action_state_hidden_size: int = 1024


class _Q99Stats:
    """q01/q99 statistics for one modality key, from DreamZero metadata.json."""

    def __init__(self, q01: np.ndarray, q99: np.ndarray):
        self.q01 = np.asarray(q01, dtype=np.float32).reshape(-1)
        self.q99 = np.asarray(q99, dtype=np.float32).reshape(-1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x -> [-1, 1] via 2*(x-q01)/(q99-q01)-1 (StateActionTransform q99)."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != self.q01.size:
            raise ValueError(
                f"q99 normalize: input dim {x.size} != stats dim {self.q01.size}"
            )
        span = self.q99 - self.q01
        out = np.zeros_like(x)
        valid = span != 0
        out[valid] = 2.0 * (x[valid] - self.q01[valid]) / span[valid] - 1.0
        out[~valid] = x[~valid]
        return out

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalize`."""
        x = np.asarray(x, dtype=np.float32)
        span = (self.q99 - self.q01).reshape(1, -1)
        base = self.q01.reshape(1, -1)
        out = np.where(span != 0, (x + 1.0) / 2.0 * span + base, x)
        return out.astype(np.float32)


def _load_q99_stats(
    metadata_path: str, embodiment_tag: str
) -> Tuple[_Q99Stats, _Q99Stats]:
    """Load state/action q99 statistics from a DreamZero ``metadata.json``.

    Actual layout (RLinf ``generate_dreamzero_metadata.py`` / Step26000
    ``experiment_cfg/metadata.json``):
    ``{embodiment_tag: {"statistics": {"state": {"state": {"q01":..., "q99":...}},
    "action": {"actions": {"q01":..., "q99":...}}}}}`` — modality-nested dicts,
    not flat ``state.state`` keys.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    entry = metadata.get(embodiment_tag) if isinstance(metadata, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(
            f"metadata.json at {metadata_path} has no entry for embodiment {embodiment_tag!r}"
        )
    stats = entry.get("statistics") or {}
    try:
        state_stats = stats["state"]["state"]
        action_stats = stats["action"]["actions"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"metadata.json statistics missing {exc!r} (keys: {sorted(stats)})"
        ) from exc
    return _Q99Stats(state_stats["q01"], state_stats["q99"]), _Q99Stats(
        action_stats["q01"], action_stats["q99"]
    )


class DreamZeroLiberoEvalModel:
    """Stateful ``predict_action`` adapter over ``DreamZeroInferenceModel``.

    Owns the LIBERO rollout protocol so the shared chunk-cached RPC policy can
    stay model-agnostic:

    - Called once per env step (``disable_action_cache`` in the PayloadBuilder).
    - Applies the official libero_sim per-frame preprocessing (RLinf
      ``libero_sim.py`` transform chain): 95% center crop then bilinear resize
      back to 256x256, before the exterior|wrist horizontal grid concat.
    - Appends the current two-view frame to a per-episode history deque and
      left-pads the window by repeating the oldest frame at episode start.
    - When its action queue is empty, builds the 33-frame grid
      (exterior-left | wrist-right), the official prompt, the normalized state,
      runs one ``predict`` call and queues the denormalized 16-step chunk.
    - Returns exactly one action per call, shape ``(1, 7)``.
    """

    def __init__(
        self,
        infer_model: DreamZeroInferenceModel,
        state_stats: _Q99Stats,
        action_stats: _Q99Stats,
        num_frames: int = 33,
        action_horizon: int = 16,
        action_dim: int = 7,
        max_state_dim: int = 64,
        binarize_gripper: bool = True,
    ):
        self._infer_model = infer_model
        self._state_stats = state_stats
        self._action_stats = action_stats
        self._num_frames = int(num_frames)
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._max_state_dim = int(max_state_dim)
        # Official LIBERO rollout binarizes the gripper command (>0 -> +1).
        self._binarize_gripper = bool(binarize_gripper)
        # episode_id -> (frame deque of (H, W, 3) grids, action queue [t, D]).
        # Single-session state (see _episode); DreamZero's caches are global.
        self._episode_id: Optional[str] = None
        self._frames: deque = deque()
        self._queue: deque()
        self._first_call = True

    def reset(self) -> None:
        """Drop session state (frames, queue, model KV/language caches)."""
        self._episode_id = None
        self._frames = deque()
        self._queue = deque()
        self._first_call = True
        self._infer_model.reset()

    def _episode(self, episode_id: str) -> Tuple[deque, deque]:
        """Return this episode's buffers, resetting the model session on change.

        DreamZero's KV / language caches are global to the single model
        instance (same as the verified DROID eval: episode keys are NOT
        independent model sessions), so a new episode_id resets everything.
        """
        if episode_id != self._episode_id:
            if self._episode_id is not None:
                self._infer_model.reset()
            self._episode_id = episode_id
            self._frames = deque(maxlen=1024)
            self._queue = deque()
            self._first_call = True
        return self._frames, self._queue

    def _build_video(self, frames: deque) -> np.ndarray:
        """Select the frame block for this inference call.

        Official LIBERO rollout protocol (RLinf ``libero_sim.py``
        ``eval_delta_indices=[0]`` + action-head session logic): EVERY replan
        sends only the current frame — the action head detects a single-frame
        input, resets its autoregressive session, and re-initializes the
        33-frame trajectory from that frame. No cross-chunk KV reuse.
        """
        return np.stack([frames[-1]]).astype(np.uint8)

    def predict_action(
        self,
        images: list,
        instructions: list,
        state: Optional[np.ndarray] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        episode_id: str = "default",
        episode_step: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        """One env step: observe, refill the chunk if needed, pop one action."""
        if state is None:
            raise ValueError("DreamZero LIBERO eval requires the 8D proprio state")
        views = images[0] if images and isinstance(images[0], list) else images
        if len(views) < 2:
            raise ValueError("DreamZero LIBERO eval requires exterior + wrist views")
        exterior, wrist = np.asarray(views[0]), np.asarray(views[1])
        if exterior.shape != wrist.shape:
            raise ValueError(
                f"exterior/wrist shape mismatch: {exterior.shape} vs {wrist.shape}"
            )
        # Official grid: exterior left, wrist right (matches prompt text). The
        # eval image transform (95% center crop) lives in the inference facade.
        grid = np.concatenate([exterior, wrist], axis=1)
        if grid.ndim != 3:
            raise ValueError(f"expected HWC frame grid, got shape {grid.shape}")
        frames, queue = self._episode(str(episode_id))
        frames.append(grid)

        if not queue:
            # Match DreamTransform's whitespace cleaning before templating.
            # (No lowercasing: the RLinf libero_sim template does not lowercase
            # and LIBERO instructions are already lowercase.)
            instruction = " ".join(str(instructions[0]).strip().split())
            prompt = _LIBERO_PROMPT_TEMPLATE.format(instruction=instruction)
            video = self._build_video(frames)
            state_norm = self._state_stats.normalize(state)
            state_norm = np.clip(state_norm, -1.0, 1.0)
            # Training collator zero-pads proprio to ``max_state_dim``; the
            # checkpoint's state projector expects the padded width.
            if state_norm.size < self._max_state_dim:
                pad = np.zeros(self._max_state_dim, dtype=np.float32)
                pad[: state_norm.size] = state_norm
                state_norm = pad
            chunk = self._infer_model.predict(
                video=video,
                prompt=prompt,
                negative_prompt=_DREAMZERO_NEGATIVE_PROMPT,
                state=state_norm,
            )
            self._first_call = False
            # (1, H, 32) normalized padded actions -> (H, 7) raw. The model
            # denoises actions padded to ``max_action_dim`` (32); the first
            # ``action_dim`` (7) columns carry the LIBERO action.
            chunk = np.asarray(chunk, dtype=np.float32)
            chunk = chunk.reshape(-1, chunk.shape[-1])
            chunk = self._action_stats.unnormalize(chunk[:, : self._action_dim])
            if self._binarize_gripper:
                # Official dreamzero_policy.py: actions[..., -1] = >0 ? +1 : -1
                # (LIBERO gripper: -1 open, +1 close).
                chunk[:, -1] = np.where(chunk[:, -1] > 0, 1.0, -1.0)
            for step in chunk[: self._action_horizon]:
                queue.append(step)
            if not queue:
                raise RuntimeError("DreamZero predict returned an empty action chunk")
        action = queue.popleft()
        if action.shape[0] != self._action_dim:
            raise ValueError(
                f"DreamZero action dim {action.shape[0]} != expected {self._action_dim}"
            )
        return action[None, :]


@register_factory("dreamzero")
class DreamZeroModelFactory:
    """Build a DreamZero model instance implementing the predict_action interface."""

    model_config_cls = DreamZeroEvalConfig

    @classmethod
    def build(
        cls,
        model_cfg: DreamZeroEvalConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create the DreamZero LIBERO eval model and its metadata."""
        import torch

        ckpt_path = str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        # The full DreamZero checkpoint (action_head.model.* keys) loads through
        # the provider's dit_init path; encoder paths come from the model
        # section (Wan release dirs), matching the training YAML convention.
        if ckpt_path and not server_args.random_init:
            model_cfg.dit_init_checkpoint_path = ckpt_path
            model_cfg.action_state_init_checkpoint_path = ckpt_path

        model = build_model(model_cfg)
        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        tokenizer_path = server_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
        infer_model = DreamZeroInferenceModel(
            model=model,
            tokenizer_path=tokenizer_path,
            text_len=model_cfg.text_len,
            embodiment_id=DREAMZERO_RUNTIME_EMBODIMENT_ID,
            device=resolved_device,
        )

        state_stats, action_stats = _load_q99_stats(
            server_args.dataset_statistics_path, "libero_sim"
        )
        eval_model = DreamZeroLiberoEvalModel(
            infer_model=infer_model,
            state_stats=state_stats,
            action_stats=action_stats,
            num_frames=model_cfg.num_frames,
            action_horizon=model_cfg.action_horizon,
            action_dim=7,
            max_state_dim=model_cfg.max_state_dim,
            binarize_gripper=getattr(model_cfg, "binarize_gripper", True),
        )

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "dreamzero",
            "ckpt_path": ckpt_path if not server_args.random_init else "random_init://dreamzero",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": 7,
            "action_horizon": model_cfg.action_horizon,
            "embodiment_tag": "libero_sim",
            "embodiment_id": LIBERO_SIM_EMBODIMENT_ID,
            "dataset_statistics_path": server_args.dataset_statistics_path,
            "tokenizer_path": tokenizer_path,
        }
        return PredictActionModelSpec(model=eval_model, metadata=metadata)
