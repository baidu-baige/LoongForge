# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
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

"""Motus per-sample transform.

:class:`LeRobotV3Dataset` (motus mode) decodes a multi-frame video stack + a
strided action chunk natively via ``delta_timestamps``. This transform turns
that raw lerobot sample into the model-ready dict consumed by
:class:`MotusPreprocessor` / ``MotusPolicy.forward``:

    first_frame        [C, H, W]              condition frame, resized/padded
    video_frames       [F, C, H, W]           future frames, resized/padded
    initial_state      [state_dim]            normalized condition-frame state
    action_sequence    [action_chunk, dim]    normalized strided action chunk
    language_embedding [S, D]                  T5 (parquet | external pt | on-the-fly)
    vlm_inputs         {...} | None           Qwen VLM tokens (first frame)

All numerics (view stitch, resize/pad, [0,1] action/state normalization, T5
resolution, VLM preprocessing) are ported verbatim from the source Motus
``LeRobotMotusDataset``; only the plumbing (dataset decode via lerobot, config
sourcing via loongforge) differs.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.embodied.data.datasets.motus.transforms.image_utils import tensor_to_pil
from loongforge.embodied.data.datasets.motus.transforms.norm import (
    load_normalization_stats,
    normalize_actions,
)
from loongforge.embodied.data.datasets.motus.transforms.views import assemble_first_and_video

logger = logging.getLogger(__name__)


class _MotusT5Resolver:
    """Resolve a per-sample T5 language embedding (verbatim source logic).

    Resolution order:
      1. parquet column (``language_embedding`` /
         ``observation.feature.language_embedding``) present in the decoded item.
      2. external per-episode ``.pt`` referenced by ``t5_embedding_path`` in
         ``meta/episodes.jsonl`` (loaded by ``episode_index``, cached in-process).
      3. on-the-fly encode with the vendored WAN T5 encoder (MainProcess only),
         cached to disk + written back into ``episodes.jsonl``.
    """

    def __init__(
        self,
        dataset,
        enable_t5_fallback: bool,
        t5_wan_path: Optional[str],
        t5_folder_name: str,
        t5_text_len: int,
    ) -> None:
        """Store the dataset handle and T5 fallback settings (WAN path, cache folder, text len)."""
        self.dataset = dataset
        self.enable_t5_fallback = bool(enable_t5_fallback)
        self.t5_wan_path = t5_wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
        self.t5_folder_name = str(t5_folder_name)
        self.t5_text_len = int(t5_text_len)
        self.t5_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._t5_encoder = None
        self._episode_embedding_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # ---- path helpers ----
    def _dataset_root(self) -> Path:
        """Return the dataset's root directory as a ``Path``."""
        return Path(self.dataset.root)

    def _sub_dataset_for(self, item: Dict[str, Any]):
        """Return the concrete lerobot dataset owning ``item``.

        For a single ``LeRobotV3Dataset`` this is the dataset itself. For a
        ``MultiLeRobotV3Dataset`` the sample carries a ``dataset_index`` naming
        which sub-repo it came from, so external ``.pt`` / ``meta`` lookups hit
        the right task's root + metadata.
        """
        if hasattr(self.dataset, "_datasets"):
            di_raw = item.get("dataset_index", 0)
            di = int(di_raw.item()) if hasattr(di_raw, "item") else int(di_raw)
            return self.dataset._datasets[di]
        return self.dataset

    def _root_for(self, item: Dict[str, Any]) -> Path:
        """Return the root directory of the sub-dataset owning ``item``."""
        return Path(self._sub_dataset_for(item).root)

    def _meta_episodes_for(self, item: Dict[str, Any]):
        """Return the episode metadata of the sub-dataset owning ``item``."""
        return self._sub_dataset_for(item).meta.episodes

    def _t5_cache_file_path(self, episode_index: int, root: Optional[Path] = None) -> Path:
        """Return the on-disk T5 embedding cache path for ``episode_index``."""
        base = root if root is not None else self._dataset_root()
        return base / self.t5_folder_name / f"episode_{episode_index:06d}.pt"

    def _t5_lock_file_path(self, episode_index: int, root: Optional[Path] = None) -> Path:
        """Return the lock-file path guarding the T5 cache write for ``episode_index``."""
        base = root if root is not None else self._dataset_root()
        return base / self.t5_folder_name / f"episode_{episode_index:06d}.pt.lock"

    def _episodes_jsonl_path(self, root: Optional[Path] = None) -> Path:
        """Return the ``meta/episodes.jsonl`` path under ``root`` (or the dataset root)."""
        base = root if root is not None else self._dataset_root()
        return base / "meta" / "episodes.jsonl"

    # placeholder-t5-methods

    def _atomic_update_episodes_jsonl(
        self, episode_index: int, updates: Dict[str, Any], root: Optional[Path] = None
    ) -> None:
        """In-place update of one episode entry in meta/episodes.jsonl (temp + replace)."""
        path = self._episodes_jsonl_path(root)
        if not path.exists():
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        found = False
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if int(obj.get("episode_index", -1)) == int(episode_index):
                    obj.update(updates)
                    found = True
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if not found:
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return
        tmp.replace(path)

    def _ensure_t5_encoder(self):
        """Lazily create the WAN T5 encoder (MainProcess only) and cache it on the instance."""
        if self._t5_encoder is not None:
            return self._t5_encoder
        current_process = multiprocessing.current_process()
        if current_process.name != "MainProcess":
            raise RuntimeError(
                f"T5 encoder initialization in DataLoader worker process "
                f"({current_process.name}) is disabled to avoid memory issues. "
                "Please pre-generate T5 embeddings offline."
            )
        from loongforge.embodied.model.motus.motus_impl.wan.modules.t5 import T5EncoderModel

        if not self.t5_wan_path:
            raise ValueError(
                "enable_t5_fallback=True but t5_wan_path is not provided and "
                "WAN_PATH/WAN_ROOT is not set."
            )
        ckpt = os.path.join(self.t5_wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
        tok = os.path.join(self.t5_wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
        dtype = torch.bfloat16 if self.t5_device.startswith("cuda") else torch.float32
        self._t5_encoder = T5EncoderModel(
            text_len=self.t5_text_len, dtype=dtype, device=self.t5_device,
            checkpoint_path=ckpt, tokenizer_path=tok,
        )
        return self._t5_encoder

    # placeholder-encode-cache

    def _encode_and_cache_t5_embedding(
        self,
        episode_index: int,
        instruction: str,
        root: Optional[Path] = None,
        meta_episodes: Optional[Dict[int, Any]] = None,
    ) -> torch.Tensor:
        """Encode on-the-fly and cache to disk (lock-file guarded), returning [S,D]/[V,S,D]."""
        out_pt = self._t5_cache_file_path(episode_index, root)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        if out_pt.exists():
            emb = torch.load(out_pt, map_location="cpu")
            return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)

        lock_path = self._t5_lock_file_path(episode_index, root)
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if out_pt.exists():
                    emb = torch.load(out_pt, map_location="cpu")
                    return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                if time.time() - start > 600:
                    raise TimeoutError(f"Timeout waiting for T5 embedding lock: {lock_path}")
                time.sleep(0.2)
        try:
            if out_pt.exists():
                emb = torch.load(out_pt, map_location="cpu")
                return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            encoder = self._ensure_t5_encoder()
            with torch.no_grad():
                t5_out = encoder([instruction], self.t5_device)
            if isinstance(t5_out, list):
                emb = t5_out[0]
            elif isinstance(t5_out, torch.Tensor):
                emb = t5_out
            else:
                raise ValueError(f"Unexpected T5 encoder output type: {type(t5_out)}")
            if isinstance(emb, torch.Tensor) and emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            torch.save(emb.detach().cpu(), out_pt)
            rel = f"{self.t5_folder_name}/episode_{episode_index:06d}.pt"
            try:
                self._atomic_update_episodes_jsonl(
                    episode_index, {"t5_embedding_path": rel}, root
                )
                if meta_episodes is not None and episode_index in meta_episodes:
                    meta_episodes[episode_index]["t5_embedding_path"] = rel
            except Exception as e:
                logger.warning(f"Failed to update episodes.jsonl for episode {episode_index}: {e}")
            return emb
        finally:
            try:
                lock_path.unlink()
            except Exception:
                pass

    # placeholder-resolve

    def resolve(self, item: Dict[str, Any]) -> torch.Tensor:
        """Return a ``[S, D]`` T5 embedding for the sample's episode."""
        all_embeddings = item.get("language_embedding", None)
        if all_embeddings is None:
            all_embeddings = item.get("observation.feature.language_embedding", None)

        if all_embeddings is None:
            ep_index_raw = item.get("episode_index", None)
            if ep_index_raw is None:
                raise KeyError("episode_index not found in item; cannot load external embedding")
            ep_index = int(ep_index_raw.item()) if hasattr(ep_index_raw, "item") else int(ep_index_raw)

            # In multi mode episode_index is local per sub-repo and can collide
            # across tasks, so key the in-process cache by (dataset_index, ep).
            di_raw = item.get("dataset_index", 0)
            di = int(di_raw.item()) if hasattr(di_raw, "item") else int(di_raw)
            cache_key = (di, ep_index)

            cached = self._episode_embedding_cache.get(cache_key, None)
            if cached is None:
                meta_episodes = self._meta_episodes_for(item)
                ep_meta = meta_episodes[ep_index]
                if ep_meta is None:
                    raise KeyError(f"episode {ep_index} not found in meta.episodes")
                rel_path = ep_meta.get("t5_embedding_path", None)
                item_root = self._root_for(item)
                if rel_path is None:
                    if not self.enable_t5_fallback:
                        raise KeyError(
                            "language_embedding not found in item and t5_embedding_path not found "
                            "in meta/episodes.jsonl; set enable_t5_fallback=True to encode on-the-fly."
                        )
                    instr = item.get("language_instruction", None)
                    if instr is None or (isinstance(instr, str) and len(instr.strip()) == 0):
                        instr = item.get("task", "")
                    if not isinstance(instr, str):
                        instr = str(instr)
                    emb = self._encode_and_cache_t5_embedding(
                        ep_index, instr, item_root, meta_episodes
                    )
                    cached = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                    self._episode_embedding_cache[cache_key] = cached
                else:
                    abs_path = item_root / str(rel_path)
                    emb = torch.load(abs_path, map_location="cpu")
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    if emb.ndim == 2:
                        emb = emb.unsqueeze(0)
                    self._episode_embedding_cache[cache_key] = emb
                    cached = emb
            all_embeddings = cached

        if not isinstance(all_embeddings, torch.Tensor):
            all_embeddings = torch.tensor(all_embeddings)
        if all_embeddings.ndim == 2:
            all_embeddings = all_embeddings.unsqueeze(0)
        return all_embeddings[0].float()


class MotusAssembleTransform(BaseTransform):
    """Assemble a decoded lerobot sample into a model-ready Motus sample dict."""

    def __init__(
        self,
        view_mode: str,
        decode_keys: List[str],
        action_key: str,
        target_size: Tuple[int, int],
        action_min: np.ndarray,
        action_max: np.ndarray,
        t5_resolver: _MotusT5Resolver,
        vlm_processor: Any = None,
    ) -> None:
        """Store view/decode keys, target size, action normalisation bounds and T5/VLM helpers."""
        super().__init__(apply_to=[], training=True)
        self.view_mode = view_mode
        self.decode_keys = decode_keys
        self.action_key = action_key
        self.target_size = target_size
        self.action_min = action_min
        self.action_max = action_max
        self.t5_resolver = t5_resolver
        self.vlm_processor = vlm_processor

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Turn a decoded lerobot sample into a model-ready Motus sample dict.

        Assembles the condition frame + future video frames, normalises the action
        sequence and initial state, resolves the T5 language embedding, and (optionally)
        builds the VLM inputs.
        """
        first_frame, video_frames = assemble_first_and_video(
            data, self.view_mode, self.decode_keys, self.target_size
        )

        # data[action_key] is [anchor(delta-0)] + [future action chunk]. Index 0
        # is the condition-frame action; indices 1: are the future chunk that
        # matches the source geometry.
        full_actions = torch.as_tensor(data[self.action_key]).float()
        condition_frame_action = full_actions[0]
        action_sequence = full_actions[1:]

        if "observation.state" in data and data["observation.state"] is not None:
            initial_state = torch.as_tensor(data["observation.state"]).float()
        else:
            # Stateless datasets: use the delta-0 condition-frame action as the
            # state proxy (matches base lerobot_dataset.py:700-707).
            initial_state = condition_frame_action.float()

        normalized_actions = normalize_actions(action_sequence, self.action_min, self.action_max)
        normalized_initial_state = normalize_actions(
            initial_state.unsqueeze(0), self.action_min, self.action_max
        ).squeeze(0)

        language_embedding = self.t5_resolver.resolve(data)

        vlm_inputs = None
        if self.vlm_processor is not None:
            from loongforge.embodied.data.datasets.motus.transforms.vlm_utils import (
                preprocess_vlm_messages,
            )

            # Align to base: base's condition-frame row (item_cond) carries no
            # task text, so its VLM prompt is image-only (empty text). The task
            # instruction reaches the model solely through the T5
            # language_embedding path, NOT the VLM. Feed an empty instruction so
            # the VLM token stream (input_ids/attention_mask) matches base
            # bit-for-bit; otherwise the extra instruction tokens perturb the
            # und branch of joint attention and the video loss.
            text_instr = ""
            first_frame_pil = tensor_to_pil(first_frame)
            vlm_inputs = preprocess_vlm_messages(text_instr, first_frame_pil, self.vlm_processor)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames,
            "initial_state": normalized_initial_state,
            "action_sequence": normalized_actions,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_inputs,
        }


@register_transform_builder("motus")
def build_motus_transforms(ctx: TransformBuilderContext) -> Iterable[BaseTransform]:
    """Build the Motus assembly transform from typed configs + the dataset.

    View geometry (mode / decode keys / action key) is read from the motus-mode
    :class:`LeRobotV3Dataset` so the dataset and transform never disagree. Video
    resolution + VLM checkpoint come from ``model_cfg``; normalization embodiment
    and T5 plumbing come from ``data_cfg``.
    """
    dataset = ctx.dataset
    model_cfg = ctx.model_cfg
    data_cfg = ctx.data_cfg

    if not getattr(dataset, "_motus_mode", False):
        # Dataset is not in motus multi-frame mode; nothing to assemble.
        return []

    target_size = (model_cfg.video_height, model_cfg.video_width)

    stat_path = Path(__file__).parent / "stat.json"
    action_min, action_max = load_normalization_stats(str(stat_path), data_cfg.embodiment_type)
    if action_min is None or action_max is None:
        raise ValueError(
            f"Failed to load normalization stats for embodiment "
            f"'{data_cfg.embodiment_type}' from {stat_path}"
        )

    vlm_processor = None
    if model_cfg.vlm_checkpoint_path:
        from transformers import AutoProcessor

        vlm_processor = AutoProcessor.from_pretrained(model_cfg.vlm_checkpoint_path)
        logger.info(f"Loaded VLM processor from {model_cfg.vlm_checkpoint_path}")

    t5_resolver = _MotusT5Resolver(
        dataset=dataset,
        enable_t5_fallback=data_cfg.enable_t5_fallback,
        t5_wan_path=data_cfg.t5_wan_path or None,
        t5_folder_name=data_cfg.t5_folder_name,
        t5_text_len=data_cfg.t5_text_len,
    )

    return [
        MotusAssembleTransform(
            view_mode=dataset._view_mode,
            decode_keys=dataset._decode_keys,
            action_key=dataset._action_key,
            target_size=target_size,
            action_min=action_min,
            action_max=action_max,
            t5_resolver=t5_resolver,
            vlm_processor=vlm_processor,
        )
    ]
