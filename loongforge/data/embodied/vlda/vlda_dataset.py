# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read VLDA's LeRobot v2 episodes with future images and same-episode negatives."""

import hashlib
import io
import json
import random
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_norm_stats(path: str | Path) -> dict:
    """Read dataset metadata statistics or a checkpoint's statistics snapshot."""
    with Path(path).open() as stream:
        stats = json.load(stream)
    stats = stats.get("norm_stats", stats)
    return {key: {name: np.asarray(value) for name, value in values.items()} for key, values in stats.items()}


class VLDADataset(Dataset):
    """Map each frame to one action chunk, without crossing episode boundaries.

    This reader targets the v2 embedded-image datasets used by VLDA. It leaves
    sampling, distributed sharding and worker snapshots to StatefulDataLoader.
    """

    def __init__(self, root, model_cfg, data_cfg, *, seed=42):
        self.root = Path(root)
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.seed = seed
        with (self.root / "meta/info.json").open() as stream:
            self.info = json.load(stream)
        if not self.info["codebase_version"].startswith("v2"):
            raise ValueError("VLDA's dataset strategy expects a LeRobot v2 embedded-image dataset")
        if len(data_cfg.camera_keys) != 3 or not data_cfg.camera_keys[0]:
            raise ValueError("camera_keys must contain base, left wrist and right wrist (empty if absent)")
        for key in filter(None, data_cfg.camera_keys):
            if self.info["features"][key]["dtype"] != "image":
                raise ValueError(f"VLDA expects embedded images for {key}; video datasets need a video reader")
        with (self.root / "meta/episodes.jsonl").open() as stream:
            self.episodes = sorted(
                (json.loads(line) for line in stream if line.strip()), key=lambda row: row["episode_index"]
            )
        self.ends = np.cumsum([episode["length"] for episode in self.episodes]).tolist()
        with (self.root / "meta/tasks.jsonl").open() as stream:
            self.tasks = {row["task_index"]: row["task"] for row in map(json.loads, stream)}
        stats_path = Path(data_cfg.norm_stats_path) if data_cfg.norm_stats_path else self.root / "meta/stats.json"
        self.meta = SimpleNamespace(stats=load_norm_stats(stats_path))
        self.fingerprint = {
            name: hashlib.sha256((self.root / "meta" / name).read_bytes()).hexdigest()
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
        }
        self._episodes_cache = OrderedDict()
        # LF attaches the transform here, keeping this dataset's Stateful
        # methods visible to Python 3.12's static protocol checks.
        self._transform = None

    def __len__(self):
        return self.ends[-1]

    def _episode(self, position):
        if position not in self._episodes_cache:
            import pandas as pd

            episode_index = self.episodes[position]["episode_index"]
            path = self.info["data_path"].format(
                episode_chunk=episode_index // self.info["chunks_size"],
                episode_index=episode_index,
            )
            self._episodes_cache[position] = pd.read_parquet(self.root / path)
            if len(self._episodes_cache) > 2:
                self._episodes_cache.popitem(last=False)
        self._episodes_cache.move_to_end(position)
        return self._episodes_cache[position]

    def _image(self, value):
        if isinstance(value, dict):
            source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else self.root / value["path"]
            with Image.open(source) as image:
                return np.asarray(image.convert("RGB")).copy()
        return np.asarray(value, dtype=np.uint8)

    def _vjepa_image(self, frame, global_index):
        cache = self.data_cfg.vjepa_image_cache_dir
        if cache:
            with Image.open(self.root / cache / f"{global_index:09d}.png") as image:
                return np.asarray(image.convert("RGB")).copy()
        return self._image(frame[self.data_cfg.camera_keys[0]])

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        position = bisect_right(self.ends, index)
        start = self.ends[position - 1] if position else 0
        episode = self._episode(position)
        local_index = index - start
        frame = episode.iloc[local_index]
        horizon = self.model_cfg.action_horizon
        future = min(local_index + horizon, len(episode) - 1)
        action_indices = np.minimum(np.arange(local_index, local_index + horizon), len(episode) - 1)
        actions = np.stack(episode.iloc[action_indices][self.data_cfg.action_key].to_list())
        images = [self._image(frame[key]) if key else None for key in self.data_cfg.camera_keys]
        sample = {
            "images": images,
            "state": np.asarray(frame[self.data_cfg.state_key]),
            "actions": actions,
            "prompt": self.tasks[int(frame["task_index"])],
            "vjepa_image": self._vjepa_image(frame, index),
            "vjepa_future_image": self._vjepa_image(episode.iloc[future], start + future),
            "index": index,
            "future_index": start + future,
            "episode_index": int(frame["episode_index"]),
        }
        if self.model_cfg.stage == 2:
            count = min(horizon, self.model_cfg.crop_action_sequences or horizon)
            right_start = min(local_index + count, len(episode))
            total_valid = local_index + len(episode) - right_start
            if total_valid < 4 * count:
                raise ValueError(f"Episode {sample['episode_index']} has too few distinct negative actions")
            key = f"{self.seed}:{epoch}:{index}".encode()
            rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little"))
            offsets = rng.sample(range(total_valid), 4 * count)
            negatives = [offset if offset < local_index else right_start + offset - local_index for offset in offsets]
            rows = episode.iloc[negatives]
            sample["neg_actions"] = np.stack(rows[self.data_cfg.action_key].to_list())
            sample["neg_states"] = np.stack(rows[self.data_cfg.state_key].to_list())
            sample["negative_indices"] = np.asarray(negatives) + start
        return self._transform(sample) if self._transform is not None else sample

    def state_dict(self):
        """Snapshot worker randomness alongside the loader's prefetch/cursor state."""
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}

    def load_state_dict(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
