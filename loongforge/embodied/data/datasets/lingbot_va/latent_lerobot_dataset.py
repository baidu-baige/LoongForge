# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Latent LeRobot dataset implementation for LingBot-VA."""

from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterator, List
import json
import os
import time

import datasets
import numpy as np
import torch

from loongforge.embodied.model.lingbot_va.features import (
    REPO_DISCOVERY_CACHE_POLL_SECONDS,
    REPO_DISCOVERY_CACHE_WAIT_SECONDS,
    feature_enabled,
)
from einops import rearrange
from torch.utils.data import Dataset, get_worker_info

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from loongforge.embodied.data.datasets.lerobot_dataset import LeRobotV3Dataset
except ImportError:
    LeRobotV3Dataset = object
    LeRobotDatasetMetadata = None

try:
    from lerobot.constants import HF_LEROBOT_HOME
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ImportError:
        HF_LEROBOT_HOME = None

try:
    from lerobot.datasets.utils import get_episode_data_index
except ImportError:
    get_episode_data_index = None

try:
    from scipy.spatial.transform import Rotation as Rotation
except ImportError:
    Rotation = None


_CFG_WORKER_GENERATORS: Dict[int, torch.Generator] = {}


class _LeRobotV21Metadata:
    """Read the local v2.1 metadata fields used by LingBot training."""

    def __init__(self, root: Path):
        info_path = root / "meta" / "info.json"
        episodes_path = root / "meta" / "episodes.jsonl"
        self.info = json.loads(info_path.read_text())
        self.episodes = {}
        with episodes_path.open() as f:
            for line in f:
                value = json.loads(line)
                self.episodes[int(value["episode_index"])] = value

    def get_episode_chunk(self, episode_index: int) -> int:
        """Return the metadata chunk id for an episode index."""
        return int(episode_index) // int(self.info.get("chunks_size", 1000))


def _load_lerobot_v21_actions(root: Path):
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot v2.1 parquet files found under {root / 'data'}"
        )
    return datasets.Dataset.from_parquet(
        [str(path) for path in paths], columns=["action"]
    )


def _get_episode_data_index_v21(episodes: Dict[int, Dict[str, Any]], selected_episodes):
    selected = set(selected_episodes) if selected_episodes is not None else None
    starts = {}
    ends = {}
    offset = 0
    for episode_index, value in sorted(episodes.items()):
        if selected is not None and episode_index not in selected:
            continue
        starts[episode_index] = offset
        offset += int(value["length"])
        ends[episode_index] = offset
    return {"from": starts, "to": ends}


def _cfg_dropout_rand() -> float:
    worker_info = get_worker_info()
    if worker_info is None:
        return torch.rand(1).item()

    seed = int(worker_info.seed)
    generator = _CFG_WORKER_GENERATORS.get(seed)
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        _CFG_WORKER_GENERATORS[seed] = generator
    return torch.rand(1, generator=generator).item()


@dataclass
class LingBotVALatentDatasetConfig:
    """Represent LingBotVALatentDatasetConfig."""

    dataset_path: str
    empty_emb_path: str
    obs_cam_keys: List[str]
    norm_stat: Dict[str, List[float]]
    inverse_used_action_channel_ids: List[int]
    cfg_prob: float = 0.1
    env_type: str = "none"
    num_init_worker: int = 1
    revision: str = "v2.1"
    video_backend: str = "pyav"
    action_dim: int = 30
    latent_dir_name: str = "latents"
    metadata_filename: str = "info.json"
    allow_missing_lerobot: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_configs(
        cls, model_cfg, data_cfg, training_args
    ) -> "LingBotVALatentDatasetConfig":
        """Run from configs.

        Args:
            model_cfg: Input value for this operation.
            data_cfg: Input value for this operation.
            training_args: Input value for this operation.

        Returns:
            The computed result.
        """
        dataset_path = data_cfg.dataset_path or training_args.dataset_path
        if not dataset_path:
            raise ValueError("LingBotVADataConfig.dataset_path must be configured")
        empty_emb_path = data_cfg.empty_emb_path or os.path.join(
            dataset_path, "empty_emb.pt"
        )
        obs_cam_keys = _as_list(data_cfg.obs_cam_keys)
        if not obs_cam_keys:
            raise ValueError("LingBotVADataConfig.obs_cam_keys must be configured")
        inverse_ids = _as_int_list(data_cfg.inverse_used_action_channel_ids)
        if not inverse_ids:
            raise ValueError(
                "LingBotVADataConfig.inverse_used_action_channel_ids must be configured"
            )
        return cls(
            dataset_path=dataset_path,
            empty_emb_path=empty_emb_path,
            obs_cam_keys=obs_cam_keys,
            norm_stat={
                "q01": _as_float_list(data_cfg.norm_q01),
                "q99": _as_float_list(data_cfg.norm_q99),
            },
            inverse_used_action_channel_ids=inverse_ids,
            cfg_prob=float(data_cfg.cfg_prob),
            env_type=str(data_cfg.env_type),
            num_init_worker=int(training_args.num_workers or 1),
            revision=str(data_cfg.revision),
            video_backend=str(data_cfg.video_backend),
            action_dim=int(data_cfg.action_dim or model_cfg.action_dim),
            latent_dir_name=str(data_cfg.latent_dir_name),
            metadata_filename=str(data_cfg.metadata_filename),
            allow_missing_lerobot=bool(data_cfg.allow_missing_lerobot),
        )


class LingBotVAMultiLatentDataset(Dataset):
    """Represent LingBotVAMultiLatentDataset."""

    def __init__(self, config: LingBotVALatentDatasetConfig):
        """Initialize the instance.

        Args:
            config: Input value for this operation.
        """
        repo_ids = _find_lerobot_repos(config)
        if not repo_ids:
            raise ValueError(
                f"No LeRobot repositories found under {config.dataset_path}"
            )
        construct_fn = partial(LingBotVALatentDataset, config=config)
        num_workers = max(1, min(int(config.num_init_worker), len(repo_ids)))
        if num_workers > 1:
            with Pool(num_workers) as pool:
                self.datasets = pool.map(construct_fn, repo_ids)
        else:
            self.datasets = [construct_fn(repo_id) for repo_id in repo_ids]
        self.offsets = [0]
        for dataset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __len__(self):
        """Return the number of available items.

        Returns:
            The computed result.
        """
        return self.offsets[-1]

    def __getitem__(self, index):
        """Return one item by index.

        Args:
            index: Input value for this operation.

        Returns:
            The computed result.
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_id = np.searchsorted(self.offsets, index, side="right") - 1
        local_index = index - self.offsets[dataset_id]
        output = self.datasets[dataset_id][local_index]
        if (
            feature_enabled("LINGBOT_SAMPLE_META_EXPORT")
            and "_lingbot_sample_meta" in output
        ):
            sample_meta = dict(output["_lingbot_sample_meta"])
            sample_meta["global_index"] = int(index)
            sample_meta["dataset_id"] = int(dataset_id)
            sample_meta["dataset_offset"] = int(self.offsets[dataset_id])
            output["_lingbot_sample_meta"] = sample_meta
        return output

    def estimate_sample_cost(self, index: int) -> float:
        """Return a cheap monotonic workload estimate for sampler balancing."""
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_id = np.searchsorted(self.offsets, index, side="right") - 1
        return self.datasets[dataset_id].estimate_sample_cost(
            index - self.offsets[dataset_id]
        )


class LingBotVALatentDataset(LeRobotV3Dataset):
    """Represent LingBotVALatentDataset."""

    def __init__(self, repo_id: str, config: LingBotVALatentDatasetConfig):
        """Initialize the instance.

        Args:
            repo_id: Input value for this operation.
            config: Input value for this operation.
        """
        if LeRobotDatasetMetadata is None or HF_LEROBOT_HOME is None:
            if config.allow_missing_lerobot:
                raise RuntimeError(
                    "LeRobot is required to read LingBot-VA latent datasets"
                )
            raise ImportError("Please install lerobot to use LingBotVALatentDataset")
        self.repo_id = repo_id
        self.config = config
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = config.revision
        self.video_backend = config.video_backend
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        if get_episode_data_index is None:
            self.meta = _LeRobotV21Metadata(self.root)
            self.hf_dataset = _load_lerobot_v21_actions(self.root)
            self.episode_data_index = _get_episode_data_index_v21(
                self.meta.episodes, self.episodes
            )
        else:
            self.meta = LeRobotDatasetMetadata(
                self.repo_id, self.root, self.revision, force_cache_sync=False
            )
            self.hf_dataset = self.load_hf_dataset()
            self.episode_data_index = get_episode_data_index(
                self.meta.episodes, self.episodes
            )
        self.latent_path = Path(repo_id) / config.latent_dir_name
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.q01 = np.array(config.norm_stat["q01"], dtype="float")[None]
        self.q99 = np.array(config.norm_stat["q99"], dtype="float")[None]
        self.hf_torch_view = self.hf_dataset.with_format(
            type="torch", columns=["action"], output_all_columns=False
        )
        self.sample_metas = self._parse_meta()

    def __len__(self):
        """Return the number of available items.

        Returns:
            The computed result.
        """
        return len(self.sample_metas)

    def __getitem__(self, index):
        """Return one item by index.

        Args:
            index: Input value for this operation.

        Returns:
            The computed result.
        """
        meta = self.sample_metas[index % len(self.sample_metas)]
        episode_index = meta["episode_index"]
        local_start = meta["start_frame"]
        local_end = meta["end_frame"]
        latent_data = self._get_range_latent_data(local_start, local_end, episode_index)
        latent_frame_ids = latent_data[f"{self.config.obs_cam_keys[0]}.frame_ids"]
        global_start = self._get_global_idx(episode_index, local_start)
        global_end = self._get_global_idx(episode_index, local_end)
        hf_data = self.hf_torch_view[global_start:global_end]
        latent_data.update(hf_data)
        output = self._cat_video_latents(latent_data)
        cfg_meta = output.pop("_lingbot_cfg_meta", None)
        output["actions"], output["actions_mask"] = self._action_post_process(
            local_start, local_end, latent_frame_ids, latent_data["action"]
        )
        output["latents"] = output["latents"].permute(3, 0, 1, 2).contiguous()
        output["frame_ids"] = torch.as_tensor(latent_frame_ids, dtype=torch.long)
        if feature_enabled("LINGBOT_SAMPLE_META_EXPORT"):
            frame_ids = torch.as_tensor(latent_frame_ids, dtype=torch.long)
            stride = (
                int(frame_ids[1].item() - frame_ids[0].item())
                if frame_ids.numel() > 1
                else 0
            )
            output["_lingbot_sample_meta"] = {
                "repo_id": str(self.repo_id),
                "local_index": int(index % len(self.sample_metas)),
                "episode_index": int(episode_index),
                "start_frame": int(local_start),
                "end_frame": int(local_end),
                "span": int(local_end - local_start),
                "frame_ids_len": int(frame_ids.numel()),
                "frame_ids_first": (
                    int(frame_ids[0].item()) if frame_ids.numel() else -1
                ),
                "frame_ids_last": (
                    int(frame_ids[-1].item()) if frame_ids.numel() else -1
                ),
                "frame_stride": stride,
                "latent_c": int(output["latents"].shape[0]),
                "latent_f": int(output["latents"].shape[1]),
                "latent_h": int(output["latents"].shape[2]),
                "latent_w": int(output["latents"].shape[3]),
                "action_c": int(output["actions"].shape[0]),
                "action_f": int(output["actions"].shape[1]),
                "action_tokens": int(output["actions"].shape[2]),
                "action_w": int(output["actions"].shape[3]),
                "action_mask_sum": int(output["actions_mask"].sum().item()),
            }
            if isinstance(cfg_meta, dict):
                output["_lingbot_sample_meta"].update(cfg_meta)
        return output

    def estimate_sample_cost(self, index: int) -> float:
        """Estimate compute cost without loading latent tensors.

        LingBot-VA sequence length is monotonic in the sampled clip span; use
        the metadata span as the accepted low-overhead cost proxy.
        """
        meta = self.sample_metas[index % len(self.sample_metas)]
        for key in ("num_frames", "length", "frames", "frame_count"):
            value = meta.get(key)
            if value is not None:
                return float(max(1, int(value)))
        start = int(meta.get("start_frame", 0))
        end = int(meta.get("end_frame", start + 1))
        return float(max(1, end - start))

    def _parse_meta(self):
        """Run parse meta.

        Returns:
            The computed result.
        """
        output = []
        for value in self.meta.episodes.values():
            episode_index = value["episode_index"]
            tasks = value.get("tasks", [])
            for action_config in value.get("action_config", []):
                current_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                    "episode_length": value.get("length"),
                }
                current_meta.update(action_config)
                if self._check_meta(
                    current_meta["start_frame"],
                    current_meta["end_frame"],
                    current_meta["episode_index"],
                ):
                    output.append(current_meta)
        return output

    def _check_meta(self, start_frame, end_frame, episode_index):
        """Run check meta.

        Args:
            start_frame: Input value for this operation.
            end_frame: Input value for this operation.
            episode_index: Input value for this operation.

        Returns:
            The computed result.
        """
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.config.obs_cam_keys:
            latent_file = (
                latent_path
                / key
                / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        """Run get global idx.

        Args:
            episode_index: Input value for this operation.
            local_index: Input value for this operation.

        Returns:
            The computed result.
        """
        return local_index + self.episode_data_index["from"][episode_index]

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        """Run get range latent data.

        Args:
            start_frame: Input value for this operation.
            end_frame: Input value for this operation.
            episode_index: Input value for this operation.

        Returns:
            The computed result.
        """
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        output = {}
        for key in self.config.obs_cam_keys:
            latent_file = (
                latent_path
                / key
                / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            latent_data = torch.load(latent_file, weights_only=False)
            for inner_key, inner_value in latent_data.items():
                output[f"{key}.{inner_key}"] = inner_value
        return output

    def _cat_video_latents(self, data_dict):
        """Run cat video latents.

        Args:
            data_dict: Input value for this operation.

        Returns:
            The computed result.
        """
        latent_list = []
        for key in self.config.obs_cam_keys:
            latent = data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(
                latent,
                "(frames height width) channels -> frames height width channels",
                frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
            )
            latent_list.append(latent)
        if self.config.env_type == "robotwin_tshape":
            wrist_latent = torch.cat(latent_list[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_list[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_list, dim=2)
        text_emb = data_dict[f"{self.config.obs_cam_keys[0]}.text_emb"]
        cfg_draw = _cfg_dropout_rand()
        cfg_dropped = cfg_draw < self.config.cfg_prob
        if cfg_dropped:
            text_emb = self.empty_emb
        output = {"latents": cat_latent, "text_emb": text_emb}
        if feature_enabled("LINGBOT_SAMPLE_META_EXPORT"):
            with torch.no_grad():
                text_float = (
                    text_emb.detach().float() if torch.is_tensor(text_emb) else None
                )
                empty_float = (
                    self.empty_emb.detach().float()
                    if torch.is_tensor(self.empty_emb)
                    else None
                )
                if (
                    text_float is not None
                    and empty_float is not None
                    and text_float.shape == empty_float.shape
                ):
                    text_is_empty = bool(
                        torch.equal(text_emb.cpu(), self.empty_emb.cpu())
                    )
                else:
                    text_is_empty = bool(cfg_dropped)
                output["_lingbot_cfg_meta"] = {
                    "cfg_draw": float(cfg_draw),
                    "cfg_prob": float(self.config.cfg_prob),
                    "cfg_dropped": int(cfg_dropped),
                    "text_is_empty": int(text_is_empty),
                    "text_numel": (
                        int(text_float.numel()) if text_float is not None else 0
                    ),
                    "text_sum": (
                        float(text_float.sum().item())
                        if text_float is not None and text_float.numel()
                        else 0.0
                    ),
                    "text_abs_sum": (
                        float(text_float.abs().sum().item())
                        if text_float is not None and text_float.numel()
                        else 0.0
                    ),
                    "text_sq_sum": (
                        float(text_float.pow(2).sum().item())
                        if text_float is not None and text_float.numel()
                        else 0.0
                    ),
                    "text_max_abs": (
                        float(text_float.abs().max().item())
                        if text_float is not None and text_float.numel()
                        else 0.0
                    ),
                }
        return output

    def _action_post_process(
        self, local_start_frame, local_end_frame, latent_frame_ids, action
    ):
        """Run action post process.

        Args:
            local_start_frame: Input value for this operation.
            local_end_frame: Input value for this operation.
            latent_frame_ids: Input value for this operation.
            action: Input value for this operation.

        Returns:
            The computed result.
        """
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        latent_frame_ids = torch.as_tensor(latent_frame_ids).cpu().numpy()
        action_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = int(latent_frame_ids[1] - latent_frame_ids[0])
        action = action[action_shift:]
        if self.config.env_type == "robotwin_tshape":
            action = _robotwin_relative_action(action)
        action = np.pad(
            action,
            pad_width=((frame_stride * 4, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4
        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype="bool")
        action_padded = np.pad(
            action, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )
        mask_padded = np.pad(
            action_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )
        action_aligned = action_padded[:, self.config.inverse_used_action_channel_ids]
        mask_aligned = mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
            self.q99 - self.q01 + 1e-6
        ) * 2.0 - 1.0
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(
            action_aligned,
            "(frames tokens) channels -> channels frames tokens 1",
            frames=latent_frame_num,
        )
        mask_aligned = rearrange(
            mask_aligned,
            "(frames tokens) channels -> channels frames tokens 1",
            frames=latent_frame_num,
        )
        action_aligned *= mask_aligned
        return (
            torch.from_numpy(action_aligned).float(),
            torch.from_numpy(mask_aligned).bool(),
        )


def _find_lerobot_repos(config: LingBotVALatentDatasetConfig) -> List[str]:
    """Run find lerobot repos.

    Args:
        config: Input value for this operation.

    Returns:
        The computed result.
    """
    max_repos = int(os.getenv("LINGBOT_VA_MAX_REPOS", 0))
    if feature_enabled("LINGBOT_REPO_DISCOVERY_CACHE"):
        cached = _load_or_build_repo_discovery_cache(config, max_repos)
        if cached is not None:
            return cached
    return _scan_lerobot_repos(config, max_repos)


def _scan_lerobot_repos(
    config: LingBotVALatentDatasetConfig, max_repos: int
) -> List[str]:
    """Run scan lerobot repos.

    Args:
        config: Input value for this operation.
        max_repos: Input value for this operation.

    Returns:
        The computed result.
    """
    repos = []
    # Keep the community baseline's native filesystem order for no-replay
    # per-step comparisons. Do not sort os.walk() directories here.
    for root, _, files in os.walk(config.dataset_path):
        if config.metadata_filename in files and root.endswith("meta"):
            repos.append(str(Path(root).parent))
    return repos[:max_repos] if max_repos > 0 else repos


def _load_or_build_repo_discovery_cache(
    config: LingBotVALatentDatasetConfig, max_repos: int
) -> List[str]:
    """Run load or build repo discovery cache.

    Args:
        config: Input value for this operation.
        max_repos: Input value for this operation.

    Returns:
        The computed result.
    """
    dataset_path = Path(config.dataset_path)
    cache_dir = Path(
        os.environ.get(
            "LINGBOT_REPO_DISCOVERY_CACHE_DIR", dataset_path / ".lingbot_cost_cache"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = str(dataset_path).replace("/", "__")
    cache_path = (
        cache_dir
        / f"{safe_dataset}.{config.metadata_filename}.max{max_repos}.repo_list.json"
    )
    signature = {
        "version": 3,
        "dataset_path": str(dataset_path),
        "metadata_filename": config.metadata_filename,
        "max_repos": max_repos,
        "ordering": "os.walk",
    }
    cached = _read_repo_discovery_cache(cache_path, signature)
    if cached is not None:
        return cached

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    wait_s = REPO_DISCOVERY_CACHE_WAIT_SECONDS
    poll_s = REPO_DISCOVERY_CACHE_POLL_SECONDS
    start_time = time.time()
    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            cached = _read_repo_discovery_cache(cache_path, signature)
            if cached is not None:
                return cached
            if time.time() - start_time > wait_s:
                raise TimeoutError(
                    f"Timed out waiting for LingBot repo discovery cache: {cache_path}"
                )
            time.sleep(poll_s)

    try:
        os.write(lock_fd, f"pid={os.getpid()} time={time.time()}\n".encode())
        cached = _read_repo_discovery_cache(cache_path, signature)
        if cached is not None:
            return cached
        repos = _scan_lerobot_repos(config, max_repos)
        payload = {"signature": signature, "repos": repos}
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, cache_path)
        return repos
    finally:
        os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _read_repo_discovery_cache(cache_path: Path, signature: Dict[str, Any]):
    """Run read repo discovery cache.

    Args:
        cache_path: Input value for this operation.
        signature: Input value for this operation.

    Returns:
        The computed result.
    """
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            cached_signature = cached.get("signature", {})
            repos = [str(repo) for repo in cached.get("repos", [])]
            if cached_signature == signature:
                return repos
        except Exception:
            pass
    return None


def _robotwin_relative_action(action: np.ndarray) -> np.ndarray:
    """Run robotwin relative action.

    Args:
        action: Input value for this operation.

    Returns:
        The computed result.
    """
    if Rotation is None:
        raise ImportError(
            "scipy is required for robotwin_tshape relative pose conversion"
        )
    left_action = _get_relative_pose(action[:, :7])
    right_action = _get_relative_pose(action[:, 8:15])
    return np.concatenate(
        [left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1
    )


def _get_relative_pose(pose: np.ndarray) -> np.ndarray:
    """Run get relative pose.

    Args:
        pose: Input value for this operation.

    Returns:
        The computed result.
    """
    rotation = Rotation.from_quat(pose[:, 3:7])
    first_rotation = Rotation.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    relative_translation = pose[:, :3] - pose[0:1, :3]
    relative_rotation = first_rotation.inv() * rotation
    return np.concatenate([relative_translation, relative_rotation.as_quat()], axis=1)


def _get_required_arg(args, *names):
    """Run get required arg.

    Args:
        args: Input value for this operation.
        *names: Input value for this operation.

    Returns:
        The computed result.
    """
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            if isinstance(value, list):
                return value[0]
            return value
    raise ValueError(f"Missing required argument, expected one of: {names}")


def _as_list(value: Any) -> List[Any]:
    """Run as list.

    Args:
        value: Input value for this operation.

    Returns:
        The computed result.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _as_float_list(value: Any) -> List[float]:
    """Run as float list.

    Args:
        value: Input value for this operation.

    Returns:
        The computed result.
    """
    return [float(item) for item in _as_list(value)]


def _as_int_list(value: Any) -> List[int]:
    """Run as int list.

    Args:
        value: Input value for this operation.

    Returns:
        The computed result.
    """
    return [int(item) for item in _as_list(value)]


def build_lingbot_dataset(model_cfg, data_cfg, training_args):
    """Build the LingBot-VA latent dataset for the embodied data engine."""
    config = LingBotVALatentDatasetConfig.from_configs(
        model_cfg, data_cfg, training_args
    )
    dataset = LingBotVAMultiLatentDataset(config)
    return dataset
