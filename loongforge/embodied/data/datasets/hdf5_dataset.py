# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Data processing aligned with the reference X-VLA implementation
# (https://github.com/2toinf/X-VLA), datasets/dataset.py +
# datasets/domain_handler/real_world.py (AIR-AGILEX handler).

"""Native HDF5 VLA dataset adapter (reference-aligned).

This loader reproduces the reference X-VLA data pipeline for episode HDF5
datasets so that the per-sample tensors fed to the model match the reference
``InfiniteDataReader`` exactly:

  * Actions are derived from ``observations/eef_quaternion`` ([T, 16]):
    left/right arm each → xyz(3) + rot6d(6) + gripper(1) = 10 dims, concatenated
    to a 20-dim absolute trajectory. Gripper is thresholded ``raw * 50 < 1.0``.
  * The future window is sampled by time interpolation (``interp1d``) over
    ``eef_left_time`` / ``eef_right_time`` across ``[cur, cur + qdur]`` with
    ``num_actions + 1`` points. ``proprio = abs[0]``, ``action = abs[1:]``.
  * Candidate start indices use stride 2 (``range(0, T - 30, 2)``); static
    segments (where the first action step barely moves) are skipped.
  * Images: per-view resize to 224×224 (BICUBIC) + ImageNet normalization, with
    optional ColorJitter during training — applied here so the collator's
    XVLAProcessor receives already-tensorized views ([0,1]→normalized).
  * No q99 / statistical normalization is applied to actions.

The reference uses an *infinite, weighted-random* iterable reader. Here we
expose the same per-sample tensors through a map-style :class:`Dataset`: a flat
index over ``(episode, candidate_start)`` pairs is precomputed in the original
iteration order (episode by episode, candidate by candidate), so ``__getitem__``
reproduces the deterministic non-shuffled sample sequence. Shuffling across
epochs is delegated to the DataLoader sampler.
"""

import io
import h5py
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper (mirrors X-VLA/datasets/utils.py:quat_to_rotate6d)
# ─────────────────────────────────────────────────────────────────────────────
def _quat_to_rotate6d(q: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    """
    Convert quaternion(s) to the 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation
    matrix, flattened to a 6-element vector.  This mirrors
    ``X-VLA/datasets/utils.py:quat_to_rotate6d``.

    Args:
        q: Array of shape (..., 4) representing quaternions in xyzw order
           (or wxyz when ``scalar_first=True``).
        scalar_first: If ``True``, the scalar (w) component is the first element.

    Returns:
        Array of shape (..., 6) containing the 6D rotation vectors.
    """
    from scipy.spatial.transform import Rotation as R

    return (
        R.from_quat(q, scalar_first=scalar_first)
        .as_matrix()[..., :, :2]
        .reshape(q.shape[:-1] + (6,))
    )


class HDF5VLADataset(Dataset):
    """Reference-aligned map-style dataset over episode HDF5 files."""

    # Reference AIR-AGILEX handler constants.
    FREQ = 30.0
    QDUR = 2.0
    # Future-window margin / candidate stride (AIRAgilexHandler.index_candidates).
    INDEX_MARGIN = 30
    INDEX_STRIDE = 2
    STATIC_EPS = 1e-5

    def __init__(
        self,
        root: str | Path,
        action_horizon: int = 30,
        num_views: int = 3,
        training: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset by discovering episode files and building the image pipeline.

        Reads ``metadata.json`` (or any ``*.json`` that contains ``dataset_name``
        and ``datalist`` keys) to determine the episode file list, domain id,
        observation keys, and language instruction key.  Falls back to
        globbing ``episode_*.hdf5`` when no metadata is present.

        Args:
            root: Root directory containing HDF5 episode files and optional
                  ``metadata.json``.
            action_horizon: Number of future action steps (``num_actions``) to
                            return per sample.
            num_views: Number of camera views to stack into ``image_input``.
            training: When ``True``, enables ColorJitter augmentation and
                      infinite shuffled streaming.
            transform: Optional callable applied to each sample dict after
                       assembly.

        Raises:
            FileNotFoundError: If no episode HDF5 files are found under ``root``.
        """

        # NOTE: do not store the module object on ``self`` — module objects are
        # not pickleable, which breaks DataLoader workers that pickle the dataset
        # (spawn start method). ``_open_h5`` re-imports h5py locally instead.
        _ = h5py
        self.root = Path(root)
        self.num_actions = action_horizon
        self.num_views = num_views
        self.training = training
        self._transform = transform

        metadata = self._read_metadata(self.root)
        self.dataset_name: str = metadata.get("dataset_name", "AIR-AGILEX")
        self.robot_type: str = metadata.get("robot_type", self.dataset_name)
        self.observation_keys: List[str] = metadata.get(
            "observation_key",
            ["observations/images/cam_high"],
        )
        self.language_instruction_key: str = metadata.get(
            "language_instruction_key", "language_instruction"
        )

        # Episode files come from metadata["datalist"] (absolute paths) when
        # present, otherwise discovered on disk.
        datalist = metadata.get("datalist")
        if datalist:
            self._episode_files = [Path(p if isinstance(p, str) else p[0]) for p in datalist]
        else:
            self._episode_files = self._discover_episodes(self.root)
        if not self._episode_files:
            raise FileNotFoundError(f"No episode files found under {self.root}")

        # Lazy per-episode data cache. Instead of holding open h5py file handles
        # (which are not pickleable and therefore break DataLoader workers under
        # the spawn/fork-then-pickle start method), each episode's parsed data is
        # loaded on first access via ``_get_episode_data`` and memoized here.
        # ``__getitem__`` reads only the frame it needs from the cached handle.
        self._episode_cached_data: List[Optional[Dict[str, Any]]] = [
            None
        ] * len(self._episode_files)

        # Stats kept for inference-time denormalization compatibility (identity:
        # the reference does not normalize actions, so q01/q99 span [-1, 1] etc.
        # are computed but unused for training when normalization is disabled).
        self._stats = self._compute_action_statistics()

        # Per-episode lightweight metadata (instruction, image mask, the
        # left/right interpolation callables and reference time axis). These are
        # independent of the sample index and small in size, so they are
        # precomputed once here and reused by both index construction and
        # __getitem__. Only the per-frame images are read lazily per sample.
        self._episode_meta = [
            self._build_episode_meta(ep_idx) for ep_idx in range(len(self._episode_files))
        ]

        # Flat index over (episode_index, candidate_start) pairs in the original
        # iteration order. Static segments are filtered here so that the sample
        # sequence and count match the previous __iter__ behavior exactly.
        self._index = self._build_index()

        logger.info(
            "HDF5VLADataset(reference-aligned): root=%s, dataset=%s, episodes=%d, "
            "num_actions=%d, views=%s, training=%s, samples=%d",
            root, self.dataset_name, len(self._episode_files),
            self.num_actions, self.observation_keys, training, len(self._index),
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _read_metadata(root: Path) -> Dict[str, Any]:
        """
        Load dataset metadata from the root directory.

        Checks for ``metadata.json`` first; if absent, iterates over all
        ``*.json`` files in the directory and returns the first one that
        contains both ``dataset_name`` and ``datalist`` keys.  Returns an
        empty dict when no matching file is found.

        Args:
            root: Dataset root directory.

        Returns:
            Parsed metadata dict, or ``{}`` if none is found.
        """
        meta_path = root / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        for p in sorted(root.glob("*.json")):
            with open(p) as f:
                meta = json.load(f)
            if "dataset_name" in meta and "datalist" in meta:
                return meta
        return {}

    @staticmethod
    def _discover_episodes(root: Path) -> List[Path]:
        """
        Glob and sort all ``episode_*.hdf5`` files under the root directory.

        Files are sorted numerically by the trailing integer in the stem
        (e.g. ``episode_0042.hdf5`` → 42) so iteration order is deterministic.

        Args:
            root: Dataset root directory.

        Returns:
            Sorted list of episode file paths.
        """
        files = list(root.glob("episode_*.hdf5"))

        def _key(p: Path) -> int:
            """Extract the trailing numeric index from an episode filename."""
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return 0

        return sorted(files, key=_key)

    def _open_h5(self, path: Path):
        """
        Open an HDF5 file, falling back to in-memory loading on OSError.

        Some network/FUSE file systems raise ``OSError`` when ``h5py`` tries to
        open the file directly.  In that case the file is read into a
        ``BytesIO`` buffer first.

        Args:
            path: Path to the HDF5 episode file.

        Returns:
            An open ``h5py.File`` handle in read-only mode.
        """

        try:
            return h5py.File(str(path), "r")
        except OSError:
            with open(path, "rb") as fh:
                return h5py.File(io.BytesIO(fh.read()), "r")

    def _get_episode_data(self, ep_idx: int):
        """
        Return the open HDF5 handle for episode ``ep_idx``, loading it on demand.

        The handle is memoized in ``self._episode_cached_data`` so that repeated
        accesses reuse the same open file. Because the cache starts empty (and is
        not part of the pickled state — see ``__getstate__``), each DataLoader
        worker lazily reopens files as it accesses them, which keeps the dataset
        pickleable while avoiding reopening on every ``__getitem__`` call.

        Args:
            ep_idx: Index into ``self._episode_files``.

        Returns:
            An open ``h5py.File`` handle in read-only mode.
        """
        cached = self._episode_cached_data[ep_idx]
        if cached is None:
            cached = self._open_h5(self._episode_files[ep_idx])
            self._episode_cached_data[ep_idx] = cached
        return cached

    # --------------------------------------------------------- action assembly
    def _build_left_right(self, f) -> tuple:
        """AIR-AGILEX layout: eef_quaternion [T,16] → left/right [T,10]."""
        eef = f["observations/eef_quaternion"][()]  # [T, 16]
        l_xyz, l_quat, l_grip = eef[:, :3], eef[:, 3:7], (eef[:, 7:8] * 50 < 1.0)
        r_xyz, r_quat, r_grip = eef[:, 8:11], eef[:, 11:15], (eef[:, 15:16] * 50 < 1.0)
        left = np.concatenate([l_xyz, _quat_to_rotate6d(l_quat), l_grip], axis=-1)
        right = np.concatenate([r_xyz, _quat_to_rotate6d(r_quat), r_grip], axis=-1)
        return left.astype(np.float32), right.astype(np.float32)

    def _index_candidates(self, t_left: int) -> Iterable[int]:
        """
        Return candidate start indices for the future-window sampler.

        Mirrors ``AIRAgilexHandler.index_candidates``: stride-2 range over
        ``[0, t_left - INDEX_MARGIN)`` so there is always enough trajectory
        remaining for a full ``num_actions``-step window.

        Args:
            t_left: Total number of time steps in the left-arm trajectory.

        Returns:
            Range of valid start indices.
        """
        return range(0, max(0, t_left - self.INDEX_MARGIN), self.INDEX_STRIDE)

    def _decode_instruction(self, f) -> str:
        """
        Read and decode the language instruction string from an open HDF5 file.

        The key used is ``self.language_instruction_key`` (default
        ``"language_instruction"``).  Byte strings are decoded to UTF-8.

        Args:
            f: Open ``h5py.File`` handle.

        Returns:
            Language instruction as a plain Python string.
        """
        lang = f[self.language_instruction_key][()]
        if isinstance(lang, bytes):
            lang = lang.decode()
        return str(lang)

    # ------------------------------------------------------------- statistics
    def _compute_action_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute per-dimension action statistics over all episodes.

        Loads every episode file, extracts the 20-dim absolute trajectory
        (left 10 + right 10 dims), and computes mean, std, min, max, 1st and
        99th percentiles.  These statistics are stored for optional downstream
        normalization (the reference X-VLA training path does not apply
        normalization, but the values are exposed for compatibility).

        Returns:
            Dict with a single ``"action"`` key mapping to a sub-dict of
            ``mean``, ``std``, ``min``, ``max``, ``q01``, ``q99`` arrays,
            each of shape ``(20,)``.
        """
        all_actions = []
        for ep_idx in range(len(self._episode_files)):
            left, right = self._build_left_right(self._get_episode_data(ep_idx))
            all_actions.append(np.concatenate([left, right], axis=-1))
        actions = np.concatenate(all_actions, axis=0)
        return {
            "action": {
                "mean": np.mean(actions, axis=0),
                "std": np.std(actions, axis=0),
                "min": np.min(actions, axis=0),
                "max": np.max(actions, axis=0),
                "q01": np.percentile(actions, 1, axis=0),
                "q99": np.percentile(actions, 99, axis=0),
            }
        }

    @property
    def dataset_statistics(self) -> Dict:
        """
        Return the precomputed action statistics dict.

        Keys: ``"action"`` → sub-dict with ``mean``, ``std``, ``min``, ``max``,
        ``q01``, ``q99`` as NumPy arrays of shape ``(20,)``.
        """
        return self._stats

    @property
    def meta(self):
        """
        Return a lightweight metadata object for downstream compatibility.

        Produces a simple namespace with:
        * ``stats``: action statistics as tensors (mirrors ``dataset_statistics``).
        * ``camera_keys``: list of HDF5 observation keys used for images.
        """
        class _Meta:
            pass

        m = _Meta()
        m.stats = {
            k: {sk: torch.as_tensor(sv) for sk, sv in v.items()}
            for k, v in self._stats.items()
        }
        m.camera_keys = self.observation_keys
        return m

    # ----------------------------------------------------------- indexing
    def _build_episode_meta(self, ep_idx: int) -> Dict[str, Any]:
        """
        Precompute the per-episode, index-independent sampling structures.

        Reads the language instruction and left/right EEF trajectories from the
        episode (loaded on demand via ``_get_episode_data``), then builds the
        time-interpolation callables and the shared reference time axis. Image
        frames are *not* read here; they are loaded lazily per sample in
        ``__getitem__`` to avoid pulling a whole episode's frames into memory for
        a single sample.

        Args:
            ep_idx: Index into ``self._episode_files``.

        Returns:
            Dict with keys ``ins``, ``image_mask``, ``L``, ``R``, ``ref``,
            ``ref_max``, ``n_left``, ``n_views`` — where ``L`` / ``R`` are
            ``interp1d`` callables over the left/right trajectories.
        """
        from scipy.interpolate import interp1d

        f = self._get_episode_data(ep_idx)
        n_views = len(self.observation_keys)
        ins = self._decode_instruction(f)
        left, right = self._build_left_right(f)
        lt = f["observations/eef_left_time"][()] if "observations/eef_left_time" in f else None
        rt = f["observations/eef_right_time"][()] if "observations/eef_right_time" in f else None

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: n_views] = True
        if lt is None:
            lt = np.arange(left.shape[0], dtype=np.float64) / float(self.FREQ)
        if rt is None:
            rt = np.arange(right.shape[0], dtype=np.float64) / float(self.FREQ)

        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0
        return {
            "ins": ins,
            "image_mask": image_mask,
            "L": L,
            "R": R,
            "ref": ref,
            "ref_max": float(ref.max()),
            "n_left": left.shape[0],
            "n_views": n_views,
        }

    def _is_static(self, L, R, ref, ref_max, idx: int) -> bool:
        """
        Return ``True`` if the candidate window at ``idx`` is a static segment.

        Mirrors the reference skip rule: a window is static when neither arm
        moves more than ``STATIC_EPS`` between the first two interpolated steps.

        Args:
            L: Left-arm ``interp1d`` callable.
            R: Right-arm ``interp1d`` callable.
            ref: Shared reference time axis.
            ref_max: Maximum reference time.
            idx: Candidate start index.

        Returns:
            Whether the candidate should be skipped.
        """
        cur = ref[idx]
        q = np.linspace(cur, min(cur + self.QDUR, ref_max), self.num_actions + 1, dtype=np.float32)
        lseq = torch.tensor(L(q))
        rseq = torch.tensor(R(q))
        return (
            (lseq[1] - lseq[0]).abs().max() < self.STATIC_EPS
            and (rseq[1] - rseq[0]).abs().max() < self.STATIC_EPS
        )

    def _build_index(self) -> List[Tuple[int, int]]:
        """
        Build the flat sample index in the original iteration order.

        For each episode (in file order) the non-shuffled candidate start
        indices are enumerated and static segments are dropped, yielding the
        exact same sample sequence the previous ``__iter__`` produced in
        evaluation mode (and one shuffled epoch of training).

        Returns:
            List of ``(episode_index, candidate_start)`` tuples.
        """
        index: List[Tuple[int, int]] = []
        for ep_idx, meta in enumerate(self._episode_meta):
            L, R, ref, ref_max = meta["L"], meta["R"], meta["ref"], meta["ref_max"]
            for idx in self._index_candidates(meta["n_left"]):
                if self._is_static(L, R, ref, ref_max, idx):
                    continue
                index.append((ep_idx, idx))
        return index

    # ----------------------------------------------------------- access
    def __len__(self) -> int:
        """Number of valid (non-static) samples across all episodes."""
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Assemble the ``i``-th sample as a generic VLA sample dict.

        Mirrors the output contract of :class:`LeRobotV2Dataset` /
        :class:`LeRobotV3Dataset` so this dataset is a drop-in replacement: the
        per-view images are emitted under ``observation.images.<cam>`` keys
        (raw [C, H, W] float32 in [0, 1]), alongside ``observation.state``,
        ``action``, ``task`` and ``robot_type``. Image resize /
        normalization and tokenization are left to the downstream transform +
        collator (the XVLA collator's raw-image path routes these through the
        XVLAProcessor), so that swapping in ``lerobot_dataset`` requires no
        change to ``xvla_transform`` / ``xvla_collator``.

        The X-VLA-specific action generation is performed here (unchanged):
        time interpolation over ``[cur, cur + QDUR]`` with ``num_actions + 1``
        points → 20-dim absolute trajectory, ``proprio = abs[0]``,
        ``action = abs[1:]``.

        Args:
            i: Flat sample index.

        Returns:
            Sample dict with keys: ``observation.images.<cam>`` (one per view),
            ``observation.state``, ``action``, ``task``, ``robot_type``.
        """
        ep_idx, idx = self._index[i]
        f = self._get_episode_data(ep_idx)
        meta = self._episode_meta[ep_idx]

        L, R = meta["L"], meta["R"]
        ref, ref_max = meta["ref"], meta["ref_max"]

        cur = ref[idx]
        q = np.linspace(cur, min(cur + self.QDUR, ref_max), self.num_actions + 1, dtype=np.float32)
        lseq = torch.tensor(L(q))
        rseq = torch.tensor(R(q))

        abs_traj = torch.cat([lseq, rseq], dim=-1).float()  # [H+1, 20]
        proprio = abs_traj[0]
        action = abs_traj[1:].clone()

        sample: Dict[str, Any] = {
            "observation.state": proprio,
            "action": action,
            "task": meta["ins"],
            "robot_type": self.robot_type,
        }

        # Lazily read only the frame at ``idx`` for each view (HDF5 slice) and
        # emit raw per-view images under generic keys (HWC uint8 → CHW [0, 1]),
        # matching the lerobot ``observation.images.<cam>`` contract.
        V = min(self.num_views, meta["n_views"])
        for v in range(V):
            frame = self._to_chw01(f[self.observation_keys[v]][idx])
            sample[f"observation.images.{self._cam_name(self.observation_keys[v])}"] = frame

        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    @staticmethod
    def _cam_name(observation_key: str) -> str:
        """Derive a stable camera name from an HDF5 observation key.

        Uses the trailing path component (e.g. ``observations/images/cam_high``
        → ``cam_high``) so the emitted ``observation.images.<name>`` keys are
        deterministic and human-readable.
        """
        return observation_key.rsplit("/", 1)[-1]

    def _to_chw01(self, raw) -> torch.Tensor:
        """Convert a raw HDF5 image value to a CHW float32 tensor in [0, 1].

        Handles the same storage formats as the reference loader:
        * NumPy HxWxC uint8 array → permuted to CHW and scaled by 1/255.
        * Bytes / byte-string (JPEG/PNG encoded) → decoded via PIL.
        * PIL image → converted to RGB then to tensor.
        """
        from PIL import Image

        if isinstance(raw, Image.Image):
            arr = np.asarray(raw.convert("RGB"), dtype=np.uint8)
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            arr = raw.astype(np.uint8)
        else:
            arr = np.asarray(
                Image.open(io.BytesIO(bytes(raw))).convert("RGB"), dtype=np.uint8
            )
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the open HDF5 handles when pickling.

        ``h5py.File`` handles are not pickleable, so the lazily-populated
        ``_episode_cached_data`` is reset to all-``None`` in the pickled state.
        Each worker process reopens episode files on demand via
        ``_get_episode_data`` after unpickling.
        """
        state = self.__dict__.copy()
        state["_episode_cached_data"] = [None] * len(self._episode_files)
        return state

    def __del__(self):
        """Close all open episode HDF5 handles."""
        for f in self._episode_cached_data:
            if f is None:
                continue
            try:
                f.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# Builder (called by data/__init__.py)
# ═══════════════════════════════════════════════════════════════

def build_hdf5_dataset(model_cfg, data_cfg, training_args) -> Dataset:
    """Build HDF5 dataset from typed configs + CLI training_args."""
    action_horizon = model_cfg.action_horizon

    dataset_path = training_args.dataset_path or None
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")

    action_horizon = model_cfg.action_horizon
    num_views = data_cfg.num_image_views

    return HDF5VLADataset(
        root=dataset_path,
        action_horizon=action_horizon,
        num_views=num_views,
        training=True,
    )
