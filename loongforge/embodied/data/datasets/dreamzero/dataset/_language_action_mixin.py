# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""Language-chunk sampling and state/action retrieval mixin for
``DreamZeroLeRobotDataset``.

Handles language-annotation lookup and uniform chunk sampling,
``has_full_language_chunks`` (consumed by ``DreamZeroShardedSampler``),
state/action array retrieval (including language-chunk-aware slicing and
relative-action conversion), LAPA/dream-action/RL-info/modality-keyed
retrieval, and initial-action loading. Mixed into ``DreamZeroLeRobotDataset``;
relies on attributes assigned in that class's ``__init__`` (``self.modality_configs``,
``self.relative_action``, etc.) via normal Python attribute lookup at call time.
"""

import numpy as np
import pandas as pd

from .constants import INITIAL_ACTIONS_FILENAME, METADATA_LANG_KEYS


def _load_initial_actions(path):
    """Load initial actions saved as a pickled object array in an .npz file."""
    with np.load(str(path), allow_pickle=True) as payload:
        return [
            {key: value for key, value in item.items()}
            for item in payload["arr_0"]
        ]


class _DreamZeroLanguageActionMixin:
    """Language-chunk sampling and state/action retrieval methods."""

    def _get_language_key(self) -> str | None:
        """Return the original parquet column name for the annotation language key."""
        for modality_name in self.modality_keys:
            for modality_key in self.modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    if annotation_meta is None or subkey not in annotation_meta:
                        continue
                    return annotation_meta[subkey].original_key
        return None

    def _get_language_annotations(self, trajectory_id: int) -> np.ndarray | None:
        """Return the per-step language annotation column for a trajectory, if present."""
        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = self._get_language_key()
        if language_key is None or language_key not in traj_data.columns:
            return None
        return traj_data[language_key].values

    def _sample_video_indices_for_language_chunk(
        self,
        trajectory_id: int,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Expand video indices within the anchor's language segment when enabled."""
        if not self.language_chunk_sampling:
            return step_indices
        language_annotations = self._get_language_annotations(trajectory_id)
        if language_annotations is None:
            return step_indices
        trajectory_index = self.get_trajectory_index(trajectory_id)
        return self._uniform_sample_from_language_ranges(
            step_indices,
            language_annotations,
            int(self.trajectory_lengths[trajectory_index]),
        )

    def _uniform_sample_from_language_ranges(
        self,
        step_indices: np.ndarray,
        language_annotations: np.ndarray,
        trajectory_length: int,
    ) -> np.ndarray:
        """Sample video chunks from a consistent DROID language segment."""
        if len(step_indices) == 0:
            return np.array([])

        first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
        target_language = language_annotations[first_idx]
        max_frames = 8 * int(self.max_chunk_size or 1) + 1
        per_step_offsets = [0, 3, 6, 9, 12, 15, 18, 21]
        sampled_list: list[int] = []

        def add_step_set(anchor_index: int) -> None:
            """Append an 8-frame video chunk anchored at ``anchor_index`` if under budget."""
            if anchor_index < 0 or anchor_index + 23 >= trajectory_length:
                return
            if len(sampled_list) + len(per_step_offsets) > max_frames:
                return
            for offset in per_step_offsets:
                sampled_list.append(int(anchor_index + offset))

        add_step_set(first_idx)

        step = 1
        back_done = False
        fwd_done = False
        while len(sampled_list) < max_frames and (not back_done or not fwd_done):
            if not back_done:
                back_anchor = first_idx - 24 * step
                if back_anchor < 0:
                    back_done = True
                elif language_annotations[back_anchor] != target_language:
                    back_done = True
                else:
                    add_step_set(back_anchor)
            if len(sampled_list) >= max_frames:
                break
            if not fwd_done:
                fwd_anchor = first_idx + 24 * step
                if fwd_anchor >= trajectory_length:
                    fwd_done = True
                elif language_annotations[fwd_anchor] != target_language:
                    fwd_done = True
                else:
                    add_step_set(fwd_anchor)
            step += 1

        if len(sampled_list) == 0:
            return np.array([])
        unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
        if unique_sorted.size > max_frames:
            unique_sorted = unique_sorted[:max_frames]

        if unique_sorted.size > 0:
            additional_idx = int(unique_sorted[-1]) + 3
            if additional_idx < trajectory_length and unique_sorted.size < max_frames:
                unique_sorted = np.append(unique_sorted, additional_idx)
            else:
                if unique_sorted.size <= 8:
                    return np.array([])
                unique_sorted = unique_sorted[:-7]

        assert unique_sorted.size % 8 == 1, (
            f"unique_sorted size {unique_sorted.size} is not 8n+1"
        )
        self._current_num_chunks[first_idx] = (unique_sorted.size - 1) // 8
        return unique_sorted

    def has_full_language_chunks(self, trajectory_id: int, step_index: int) -> bool:
        """Return whether language-chunk sampling produces fixed-size tensors."""
        if not self.language_chunk_sampling or not self.max_chunk_size:
            return True

        cache_key = (int(trajectory_id), int(step_index))
        cache = self._full_language_chunk_cache
        if cache_key in cache:
            return cache[cache_key]

        def remember(value: bool) -> bool:
            """Cache ``value`` for this trajectory/step under ``cache_key`` and return it."""
            cache[cache_key] = value
            return value

        language_annotations = self._get_language_annotations(trajectory_id)
        if language_annotations is None:
            return remember(True)

        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = int(self.trajectory_lengths[trajectory_index])
        if trajectory_length <= 0:
            return remember(False)

        first_idx = max(0, min(int(step_index), trajectory_length - 1))
        video_indices = self._uniform_sample_from_language_ranges(
            np.array([first_idx], dtype=int),
            language_annotations,
            trajectory_length,
        )
        if video_indices.size == 0 or video_indices.size % 8 != 1:
            return remember(False)

        max_chunks = int(self.max_chunk_size)
        num_video_chunks = (video_indices.size - 1) // 8
        previous_chunk_count = self._current_num_chunks.get(first_idx)
        self._current_num_chunks[first_idx] = num_video_chunks
        try:
            action_indices = self._language_chunk_action_indices(
                trajectory_id,
                np.array([first_idx], dtype=int),
            )
            state_indices = self._language_chunk_state_indices(
                trajectory_id,
                np.array([first_idx], dtype=int),
            )
        finally:
            if previous_chunk_count is None:
                self._current_num_chunks.pop(first_idx, None)
            else:
                self._current_num_chunks[first_idx] = previous_chunk_count

        return remember(
            num_video_chunks == max_chunks
            and action_indices.size == 24 * max_chunks
            and state_indices.size == max_chunks
        )

    def _get_state_action_array(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
    ) -> tuple[np.ndarray, object, str]:
        """Return the raw state/action array, its config, and subkey for a modality key."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        subkey = key.replace(modality + ".", "")
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey
        trajectory_data = self.get_trajectory_data(trajectory_id)
        assert le_key in trajectory_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(trajectory_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]
        return data_array, state_or_action_cfg, subkey

    def _language_chunk_state_indices(
        self,
        trajectory_id: int,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Return state-chunk anchor indices sampled within a consistent language segment."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = int(self.trajectory_lengths[trajectory_index])
        language_annotations = self._get_language_annotations(trajectory_id)
        if language_annotations is None or len(step_indices) == 0:
            return np.minimum(np.maximum(step_indices, 0), trajectory_length - 1)

        first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
        target_language = language_annotations[first_idx]
        target_num_chunks = self._current_num_chunks.get(first_idx)
        max_frames = int(self.max_chunk_size or 1)
        sampled_list: list[int] = []

        def add_anchor(anchor_index: int) -> None:
            """Append ``anchor_index`` to the sampled list if under the frame budget."""
            if len(sampled_list) >= max_frames:
                return
            if target_num_chunks is not None and len(sampled_list) >= target_num_chunks:
                return
            if 0 <= anchor_index and anchor_index + 24 < trajectory_length:
                sampled_list.append(int(anchor_index))

        add_anchor(first_idx)

        step = 1
        back_done = False
        fwd_done = False
        while len(sampled_list) < max_frames and (not back_done or not fwd_done):
            if target_num_chunks is not None and len(sampled_list) >= target_num_chunks:
                break
            if not back_done:
                back_anchor = first_idx - 24 * step
                if back_anchor < 0:
                    back_done = True
                elif language_annotations[back_anchor] != target_language:
                    back_done = True
                else:
                    add_anchor(back_anchor)
            if len(sampled_list) >= max_frames:
                break
            if not fwd_done:
                fwd_anchor = first_idx + 24 * step
                if fwd_anchor >= trajectory_length:
                    fwd_done = True
                elif language_annotations[fwd_anchor] != target_language:
                    fwd_done = True
                else:
                    add_anchor(fwd_anchor)
            step += 1

        if len(sampled_list) > 0:
            return np.array(sorted(set(sampled_list)), dtype=int)
        return np.array([], dtype=int)

    def _language_chunk_action_indices(
        self,
        trajectory_id: int,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Return action-chunk indices (24 steps per chunk) sampled within a language segment."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = int(self.trajectory_lengths[trajectory_index])
        language_annotations = self._get_language_annotations(trajectory_id)
        if language_annotations is None or len(step_indices) == 0:
            return np.minimum(np.maximum(step_indices, 0), trajectory_length - 1)

        first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
        target_language = language_annotations[first_idx]
        target_num_chunks = self._current_num_chunks.get(first_idx)
        max_frames = 24 * int(self.max_chunk_size or 1)
        sampled_list: list[int] = []

        def add_step_set(anchor_index: int) -> None:
            """Append a 24-step chunk anchored at ``anchor_index`` if under budget."""
            if anchor_index < 0 or anchor_index + 24 >= trajectory_length:
                return
            if len(sampled_list) + 24 > max_frames:
                return
            if target_num_chunks is not None and len(sampled_list) // 24 >= target_num_chunks:
                return
            for offset in range(24):
                sampled_list.append(int(anchor_index + offset))

        add_step_set(first_idx)

        step = 1
        back_done = False
        fwd_done = False
        while len(sampled_list) < max_frames and (not back_done or not fwd_done):
            if target_num_chunks is not None and len(sampled_list) // 24 >= target_num_chunks:
                break
            if not back_done:
                back_anchor = first_idx - 24 * step
                if back_anchor < 0:
                    back_done = True
                elif language_annotations[back_anchor] != target_language:
                    back_done = True
                else:
                    add_step_set(back_anchor)
            if len(sampled_list) >= max_frames:
                break
            if not fwd_done:
                fwd_anchor = first_idx + 24 * step
                if fwd_anchor >= trajectory_length:
                    fwd_done = True
                elif language_annotations[fwd_anchor] != target_language:
                    fwd_done = True
                else:
                    add_step_set(fwd_anchor)
            step += 1

        if len(sampled_list) == 0:
            return np.array([], dtype=int)
        unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
        capped_size = min(unique_sorted.size, max_frames)
        divisible_size = (capped_size // 24) * 24
        return unique_sorted[:divisible_size]

    def _get_state_language_chunk(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Return state data sampled and padded according to language-chunk anchors."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        data_array, state_or_action_cfg, _ = self._get_state_action_array(
            trajectory_id, modality, key
        )
        sampled_indices = self._language_chunk_state_indices(trajectory_id, step_indices)
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def _get_action_language_chunk(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Return action data sampled/padded per language-chunk anchors, optionally relative."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        data_array, state_or_action_cfg, subkey = self._get_state_action_array(
            trajectory_id, modality, key
        )
        sampled_indices = self._language_chunk_action_indices(trajectory_id, step_indices)
        action_data = self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

        should_convert_to_relative = (
            (self.relative_action or self.relative_action_per_horizon)
            and len(sampled_indices) > 0
            and (self.relative_action_keys is None or subkey in self.relative_action_keys)
        )
        if should_convert_to_relative:
            action_data = self._convert_to_relative_action(
                action_data=action_data,
                action_key=key,
                sampled_indices=sampled_indices,
                trajectory_id=trajectory_id,
                chunk_size=24,
            )
        return action_data

    def _convert_to_relative_action(
        self,
        action_data: np.ndarray,
        action_key: str,
        sampled_indices: np.ndarray,
        trajectory_id: int,
        chunk_size: int = 24,
    ) -> np.ndarray:
        """Convert each action chunk to be relative to its anchor state, per chunk."""
        subkey = action_key.replace("action.", "")
        le_state_cfg = self.lerobot_modality_meta.state
        if subkey not in le_state_cfg:
            return action_data

        le_state_key = le_state_cfg[subkey].original_key
        if le_state_key is None:
            le_state_key = subkey

        traj_data = self.get_trajectory_data(trajectory_id)
        if le_state_key not in traj_data.columns:
            return action_data

        state_array: np.ndarray = np.stack(traj_data[le_state_key])
        if state_array.ndim == 1:
            state_array = state_array.reshape(-1, 1)

        le_indices = np.arange(
            le_state_cfg[subkey].start,
            le_state_cfg[subkey].end,
        )
        state_array = state_array[:, le_indices]

        relative_action_data = action_data.copy()
        num_chunks = len(sampled_indices) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size
            anchor_idx = int(sampled_indices[chunk_start])
            if anchor_idx >= len(state_array):
                ref_state = state_array[-1]
            else:
                ref_state = state_array[anchor_idx]
            relative_action_data[chunk_start:chunk_end] = (
                action_data[chunk_start:chunk_end] - ref_state
            )
        return relative_action_data

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        if self.language_chunk_sampling and modality == "state":
            return self._get_state_language_chunk(trajectory_id, modality, key, step_indices)
        if self.language_chunk_sampling and modality == "action":
            return self._get_action_language_chunk(trajectory_id, modality, key, step_indices)

        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            trajectory_data = self.get_trajectory_data(trajectory_id)
            frame_index_array = trajectory_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        data_array, state_or_action_cfg, _ = self._get_state_action_array(
            trajectory_id,
            modality,
            key,
        )

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_lapa_action(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Get LAPA action data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the LAPA action data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray | None: The LAPA action data, or None if the key is not found.
        """
        return self._get_optional_padded_array(trajectory_id, key, step_indices)

    def get_dream_actions(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Get DREAM action data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the DREAM action data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray | None: The DREAM action data, or None if the key is not found.
        """
        return self._get_optional_padded_array(trajectory_id, key, step_indices)

    def _get_optional_padded_array(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Return an optional trajectory column padded to the requested indices."""
        trajectory_data = self.get_trajectory_data(trajectory_id)
        if key not in trajectory_data.columns:
            return None
        data_array: np.ndarray = np.stack(trajectory_data[key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        trajectory_index = self.get_trajectory_index(trajectory_id)
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=self.trajectory_lengths[trajectory_index],
            padding_strategy="first_last",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            list[str]: The annotation data for the trajectory and step indices.
                If no matching data is found, return empty strings.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")

        # Check if this is a metadata-based language key (detailed_global_instruction_medium/concise)
        if subkey in METADATA_LANG_KEYS:
            return self._get_language_from_metadata(trajectory_id, subkey, len(step_indices))

        # Otherwise, load from parquet columns (original behavior)
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        trajectory_data = self.get_trajectory_data(trajectory_id)
        if pd.api.types.is_numeric_dtype(trajectory_data[original_key]):
            # Stored as list of integers
            task_indices: list[int] = trajectory_data[original_key].iloc[step_indices].tolist()
            return self.tasks.loc[task_indices]["task"].tolist()
        else:
            # Stored as list of strings
            return trajectory_data[original_key].iloc[step_indices].astype(str).tolist()

    def _get_language_from_metadata(
        self,
        trajectory_id: int,
        lang_key: str,
        nframes: int,
    ) -> list[str]:
        """Get language instruction from metadata files for special language keys.

        Supports:
        - detailed_global_instruction_medium: Longer, detailed description
        - detailed_global_instruction_concise: Short summary

        Args:
            trajectory_id (int): The ID of the trajectory (episode_index).
            lang_key (str): The language key (e.g., "detailed_global_instruction_medium").
            nframes (int): Number of frames to return the instruction for.

        Returns:
            list[str]: The instruction repeated for each frame (empty string if not found).
        """
        if trajectory_id in self._detailed_global_instructions:
            instruction = self._detailed_global_instructions[trajectory_id].get(lang_key, "")
        else:
            instruction = ""
        return [instruction] * nframes

    def get_rl_info(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the reward data for a trajectory by step indices.

        If the step indices are out of range, pad with first/last step data.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the reward data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray: The reward data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        trajectory_data = self.get_trajectory_data(trajectory_id)
        data_array: np.ndarray = np.stack(trajectory_data[key])  # type: ignore

        if key == "rl_info.next.reward":
            padding_strategy = "zero"
        else:
            padding_strategy = "first_last"

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str] | None:
        """Get the data corresponding to the modality for a trajectory by step indices.

        This method dispatches to the appropriate specialized method based on the modality.
        For the language modality, empty strings are returned if no matching data is found.

        Args:
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data (video, state, action, language, etc.).
            key (str): The key of the data.
            step_indices (np.ndarray): The step indices of the trajectory.

        Returns:
            np.ndarray | list[str] | None: The data for the specified modality.
        """
        if modality == "video":
            video_indices = self._sample_video_indices_for_language_chunk(
                trajectory_id,
                step_indices,
            )
            return self.get_video(trajectory_id, key, video_indices)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        elif modality == "lapa_action":
            return self.get_lapa_action(trajectory_id, key, step_indices)
        elif modality == "dream_actions":
            return self.get_dream_actions(trajectory_id, key, step_indices)
        elif modality == "rl_info":
            return self.get_rl_info(trajectory_id, key, step_indices)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_initial_actions(self):
        """Load initial actions from the dataset if available.

        Returns:
            list: List containing initial actions if the file exists, empty list otherwise.
        """
        initial_actions_path = self.dataset_path / INITIAL_ACTIONS_FILENAME
        if not initial_actions_path.exists():
            return []
        return _load_initial_actions(initial_actions_path)
