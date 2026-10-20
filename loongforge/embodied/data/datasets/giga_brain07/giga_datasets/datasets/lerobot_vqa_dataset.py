import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .. import utils
from .base_dataset import BaseDataset
from .dataset import register_dataset
from .lerobot_dataset import _decode_video_frames_pyav_streaming

_VQA_ANNOTATION_NAME = 'vqa_llava_json.jsonl'


@dataclass(frozen=True, slots=True)
class _LeRobotVQASample:
    root_index: int
    episode_index: int
    camera: str
    video_key: str
    frame_index: int
    qa_index: int
    question: str
    answer: str


@dataclass(slots=True)
class _LeRobotVQARootInfo:
    root: Path
    fps: float
    video_path_template: str
    video_keys: set[str]
    episodes: dict[int, dict[str, Any]]


@register_dataset
class LeRobotVQADataset(BaseDataset):
    """Single-QA VQA dataset backed by LeRobot videos and
    ``meta/vqa_llava_json.jsonl``.

    This dataset intentionally does not instantiate :class:`LeRobotDataset`, so
    it avoids HuggingFace/parquet frame cache construction. It only reads
    lightweight metadata under ``meta/`` and decodes requested video frames.
    """

    def __init__(
        self,
        data_path: list[str] | tuple[str, ...],
        data_size: int | None = None,
        source_name: str = 'lerobot_vqa',
        camera_key_prefix: str = 'observation.images',
        image_tag: str = '<image>\n',
        load_image: bool = True,
        tolerance_s: float = 2e-2,
        align_robot_schema: bool = False,
        robot_image_key: str = 'observation.images.cam_high',
        observation_memory_size: int = 1,
        default_embodiment_id: int = 0,
        default_action_dim: int = 32,
        default_action_horizon: int = 50,
        config_path: str | None = None,
        transform: Any = None,
    ) -> None:
        super().__init__(config_path=config_path, data_path=None, transform=transform)
        self.data_paths = self._normalize_data_path(data_path)
        self.data_size = None if data_size is None else int(data_size)
        self.source_name = source_name
        self.camera_key_prefix = camera_key_prefix.rstrip('.')
        self.image_tag = image_tag
        self.load_image = bool(load_image)
        self.tolerance_s = float(tolerance_s)
        self.align_robot_schema = bool(align_robot_schema)
        self.robot_image_key = robot_image_key
        self.observation_memory_size = int(observation_memory_size)
        if self.observation_memory_size < 1:
            raise ValueError(f'observation_memory_size must be positive, got {self.observation_memory_size}')
        self.default_embodiment_id = int(default_embodiment_id)
        self.default_action_dim = int(default_action_dim)
        self.default_action_horizon = int(default_action_horizon)

        self.samples: list[_LeRobotVQASample] | None = None
        self.root_infos: list[_LeRobotVQARootInfo] | None = None

    @property
    def data_path(self) -> list[str]:
        return self.data_paths

    @staticmethod
    def _normalize_data_path(data_path: list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(data_path, (str, os.PathLike)):
            raise TypeError('LeRobotVQADataset data_path should be a Python list of LeRobot root paths')
        if not isinstance(data_path, (list, tuple)):
            raise TypeError(f'LeRobotVQADataset data_path should be a list, got {type(data_path).__name__}')
        if len(data_path) == 0:
            raise ValueError('LeRobotVQADataset data_path should not be empty')

        data_paths: list[str] = []
        for path in data_path:
            if not isinstance(path, (str, os.PathLike)):
                raise TypeError(f'Each LeRobot root path should be str or PathLike, got {type(path).__name__}')
            data_paths.append(os.path.abspath(os.fspath(path)))
        return data_paths

    @classmethod
    def load(cls, data_or_config: str | dict) -> 'LeRobotVQADataset':
        from .dataset import load_config

        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def save(self, save_path: str, store_rel_path: bool = True) -> None:
        from .dataset import get_rel_path

        if save_path.endswith('.json'):
            save_config_path = save_path
            save_dir = os.path.dirname(save_config_path)
        else:
            save_dir = save_path
            save_config_path = os.path.join(save_path, 'config.json')

        data_paths = self.data_paths
        if store_rel_path:
            data_paths = [get_rel_path(path, save_dir) for path in data_paths]

        config = {
            '_class_name': 'LeRobotVQADataset',
            '_key_names': [
                'answer',
                'answers',
                'camera',
                'conversations',
                'dataset_type',
                'episode_index',
                'frame_index',
                'image',
                'qa_index',
                'question',
                'sample_id',
                'source_root',
                'video_key',
            ],
            'data_path': data_paths,
            'data_size': len(self),
            'source_name': self.source_name,
            'camera_key_prefix': self.camera_key_prefix,
            'image_tag': self.image_tag,
            'load_image': self.load_image,
            'tolerance_s': self.tolerance_s,
            'align_robot_schema': self.align_robot_schema,
            'robot_image_key': self.robot_image_key,
            'observation_memory_size': self.observation_memory_size,
            'default_embodiment_id': self.default_embodiment_id,
            'default_action_dim': self.default_action_dim,
            'default_action_horizon': self.default_action_horizon,
        }
        utils.save_file(save_config_path, config)

    def _load_info(self, root: Path) -> dict[str, Any]:
        info_path = root / 'meta' / 'info.json'
        if not info_path.is_file():
            raise FileNotFoundError(f'LeRobot info.json not found: {info_path}')
        with info_path.open('r') as f:
            info = json.load(f)
        if 'fps' not in info:
            raise KeyError(f'Missing fps in {info_path}')
        if 'video_path' not in info:
            raise KeyError(f'Missing video_path in {info_path}')
        if 'features' not in info or not isinstance(info['features'], dict):
            raise KeyError(f'Missing features in {info_path}')
        return info

    def _load_episode_metadata(self, root: Path) -> dict[int, dict[str, Any]]:
        episodes_dir = root / 'meta' / 'episodes'
        paths = sorted(episodes_dir.glob('*/*.parquet'))
        if len(paths) == 0:
            raise FileNotFoundError(f'No LeRobot episode metadata parquet found under {episodes_dir}')

        tables = []
        for path in paths:
            schema = pq.read_schema(path)
            columns = [name for name in schema.names if not name.startswith('stats/')]
            tables.append(pq.read_table(path, columns=columns))
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options='default')

        if 'episode_index' not in table.column_names:
            raise KeyError(f'Missing episode_index in LeRobot episode metadata under {episodes_dir}')

        episodes: dict[int, dict[str, Any]] = {}
        columns = table.column_names
        for row_index in range(table.num_rows):
            episode = {name: table[name][row_index].as_py() for name in columns}
            episode_index = int(episode['episode_index'])
            episodes[episode_index] = episode
        return episodes

    def _camera_to_video_key(self, camera: Any, root_info: _LeRobotVQARootInfo) -> str:
        if not isinstance(camera, str) or len(camera.strip()) == 0:
            raise ValueError(f'Invalid camera value: {camera!r}')

        camera = camera.strip()
        if camera in root_info.video_keys:
            return camera

        prefixed = f'{self.camera_key_prefix}.{camera}'
        if prefixed in root_info.video_keys:
            return prefixed

        raise KeyError(
            f'Camera {camera!r} does not map to a video feature. '
            f'Tried {camera!r} and {prefixed!r}. Available video keys: {sorted(root_info.video_keys)}'
        )

    def _strip_leading_image_tag(self, text: str) -> str:
        stripped = text.strip()
        image_tokens = [self.image_tag, self.image_tag.rstrip(), '<image>\n', '<image>']

        changed = True
        while changed:
            changed = False
            candidate = stripped.lstrip()
            for token in image_tokens:
                if token and candidate.startswith(token):
                    stripped = candidate[len(token) :].lstrip()
                    changed = True
                    break

        return stripped

    def _extract_qa_pairs(self, conversations: Any, ann_path: Path, episode_index: int, frame_index: int) -> list[tuple[str, str]]:
        if not isinstance(conversations, list):
            raise TypeError(f'conversations should be a list in {ann_path}, episode_index={episode_index}, frame_index={frame_index}')

        qa_pairs: list[tuple[str, str]] = []
        pending_question: str | None = None
        for message in conversations:
            if not isinstance(message, dict):
                raise TypeError(
                    f'Each conversation message should be a dict in {ann_path}, ' f'episode_index={episode_index}, frame_index={frame_index}'
                )

            role = str(message.get('from', message.get('role', ''))).strip().lower()
            value = str(message.get('value', message.get('content', '')))
            if role in {'human', 'user'}:
                if pending_question is not None:
                    raise ValueError(
                        f'Found consecutive human messages before an answer in {ann_path}, '
                        f'episode_index={episode_index}, frame_index={frame_index}'
                    )
                pending_question = self._strip_leading_image_tag(value)
            elif role in {'gpt', 'assistant'}:
                if pending_question is None:
                    raise ValueError(
                        f'Found assistant message without preceding question in {ann_path}, '
                        f'episode_index={episode_index}, frame_index={frame_index}'
                    )
                qa_pairs.append((pending_question, value.strip()))
                pending_question = None

        if pending_question is not None:
            raise ValueError(f'Found unmatched human question in {ann_path}, episode_index={episode_index}, frame_index={frame_index}')
        if len(qa_pairs) == 0:
            raise ValueError(f'No QA pairs found in {ann_path}, episode_index={episode_index}, frame_index={frame_index}')
        return qa_pairs

    def _load_root_info(self, root: Path) -> _LeRobotVQARootInfo:
        if not root.is_dir():
            raise FileNotFoundError(f'LeRobot root not found: {root}')

        info = self._load_info(root)
        video_keys = {key for key, feature in info['features'].items() if isinstance(feature, dict) and feature.get('dtype') == 'video'}
        if len(video_keys) == 0:
            raise ValueError(f'No video features found in {root / "meta" / "info.json"}')

        return _LeRobotVQARootInfo(
            root=root,
            fps=float(info['fps']),
            video_path_template=str(info['video_path']),
            video_keys=video_keys,
            episodes=self._load_episode_metadata(root),
        )

    def _build_samples_for_root(self, root_index: int, root_info: _LeRobotVQARootInfo) -> list[_LeRobotVQASample]:
        ann_path = root_info.root / 'meta' / _VQA_ANNOTATION_NAME
        if not ann_path.is_file():
            raise FileNotFoundError(f'LeRobot VQA annotation not found: {ann_path}')

        samples: list[_LeRobotVQASample] = []
        with ann_path.open('r') as f:
            for line_index, line in enumerate(f, start=1):
                line = line.strip()
                if len(line) == 0:
                    continue
                try:
                    episode_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid jsonl line in {ann_path}:{line_index}') from exc
                if not isinstance(episode_record, dict):
                    raise TypeError(f'Each jsonl line should be a dict in {ann_path}:{line_index}')

                for key in ('episode_index', 'camera', 'frames'):
                    if key not in episode_record:
                        raise KeyError(f'Missing {key!r} in {ann_path}:{line_index}')

                episode_index = int(episode_record['episode_index'])
                if episode_index not in root_info.episodes:
                    raise IndexError(f'episode_index={episode_index} not found in episode metadata for {root_info.root}')

                camera = str(episode_record['camera'])
                video_key = self._camera_to_video_key(camera, root_info)
                frames = episode_record['frames']
                if not isinstance(frames, list):
                    raise TypeError(f'frames should be a list in {ann_path}:{line_index}')

                for frame_record in frames:
                    if not isinstance(frame_record, dict):
                        raise TypeError(f'Each frame record should be a dict in {ann_path}:{line_index}')
                    if 'frame_index' not in frame_record:
                        raise KeyError(f'Missing frame_index in {ann_path}:{line_index}')
                    if 'conversations' not in frame_record:
                        raise KeyError(f'Missing conversations in {ann_path}:{line_index}')

                    frame_index = int(frame_record['frame_index'])
                    qa_pairs = self._extract_qa_pairs(
                        frame_record['conversations'],
                        ann_path,
                        episode_index,
                        frame_index,
                    )
                    for qa_index, (question, answer) in enumerate(qa_pairs):
                        samples.append(
                            _LeRobotVQASample(
                                root_index=root_index,
                                episode_index=episode_index,
                                camera=camera,
                                video_key=video_key,
                                frame_index=frame_index,
                                qa_index=qa_index,
                                question=question,
                                answer=answer,
                            )
                        )
        return samples

    def open(self) -> None:
        if self.samples is not None and self.root_infos is not None:
            return

        root_infos: list[_LeRobotVQARootInfo] = []
        samples: list[_LeRobotVQASample] = []
        for root_index, root_path in enumerate(self.data_paths):
            root_info = self._load_root_info(Path(root_path))
            root_infos.append(root_info)
            samples.extend(self._build_samples_for_root(root_index, root_info))

        if self.data_size is not None:
            assert self.data_size == len(samples), f'data_size={self.data_size} but loaded {len(samples)} QA pairs'
        else:
            self.data_size = len(samples)

        self.root_infos = root_infos
        self.samples = samples

    def close(self) -> None:
        if self.samples is not None:
            self.samples.clear()
            self.samples = None
        if self.root_infos is not None:
            self.root_infos.clear()
            self.root_infos = None
        super().close()

    def __len__(self) -> int:
        if self.data_size is None:
            self.open()
        return int(self.data_size)

    def _get_episode(self, root_info: _LeRobotVQARootInfo, sample: _LeRobotVQASample) -> dict[str, Any]:
        episode = root_info.episodes.get(sample.episode_index)
        if episode is None:
            raise IndexError(f'episode_index={sample.episode_index} not found for root {root_info.root}')
        return episode

    def _get_video_path(self, root_info: _LeRobotVQARootInfo, sample: _LeRobotVQASample) -> Path:
        episode = self._get_episode(root_info, sample)
        chunk_key = f'videos/{sample.video_key}/chunk_index'
        file_key = f'videos/{sample.video_key}/file_index'
        if chunk_key not in episode or file_key not in episode:
            raise KeyError(
                f'Missing video chunk/file metadata for video_key={sample.video_key!r}, '
                f'episode_index={sample.episode_index}, root={root_info.root}'
            )
        video_rel_path = root_info.video_path_template.format(
            video_key=sample.video_key,
            chunk_index=int(episode[chunk_key]),
            file_index=int(episode[file_key]),
        )
        video_path = root_info.root / video_rel_path
        if not video_path.is_file():
            raise FileNotFoundError(f'Video file not found: {video_path}')
        return video_path

    def _get_frame_timestamp(self, root_info: _LeRobotVQARootInfo, sample: _LeRobotVQASample) -> float:
        episode = self._get_episode(root_info, sample)
        from_timestamp_key = f'videos/{sample.video_key}/from_timestamp'
        from_timestamp = float(episode.get(from_timestamp_key, 0.0) or 0.0)
        return from_timestamp + sample.frame_index / root_info.fps

    def _decode_image(self, root_info: _LeRobotVQARootInfo, sample: _LeRobotVQASample) -> torch.Tensor:
        video_path = self._get_video_path(root_info, sample)
        timestamp = self._get_frame_timestamp(root_info, sample)
        frames = _decode_video_frames_pyav_streaming(video_path, [timestamp], self.tolerance_s)
        return frames.squeeze(0)

    def _adapt_robot_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if not self.align_robot_schema or self.observation_memory_size == 1:
            return image_tensor
        return image_tensor.unsqueeze(0).repeat(self.observation_memory_size, 1, 1, 1).contiguous()

    def _apply_robot_schema_defaults(self, data_dict: dict[str, Any]) -> None:
        if not self.align_robot_schema:
            return

        if data_dict.get('embodiment_id', None) is None:
            data_dict['embodiment_id'] = self.default_embodiment_id
        if data_dict.get('observation.state', None) is None:
            data_dict['observation.state'] = np.zeros((self.default_action_dim,), dtype=np.float32)
        if data_dict.get('action', None) is None:
            data_dict['action'] = np.zeros((self.default_action_horizon, self.default_action_dim), dtype=np.float32)
        if data_dict.get('action_is_pad', None) is None:
            data_dict['action_is_pad'] = np.ones((self.default_action_horizon,), dtype=np.bool_)

    def _get_data(self, index: int) -> dict[str, Any]:
        if self.samples is None or self.root_infos is None:
            self.open()
        sample = self.samples[index]
        root_info = self.root_infos[sample.root_index]
        sample_id = f'{sample.root_index}:{sample.episode_index}:{sample.frame_index}:' f'{sample.camera}:{sample.qa_index}'

        data_dict: dict[str, Any] = {
            'data_index': index,
            'sample_id': sample_id,
            'dataset_type': self.source_name,
            'source_root': str(root_info.root),
            'root_index': sample.root_index,
            'episode_index': sample.episode_index,
            'frame_index': sample.frame_index,
            'qa_index': sample.qa_index,
            'camera': sample.camera,
            'video_key': sample.video_key,
            'question': sample.question,
            'answer': sample.answer,
            'answers': [sample.answer],
            'conversations': [
                {'from': 'human', 'value': f'{self.image_tag}{sample.question}'.rstrip()},
                {'from': 'gpt', 'value': sample.answer},
            ],
            'task': f'Question: {sample.question}\nAnswer:',
            'vqa_prompt': f'Question: {sample.question}\nAnswer:',
            'vqa_text': f'Question: {sample.question}\nAnswer: {sample.answer}',
            'vqa_language_only': False,
        }

        if self.load_image or self.align_robot_schema:
            image_tensor = self._decode_image(root_info, sample)
            data_dict[sample.video_key] = image_tensor
            adapted_image_tensor = self._adapt_robot_image(image_tensor)
            if self.align_robot_schema:
                data_dict[self.robot_image_key] = adapted_image_tensor
                if self.observation_memory_size > 1:
                    image_is_pad = torch.ones((self.observation_memory_size,), dtype=torch.bool)
                    image_is_pad[-1] = False
                    data_dict[f'{self.robot_image_key}_is_pad'] = image_is_pad
            data_dict['image'] = adapted_image_tensor

        self._apply_robot_schema_defaults(data_dict)
        return data_dict
