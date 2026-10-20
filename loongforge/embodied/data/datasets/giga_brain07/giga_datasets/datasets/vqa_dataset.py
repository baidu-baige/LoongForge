import copy
import hashlib
import json
import logging
import math
import os
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import utils
from .base_dataset import BaseDataset
from .dataset import register_dataset


class _SkipVQAImageError(RuntimeError):
    pass


@register_dataset
class VQADataset(BaseDataset):
    """VQA dataset backed by annotation directories or single
    ``.json``/``.jsonl`` files."""

    def __init__(
        self,
        data_path: str,
        data_size: int | None = None,
        annotation_files: list[str] | None = None,
        split: str | list[str] | None = None,
        with_conversations: bool = True,
        image_tag: str = '<image>\n',
        source_name: str = 'vqa',
        load_image: bool = False,
        align_robot_schema: bool = False,
        robot_image_key: str = 'observation.images.cam_high',
        observation_memory_size: int = 1,
        default_embodiment_id: int = 0,
        default_action_dim: int = 32,
        default_action_horizon: int = 50,
        qa_prompt_template: str = 'Question: {question}\nAnswer:',
        qa_full_template: str = 'Question: {question}\nAnswer: {answer}',
        language_only: bool = True,
        lazy_jsonl: bool = False,
        jsonl_index_cache_dir: str | None = None,
        multi_image_mode: str = 'error',
        skip_oversized_images: bool = True,
        max_image_pixels: int | None = None,
        max_image_load_retries: int = 32,
        image_path_prefix_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super(VQADataset, self).__init__(data_path=data_path, **kwargs)
        self.data_size = data_size
        self.annotation_files = annotation_files
        self.split = split
        self.with_conversations = with_conversations
        self.image_tag = image_tag
        self.source_name = source_name
        self.load_image = load_image
        self.align_robot_schema = align_robot_schema
        self.robot_image_key = robot_image_key
        self.observation_memory_size = int(observation_memory_size)
        if self.observation_memory_size < 1:
            raise ValueError(f'observation_memory_size must be positive, got {self.observation_memory_size}')
        self.default_embodiment_id = int(default_embodiment_id)
        self.default_action_dim = int(default_action_dim)
        self.default_action_horizon = int(default_action_horizon)
        self.qa_prompt_template = qa_prompt_template
        self.qa_full_template = qa_full_template
        self.language_only = language_only
        self.lazy_jsonl = bool(lazy_jsonl)
        self.jsonl_index_cache_dir = jsonl_index_cache_dir
        self.multi_image_mode = str(multi_image_mode).lower()
        if self.multi_image_mode not in {'error', 'first', 'grid'}:
            raise ValueError(f'multi_image_mode must be one of error/first/grid, got {multi_image_mode!r}')
        if max_image_pixels is None:
            max_image_pixels_env = os.environ.get('GIGA_DATASETS_VQA_MAX_IMAGE_PIXELS', None)
            if max_image_pixels_env is not None and len(max_image_pixels_env) > 0:
                max_image_pixels = int(max_image_pixels_env)
        self.skip_oversized_images = bool(skip_oversized_images)
        self.max_image_pixels = None if max_image_pixels is None else int(max_image_pixels)
        if self.max_image_pixels is not None and self.max_image_pixels < 1:
            raise ValueError(f'max_image_pixels must be positive, got {self.max_image_pixels}')
        self.max_image_load_retries = int(max_image_load_retries)
        self.image_path_prefix_map = dict(image_path_prefix_map or {})
        if self.max_image_load_retries < 0:
            raise ValueError(f'max_image_load_retries must be non-negative, got {self.max_image_load_retries}')

        self.data_list: list[tuple[dict, str, int]] | None = None
        self._lazy_jsonl_paths: list[str] = []
        self._lazy_jsonl_offsets: list[np.ndarray] = []
        self._lazy_jsonl_qa_indices: list[np.ndarray] = []
        self._lazy_jsonl_cum_sizes: list[int] = []
        self._lazy_jsonl_handles: dict[str, Any] = {}
        self._logged_skipped_image_errors: set[str] = set()

    @classmethod
    def load(cls, data_or_config: str | dict) -> 'VQADataset':
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

        annotation_files = self._resolve_annotation_files()
        if self.data_path is not None and store_rel_path:
            annotation_root = self._annotation_root()
            annotation_files = [os.path.relpath(path, annotation_root) for path in annotation_files]

        config = {
            '_class_name': 'VQADataset',
            '_key_names': [
                'answer',
                'answers',
                'action',
                'action_is_pad',
                'conversations',
                'dataset_type',
                'embodiment_id',
                'image',
                'image_path',
                'images',
                'observation.state',
                'qa_index',
                'question',
                'sample_id',
                'split',
                'task',
                'vqa_language_only',
                'vqa_prompt',
                'vqa_text',
            ],
            'data_size': len(self),
            'annotation_files': annotation_files,
            'split': self.split,
            'with_conversations': self.with_conversations,
            'image_tag': self.image_tag,
            'source_name': self.source_name,
            'load_image': self.load_image,
            'align_robot_schema': self.align_robot_schema,
            'robot_image_key': self.robot_image_key,
            'observation_memory_size': self.observation_memory_size,
            'default_embodiment_id': self.default_embodiment_id,
            'default_action_dim': self.default_action_dim,
            'default_action_horizon': self.default_action_horizon,
            'qa_prompt_template': self.qa_prompt_template,
            'qa_full_template': self.qa_full_template,
            'language_only': self.language_only,
            'lazy_jsonl': self.lazy_jsonl,
            'jsonl_index_cache_dir': self.jsonl_index_cache_dir,
            'multi_image_mode': self.multi_image_mode,
            'skip_oversized_images': self.skip_oversized_images,
            'max_image_pixels': self.max_image_pixels,
            'max_image_load_retries': self.max_image_load_retries,
            'image_path_prefix_map': self.image_path_prefix_map,
        }
        if self.data_path is not None:
            config['data_path'] = get_rel_path(self.data_path, save_dir) if store_rel_path else self.data_path

        utils.save_file(save_config_path, config)

    def _split_set(self) -> set[str] | None:
        if self.split is None:
            return None
        if isinstance(self.split, str):
            return {self.split.lower()}
        return {str(s).lower() for s in self.split}

    @staticmethod
    def _is_annotation_path(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in {'.json', '.jsonl'}

    def _annotation_root(self) -> str:
        data_path = os.path.abspath(self.data_path)
        if self._is_annotation_path(data_path):
            return os.path.dirname(data_path)
        return data_path

    def _resolve_annotation_files(self) -> list[str]:
        data_path = os.path.abspath(self.data_path)
        annotation_root = self._annotation_root()

        if self.annotation_files is not None:
            ann_paths = []
            for file_name in self.annotation_files:
                if os.path.isabs(file_name):
                    ann_paths.append(file_name)
                else:
                    ann_paths.append(os.path.abspath(os.path.join(annotation_root, file_name)))
        elif self._is_annotation_path(data_path):
            if not os.path.isfile(data_path):
                raise FileNotFoundError(f'Annotation file not found: {self.data_path}')
            ann_paths = [data_path]
        elif os.path.isdir(data_path):
            ann_paths = utils.list_dir(data_path, recursive=False, exts={'.json', '.jsonl'})
        else:
            raise FileNotFoundError(f'VQA data_path should be a .json/.jsonl file or a directory: {self.data_path}')

        if len(ann_paths) == 0:
            raise FileNotFoundError(f'No .json/.jsonl annotation file found in {self.data_path}')

        split_set = self._split_set()
        if split_set is not None:
            split_file_paths = []
            for path in ann_paths:
                file_name = os.path.basename(path).lower()
                if any(split in file_name for split in split_set):
                    split_file_paths.append(path)
            # If split cannot be inferred from file names, fallback to all files and rely on per-record split filtering.
            if len(split_file_paths) > 0:
                ann_paths = split_file_paths

        ann_paths = [os.path.abspath(path) for path in ann_paths]
        ann_paths.sort()
        return ann_paths

    def _load_annotation_file(self, file_path: str) -> list[dict]:
        if file_path.endswith('.jsonl'):
            records = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Invalid jsonl line in {file_path}:{i + 1}') from exc
                    if not isinstance(data, dict):
                        raise TypeError(f'Each record in jsonl should be a dict, got {type(data).__name__}')
                    records.append(data)
            return records

        data = utils.load_file(file_path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ('data', 'annotations', 'items', 'samples'):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]

        raise TypeError(f'Unsupported annotation content type: {type(data).__name__}')

    def _jsonl_index_cache_path(self, file_path: str) -> str:
        cache_dir = self.jsonl_index_cache_dir
        if cache_dir is None:
            cache_dir = os.environ.get('GIGA_DATASETS_JSONL_INDEX_CACHE', '/tmp/giga_datasets_jsonl_offsets')
        os.makedirs(cache_dir, exist_ok=True)

        stat = os.stat(file_path)
        digest = hashlib.sha1(os.path.abspath(file_path).encode('utf-8')).hexdigest()[:16]
        base_name = os.path.basename(file_path)
        return os.path.join(cache_dir, f'{base_name}.{digest}.{stat.st_size}.{int(stat.st_mtime_ns)}.offsets.npy')

    def _load_or_build_jsonl_offsets(self, file_path: str) -> np.ndarray:
        cache_path = self._jsonl_index_cache_path(file_path)
        if os.path.exists(cache_path):
            return np.load(cache_path, mmap_mode='r')

        offsets = array('Q')
        with open(file_path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        offsets_array = np.asarray(offsets, dtype=np.uint64)
        tmp_path = f'{cache_path}.{os.getpid()}.tmp'
        np.save(tmp_path, offsets_array)
        saved_tmp_path = tmp_path if tmp_path.endswith('.npy') else f'{tmp_path}.npy'
        os.replace(saved_tmp_path, cache_path)
        return np.load(cache_path, mmap_mode='r')

    def _jsonl_sample_index_cache_paths(self, file_path: str) -> tuple[str, str]:
        offsets_path = self._jsonl_index_cache_path(file_path)
        if offsets_path.endswith('.offsets.npy'):
            base_path = offsets_path[: -len('.offsets.npy')]
        else:
            base_path = offsets_path[: -len('.npy')] if offsets_path.endswith('.npy') else offsets_path
        return f'{base_path}.sample_offsets.npy', f'{base_path}.qa_indices.npy'

    def _load_or_build_jsonl_sample_index(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        offsets_cache_path, qa_indices_cache_path = self._jsonl_sample_index_cache_paths(file_path)
        if os.path.exists(offsets_cache_path) and os.path.exists(qa_indices_cache_path):
            return np.load(offsets_cache_path, mmap_mode='r'), np.load(qa_indices_cache_path, mmap_mode='r')

        offsets = array('Q')
        qa_indices = array('I')
        with open(file_path, 'rb') as f:
            line_index = 0
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line_index += 1
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid jsonl line in {file_path}:{line_index}') from exc
                if not isinstance(data, dict):
                    raise TypeError(f'Each record in jsonl should be a dict, got {type(data).__name__}')

                qa_count = self._count_record_qa_samples(data)
                for qa_index in range(qa_count):
                    offsets.append(offset)
                    qa_indices.append(qa_index)

        offsets_array = np.asarray(offsets, dtype=np.uint64)
        qa_indices_array = np.asarray(qa_indices, dtype=np.uint32)

        offsets_tmp_path = f'{offsets_cache_path}.{os.getpid()}.tmp'
        qa_indices_tmp_path = f'{qa_indices_cache_path}.{os.getpid()}.tmp'
        np.save(offsets_tmp_path, offsets_array)
        np.save(qa_indices_tmp_path, qa_indices_array)
        saved_offsets_tmp_path = offsets_tmp_path if offsets_tmp_path.endswith('.npy') else f'{offsets_tmp_path}.npy'
        saved_qa_indices_tmp_path = qa_indices_tmp_path if qa_indices_tmp_path.endswith('.npy') else f'{qa_indices_tmp_path}.npy'
        os.replace(saved_offsets_tmp_path, offsets_cache_path)
        os.replace(saved_qa_indices_tmp_path, qa_indices_cache_path)
        return np.load(offsets_cache_path, mmap_mode='r'), np.load(qa_indices_cache_path, mmap_mode='r')

    def _read_lazy_jsonl_record(self, file_path: str, offset: int) -> dict:
        handle = self._lazy_jsonl_handles.get(file_path)
        if handle is None:
            handle = open(file_path, 'rb')
            self._lazy_jsonl_handles[file_path] = handle

        handle.seek(offset)
        line = handle.readline()
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid jsonl line at byte offset {offset} in {file_path}') from exc
        if not isinstance(data, dict):
            raise TypeError(f'Each record in jsonl should be a dict, got {type(data).__name__}')
        return data

    def _infer_split_from_path(self, file_path: str) -> str | None:
        file_name = os.path.basename(file_path).lower()
        if 'train' in file_name:
            return 'train'
        if 'val' in file_name or 'valid' in file_name:
            return 'val'
        if 'test' in file_name:
            return 'test'
        return None

    def _resolve_image_path(self, image_path: str | list[str] | None) -> str | list[str] | None:
        if image_path is None:
            return None
        if isinstance(image_path, list):
            if self.multi_image_mode == 'error':
                raise TypeError('VQA records with multiple images require ' 'multi_image_mode="first" or multi_image_mode="grid"')
            return [self._resolve_single_image_path(path) for path in image_path]
        return self._resolve_single_image_path(image_path)

    def _resolve_single_image_path(self, image_path: str) -> str:
        if not isinstance(image_path, str):
            raise TypeError(f'VQA image path must be a string, got {type(image_path).__name__}')
        if os.path.isabs(image_path):
            resolved_path = image_path
        else:
            resolved_path = os.path.abspath(os.path.join(self._annotation_root(), image_path))

        for source_prefix, target_prefix in self.image_path_prefix_map.items():
            source_path = Path(os.path.abspath(os.path.expanduser(source_prefix)))
            try:
                relative_path = Path(resolved_path).relative_to(source_path)
            except ValueError:
                continue
            target_path = Path(os.path.abspath(os.path.expanduser(target_prefix)))
            resolved_path = str(target_path / relative_path)
            break

        if not os.path.exists(resolved_path):
            resolved_path = self._resolve_nested_duplicate_dir_path(resolved_path)

        return resolved_path

    @staticmethod
    def _compose_image_grid(images: list[Any]):
        from PIL import Image

        if not images:
            raise ValueError('Cannot compose an empty image grid')
        if len(images) == 1:
            return images[0]

        columns = int(math.ceil(math.sqrt(len(images))))
        rows = int(math.ceil(len(images) / columns))
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        canvas = Image.new('RGB', (columns * cell_width, rows * cell_height))
        for image_index, image in enumerate(images):
            row, column = divmod(image_index, columns)
            x = column * cell_width + (cell_width - image.width) // 2
            y = row * cell_height + (cell_height - image.height) // 2
            canvas.paste(image, (x, y))
        return canvas

    @staticmethod
    def _resolve_nested_duplicate_dir_path(path: str, max_depth: int = 4) -> str:
        """Recover mirrored datasets where the last directory name is
        duplicated.

        Example:
        ``.../flickr30k-images/100.jpg`` ->
        ``.../flickr30k-images/flickr30k-images/flickr30k-images/100.jpg``
        """

        parent_dir = os.path.dirname(path)
        file_name = os.path.basename(path)
        repeated_dir_name = os.path.basename(parent_dir)
        if len(repeated_dir_name) == 0:
            return path

        probe_dir = parent_dir
        for _ in range(max_depth):
            probe_dir = os.path.join(probe_dir, repeated_dir_name)
            if not os.path.isdir(probe_dir):
                break

            candidate_path = os.path.join(probe_dir, file_name)
            if os.path.exists(candidate_path):
                return candidate_path

        return path

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

    def _extract_qa_pairs_from_conversations(self, conversations: Any) -> list[tuple[str, str]]:
        if not isinstance(conversations, list):
            return []

        qa_pairs: list[tuple[str, str]] = []
        pending_question: str | None = None
        for message in conversations:
            if not isinstance(message, dict):
                continue

            role = str(message.get('from', message.get('role', ''))).strip().lower()
            value = str(message.get('value', message.get('content', '')))
            if len(value) == 0:
                continue

            if role in {'human', 'user'}:
                if pending_question is None:
                    pending_question = self._strip_leading_image_tag(value)
                continue

            if role in {'gpt', 'assistant'} and pending_question is not None:
                qa_pairs.append((pending_question, value.strip()))
                pending_question = None

        return qa_pairs

    def _extract_qa_from_conversations(self, conversations: Any) -> tuple[str, str]:
        qa_pairs = self._extract_qa_pairs_from_conversations(conversations)
        if len(qa_pairs) == 0:
            return '', ''
        return qa_pairs[0]

    def _count_record_qa_samples(self, record: dict) -> int:
        qa_count = len(self._extract_qa_pairs_from_conversations(record.get('conversations', None)))
        return max(1, qa_count)

    def _normalize_record(self, record: dict, ann_path: str, data_index: int, qa_index: int = 0) -> dict:
        if not isinstance(record, dict):
            raise TypeError(f'Each annotation record should be a dict, got {type(record).__name__}')

        data_dict = copy.deepcopy(record)

        image_path = data_dict.get('image_path', data_dict.get('image', None))
        image_path = self._resolve_image_path(image_path)

        qa_pairs = self._extract_qa_pairs_from_conversations(data_dict.get('conversations', None))
        qa_count = len(qa_pairs)
        if qa_count > 0:
            if qa_index < 0 or qa_index >= qa_count:
                raise IndexError(f'qa_index={qa_index} out of range for {ann_path}, data_index={data_index}')
            parsed_question, parsed_answer = qa_pairs[qa_index]
        else:
            parsed_question, parsed_answer = '', ''

        if qa_count > 1:
            question = parsed_question
            answer = parsed_answer
            answers = [answer]
        else:
            question = str(data_dict.get('question', parsed_question))
            answer = data_dict.get('answer', parsed_answer)
            answers = data_dict.get('answers', None)
        if answers is None:
            answers = [answer] if len(str(answer)) > 0 else []

        explicit_sample_id = data_dict.get('sample_id', data_dict.get('id', None))
        if explicit_sample_id is None:
            sample_id = f'{os.path.basename(ann_path)}:{data_index}'
        elif qa_count > 1:
            sample_id = f'{explicit_sample_id}:{qa_index}'
        else:
            sample_id = explicit_sample_id

        data_dict['data_index'] = data_index
        data_dict['sample_id'] = sample_id
        data_dict['qa_index'] = qa_index
        data_dict['question'] = question
        data_dict['answer'] = answer
        data_dict['answers'] = answers
        data_dict['image_path'] = image_path
        data_dict['image'] = image_path
        data_dict['dataset_type'] = data_dict.get('dataset_type', self.source_name)
        data_dict['vqa_prompt'] = data_dict.get('vqa_prompt', self.qa_prompt_template.format(question=question, answer=answer))
        data_dict['vqa_text'] = data_dict.get('vqa_text', self.qa_full_template.format(question=question, answer=answer))
        data_dict['task'] = data_dict.get('task', data_dict['vqa_prompt'])
        data_dict['vqa_language_only'] = bool(data_dict.get('vqa_language_only', self.language_only))

        if 'split' not in data_dict or data_dict['split'] is None:
            data_dict['split'] = self._infer_split_from_path(ann_path)

        if self.with_conversations:
            if qa_count > 1 or 'conversations' not in data_dict:
                prompt = f'{self.image_tag}{question}'.rstrip()
                data_dict['conversations'] = [
                    {'from': 'human', 'value': prompt},
                    {'from': 'gpt', 'value': str(answer)},
                ]

            if image_path is not None and 'images' not in data_dict:
                data_dict['images'] = image_path

            # Keep empty data_path so external loaders using os.path.join can
            # still work with absolute image paths.
            data_dict.setdefault('data_path', '')
        else:
            data_dict.pop('conversations', None)

        return data_dict

    def _apply_robot_schema_defaults(self, data_dict: dict) -> None:
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

    def _adapt_robot_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if not self.align_robot_schema or self.observation_memory_size == 1:
            return image_tensor
        return image_tensor.unsqueeze(0).repeat(self.observation_memory_size, 1, 1, 1).contiguous()

    def open(self) -> None:
        if self.data_list is None and len(self._lazy_jsonl_offsets) == 0:
            split_set = self._split_set()
            self.data_list = []
            for ann_path in self._resolve_annotation_files():
                inferred_split = self._infer_split_from_path(ann_path)
                if self.lazy_jsonl and ann_path.endswith('.jsonl'):
                    if split_set is not None and inferred_split is not None and inferred_split.lower() not in split_set:
                        continue
                    offsets, qa_indices = self._load_or_build_jsonl_sample_index(ann_path)
                    self._lazy_jsonl_paths.append(ann_path)
                    self._lazy_jsonl_offsets.append(offsets)
                    self._lazy_jsonl_qa_indices.append(qa_indices)
                    total_size = (self._lazy_jsonl_cum_sizes[-1] if self._lazy_jsonl_cum_sizes else 0) + len(offsets)
                    self._lazy_jsonl_cum_sizes.append(total_size)
                    continue

                records = self._load_annotation_file(ann_path)
                for record in records:
                    if split_set is not None:
                        record_split = record.get('split', inferred_split)
                        if record_split is not None and str(record_split).lower() not in split_set:
                            continue
                    qa_count = self._count_record_qa_samples(record)
                    for qa_index in range(qa_count):
                        self.data_list.append((record, ann_path, qa_index))

            if self.data_size is not None:
                assert self.data_size == len(self.data_list) + (self._lazy_jsonl_cum_sizes[-1] if self._lazy_jsonl_cum_sizes else 0)
            else:
                self.data_size = len(self.data_list) + (self._lazy_jsonl_cum_sizes[-1] if self._lazy_jsonl_cum_sizes else 0)

    def close(self) -> None:
        if self.data_list is not None:
            self.data_list.clear()
            self.data_list = None
        for handle in self._lazy_jsonl_handles.values():
            handle.close()
        self._lazy_jsonl_handles.clear()
        self._lazy_jsonl_paths.clear()
        self._lazy_jsonl_offsets.clear()
        self._lazy_jsonl_qa_indices.clear()
        self._lazy_jsonl_cum_sizes.clear()
        self._logged_skipped_image_errors.clear()
        super(VQADataset, self).close()

    def filter(self, mode: str, dataset_index: int = 0, **kwargs: Any) -> None:
        self.open()
        if self.lazy_jsonl and len(self._lazy_jsonl_offsets) > 0:
            raise NotImplementedError('VQADataset.filter is not supported with lazy_jsonl=True')
        if mode == 'index':
            start = kwargs.get('start', 0)
            end = kwargs.get('end', None)
            step = kwargs.get('step', 1)
            self.data_list = self.data_list[start:end:step]
        elif mode == 'func':
            func = kwargs.pop('func')
            if isinstance(func, str):
                func = utils.import_function(func)
            self.data_list = func(self.data_list, dataset_index=dataset_index, **kwargs)
        else:
            assert False

        logging.info(f'filter dataset {dataset_index} from {self.data_size} to {len(self.data_list)}')
        self.data_size = len(self.data_list)

    def __len__(self) -> int:
        if self.data_size is None:
            self.open()
        return self.data_size

    def _read_record(self, index: int) -> tuple[dict, str, int]:
        if len(self._lazy_jsonl_offsets) > 0:
            lazy_size = self._lazy_jsonl_cum_sizes[-1]
            if index < lazy_size:
                segment_index = int(np.searchsorted(self._lazy_jsonl_cum_sizes, index, side='right'))
                prev_size = 0 if segment_index == 0 else self._lazy_jsonl_cum_sizes[segment_index - 1]
                local_index = index - prev_size
                ann_path = self._lazy_jsonl_paths[segment_index]
                offset = int(self._lazy_jsonl_offsets[segment_index][local_index])
                qa_index = int(self._lazy_jsonl_qa_indices[segment_index][local_index])
                raw_record = self._read_lazy_jsonl_record(ann_path, offset)
            else:
                raw_record, ann_path, qa_index = self.data_list[index - lazy_size]
        else:
            raw_record, ann_path, qa_index = self.data_list[index]
        return raw_record, ann_path, qa_index

    def _next_retry_index(self, index: int, retry: int) -> int:
        data_size = len(self)
        if data_size <= 1:
            return index
        if retry == 1:
            return (index + 1) % data_size

        stride = max(1, data_size // max(2, self.max_image_load_retries + 1))
        while math.gcd(stride, data_size) != 1:
            stride += 1
        return (index + retry * stride) % data_size

    def _log_skipped_image(self, message: str) -> None:
        if message in self._logged_skipped_image_errors:
            return
        self._logged_skipped_image_errors.add(message)
        print(message, file=sys.stderr, flush=True)
        logging.warning(message)

    def _get_data(self, index: int) -> dict:
        last_exc = None
        for retry in range(self.max_image_load_retries + 1):
            retry_index = index if retry == 0 else self._next_retry_index(index, retry)
            try:
                return self._get_data_once(retry_index)
            except _SkipVQAImageError as exc:
                last_exc = exc
                if not self.skip_oversized_images or retry >= self.max_image_load_retries:
                    raise

                message = f'{exc}; resampling VQA sample from data_index={retry_index}'
                self._log_skipped_image(message)

        raise RuntimeError(f'Failed to load VQA sample after {self.max_image_load_retries} retries') from last_exc

    def _get_data_once(self, index: int) -> dict:
        raw_record, ann_path, qa_index = self._read_record(index)
        data_dict = self._normalize_record(raw_record, ann_path, index, qa_index=qa_index)

        self._apply_robot_schema_defaults(data_dict)

        if self.load_image or self.align_robot_schema:
            image_path = data_dict.get('image_path', None)
            if image_path is not None:
                from PIL import Image

                try:
                    image_paths = image_path if isinstance(image_path, list) else [image_path]
                    if self.multi_image_mode == 'first':
                        image_paths = image_paths[:1]
                    if not image_paths:
                        raise _SkipVQAImageError(
                            'Skipping VQA sample with no images: '
                            f'annotation_path={ann_path}, '
                            f'data_index={index}, '
                            f'sample_id={data_dict.get("sample_id", None)}'
                        )

                    image_rgbs = []
                    for current_image_path in image_paths:
                        if not os.path.isfile(current_image_path) or os.path.getsize(current_image_path) == 0:
                            raise _SkipVQAImageError(
                                'Skipping missing or empty VQA image: '
                                f'image_path={current_image_path}, '
                                f'annotation_path={ann_path}, '
                                f'data_index={index}, '
                                f'sample_id={data_dict.get("sample_id", None)}'
                            )
                        with Image.open(current_image_path) as image:
                            width, height = image.size
                            image_pixels = width * height
                            if self.max_image_pixels is not None and image_pixels > self.max_image_pixels:
                                raise _SkipVQAImageError(
                                    'Skipping oversized VQA image: '
                                    f'image_path={current_image_path}, '
                                    f'annotation_path={ann_path}, '
                                    f'data_index={index}, '
                                    f'sample_id={data_dict.get("sample_id", None)}, '
                                    f'image_size={width}x{height}, '
                                    f'image_pixels={image_pixels}, '
                                    f'max_image_pixels={self.max_image_pixels}'
                                )
                            image_rgbs.append(image.convert('RGB'))

                    image_rgb = self._compose_image_grid(image_rgbs)
                    if self.align_robot_schema:
                        image_np = np.asarray(image_rgb, dtype=np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                        image_tensor = self._adapt_robot_image(image_tensor)
                        data_dict[self.robot_image_key] = image_tensor
                        if self.observation_memory_size > 1:
                            image_is_pad = torch.ones(
                                (self.observation_memory_size,),
                                dtype=torch.bool,
                            )
                            image_is_pad[-1] = False
                            data_dict[f'{self.robot_image_key}_is_pad'] = image_is_pad
                        data_dict['image'] = image_tensor
                    else:
                        data_dict['image'] = image_rgb.copy()
                except _SkipVQAImageError:
                    raise
                except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                    error_msg = (
                        'Skipping PIL decompression-bomb VQA image: '
                        f'image_path={image_path}, '
                        f'annotation_path={ann_path}, '
                        f'data_index={index}, '
                        f'sample_id={data_dict.get("sample_id", None)}'
                    )
                    raise _SkipVQAImageError(error_msg) from exc
                except Image.UnidentifiedImageError as exc:
                    error_msg = (
                        'Skipping unreadable VQA image: '
                        f'image_path={image_path}, '
                        f'annotation_path={ann_path}, '
                        f'data_index={index}, '
                        f'sample_id={data_dict.get("sample_id", None)}'
                    )
                    raise _SkipVQAImageError(error_msg) from exc
                except Exception as exc:
                    error_msg = (
                        'Failed to load VQA image: '
                        f'image_path={image_path}, '
                        f'annotation_path={ann_path}, '
                        f'data_index={index}, '
                        f'sample_id={data_dict.get("sample_id", None)}'
                    )
                    print(error_msg, file=sys.stderr, flush=True)
                    logging.exception(error_msg)
                    raise RuntimeError(error_msg) from exc

        return data_dict
