import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

from giga_models.utils.action_horizon import flow_action_horizon_indices

PROPRI_TOKEN = '<|propri|>'


def register_propri_token(tokenizer) -> int:
    """Register and validate the continuous-state anchor token."""
    tokenizer.add_tokens([PROPRI_TOKEN])
    token_id = int(tokenizer.convert_tokens_to_ids(PROPRI_TOKEN))
    encoded = tokenizer.encode(PROPRI_TOKEN, add_special_tokens=False)
    if encoded != [token_id]:
        raise ValueError(f'{PROPRI_TOKEN} must encode to one token, got ids={encoded}')
    return token_id


def resolve_propri_token_id(tokenizer_model_path: str) -> tuple[int, int, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    token_id = register_propri_token(tokenizer)
    return token_id, len(tokenizer), int(tokenizer.vocab_size)


def load_inference_config(ckpt_dir: str) -> dict:
    """Read inference_config.json from a checkpoint dir.

    Sidecar to diffusers config.json. Written by GigaBrain07Trainer.save_model_hook on every new checkpoint. Missing means the ckpt predates the hook
    — manually create the sidecar (copy the four transform sub-dicts from the train config: image_cfg, prompt_cfg, norm_cfg, delta_action_cfg) before
    deploying.
    """
    path = os.path.join(ckpt_dir, 'inference_config.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. This ckpt predates the inference_config.json sidecar — '
            f'either re-train with the updated trainer (auto-dumps it) or hand-create '
            f'inference_config.json beside config.json containing the train-time '
            f'image_cfg / prompt_cfg / norm_cfg / delta_action_cfg sub-dicts.'
        )
    with open(path) as f:
        cfg = json.load(f)
    print(f'Loaded inference_config.json from: {path}')
    return cfg


def _load_fast_tokenizer(path: str):
    """Load FAST action tokenizer directly, bypassing AutoProcessor.

    The local transformers dev version (5.3.0.dev0) has import chain issues
    that break ``AutoProcessor.from_pretrained`` for the FAST tokenizer
    (mistral_common import error). This helper loads the BPE tokenizer and
    wraps it with a minimal encode/decode interface compatible with the
    ``UniversalActionProcessor`` API.
    """
    from scipy.fft import dct, idct

    with open(f'{path}/processor_config.json') as f:
        cfg = json.load(f)

    bpe_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'{path}/tokenizer.json',
        clean_up_tokenization_spaces=False,
    )

    class FastActionTokenizer:
        """Minimal drop-in replacement for ``UniversalActionProcessor``."""

        def __init__(self, bpe_tok, scale, vocab_size, min_token):
            self.bpe_tokenizer = bpe_tok
            self.scale = scale
            self.vocab_size = vocab_size
            self.min_token = min_token
            self._time_horizon = None
            self._action_dim = None

        def __call__(self, action_chunk):
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()
            action_chunk = np.asarray(action_chunk, dtype=np.float32)
            if action_chunk.ndim == 2:
                action_chunk = action_chunk[None, ...]
            self._time_horizon = action_chunk.shape[-2]
            self._action_dim = action_chunk.shape[-1]
            dct_coeff = dct(action_chunk, axis=1, norm='ortho')
            dct_coeff = np.around(dct_coeff * self.scale)
            tokens = []
            for elem in dct_coeff:
                token_str = ''.join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
                tokens.append(self.bpe_tokenizer(token_str)['input_ids'])
            return tokens

        def decode(self, tokens, *, time_horizon=None, action_dim=None):
            th = time_horizon or self._time_horizon
            ad = action_dim or self._action_dim
            assert th is not None and ad is not None, 'time_horizon and action_dim required'
            decoded_actions = []
            for token in tokens:
                eff_ad = ad
                try:
                    decoded_str = self.bpe_tokenizer.decode(token)
                    coeffs = np.array(list(map(ord, decoded_str))) + self.min_token
                    total = coeffs.shape[0]
                    # FAST stores (time_horizon, action_dim) DCT coefficients flattened, so
                    # the true encode-time action_dim is total//th. Prefer it over the caller's
                    # `ad` when they differ (e.g. caller passed a delta-masked / trimmed dim
                    # instead of the raw data action dim) — reshape to the native dim and let
                    # the caller trim. Avoids silently returning zeros on a dim mismatch.
                    if th > 0 and total % th == 0 and total // th != ad:
                        eff_ad = total // th
                    decoded_dct = coeffs.reshape(th, eff_ad)
                except Exception as e:
                    print(
                        f'WARNING: FAST decode failed: {e}. BPE decoded {len(decoded_str)} chars, '
                        f'not a multiple of time_horizon={th}. Returning zeros.'
                    )
                    eff_ad = ad
                    decoded_dct = np.zeros((th, eff_ad))
                decoded_actions.append(idct(decoded_dct / self.scale, axis=0, norm='ortho'))
            return np.stack(decoded_actions)

    return FastActionTokenizer(
        bpe_tok=bpe_tokenizer,
        scale=cfg['scale'],
        vocab_size=cfg['vocab_size'],
        min_token=cfg['min_token'],
    )


CONTROL_MODE_JOINT = 'joint'
CONTROL_MODE_END_EFFECTOR = 'end effector'
END_EFFECTOR_CONTROL_EMBODIMENT_IDS = frozenset((3, 4, 5))
END_EFFECTOR_TYPE_GRIPPER = 'gripper'
END_EFFECTOR_TYPE_DEX_HAND = 'dex hand'


def split_task_and_subtask(task: Any) -> tuple[str, str | None]:
    task_text = _as_nonempty_text(task)
    if task_text is None:
        raise TypeError(f'Expected non-empty task text, got {type(task).__name__}: {task!r}')

    cleaned = task_text.lower().strip().replace('_', ' ')
    marker = ' subtask:'
    marker_index = cleaned.find(marker)
    if marker_index == -1:
        return cleaned, None

    main_task = cleaned[:marker_index].strip()
    subtask_text = cleaned[marker_index + len(marker) :]
    if not subtask_text:
        return main_task, None

    for line in subtask_text.splitlines():
        sub_task = line.strip()
        if sub_task:
            return main_task, sub_task
    return main_task, None


def _format_prompt_context(data: dict[str, Any] | None) -> str:
    if data is None:
        return ''

    meta = data.get('meta')
    repo_id = getattr(meta, 'repo_id', None)
    root = getattr(meta, 'root', None)
    episode_index = data.get('episode_index')
    frame_index = data.get('frame_index')
    task_index = data.get('task_index')
    tasks = getattr(meta, 'tasks', None)
    tasks_type = type(tasks).__name__ if tasks is not None else None
    tasks_columns = getattr(tasks, 'columns', None)
    if tasks_columns is not None:
        tasks_columns = list(tasks_columns)
    return (
        f' repo_id={repo_id!r}, root={root!r}, episode_index={episode_index!r}, '
        f'frame_index={frame_index!r}, task_index={task_index!r}, '
        f'meta_tasks_type={tasks_type!r}, meta_tasks_columns={tasks_columns!r}'
    )


def _preview_text(text: str, max_chars: int = 240) -> str:
    text = ' '.join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + '...'


def _as_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value

    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _as_nonempty_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _task_text_from_row(row: Any) -> str | None:
    if (task_text := _as_nonempty_text(row)) is not None:
        return task_text

    get = getattr(row, 'get', None)
    if callable(get):
        for key in ('task', 'name'):
            try:
                if (task_text := _as_nonempty_text(get(key))) is not None:
                    return task_text
            except (KeyError, TypeError):
                pass

    return _as_nonempty_text(getattr(row, 'name', None))


def _lookup_task_text(tasks: Any, task_index: Any) -> str | None:
    task_index = _as_scalar(task_index)

    if isinstance(tasks, dict):
        for key in (task_index, str(task_index)):
            if key in tasks and (task_text := _task_text_from_row(tasks[key])) is not None:
                return task_text
        return None

    if isinstance(tasks, (list, tuple)):
        try:
            return _task_text_from_row(tasks[int(task_index)])
        except (IndexError, TypeError, ValueError):
            return None

    columns = getattr(tasks, 'columns', ())
    if 'task_index' in columns:
        try:
            matches = tasks[tasks['task_index'] == task_index]
            if len(matches) > 0 and (task_text := _task_text_from_row(matches.iloc[0])) is not None:
                return task_text
        except (KeyError, TypeError, ValueError):
            pass

    if 'task' in columns:
        try:
            if (task_text := _as_nonempty_text(tasks.iloc[int(task_index)]['task'])) is not None:
                return task_text
        except (IndexError, KeyError, TypeError, ValueError):
            pass

    loc = getattr(tasks, 'loc', None)
    if loc is not None:
        for key in (task_index, str(task_index)):
            try:
                if (task_text := _task_text_from_row(loc[key])) is not None:
                    return task_text
            except (KeyError, TypeError, ValueError):
                pass

    iloc = getattr(tasks, 'iloc', None)
    if iloc is not None:
        try:
            return _task_text_from_row(iloc[int(task_index)])
        except (IndexError, TypeError, ValueError):
            return None

    return None


def _task_text_from_path(root: Any) -> str | None:
    if root is None:
        return None

    path_parts = [part for part in os.fspath(root).split(os.sep) if part]
    for part in reversed(path_parts):
        normalized = part.strip().lower()
        if not normalized or normalized.endswith('_merged') or normalized.isdigit():
            continue
        if any(char.isalpha() for char in normalized):
            return normalized.replace('_', ' ').replace('-', ' ')
    return None


def resolve_task_text(task: Any, sample_context: dict[str, Any] | None = None) -> str:
    if (task_text := _as_nonempty_text(task)) is not None:
        return task_text

    task_index = task
    meta = None
    if sample_context is not None:
        task_index = sample_context.get('task_index', task)
        meta = sample_context.get('meta')

    tasks = getattr(meta, 'tasks', None)
    if tasks is not None and (task_text := _lookup_task_text(tasks, task_index)) is not None:
        return task_text

    info = getattr(meta, 'info', None)
    if isinstance(info, dict):
        for key in ('tasks', 'task_map', 'task_mapping'):
            tasks = info.get(key)
            if tasks is not None and (task_text := _lookup_task_text(tasks, task_index)) is not None:
                return task_text
        for key in ('task', 'task_name', 'description'):
            if (task_text := _as_nonempty_text(info.get(key))) is not None:
                return task_text

    root = getattr(meta, 'root', None)
    if (task_text := _task_text_from_path(root)) is not None:
        return task_text

    raise TypeError(f'Expected non-empty task text, got {type(task).__name__}: {task!r}{_format_prompt_context(sample_context)}')


def normalize_control_mode(control_mode: str | None) -> str | None:
    if control_mode is None:
        return None

    normalized = control_mode.strip().lower().replace('_', ' ').replace('-', ' ')
    normalized = ' '.join(normalized.split())
    if normalized == CONTROL_MODE_JOINT:
        return CONTROL_MODE_JOINT
    if normalized in {CONTROL_MODE_END_EFFECTOR, 'ee', 'endeffector'}:
        return CONTROL_MODE_END_EFFECTOR

    raise ValueError(f'Unsupported control mode: {control_mode}')


def normalize_end_effector_type(end_effector_type: str | None) -> str | None:
    if end_effector_type is None:
        return None

    normalized = end_effector_type.strip().lower().replace('_', ' ').replace('-', ' ')
    normalized = ' '.join(normalized.split())
    if normalized == END_EFFECTOR_TYPE_GRIPPER:
        return END_EFFECTOR_TYPE_GRIPPER
    if normalized in {END_EFFECTOR_TYPE_DEX_HAND, 'dexhand', 'dexterous hand', 'dexterous hands'}:
        return END_EFFECTOR_TYPE_DEX_HAND

    raise ValueError(f'Unsupported end effector type: {end_effector_type}')


def infer_control_mode_from_data(data: dict[str, Any]) -> str | None:
    explicit_control_mode = normalize_control_mode(data.get('control_mode'))
    if explicit_control_mode is not None:
        return explicit_control_mode

    embodiment_id = data.get('embodiment_id')
    if embodiment_id is not None:
        try:
            return CONTROL_MODE_END_EFFECTOR if int(embodiment_id) in END_EFFECTOR_CONTROL_EMBODIMENT_IDS else CONTROL_MODE_JOINT
        except (TypeError, ValueError):
            pass

    if (
        'action_tcp_endpose_quat' in data
        or 'observation.state_tcp_endpose_quat' in data
        or 'action_tcp_endpose_quat_is_pad' in data
        or 'action_hands_episode_first' in data
        or 'observation.state_action_hands_episode_first' in data
        or 'action_hands_episode_first_is_pad' in data
        or 'observation.state_action' in data
    ):
        return CONTROL_MODE_END_EFFECTOR
    if 'action' in data and 'action_tcp_endpose_quat' not in data:
        return CONTROL_MODE_JOINT

    return None


def canonicalize_state_action_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize dataset keys to canonical state/action names.

    Some datasets use TCP quaternion key variants (for example
    ``action_tcp_endpose_quat`` or ``action_hands_episode_first``) while the
    downstream pipeline expects ``action`` / ``observation.state`` /
    ``action_is_pad``. This helper copies available variant keys into the
    canonical names when needed.
    """
    canonical_aliases = (
        ('action', 'action_tcp_endpose_quat'),
        ('action', 'action_hands_episode_first'),
        ('action', 'action_cam_episode_first'),
        ('action_is_pad', 'action_tcp_endpose_quat_is_pad'),
        ('action_is_pad', 'action_hands_episode_first_is_pad'),
        ('action_is_pad', 'action_cam_episode_first_is_pad'),
        ('observation.state', 'observation.state_tcp_endpose_quat'),
        ('observation.state', 'observation.state_action_hands_episode_first'),
        ('observation.state', 'observation.state_action_cam_episode_first'),
        ('observation.state', 'observation.state_action'),
    )
    for canonical_key, alias_key in canonical_aliases:
        if canonical_key not in data and alias_key in data:
            data[canonical_key] = data[alias_key]
    return data


def _longest_false_run(mask: torch.Tensor) -> int:
    longest_run = 0
    current_run = 0
    for item in mask.reshape(-1):
        if bool(item):
            current_run = 0
        else:
            current_run += 1
            longest_run = max(longest_run, current_run)
    return longest_run


def infer_repeated_front_pair_side_mask(delta_mask: torch.Tensor | list[bool]) -> torch.Tensor | None:
    mask = torch.as_tensor(delta_mask, dtype=torch.bool).reshape(-1)
    mask_dim = int(mask.shape[-1])
    for single_side_dim in range(mask_dim // 2, 0, -1):
        side_mask = mask[:single_side_dim]
        next_side_mask = mask[single_side_dim : 2 * single_side_dim]
        if bool(side_mask.any()) and bool((~side_mask).any()) and torch.equal(side_mask, next_side_mask):
            return side_mask
    return None


def infer_repeated_front_pair_dim(delta_mask: torch.Tensor | list[bool]) -> int | None:
    """Return the width of the repeated two-side prefix in a delta mask."""
    side_mask = infer_repeated_front_pair_side_mask(delta_mask)
    if side_mask is None:
        return None
    return int(side_mask.numel()) * 2


@dataclass(frozen=True)
class ActionStateDimLayout:
    """Resolved state-conditioning and action-supervision widths for one
    sample."""

    full_action_dim: int
    full_state_dim: int
    state_aligned_action_dim: int
    front_pair_dim: int | None
    action_supervised_dim: int
    state_input_dim: int
    is_robot_moving: bool
    is_body_moving: bool


def resolve_action_state_dim_layout(
    delta_mask: torch.Tensor | list[bool],
    *,
    raw_action_dim: int,
    raw_state_dim: int,
    max_action_dim: int,
    is_robot_moving: bool = False,
    is_body_moving: bool = False,
) -> ActionStateDimLayout:
    """Resolve valid prefixes without changing delta/absolute action semantics.

    Mobile-base motion enables the complete configured action schema. When the base is stationary, body motion keeps every state-aligned action. State
    conditioning is independent of both motion flags and always retains the complete observed state schema. A repeated two-side prefix (2N) only
    limits action supervision while the body and mobile base are stationary.
    """
    mask = torch.as_tensor(delta_mask, dtype=torch.bool).reshape(-1)
    max_dim = int(max_action_dim)
    if max_dim <= 0:
        raise ValueError(f'max_action_dim must be positive, got {max_action_dim}')

    action_dim = max(0, int(raw_action_dim))
    state_dim = max(0, int(raw_state_dim))
    full_action_dim = min(int(mask.numel()), action_dim, max_dim)
    full_state_dim = min(state_dim, max_dim)
    state_aligned_action_dim = min(full_action_dim, full_state_dim)

    front_pair_dim = None
    if not bool(is_robot_moving):
        front_pair_dim = infer_repeated_front_pair_dim(mask[:state_aligned_action_dim])

    if bool(is_robot_moving):
        action_supervised_dim = full_action_dim
    elif bool(is_body_moving) or front_pair_dim is None:
        action_supervised_dim = state_aligned_action_dim
    else:
        action_supervised_dim = min(front_pair_dim, state_aligned_action_dim)

    state_input_dim = full_state_dim

    return ActionStateDimLayout(
        full_action_dim=full_action_dim,
        full_state_dim=full_state_dim,
        state_aligned_action_dim=state_aligned_action_dim,
        front_pair_dim=front_pair_dim,
        action_supervised_dim=action_supervised_dim,
        state_input_dim=state_input_dim,
        is_robot_moving=bool(is_robot_moving),
        is_body_moving=bool(is_body_moving),
    )


def infer_end_effector_type_from_delta_mask(delta_mask: torch.Tensor | list[bool]) -> str | None:
    side_mask = infer_repeated_front_pair_side_mask(delta_mask)
    if side_mask is None:
        return None

    return END_EFFECTOR_TYPE_DEX_HAND if _longest_false_run(side_mask) > 1 else END_EFFECTOR_TYPE_GRIPPER


def extract_available_images(
    data: dict[str, Any],
    present_img_keys: list[str] | None = None,
    *,
    enable_depth_img: bool = False,
    depth_img_prefix_name: str | None = None,
) -> dict[str, torch.Tensor]:
    if present_img_keys is None:
        present_img_keys = [
            'observation.images.cam_high',
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist',
        ]

    images: dict[str, torch.Tensor] = {}
    for key in present_img_keys:
        if key in data:
            images[key] = data[key]

        if enable_depth_img:
            assert depth_img_prefix_name is not None, 'depth_img_prefix_name is required when enable_depth_img is True'
            depth_img_key = key.replace('observation.images', depth_img_prefix_name)
            if depth_img_key in data:
                images[depth_img_key] = data[depth_img_key]

    return images


def resolve_delta_mask_dim(mask_dim: int, *candidate_dims: Any) -> int:
    """Resolve the effective prefix length for a shared embodiment delta mask.

    Some single-arm robots reuse a multi-arm embodiment id but only occupy the leading 6 or 7 action/state dims. In that case we treat the embodiment
    mask as a prefix mask and truncate it to the available tensor width.
    """
    valid_dims: list[int] = []
    for dim in candidate_dims:
        if dim is None:
            continue
        normalized_dim = int(dim)
        if normalized_dim > 0:
            valid_dims.append(normalized_dim)

    if not valid_dims:
        return mask_dim

    return min(mask_dim, *valid_dims)


def resolve_delta_mask_prefix(mask: torch.Tensor, *candidate_dims: Any) -> torch.Tensor:
    effective_dim = resolve_delta_mask_dim(mask.shape[-1], *candidate_dims)
    return mask[..., :effective_dim]


EGODEX_DUAL_HAND_TCP_QUAT_DIM = 16
EGODEX_TCP_POSE_BLOCK_DIM = 8


def _invert_rigid_transform_batch(t: torch.Tensor) -> torch.Tensor:
    """Invert batched SE(3) matrices ``[..., 4, 4]`` (local->world)."""
    r = t[..., :3, :3]
    p = t[..., :3, 3]
    r_t = r.transpose(-1, -2)
    p_inv = -torch.einsum('...ij,...j->...i', r_t, p)
    out = torch.zeros_like(t)
    out[..., :3, :3] = r_t
    out[..., :3, 3] = p_inv
    out[..., 3, 3] = 1.0
    return out


def _compose_rigid_transform_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched SE(3) compose: ``a @ b`` with shape ``[..., 4, 4]``."""
    out = torch.zeros_like(a)
    out[..., :3, :3] = torch.matmul(a[..., :3, :3], b[..., :3, :3])
    out[..., :3, 3] = torch.einsum('...ij,...j->...i', a[..., :3, :3], b[..., :3, 3]) + a[..., :3, 3]
    out[..., 3, 3] = 1.0
    return out


def _matrix_to_quaternion_xyzw(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``[..., 3, 3]`` to quaternions ``[..., 4]``
    (xyzw)."""
    batch_shape = rot.shape[:-2]
    rot = rot.reshape(-1, 3, 3)
    n = rot.shape[0]
    quat = torch.zeros((n, 4), dtype=rot.dtype, device=rot.device)
    tr = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    mask = tr > 0.0
    if mask.any():
        s = torch.sqrt(tr[mask] + 1.0) * 2.0
        quat[mask, 3] = 0.25 * s
        quat[mask, 0] = (rot[mask, 2, 1] - rot[mask, 1, 2]) / s
        quat[mask, 1] = (rot[mask, 0, 2] - rot[mask, 2, 0]) / s
        quat[mask, 2] = (rot[mask, 1, 0] - rot[mask, 0, 1]) / s

    mask_x = (~mask) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    if mask_x.any():
        s = torch.sqrt(1.0 + rot[mask_x, 0, 0] - rot[mask_x, 1, 1] - rot[mask_x, 2, 2]) * 2.0
        quat[mask_x, 0] = 0.25 * s
        quat[mask_x, 1] = (rot[mask_x, 0, 1] + rot[mask_x, 1, 0]) / s
        quat[mask_x, 2] = (rot[mask_x, 0, 2] + rot[mask_x, 2, 0]) / s
        quat[mask_x, 3] = (rot[mask_x, 2, 1] - rot[mask_x, 1, 2]) / s

    mask_y = (~mask) & (~mask_x) & (rot[:, 1, 1] > rot[:, 2, 2])
    if mask_y.any():
        s = torch.sqrt(1.0 + rot[mask_y, 1, 1] - rot[mask_y, 0, 0] - rot[mask_y, 2, 2]) * 2.0
        quat[mask_y, 0] = (rot[mask_y, 0, 1] + rot[mask_y, 1, 0]) / s
        quat[mask_y, 1] = 0.25 * s
        quat[mask_y, 2] = (rot[mask_y, 1, 2] + rot[mask_y, 2, 1]) / s
        quat[mask_y, 3] = (rot[mask_y, 0, 2] - rot[mask_y, 2, 0]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + rot[mask_z, 2, 2] - rot[mask_z, 0, 0] - rot[mask_z, 1, 1]) * 2.0
        quat[mask_z, 0] = (rot[mask_z, 0, 2] + rot[mask_z, 2, 0]) / s
        quat[mask_z, 1] = (rot[mask_z, 1, 2] + rot[mask_z, 2, 1]) / s
        quat[mask_z, 2] = 0.25 * s
        quat[mask_z, 3] = (rot[mask_z, 1, 0] - rot[mask_z, 0, 1]) / s

    quat = F.normalize(quat, dim=-1)
    return quat.reshape(*batch_shape, 4)


def _ego_quaternion_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    row0 = torch.stack((1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)), dim=-1)
    row1 = torch.stack((2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)), dim=-1)
    row2 = torch.stack((2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _dual_hand_tcp_vec_to_mat4(hand_vec: torch.Tensor) -> torch.Tensor:
    """``[..., 8]`` TCP ``xyz + quat_xyzw + grip`` → ``[..., 4, 4]`` (grip ignored)."""
    pos = hand_vec[..., :3]
    quat = hand_vec[..., 3:7]
    rot = _ego_quaternion_xyzw_to_matrix(quat)
    out = torch.zeros(*hand_vec.shape[:-1], 4, 4, dtype=hand_vec.dtype, device=hand_vec.device)
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def _mat4_to_dual_hand_tcp_vec(transform: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
    """``[..., 4, 4]`` + grip ``[..., 1]`` → ``[..., 8]``."""
    pos = transform[..., :3, 3]
    quat = _matrix_to_quaternion_xyzw(transform[..., :3, :3])
    return torch.cat((pos, quat, grip), dim=-1)


def reframe_dual_hand_tcp_chunk_to_anchor_camera(
    action: torch.Tensor,
    camera_to_world: torch.Tensor,
    *,
    tcp_dim: int = EGODEX_DUAL_HAND_TCP_QUAT_DIM,
) -> torch.Tensor:
    """Express each action timestep in the chunk anchor (index 0) camera frame.

    EgoDex stores ``camera_to_world`` as local->world. Per-frame hand TCP in camera ``k`` is
    ``H_k``; reframed pose in anchor camera ``0`` is ``inv(T_cam_0) @ T_cam_k @ H_k``.
    """
    if action.shape[-1] < tcp_dim:
        raise ValueError(f'Expected action last dim >= {tcp_dim}, got {action.shape[-1]}')
    if action.ndim < 2:
        raise ValueError(f'Expected action with a time dimension, got shape {tuple(action.shape)}')
    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError(f'Expected camera_to_world [..., 4, 4], got {tuple(camera_to_world.shape)}')
    if action.shape[-2] != camera_to_world.shape[-3]:
        raise ValueError(f'Action time dim {action.shape[-2]} != camera time dim {camera_to_world.shape[-3]}')

    x = action[..., :tcp_dim]
    suffix = action[..., tcp_dim:] if action.shape[-1] > tcp_dim else None

    left = x[..., :EGODEX_TCP_POSE_BLOCK_DIM]
    right = x[..., EGODEX_TCP_POSE_BLOCK_DIM:EGODEX_DUAL_HAND_TCP_QUAT_DIM]
    left_grip = left[..., 7:8]
    right_grip = right[..., 7:8]

    left_m = _dual_hand_tcp_vec_to_mat4(left)
    right_m = _dual_hand_tcp_vec_to_mat4(right)

    t_cam0_inv = _invert_rigid_transform_batch(camera_to_world[..., 0, :, :]).unsqueeze(-3)
    t_cam_k_to_cam0 = _compose_rigid_transform_batch(t_cam0_inv.expand_as(camera_to_world), camera_to_world)
    left_new = _mat4_to_dual_hand_tcp_vec(_compose_rigid_transform_batch(t_cam_k_to_cam0, left_m), left_grip)
    right_new = _mat4_to_dual_hand_tcp_vec(_compose_rigid_transform_batch(t_cam_k_to_cam0, right_m), right_grip)
    out = torch.cat((left_new, right_new), dim=-1)
    if suffix is not None:
        out = torch.cat((out, suffix), dim=-1)
    return out


def reframe_dual_hand_tcp_chunk_to_anchor_camera_numpy(
    action: Any,
    camera_to_world: Any,
    *,
    tcp_dim: int = EGODEX_DUAL_HAND_TCP_QUAT_DIM,
) -> Any:
    """NumPy wrapper around
    :func:`reframe_dual_hand_tcp_chunk_to_anchor_camera`."""
    import numpy as _np

    out = reframe_dual_hand_tcp_chunk_to_anchor_camera(
        torch.from_numpy(_np.asarray(action, dtype=_np.float32)),
        torch.from_numpy(_np.asarray(camera_to_world, dtype=_np.float32)),
        tcp_dim=tcp_dim,
    )
    return out.numpy()


REMOVED_ACTION_DIM_SUPERVISION_KEYS = frozenset(
    {
        'extra_loss_mask',
        'supervision_mask',
        'use_action_result_dim_supervision_mask',
        'zero_non_action_dims_when_supervision_mask_short',
    }
)


def validate_full_action_supervision_config(delta_cfg: dict[str, Any] | None) -> None:
    configured_removed_keys = sorted(REMOVED_ACTION_DIM_SUPERVISION_KEYS.intersection(delta_cfg or {}))
    if configured_removed_keys:
        raise ValueError(
            'Legacy per-dimension action supervision options are no longer supported. '
            'Use delta_action_cfg.mask_unsupervised_action_dims_for_noise instead, and remove '
            f'delta_action_cfg keys: {configured_removed_keys}'
        )


def resolve_action_output_dim_mask(
    delta_cfg: dict[str, Any] | None,
    *,
    delta_mask_key: int | str | None,
    delta_mask: torch.Tensor | list[bool],
    original_action_dim: int,
    max_action_dim: int,
    embodiment_id: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Resolve dimensions that should remain non-zero in the final action
    output."""
    validate_full_action_supervision_config(delta_cfg)
    if int(max_action_dim) != 32:
        raise ValueError(f'GigaBrain action heads require max_action_dim=32, got {max_action_dim}')
    if delta_cfg is not None and not bool(delta_cfg.get('zero_non_action_dims_when_action_mask_short', True)):
        return torch.ones(32, dtype=torch.bool, device=device)

    configured_mask = None
    mask_cfg = None if delta_cfg is None else delta_cfg.get('mask')
    if mask_cfg:
        if delta_mask_key is not None:
            configured_mask = mask_cfg.get(delta_mask_key, mask_cfg.get(str(delta_mask_key)))
        if configured_mask is None and len(mask_cfg) == 1:
            configured_mask = next(iter(mask_cfg.values()))
    if configured_mask is None:
        configured_mask = delta_mask

    mask = torch.as_tensor(configured_mask, dtype=torch.bool, device=device)
    if embodiment_id is not None and Embodiment3QuaternionTo6D.supports_embodiment(int(embodiment_id)):
        mask = Embodiment3QuaternionTo6D.transform_mask(mask, int(embodiment_id))
    effective_mask = resolve_delta_mask_prefix(mask, int(original_action_dim), int(original_action_dim))
    aligned = torch.zeros(32, dtype=torch.bool, device=device)
    valid_cols = min(int(effective_mask.shape[-1]), 32)
    if valid_cols > 0:
        aligned[:valid_cols] = True
    return aligned


def resolve_robot_type_mask_key(mask: dict, robot_type: Any, *candidate_dims: Any) -> str:
    robot_type_key = str(robot_type)
    for dim in candidate_dims:
        if dim is None:
            continue
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            continue
        dim_key = f'{robot_type_key}_{dim}d'
        if dim_key in mask:
            return dim_key
    return robot_type_key


class Embodiment3QuaternionTo6D:
    """Convert pose quaternions into a 6D rotation representation.

    Embodiments **3** (UMI) and **5** (``robot_type`` = ``egodex_eef_handbase``) use the same TCP
    layout per hand: ``[xyz, quat_xyzw, gripper]`` → ``[xyz, rot6d, gripper]``.
    Embodiment **4** uses a different pose block size (see ``pose_dims_by_embodiment``).
    """

    embodiment_id = 3
    pose_dims_by_embodiment = {
        3: (8, 10),
        4: (13, 15),
        5: (8, 10),
    }
    embodiment_ids = frozenset(pose_dims_by_embodiment)
    source_pose_dim = 8
    target_pose_dim = 10

    @classmethod
    def _normalize_embodiment_id(cls, embodiment_id: Any) -> int:
        if embodiment_id is None:
            return cls.embodiment_id
        return int(embodiment_id)

    @classmethod
    def supports_embodiment(cls, embodiment_id: Any) -> bool:
        if embodiment_id is None:
            return False
        try:
            return cls._normalize_embodiment_id(embodiment_id) in cls.embodiment_ids
        except (TypeError, ValueError):
            return False

    @classmethod
    def get_pose_dims(cls, embodiment_id: Any) -> tuple[int, int]:
        normalized_embodiment_id = cls._normalize_embodiment_id(embodiment_id)
        if normalized_embodiment_id not in cls.pose_dims_by_embodiment:
            raise ValueError(f'Unsupported quaternion pose embodiment: {normalized_embodiment_id}')
        return cls.pose_dims_by_embodiment[normalized_embodiment_id]

    @classmethod
    def has_source_pose_dim(cls, dim: int, embodiment_id: Any) -> bool:
        if not cls.supports_embodiment(embodiment_id):
            return False
        source_pose_dim, _ = cls.get_pose_dims(embodiment_id)
        return dim % source_pose_dim == 0

    @staticmethod
    def _quaternion_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(quat, dim=-1)
        x, y, z, w = quat.unbind(dim=-1)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w

        row0 = torch.stack((1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)), dim=-1)
        row1 = torch.stack((2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)), dim=-1)
        row2 = torch.stack((2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)), dim=-1)
        return torch.stack((row0, row1, row2), dim=-2)

    @classmethod
    def transform_dim(cls, dim: int, embodiment_id: Any = None) -> int:
        normalized_embodiment_id = cls._normalize_embodiment_id(embodiment_id)
        source_pose_dim, target_pose_dim = cls.get_pose_dims(normalized_embodiment_id)
        if dim % source_pose_dim != 0:
            raise ValueError(f'Expected a multiple of {source_pose_dim} dims for embodiment {normalized_embodiment_id}, but got {dim}.')
        return dim // source_pose_dim * target_pose_dim

    @classmethod
    def _expand_rotation_flags(cls, x: torch.Tensor) -> torch.Tensor:
        rot_flags = x[..., 3:7]
        if x.dtype == torch.bool:
            rot_flags = rot_flags.any(dim=-1, keepdim=True)
        else:
            rot_flags = rot_flags.to(torch.bool).any(dim=-1, keepdim=True).to(dtype=x.dtype)
        return rot_flags.expand(*rot_flags.shape[:-1], 6)

    @classmethod
    def _transform_pose_tensor(cls, x: torch.Tensor, embodiment_id: Any = None) -> torch.Tensor:
        normalized_embodiment_id = cls._normalize_embodiment_id(embodiment_id)
        source_pose_dim, target_pose_dim = cls.get_pose_dims(normalized_embodiment_id)
        if x.shape[-1] % source_pose_dim != 0:
            raise ValueError(f'Expected a multiple of {source_pose_dim} dims for embodiment {normalized_embodiment_id}, but got {x.shape[-1]}.')

        pose_count = x.shape[-1] // source_pose_dim
        x = x.reshape(*x.shape[:-1], pose_count, source_pose_dim)
        pos = x[..., :3]
        quat = x[..., 3:7]
        tail = x[..., 7:]
        rotmat = cls._quaternion_xyzw_to_matrix(quat)
        rot6d = torch.cat((rotmat[..., :, 0], rotmat[..., :, 1]), dim=-1)
        x = torch.cat((pos, rot6d, tail), dim=-1)
        return x.reshape(*x.shape[:-2], pose_count * target_pose_dim)

    @classmethod
    def transform_mask(cls, mask: torch.Tensor, embodiment_id: Any = None) -> torch.Tensor:
        normalized_embodiment_id = cls._normalize_embodiment_id(embodiment_id)
        source_pose_dim, target_pose_dim = cls.get_pose_dims(normalized_embodiment_id)
        if mask.shape[-1] % source_pose_dim != 0:
            raise ValueError(
                f'Expected a multiple of {source_pose_dim} mask dims for embodiment {normalized_embodiment_id}, but got {mask.shape[-1]}.'
            )

        pose_count = mask.shape[-1] // source_pose_dim
        mask = mask.reshape(*mask.shape[:-1], pose_count, source_pose_dim)
        mask = torch.cat((mask[..., :3], cls._expand_rotation_flags(mask), mask[..., 7:]), dim=-1)
        return mask.reshape(*mask.shape[:-2], pose_count * target_pose_dim)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        embodiment_id = data.get('embodiment_id')
        if not self.supports_embodiment(embodiment_id):
            return data

        source_pose_dim, _ = self.get_pose_dims(embodiment_id)

        if 'observation.state' in data:
            data['observation.state'] = self._transform_pose_tensor(data['observation.state'], embodiment_id)
        if 'action' in data:
            data['action'] = self._transform_pose_tensor(data['action'], embodiment_id)
        if 'action_is_pad' in data and data['action_is_pad'].ndim > 0 and data['action_is_pad'].shape[-1] % source_pose_dim == 0:
            data['action_is_pad'] = self.transform_mask(data['action_is_pad'], embodiment_id)
        return data

    # --- Dual-hand TCP quat→6D on a **prefix** only (layout matches embodiment **5**), suffix unchanged ---
    DUAL_HAND_QUAT_PREFIX_SOURCE_DIM: int = 16
    DUAL_HAND_QUAT_LAYOUT_EMBODIMENT_ID: int = 5

    @classmethod
    def transform_tensor_dual_hand_quat_prefix(
        cls,
        x: torch.Tensor,
        *,
        prefix_source_dims: int = DUAL_HAND_QUAT_PREFIX_SOURCE_DIM,
        layout_embodiment_id: int = DUAL_HAND_QUAT_LAYOUT_EMBODIMENT_ID,
    ) -> torch.Tensor:
        """Apply quat→6D only to the leading ``prefix_source_dims`` (two 8-D
        TCP blocks); keep suffix dims as-is."""
        ps = int(prefix_source_dims)
        if int(x.shape[-1]) <= ps:
            return cls._transform_pose_tensor(x, layout_embodiment_id)
        source_pose_dim, _ = cls.get_pose_dims(layout_embodiment_id)
        prefix = x[..., :ps]
        if prefix.shape[-1] % source_pose_dim != 0:
            raise ValueError(f'Prefix {ps} is not a multiple of {source_pose_dim} for layout embodiment {layout_embodiment_id}.')
        suffix = x[..., ps:]
        out_prefix = cls._transform_pose_tensor(prefix, layout_embodiment_id)
        return torch.cat((out_prefix, suffix), dim=-1)

    @classmethod
    def transform_mask_dual_hand_quat_prefix(
        cls,
        mask: torch.Tensor,
        *,
        prefix_source_dims: int = DUAL_HAND_QUAT_PREFIX_SOURCE_DIM,
        layout_embodiment_id: int = DUAL_HAND_QUAT_LAYOUT_EMBODIMENT_ID,
    ) -> torch.Tensor:
        """Mirror ``transform_tensor_dual_hand_quat_prefix`` for boolean / pad
        masks on the last axis."""
        ps = int(prefix_source_dims)
        d = int(mask.shape[-1])
        if d <= ps:
            return cls.transform_mask(mask, layout_embodiment_id)
        pref = mask[..., :ps]
        suff = mask[..., ps:]
        return torch.cat((cls.transform_mask(pref, layout_embodiment_id), suff), dim=-1)

    @classmethod
    def transform_output_dim_dual_hand_prefix(
        cls,
        dim: int,
        *,
        prefix_source_dims: int = DUAL_HAND_QUAT_PREFIX_SOURCE_DIM,
        layout_embodiment_id: int = DUAL_HAND_QUAT_LAYOUT_EMBODIMENT_ID,
    ) -> int:
        """Last-axis width after ``transform_tensor_dual_hand_quat_prefix``."""
        ps = int(prefix_source_dims)
        if dim <= ps:
            return cls.transform_dim(dim, layout_embodiment_id)
        return cls.transform_dim(ps, layout_embodiment_id) + (dim - ps)


class Normalize:
    """Normalizes a tensor using mean/std or quantile-based scaling."""

    @staticmethod
    def _normalize_stats_key(key: int | str | os.PathLike[str]) -> int | str:
        if isinstance(key, os.PathLike):
            key = os.fspath(key)
        if isinstance(key, str):
            normalized_key = os.path.normpath(key)
            if normalized_key.lstrip('-').isdigit():
                return int(normalized_key)
            return normalized_key
        return int(key)

    def __init__(self, stats: dict[int | str, dict[str, list[float]]], *, use_quantiles: bool = False, enable_clamp: bool = False):
        """Initializes the normalization transform.

        Args:
            stats: A dictionary mapping normalization keys to normalization statistics.
            use_quantiles: If True, use 1% and 99% quantiles for scaling.
                Otherwise, use mean and standard deviation.
            enable_clamp: If True, clamp the output to [-1, 1].
        """
        self.EPSILON = 1e-6
        self.use_quantiles = use_quantiles
        self.enable_clamp = enable_clamp

        required_attrs = ['mean', 'std']
        if self.use_quantiles:
            required_attrs = ['q01', 'q99']

        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = dict()
            self.q99 = dict()
            for key in stats:
                normalized_key = self._normalize_stats_key(key)
                self.q01[normalized_key] = torch.tensor(stats[key]['q01'], dtype=torch.float32)
                self.q99[normalized_key] = torch.tensor(stats[key]['q99'], dtype=torch.float32)
        else:
            self.mean = dict()
            self.std = dict()
            for key in stats:
                normalized_key = self._normalize_stats_key(key)
                self.mean[normalized_key] = torch.tensor(stats[key]['mean'], dtype=torch.float32)
                self.std[normalized_key] = torch.tensor(stats[key]['std'], dtype=torch.float32)

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int | str = 0) -> torch.Tensor:
        """Applies normalization to the input tensor.

        Args:
            x: The input tensor to normalize.
            embodiment_id: The normalization key used to select normalization stats.

        Returns:
            The normalized tensor.
        """
        x_dim = x.shape[-1]
        stats_key = self._normalize_stats_key(embodiment_id)
        if self.use_quantiles:
            if stats_key not in self.q01:
                raise KeyError(f'Normalization stats not found for key {stats_key!r}. Available keys: {sorted(self.q01)}')
            x = (x - self.q01[stats_key][..., :x_dim]) / (
                self.q99[stats_key][..., :x_dim] - self.q01[stats_key][..., :x_dim] + self.EPSILON
            ) * 2.0 - 1.0
        else:
            if stats_key not in self.mean:
                raise KeyError(f'Normalization stats not found for key {stats_key!r}. Available keys: {sorted(self.mean)}')
            x = (x - self.mean[stats_key][..., :x_dim]) / (self.std[stats_key][..., :x_dim] + self.EPSILON)

        if self.enable_clamp:
            x = x.clamp(-1.0, 1.0)

        return x


class Unnormalize:
    """Unnormalizes a tensor using mean/std or quantile-based scaling."""

    @staticmethod
    def _normalize_stats_key(key: int | str | os.PathLike[str]) -> int | str:
        return Normalize._normalize_stats_key(key)

    def __init__(self, stats: dict[int | str, dict[str, list[float]]], *, use_quantiles: bool = False):
        """Initializes the unnormalization transform.

        Args:
            stats: A dictionary mapping normalization keys to normalization statistics.
            use_quantiles: If True, use 1% and 99% quantiles for scaling.
                Otherwise, use mean and standard deviation.
        """
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        required_attrs = ['mean', 'std']
        if self.use_quantiles:
            required_attrs = ['q01', 'q99']

        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = dict()
            self.q99 = dict()
            for key in stats:
                normalized_key = self._normalize_stats_key(key)
                self.q01[normalized_key] = torch.tensor(stats[key]['q01'], dtype=torch.float32)
                self.q99[normalized_key] = torch.tensor(stats[key]['q99'], dtype=torch.float32)
        else:
            self.mean = dict()
            self.std = dict()
            for key in stats:
                normalized_key = self._normalize_stats_key(key)
                self.mean[normalized_key] = torch.tensor(stats[key]['mean'], dtype=torch.float32)
                self.std[normalized_key] = torch.tensor(stats[key]['std'], dtype=torch.float32)

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int | str = 0) -> torch.Tensor:
        """Applies unnormalization to the input tensor.

        Args:
            x: The input tensor to unnormalize.
            embodiment_id: The normalization key used to select normalization stats.

        Returns:
            The unnormalized tensor.
        """
        x_dim = x.shape[-1]
        stats_key = self._normalize_stats_key(embodiment_id)
        if self.use_quantiles:
            if stats_key not in self.q01:
                raise KeyError(f'Normalization stats not found for key {stats_key!r}. Available keys: {sorted(self.q01)}')
            return (x + 1.0) / 2.0 * (self.q99[stats_key][..., :x_dim] - self.q01[stats_key][..., :x_dim] + self.EPSILON) + self.q01[stats_key][
                ..., :x_dim
            ]
        else:
            if stats_key not in self.mean:
                raise KeyError(f'Normalization stats not found for key {stats_key!r}. Available keys: {sorted(self.mean)}')
            return x * (self.std[stats_key][..., :x_dim] + self.EPSILON) + self.mean[stats_key][..., :x_dim]


class DeltaActions:
    """Repacks absolute actions into delta action space."""

    def __init__(self, mask: dict, selector: str = 'embodiment_id'):
        assert mask is not None, 'mask is required'
        if selector not in ('embodiment_id', 'robot_type'):
            raise ValueError(f'Unsupported DeltaActions selector: {selector!r}')
        self.selector = selector
        self.mask: dict = dict()
        for key, value in mask.items():
            if selector == 'embodiment_id':
                mask_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                if Embodiment3QuaternionTo6D.supports_embodiment(key):
                    mask_tensor = Embodiment3QuaternionTo6D.transform_mask(mask_tensor, key)
                self.mask[int(key)] = mask_tensor
            else:
                # robot_type mode: caller is responsible for any quat->6D mask transform
                # (avoids importing robot_type_mapping into utils to prevent circular deps).
                mask_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                self.mask[str(key)] = mask_tensor

    def _lookup_key(self, data: dict):
        if self.selector == 'embodiment_id':
            return data['embodiment_id']
        robot_type = data.get('robot_type')
        if robot_type is None:
            meta = data.get('meta')
            robot_type = getattr(meta, 'info', {}).get('robot_type') if meta is not None else None
        if robot_type is None:
            raise KeyError("robot_type is required when DeltaActions.selector='robot_type'")
        action = data.get('action')
        state = data.get('observation.state')
        action_dim = action.shape[-1] if action is not None else None
        state_dim = state.shape[-1] if state is not None else None
        return resolve_robot_type_mask_key(self.mask, robot_type, action_dim, state_dim)

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data

        lookup_key = self._lookup_key(data)

        state, action = data['observation.state'], data['action']
        effective_mask = resolve_delta_mask_prefix(self.mask[lookup_key], state.shape[-1], action.shape[-1])
        dims = effective_mask.shape[-1]
        masked_state = torch.where(effective_mask, state[..., :dims], torch.zeros_like(state[..., :dims]))
        while masked_state.ndim < action.ndim:
            masked_state = masked_state.unsqueeze(-2)
        action[..., :dims] -= masked_state
        data['action'] = action
        return data


class AbsoluteActions:
    """Repacks delta actions into absolute action space."""

    def __init__(self, mask: dict, selector: str = 'embodiment_id'):
        assert mask is not None, 'mask is required'
        if selector not in ('embodiment_id', 'robot_type'):
            raise ValueError(f'Unsupported AbsoluteActions selector: {selector!r}')
        self.selector = selector
        self.mask: dict = dict()
        for key, value in mask.items():
            if selector == 'embodiment_id':
                mask_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                if Embodiment3QuaternionTo6D.supports_embodiment(key):
                    mask_tensor = Embodiment3QuaternionTo6D.transform_mask(mask_tensor, key)
                self.mask[int(key)] = mask_tensor
            else:
                mask_tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                self.mask[str(key)] = mask_tensor

    def _lookup_key(self, data: dict):
        if self.selector == 'embodiment_id':
            return data['embodiment_id']
        robot_type = data.get('robot_type')
        if robot_type is None:
            meta = data.get('meta')
            robot_type = getattr(meta, 'info', {}).get('robot_type') if meta is not None else None
        if robot_type is None:
            raise KeyError("robot_type is required when AbsoluteActions.selector='robot_type'")
        action = data.get('action')
        state = data.get('observation.state')
        action_dim = action.shape[-1] if action is not None else None
        state_dim = state.shape[-1] if state is not None else None
        return resolve_robot_type_mask_key(self.mask, robot_type, action_dim, state_dim)

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data

        lookup_key = self._lookup_key(data)

        state, action = data['observation.state'], data['action']
        effective_mask = resolve_delta_mask_prefix(self.mask[lookup_key], state.shape[-1], action.shape[-1])
        dims = effective_mask.shape[-1]
        masked_state = torch.where(effective_mask, state[..., :dims], torch.zeros_like(state[..., :dims]))
        while masked_state.ndim < action.ndim:
            masked_state = masked_state.unsqueeze(-2)
        action[..., :dims] += masked_state
        data['action'] = action
        return data


class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int):
        """Initializes the padding transform.

        Args:
            action_dim: The target dimension to pad to.
        """
        self.action_dim = action_dim

    def _pad_to_dim(self, x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
        """Pad an array to the target dimension with zeros along the specified
        axis."""
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            shape = list(x.shape)
            shape[-1] = target_dim
            new_vector = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            new_vector[..., :current_dim] = x
            x = new_vector
        return x

    def __call__(self, data: dict) -> dict:
        """Pads 'observation.state' and 'action' tensors in the data dict.

        Args:
            data: A dictionary containing 'observation.state' and optionally 'action'.

        Returns:
            The data dictionary with padded tensors.
        """
        data['observation.state'] = self._pad_to_dim(data['observation.state'], self.action_dim, axis=-1)
        if 'observation.state_memory' in data:
            data['observation.state_memory'] = self._pad_to_dim(data['observation.state_memory'], self.action_dim, axis=-1)
        if 'action' in data:
            data['action'] = self._pad_to_dim(data['action'], self.action_dim, axis=-1)
        return data


# This function is used for the original PaliGemma model inference.
def resize_image(img: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize an image to the given (width, height) without preserving aspect
    ratio.

    Args:
        img: Input image, shape (C, H, W), with values typically in [0, 1].
        width: Target width (W).
        height: Target height (H).

    Returns:
        A torch.Tensor of shape (C, height, width).
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f'(C,H,W) expected, but got {img.shape}')

    resized_img = F.interpolate(img.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    return resized_img


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> tuple[torch.Tensor, dict]:
    """Resize an image to fit inside the given (width, height) while preserving
    aspect ratio, then pad with the specified value so that the final image
    exactly matches the target size.

    Args:
        img: Input image, shape (C, H, W), with values typically in [0, 1].
        width: Target width (W).
        height: Target height (H).
        pad_value: Value to use for padding, defaults to -1.

    Returns:
        A tuple containing:
        - A torch.Tensor of shape (C, height, width).
        - A dictionary with transformation parameters.
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f'(C,H,W) expected, but got {img.shape}')

    cur_height, cur_width = img.shape[1:]
    if cur_height <= 0 or cur_width <= 0:
        raise ValueError(f'Image dimensions must be greater than 0, got H={cur_height}, W={cur_width}')

    ratio = max(cur_width / width, cur_height / height)
    resized_height = min(height, max(1, round(cur_height / ratio)))
    resized_width = min(width, max(1, round(cur_width / ratio)))
    resized_img = F.interpolate(img.unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    transform_params = {
        'original_size': (cur_width, cur_height),
        'ratio': ratio,
        'padding': (pad_left, pad_top),
    }
    return padded_img, transform_params


class RandomPoseTransform:
    """Applies a random crop, resize, and rotation to an image."""

    def __init__(self, crop_fraction: float, rotation_degrees: tuple[float, float]):
        """Initializes the random pose transform.

        Args:
            crop_fraction: Fraction (0, 1] of each side kept by the random
                crop; crop / resize sizes are derived per-image at apply time
                so the transform works for any camera resolution.
            rotation_degrees: The range of degrees for random rotation.
        """
        self.crop_fraction = crop_fraction
        self.rotation_degrees = rotation_degrees

    def generate_params(self, h: int, w: int) -> dict[str, Any]:
        """Generates random transform parameters for an image of a given size.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            A dictionary of transform parameters.
        """
        # Derive crop / resize sizes from the actual input resolution so the
        # transform follows whatever camera it is applied to (the runtime
        # anchor may be the high-res cam_high or a lower-res wrist cam). Resize
        # back to the input size so the frame keeps that camera's resolution.
        crop_size_h = max(1, int(h * self.crop_fraction))
        crop_size_w = max(1, int(w * self.crop_fraction))
        i = torch.randint(0, h - crop_size_h + 1, size=(1,)).item()
        j = torch.randint(0, w - crop_size_w + 1, size=(1,)).item()
        crop_box = (j, i, crop_size_w, crop_size_h)  # x,y,w,h
        angle = transforms.RandomRotation.get_params(self.rotation_degrees)
        return {
            'crop_box': crop_box,
            'crop_size': (crop_size_w, crop_size_h),
            'resize_size': (w, h),
            'angle': angle,
        }

    def apply_with_params(self, img: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Applies the transform to an image using pre-generated parameters.

        Args:
            img: The input image.
            params: A dictionary of transform parameters from `generate_params`.

        Returns:
            The transformed image.
        """
        j, i, tw, th = params['crop_box']
        img = transforms.functional.crop(img, i, j, th, tw)
        resize_w, resize_h = params['resize_size']
        img = transforms.functional.resize(img, (resize_h, resize_w))
        if params.get('angle') is not None:
            img = transforms.functional.rotate(img, params['angle'])
        return img

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Applies random transformations to the image.

        Args:
            img: The input image.

        Returns:
            A tuple of the transformed image and the applied parameters.
        """
        h, w = img.shape[-2:]
        params = self.generate_params(h, w)
        transformed_img = self.apply_with_params(img, params)
        return transformed_img, params


class ImageTransform:
    """Preprocesses a dictionary of images with optional augmentation."""

    def __init__(
        self,
        is_train: bool,
        resize_imgs_with_padding: tuple[int, int],
        present_img_keys: list[str] | None = None,
        primary_img_key: str = 'observation.images.cam_high',
        enable_image_aug: bool = False,
        enable_depth_img: bool = False,
        depth_img_prefix_name: str | None = None,
        depth_img_mask_ratio: float = 0.5,
        history_frame_dropout_prob: float = 0.0,
        vlm_type: str = 'paligemma2',
        history_frame_dropout_mode: str = 'all',
    ):
        """Initializes the image transform pipeline.

        Args:
            is_train: Whether the transform is used for training.
            resize_imgs_with_padding: Target size (w, h) for resizing with padding.
            present_img_keys: List of image keys to process from the input data dict.
            enable_image_aug: If True, applies color jitter and random pose transforms.
            enable_depth_img: If True, concatenates depth images to RGB images.
            depth_img_prefix_name: The prefix key for depth images in the data dict.
            depth_img_mask_ratio: The ratio of depth images to mask out during training.
            history_frame_dropout_prob: Probability of dropping all historical
                frames from every camera together. Only active during training; the
                current frame is always kept.
            history_frame_dropout_mode: Historical-frame dropout mode. The only
                supported value is ``'all'``.
            vlm_type: VLM backend type. Supported image preprocessing uses
                [-1, 1] SigLIP normalization.
        """
        self.vlm_type = vlm_type
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.present_img_keys = present_img_keys
        if self.present_img_keys is None:
            self.present_img_keys = [
                'observation.images.cam_high',
                'observation.images.cam_left_wrist',
                'observation.images.cam_right_wrist',
            ]
        self.primary_img_key = primary_img_key
        self.enable_image_aug = enable_image_aug
        self.width, self.height = resize_imgs_with_padding
        self.enable_depth_img = enable_depth_img
        self.depth_img_prefix_name = depth_img_prefix_name
        self.depth_img_mask_ratio = depth_img_mask_ratio if is_train else 0.0
        history_frame_dropout_prob = float(history_frame_dropout_prob)
        if not math.isfinite(history_frame_dropout_prob) or not 0.0 <= history_frame_dropout_prob <= 1.0:
            raise ValueError('history_frame_dropout_prob must be finite and in [0, 1], ' f'got {history_frame_dropout_prob}')
        if history_frame_dropout_mode != 'all':
            raise ValueError("history_frame_dropout_mode must be 'all', " f'got {history_frame_dropout_mode!r}')
        self.history_frame_dropout_prob = history_frame_dropout_prob if is_train else 0.0
        self.history_frame_dropout_mode = history_frame_dropout_mode

        self.cam_resolutions = {key: self.resize_imgs_with_padding for key in self.present_img_keys}

        if self.enable_image_aug:
            self.color_jitter_transform = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
            )
            # Pose aug is applied to the runtime anchor camera, whose resolution can
            # differ per sample (cam_high@768 for robot data, or a 384 wrist cam when
            # cam_high is absent, e.g. UMI). Use a resolution-agnostic crop fraction
            # so crop / resize sizes are derived from the actual image at apply time.
            self.pose_transform = RandomPoseTransform(
                crop_fraction=0.95,
                rotation_degrees=(-5, 5),
            )

    def _find_reference_image(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        for key in self.present_img_keys:
            if key in data:
                return data[key]
        for key in self.present_img_keys:
            pad_key = f'{key}_is_pad'
            if pad_key not in data:
                continue
            pad_mask = torch.as_tensor(data[pad_key])
            if pad_mask.ndim > 0 and int(pad_mask.numel()) > 1:
                return torch.zeros(int(pad_mask.numel()), 3, self.height, self.width, dtype=torch.float32)
        return torch.zeros(3, self.height, self.width, dtype=torch.float32)

    def _resolve_anchor_image_key(self, data: dict[str, torch.Tensor]) -> str:
        if self.primary_img_key in data:
            return self.primary_img_key

        for key in self.present_img_keys:
            if key in data:
                return key

        return self.primary_img_key

    @staticmethod
    def _as_frame_sequence(img: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if img.ndim == 3:
            return img.unsqueeze(0), False
        if img.ndim == 4:
            return img, True
        raise ValueError(f'Expected image shape (C,H,W) or (T,C,H,W), got {tuple(img.shape)}')

    def _build_image_mask(self, data: dict, key: str, img_exists: bool, is_temporal: bool, num_frames: int, device: torch.device) -> torch.Tensor:
        if not img_exists:
            if is_temporal:
                return torch.zeros(num_frames, dtype=torch.bool, device=device)
            return torch.tensor(False, dtype=torch.bool, device=device)

        pad_key = f'{key}_is_pad'
        if pad_key not in data:
            if is_temporal:
                return torch.ones(num_frames, dtype=torch.bool, device=device)
            return torch.tensor(True, dtype=torch.bool, device=device)

        mask = ~torch.as_tensor(data[pad_key], dtype=torch.bool, device=device)
        if is_temporal:
            if mask.ndim == 0:
                mask = mask.expand(num_frames)
            if mask.shape[-1] != num_frames:
                raise ValueError(f'{pad_key} must have {num_frames} entries for temporal image input, got {tuple(mask.shape)}')
            return mask.reshape(num_frames)
        return mask.reshape(-1)[-1]

    def _drop_history_frames(
        self,
        img_mask: torch.Tensor,
        is_temporal: bool,
        *,
        drop_all_history: bool = False,
    ) -> torch.Tensor:
        if not is_temporal or img_mask.numel() <= 1 or self.history_frame_dropout_prob == 0.0:
            return img_mask

        if not drop_all_history:
            return img_mask
        dropped_mask = img_mask.clone()
        dropped_mask[:-1] = False
        return dropped_mask

    @staticmethod
    def _has_current_frame_pixels(frames: torch.Tensor, is_temporal: bool) -> bool:
        current_frame = frames[-1] if is_temporal else frames[0]
        rgb_frame = current_frame[:3]
        return bool(torch.any(rgb_frame != 0).item())

    @staticmethod
    def _has_current_frame_mask(mask: torch.Tensor, is_temporal: bool) -> bool:
        current_mask = mask[-1] if is_temporal else mask
        return bool(current_mask.item())

    def _resize_frame_sequence(
        self, frames: torch.Tensor, img_exists: bool, key: str, anchor_img_key: str, image_transform_params: dict
    ) -> torch.Tensor:
        # Per-camera target resolution (high_res_cam override or global default).
        target = self.cam_resolutions.get(key) or self.resize_imgs_with_padding
        if target is None:
            return frames

        target_w, target_h = target
        original_h, original_w = frames.shape[-2:]
        if original_h == target_h and original_w == target_w:
            return frames

        resized_frames = []
        last_params = None
        for frame in frames:
            resized_frame, rwp_params = resize_with_pad(frame, target_w, target_h, pad_value=0)
            resized_frames.append(resized_frame)
            last_params = rwp_params

        if img_exists and key == anchor_img_key and last_params is not None:
            image_transform_params['resize_with_pad'] = last_params
        return torch.stack(resized_frames, dim=0)

    @staticmethod
    def _apply_color_jitter(img: torch.Tensor, color_jitter_params) -> torch.Tensor:
        if callable(color_jitter_params):
            return color_jitter_params(img)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = color_jitter_params
        for fn_id in fn_idx:
            fn_id = int(fn_id)
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)
        return img

    def _augment_frame_sequence(
        self, frames: torch.Tensor, img_exists: bool, key: str, anchor_img_key: str, image_transform_params: dict
    ) -> torch.Tensor:
        if not self.enable_image_aug:
            return frames

        if img_exists and key == anchor_img_key:
            h, w = frames.shape[-2:]
            pose_params = self.pose_transform.generate_params(h, w)
            frames = torch.stack([self.pose_transform.apply_with_params(frame, pose_params) for frame in frames], dim=0)
            image_transform_params['pose_transform'] = pose_params

        color_jitter = transforms.ColorJitter.get_params(
            self.color_jitter_transform.brightness,
            self.color_jitter_transform.contrast,
            self.color_jitter_transform.saturation,
            self.color_jitter_transform.hue,
        )
        augmented_frames = []
        for frame in frames:
            frame = frame.clone()
            frame[:3, :, :] = self._apply_color_jitter(frame[:3, :, :], color_jitter)
            augmented_frames.append(frame)
        return torch.stack(augmented_frames, dim=0)

    def __call__(self, data: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
        """Preprocesses input images from a data dictionary.

        This transform selects images based on `present_img_keys`, optionally
        concatenates depth, applies augmentations if in training mode, resizes
        and pads, and normalizes pixel values to [-1, 1].

        Args:
            data: A dictionary containing image data.

        Returns:
            A tuple containing:
            - images: The list of processed image tensors (C, H, W) or (T, C, H, W).
            - img_masks: A list of boolean masks, one for each image or timestep.
            - image_transform_params: A dictionary of applied transformation parameters.
        """
        images = []
        img_masks = []
        present_camera_flags = []  # 方案 B: current-frame validity per camera, present_img_keys order
        image_transform_params = {}
        reference_img = self._find_reference_image(data)
        anchor_img_key = self._resolve_anchor_image_key(data)
        has_history = any(
            key in data and isinstance(data[key], torch.Tensor) and data[key].ndim == 4 and data[key].shape[0] > 1 for key in self.present_img_keys
        )
        drop_all_history = self.history_frame_dropout_prob > 0.0 and has_history and bool(torch.rand(()).item() < self.history_frame_dropout_prob)
        has_valid_current_image = False
        has_current_image_mask = False
        has_current_image_pixels = False

        for key in self.present_img_keys:
            img_exists = key in data
            img = data[key] if img_exists else torch.zeros_like(reference_img)
            frames, is_temporal = self._as_frame_sequence(img)
            img_mask = self._build_image_mask(data, key, img_exists, is_temporal, frames.shape[0], frames.device)
            current_mask_valid = self._has_current_frame_mask(img_mask, is_temporal)
            current_frame_has_pixels = self._has_current_frame_pixels(frames, is_temporal)
            has_current_image_mask = has_current_image_mask or current_mask_valid
            has_current_image_pixels = has_current_image_pixels or current_frame_has_pixels
            has_valid_current_image = has_valid_current_image or (current_mask_valid and current_frame_has_pixels)
            present_camera_flags.append(bool(current_mask_valid))

            if self.enable_depth_img:
                assert self.depth_img_prefix_name is not None, 'depth_img_prefix_name is required'
                depth_img_key = key.replace('observation.images', self.depth_img_prefix_name)
                if img_exists and depth_img_key in data and random.random() >= self.depth_img_mask_ratio:
                    depth_frames, depth_is_temporal = self._as_frame_sequence(data[depth_img_key])
                    if depth_is_temporal != is_temporal:
                        raise ValueError(
                            f'{depth_img_key} temporal rank must match {key}: got {tuple(data[depth_img_key].shape)} vs {tuple(img.shape)}'
                        )
                    depth_img = depth_frames[:, 0:1]
                else:
                    depth_img = torch.zeros_like(frames[:, 0:1])
                frames = torch.cat([frames, depth_img], dim=1)

            frames = self._resize_frame_sequence(frames, img_exists, key, anchor_img_key, image_transform_params)
            frames = self._augment_frame_sequence(frames, img_exists, key, anchor_img_key, image_transform_params)

            if frames.dtype == torch.uint8:
                frames = frames.float() / 255.0
            frames = frames * 2.0 - 1.0
            img_mask = self._build_image_mask(data, key, img_exists, is_temporal, frames.shape[0], frames.device)
            img_mask = self._drop_history_frames(
                img_mask,
                is_temporal,
                drop_all_history=drop_all_history,
            )

            images.append(frames if is_temporal else frames[0])
            img_masks.append(img_mask)

        data['_all_current_image_masks_invalid'] = not has_current_image_mask
        data['_all_current_images_zero'] = not has_current_image_pixels
        data['_skip_loss_for_invalid_images'] = not has_valid_current_image

        return images, img_masks, image_transform_params


class TrajectoryTransform:
    """Transforms 2D trajectory data, including coordinate adjustments and
    normalization."""

    def __init__(self, step_interval: int | None = None, minmax_value: list[float] | None = None):
        """Initializes the trajectory transform.

        Args:
            step_interval: Interval for subsampling trajectory points.
            minmax_value: A list or tuple of [x_min, y_min, x_max, y_max] for
                clipping and normalization.
        """
        self.step_interval = step_interval
        self.minmax_value = minmax_value
        if minmax_value is not None:
            assert minmax_value[2] > 0 and minmax_value[3] > 0, 'x_max and y_max must be greater than 0'
            self.min_value = torch.tensor([minmax_value[0], minmax_value[1], minmax_value[0], minmax_value[1]])
            self.max_value = torch.tensor([minmax_value[2], minmax_value[3], minmax_value[2], minmax_value[3]])
        else:
            self.min_value = None
            self.max_value = None
        self.traj_size = 4  # x_left, y_left, x_right, y_right

    def to(self, device: str | torch.device):
        if self.min_value is not None:
            self.min_value = self.min_value.to(device)
        if self.max_value is not None:
            self.max_value = self.max_value.to(device)
        return self

    def __call__(self, data: dict, chunk_size: int, image_transform_params: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes 2D trajectory data from a dictionary.

        This includes subsampling, applying inverse geometric transforms from image
        augmentations, and normalizing coordinates.

        Args:
            data: A dictionary containing trajectory data.
            chunk_size: The size of the trajectory chunk.
            image_transform_params: Optional dictionary of image transforms to invert.

        Returns:
            A tuple of (transformed trajectory, padding mask).
        """
        if 'perception.2d_traj' not in data or 'perception.2d_traj_is_pad' not in data:
            traj_chunk_size = (chunk_size // self.step_interval) if self.step_interval is not None else chunk_size
            return -torch.ones(traj_chunk_size, self.traj_size, dtype=torch.float32), torch.ones(traj_chunk_size, self.traj_size, dtype=torch.bool)

        if self.step_interval is not None:
            traj = data['perception.2d_traj'][:: self.step_interval]
            traj_is_pad = data['perception.2d_traj_is_pad'][:: self.step_interval]
        else:
            traj = data['perception.2d_traj']
            traj_is_pad = data['perception.2d_traj_is_pad']

        traj[torch.isnan(traj)] = -100

        if image_transform_params is not None:
            coords = traj.view(-1, 2, 2)

            if 'resize_with_pad' in image_transform_params:
                rwp = image_transform_params['resize_with_pad']
                ratio = rwp['ratio']
                pad_x, pad_y = rwp['padding']
                coords = coords / ratio
                coords[..., 0] += pad_x
                coords[..., 1] += pad_y

            if 'pose_transform' in image_transform_params:
                pose_p = image_transform_params['pose_transform']

                if pose_p.get('crop_box'):
                    crop_x, crop_y, _, _ = pose_p['crop_box']
                    coords[..., 0] -= crop_x
                    coords[..., 1] -= crop_y

                crop_w, crop_h = pose_p['crop_size']
                resize_w, resize_h = pose_p['resize_size']
                if crop_w > 0 and crop_h > 0:
                    scale_x = resize_w / crop_w
                    scale_y = resize_h / crop_h
                    coords[..., 0] *= scale_x
                    coords[..., 1] *= scale_y

                if pose_p.get('angle') is not None:
                    angle_rad = -math.radians(pose_p['angle'])
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)

                    center_x, center_y = resize_w / 2, resize_h / 2

                    coords[..., 0] -= center_x
                    coords[..., 1] -= center_y

                    x_new = coords[..., 0] * cos_a - coords[..., 1] * sin_a
                    y_new = coords[..., 0] * sin_a + coords[..., 1] * cos_a
                    coords[..., 0] = x_new
                    coords[..., 1] = y_new

                    coords[..., 0] += center_x
                    coords[..., 1] += center_y

            traj = coords.view(-1, self.traj_size)

        traj_is_pad = traj_is_pad[:, None].expand(traj_is_pad.shape[0], self.traj_size)
        if self.minmax_value is not None:
            # set traj_is_pad to True if the traj value is out of the minmax value
            traj_is_pad = traj_is_pad | (traj < self.min_value[None, ...]) | (traj > self.max_value[None, ...])
            traj = traj.clamp(self.min_value[None, ...], self.max_value[None, ...])
            traj = traj / self.max_value[None, ...]
        return traj, traj_is_pad


class PromptTokenizerTransform:
    """Encodes task, state, and action information into token sequences for the
    policy model."""

    def __init__(
        self,
        is_train: bool,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        max_length: int,
        discrete_state_input: bool = True,
        discrete_state_input_for_pose_embodiments: bool = False,
        encode_action_input: bool = False,
        fast_token_vocab_mode: str | None = None,
        fast_token_tail_skip_tokens: int = 128,
        fast_token_tail_vocab_size: int | None = None,
        encoded_action_horizon: int | None = None,
        encode_sub_task_input: bool = False,
        enable_control_mode_token: bool = False,
        control_mode_override: str | None = None,
        enable_end_effector_token: bool = False,
        end_effector_override: str | None = None,
        text_token_length: int | None = 257152,
        autoregressive_inference_mode: bool = False,
        sample_ratios: dict | None = None,
        vlm_type: str = 'paligemma2',
        prefix_lm_text: bool = False,
        image_attn: str = 'bidirectional',
        resize_imgs_with_padding: tuple[int, int] | list[int] | None = None,
        use_chat_template: bool = False,
        chat_system_prompt: str | None = None,
        prompt_filler_text: str | None = None,
        state_input_mode: str = 'prompt',
    ):
        """Initializes the prompt and tokenizer transform.

        Args:
            is_train: Whether the transform is used for training. This affects
                whether random sampling of prompt formats is used.
            tokenizer_model_path: Path to the main tokenizer model.
            fast_tokenizer_path: Path to the fast tokenizer for actions.
            max_length: Maximum sequence length for padding.
            discrete_state_input: If True, discretize and include state in the prompt.
            discrete_state_input_for_pose_embodiments: When True, also include discretized state
                for pose-quaternion embodiments (3/4/5). Default False preserves legacy behavior.
            encode_action_input: If True, encode actions into the prompt.
            fast_token_vocab_mode: How to map FAST BPE ids. Defaults to
                ``tail`` for PaliGemma2 and ``expanded`` for other backbones.
                ``tail`` reuses an existing vocab tail slice,
                matching the legacy Pali/Gemma2 style mapping. ``expanded``
                appends ``<|action_token_i|>`` tokens to the HF tokenizer and
                resizes the model vocab.
            fast_token_tail_skip_tokens: Number of final vocab ids to leave
                unused before the tail FAST slice when
                ``fast_token_vocab_mode='tail'``. Defaults to the Gemma2-style
                reserve of 128 ids.
            fast_token_tail_vocab_size: Optional vocab size whose tail should
                hold FAST tokens. Defaults to ``tokenizer.vocab_size`` when
                available, otherwise ``len(tokenizer)``.
            encoded_action_horizon: The horizon for downsampling actions before encoding.
            encode_sub_task_input: If True, include sub-tasks in the prompt.
            text_token_length: The vocabulary size to consider for text tokens.
            autoregressive_inference_mode: If True, configure for autoregressive inference.
            sample_ratios: A dictionary of ratios for sampling different prompt formats.
            vlm_type: The VLM backbone type.
            prefix_lm_text: If True, fold the "Task: ..." text prefix into the
                bidirectional vision block (PaliGemma2-style prefix-LM). Subtask /
                FAST-action / eos remain causal. PaliGemma2's attention mask is
                already prefix-LM by construction.
            resize_imgs_with_padding: Reserved compatibility argument.
            use_chat_template: Reserved compatibility argument.
            chat_system_prompt: Optional system message used when
                ``use_chat_template=True``. If None or empty, no system turn is
                emitted (the sequence opens directly with the user turn).
            prompt_filler_text: Optional fixed text appended to the task prefix.
        """
        self.is_train = is_train
        self.device = 'cpu'
        self.vlm_type = vlm_type
        if state_input_mode not in ('prompt', 'proprio_memory', 'proprio_anchor'):
            raise ValueError(f'Unsupported state_input_mode: {state_input_mode!r}')
        if state_input_mode == 'proprio_anchor' and vlm_type != 'paligemma2':
            raise ValueError("state_input_mode='proprio_anchor' is supported only for vlm_type='paligemma2'")
        self.state_input_mode = state_input_mode
        self.propri_token_id = None
        self.prefix_lm_text = prefix_lm_text
        # Image tokens form one bidirectional prefix block in release backbones.
        if image_attn not in ('bidirectional', 'causal'):
            raise ValueError(f"image_attn must be 'bidirectional' or 'causal', got {image_attn!r}")
        if image_attn == 'causal':
            raise ValueError("image_attn='causal' is not supported by the release backbones")
        self.image_attn = image_attn
        self.prompt_filler_text = prompt_filler_text

        if vlm_type != 'paligemma2':
            raise ValueError(f"GigaBrain-0.7 requires vlm_type='paligemma2', got {vlm_type!r}")
        self.paligemma_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.paligemma_tokenizer.add_bos_token = True
        if self.state_input_mode == 'proprio_anchor':
            self.propri_token_id = register_propri_token(self.paligemma_tokenizer)
        self.processor = AutoProcessor.from_pretrained(tokenizer_model_path)

        self._fast_tokenizer_path = fast_tokenizer_path
        # Lazy load FAST tokenizer for all VLM types to avoid AutoProcessor conversion issues
        self._fast_tokenizer = None

        self.encode_action_input = encode_action_input
        if fast_token_vocab_mode is None:
            fast_token_vocab_mode = 'tail'
        self.fast_token_vocab_mode = fast_token_vocab_mode
        if self.fast_token_vocab_mode not in ('expanded', 'tail'):
            raise ValueError(f"fast_token_vocab_mode must be 'expanded' or 'tail', got {self.fast_token_vocab_mode!r}")
        self.discrete_state_input = discrete_state_input
        self.discrete_state_input_for_pose_embodiments = discrete_state_input_for_pose_embodiments
        self.encode_sub_task_input = encode_sub_task_input
        self.enable_control_mode_token = enable_control_mode_token
        self.control_mode_override = normalize_control_mode(control_mode_override)
        self.enable_end_effector_token = enable_end_effector_token
        self.end_effector_override = normalize_end_effector_type(end_effector_override)

        self.encoded_action_horizon = encoded_action_horizon
        self.fast_skip_tokens = int(fast_token_tail_skip_tokens)
        self.fast_token_tail_vocab_size = fast_token_tail_vocab_size
        self.max_length = max_length
        self.text_token_length = text_token_length

        self.autoregressive_inference_mode = autoregressive_inference_mode

        self.sample_generator = None
        if is_train and sample_ratios is not None:
            self.sample_generator = SampleGenerator(sample_ratios)
            if not self.encode_sub_task_input:
                subtask_ratios = self.sample_generator.subtask_sample_ratios()
                assert not subtask_ratios, (
                    'prompt_cfg.encode_sub_task_input=False conflicts with sample_ratios that '
                    f'include subtask prompt/target modes: {subtask_ratios}. Set those ratios to '
                    '0, e.g. use input_task_target_action=1 for no-subtask action training.'
                )

        self.use_chat_template = use_chat_template
        self.chat_system_prompt = chat_system_prompt  # None / '' → omit system turn
        if self.use_chat_template:
            raise ValueError('use_chat_template is not supported by the release backbones')

    def _resolve_control_mode(self, data: dict[str, Any]) -> str | None:
        if not self.enable_control_mode_token:
            return None

        if 'control_mode_override' in data and data['control_mode_override'] is not None:
            return normalize_control_mode(data['control_mode_override'])

        if self.control_mode_override is not None:
            return self.control_mode_override

        return infer_control_mode_from_data(data)

    def _resolve_end_effector_type(self, data: dict[str, Any]) -> str | None:
        if not self.enable_end_effector_token:
            return None

        if 'end_effector_override' in data and data['end_effector_override'] is not None:
            return normalize_end_effector_type(data['end_effector_override'])

        if self.end_effector_override is not None:
            return self.end_effector_override

        if 'end_effector_type' in data and data['end_effector_type'] is not None:
            return normalize_end_effector_type(data['end_effector_type'])

        return None

    def to(self, device: str | torch.device):
        self.device = device
        return self

    def _skip_discrete_state_in_prompt(self, embodiment_id: int | None) -> bool:
        return Embodiment3QuaternionTo6D.supports_embodiment(embodiment_id) and not self.discrete_state_input_for_pose_embodiments

    @property
    def fast_tokenizer(self):
        if self._fast_tokenizer is None:
            self._fast_tokenizer = _load_fast_tokenizer(self._fast_tokenizer_path)
        return self._fast_tokenizer

    def _select_encoded_action(self, action: torch.Tensor, action_fps: Any = None) -> torch.Tensor:
        if self.encoded_action_horizon is None:
            return action

        indices = flow_action_horizon_indices(
            action.shape[1],
            int(self.encoded_action_horizon),
            action_fps=action_fps,
            device=action.device,
        )
        assert indices is not None
        if indices.ndim != 1:
            if indices.shape[0] != action.shape[0]:
                raise ValueError(f'action_fps batch size {indices.shape[0]} does not match action batch size {action.shape[0]}')
            gather_indices = indices[:, :, None].expand(-1, -1, action.shape[2])
            return action.gather(1, gather_indices)
        return action.index_select(1, indices)

    def encode_action(self, action: torch.Tensor, action_fps: Any = None) -> dict:
        """Encodes a continuous action tensor into a sequence of discrete
        tokens.

        Args:
            action: The action tensor to encode. Shape (1, horizon, dim).

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask' for the
            encoded action.
        """
        action = self._select_encoded_action(action, action_fps=action_fps)

        batch_tokens = self.fast_tokenizer(action.to(torch.float32))
        fast_out = self.processor.tokenizer.pad({'input_ids': batch_tokens}, return_tensors='pt')

        # Assuming batch size is 1.
        act_ids = fast_out['input_ids'].squeeze(0)
        act_mask = fast_out['attention_mask'].squeeze(0)

        # Remap action tokens to the PaliGemma token space.
        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)

        act_ids = vocab_size - 1 - self.fast_skip_tokens - act_ids
        act_ids[act_mask == 0] = self.paligemma_tokenizer.pad_token_id

        # Prepare BOS, separator, and EOS tokens.
        bos = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')
        eos = self.paligemma_tokenizer('|<eos>', add_special_tokens=False, return_tensors='pt')

        # Concatenate all parts to form the final sequence.
        final_act_ids = torch.cat(
            [
                bos['input_ids'].squeeze(0).to(act_ids.device),
                act_ids,
                eos['input_ids'].squeeze(0).to(act_ids.device),
            ],
            dim=0,
        )

        final_act_mask = torch.cat(
            [
                bos['attention_mask'].squeeze(0).to(act_mask.device),
                act_mask,
                eos['attention_mask'].squeeze(0).to(act_mask.device),
            ],
            dim=0,
        )

        return {'input_ids': final_act_ids, 'attention_mask': final_act_mask}

    def encode_sub_task(self, sub_task: str, add_eos: bool = True) -> dict:
        """Encodes a sub-task string into a sequence of tokens.

        Args:
            sub_task: The sub-task string to encode.
            add_eos: If True, append an EOS token.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask' for the
            encoded sub-task.
        """
        assert self.encode_sub_task_input, (
            'Attempted to encode subtask target while prompt_cfg.encode_sub_task_input=False. ' 'Check prompt_cfg.sample_ratios.'
        )
        bos = self.paligemma_tokenizer('Subtask: ', add_special_tokens=False, return_tensors='pt')
        subtask_out = self.paligemma_tokenizer(
            [sub_task],
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        final_subtask_ids = torch.cat(
            [
                bos['input_ids'].squeeze(0),
                subtask_out['input_ids'].squeeze(0),
            ],
            dim=0,
        )
        final_subtask_mask = torch.cat(
            [
                bos['attention_mask'].squeeze(0),
                subtask_out['attention_mask'].squeeze(0),
            ],
            dim=0,
        )

        if add_eos:
            eos = self.paligemma_tokenizer('<eos>', add_special_tokens=False, return_tensors='pt')
            final_subtask_ids = torch.cat([final_subtask_ids, eos['input_ids'].squeeze(0)], dim=0)
            final_subtask_mask = torch.cat([final_subtask_mask, eos['attention_mask'].squeeze(0)], dim=0)

        return {'input_ids': final_subtask_ids, 'attention_mask': final_subtask_mask}

    def create_input_tokens(
        self,
        task: str,
        state: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        control_mode: str | None = None,
        end_effector_type: str | None = None,
        embodiment_id: int | None = None,
        sample_context: dict[str, Any] | None = None,
        image_grid_thw: torch.Tensor | None = None,
        action_fps: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Creates the final token sequence from state, task, and optional
        action.

        This method combines different modalities into a single token sequence based
        on the configured encoding scheme (e.g., including discretized state,
        encoded actions, or predicting sub-tasks).

        Args:
            task: The task description string.
            state: The optional state tensor.
            action: The optional action tensor.
            embodiment_id: The optional embodiment id used to customize prompt format.
            image_grid_thw: Reserved compatibility argument.

        Returns:
            A tuple containing:
            - final_ids: The final token IDs.
            - padded_mask: The attention mask for the padded sequence.
            - att_mask: Attention mask for language modeling loss.
            - loss_mask: Mask to indicate which tokens to use for loss calculation.
            - fast_action_indicator: A mask indicating tokens that correspond to actions.
            - predict_subtask_only: A boolean indicating if ONLY the sub-task is being predicted
              (no FAST action tokens). Used to zero out diffusion loss when there are no action tokens.
        """
        prefix_texts = []
        control_mode_text = '' if control_mode is None else f', Control mode: {control_mode}'
        end_effector_text = '' if end_effector_type is None else f', End effector: {end_effector_type}'

        task = resolve_task_text(task, sample_context)
        main_task, sub_task = split_task_and_subtask(task)

        # Randomly sample the input type
        encode_sub_task_input = self.encode_sub_task_input
        is_sub_task_train = self.is_train
        encode_action_input = self.encode_action_input
        if self.sample_generator is not None:
            encode_sub_task_input, is_sub_task_train, encode_action_input = self.sample_generator.get_sample()
        encode_sub_task_input = encode_sub_task_input and sub_task is not None

        # Create the prefix text
        # The version of pretrained model to init the model is `pt` instead of `mix`, so the prefix text should end with a `\n`
        # Situation 1: Only main task
        predict_subtask = encode_sub_task_input and is_sub_task_train
        # predict_subtask_only: subtask is predicted but no FAST action tokens follow.
        # Used downstream to zero out diffusion loss only when there are no action tokens.
        predict_subtask_only = predict_subtask and not encode_action_input
        if getattr(self, 'state_input_mode', 'prompt') == 'proprio_anchor':
            if encode_sub_task_input and not is_sub_task_train:
                prefix_texts.append(f'Subtask: {sub_task}{control_mode_text}{end_effector_text}, ' f'State: {PROPRI_TOKEN};\n')
            else:
                prefix_texts.append(f'Task: {main_task}{control_mode_text}{end_effector_text}, ' f'State: {PROPRI_TOKEN};\n')
        elif predict_subtask_only or not self.discrete_state_input:
            prefix_texts.append(f'Task: {main_task}{control_mode_text}{end_effector_text}\n')

        # Situation 2: Main task and state
        # Situation 3: Main task, subtask and state
        elif self.discrete_state_input:
            skip_discrete_state = self._skip_discrete_state_in_prompt(embodiment_id)
            if skip_discrete_state:
                if encode_sub_task_input and not is_sub_task_train:
                    prefix_texts.append(f'Subtask: {sub_task}{control_mode_text}{end_effector_text};\n')
                else:
                    prefix_texts.append(f'Task: {main_task}{control_mode_text}{end_effector_text}\n')
            else:
                assert state is not None, 'state is required when discrete_state_input is True'
                bins = torch.linspace(-1, 1, 256 + 1, device=self.device)[:-1]
                discretized = torch.bucketize(state, bins) - 1
                state_str = ' '.join(str(val.item()) for val in discretized)
                if encode_sub_task_input and not is_sub_task_train:
                    prefix_texts.append(f'Subtask: {sub_task}{control_mode_text}{end_effector_text}, State: {state_str};\n')
                else:
                    prefix_texts.append(f'Task: {main_task}{control_mode_text}{end_effector_text}, State: {state_str};\n')

        else:
            raise ValueError('Invalid prefix text mode')

        prefix_out = self.paligemma_tokenizer(
            prefix_texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        prefix_ids = prefix_out['input_ids'][0]
        prefix_mask = prefix_out['attention_mask'][0]

        # Create the suffix text first so long prefixes can leave room for it.
        sub_task_ids = None
        sub_task_mask = None
        if predict_subtask:
            encoded_sub_task = self.encode_sub_task(sub_task, add_eos=True)
            sub_task_ids = encoded_sub_task['input_ids']
            sub_task_mask = encoded_sub_task['attention_mask']

        act_ids = None
        act_mask = None
        if encode_action_input and action is not None:
            encoded_action = self.encode_action(action[None], action_fps=action_fps)
            act_ids = encoded_action['input_ids']
            act_mask = encoded_action['attention_mask']

            subtask_length = 0 if sub_task_ids is None else int(sub_task_ids.shape[0])
            prompt_length = int(prefix_ids.shape[0]) + subtask_length
            if prompt_length + int(act_ids.shape[0]) > self.max_length:
                print(
                    '[prompt_tokenizer] skipping FAST action target that would overflow prompt: '
                    f'prompt_length={prompt_length}, action_length={int(act_ids.shape[0])}, max_length={self.max_length};'
                    f'{_format_prompt_context(sample_context)} task_preview={_preview_text(task)!r}',
                    flush=True,
                )
                act_ids = None
                act_mask = None

        suffix_length = 0
        if sub_task_ids is not None:
            suffix_length += int(sub_task_ids.shape[0])
        if act_ids is not None:
            suffix_length += int(act_ids.shape[0])

        prefix_length_before_truncation = int(prefix_ids.shape[0])
        prefix_budget = self.max_length if suffix_length == 0 else max(1, self.max_length - suffix_length)
        if prefix_length_before_truncation > prefix_budget:
            print(
                '[prompt_tokenizer] truncating long prefix: '
                f'prefix_length={prefix_length_before_truncation}, prefix_budget={prefix_budget}, '
                f'max_length={self.max_length}, suffix_length={suffix_length};'
                f'{_format_prompt_context(sample_context)} task_preview={_preview_text(task)!r}',
                flush=True,
            )
            if getattr(self, 'state_input_mode', 'prompt') == 'proprio_anchor':
                anchor_positions = torch.nonzero(prefix_ids == self.propri_token_id, as_tuple=False).flatten()
                if anchor_positions.numel() != 1:
                    raise ValueError(f'Expected one {PROPRI_TOKEN} before prefix truncation, ' f'got {int(anchor_positions.numel())}')
                anchor_index = int(anchor_positions.item())
                if anchor_index >= prefix_budget:
                    kept_prefix_length = prefix_budget - 1
                    prefix_ids = torch.cat(
                        [
                            prefix_ids[:kept_prefix_length],
                            prefix_ids[anchor_index : anchor_index + 1],
                        ]
                    )
                    prefix_mask = torch.cat(
                        [
                            prefix_mask[:kept_prefix_length],
                            prefix_mask[anchor_index : anchor_index + 1],
                        ]
                    )
                else:
                    prefix_ids = prefix_ids[:prefix_budget]
                    prefix_mask = prefix_mask[:prefix_budget]
            else:
                prefix_ids = prefix_ids[:prefix_budget]
                prefix_mask = prefix_mask[:prefix_budget]

        prefix_length = len(prefix_ids)
        fast_action_indicator = torch.zeros(prefix_length, dtype=torch.int32)
        subtask_indicator = torch.zeros(prefix_length, dtype=torch.int32)
        loss_indicator = torch.zeros(prefix_length, dtype=torch.int32)

        final_ids = prefix_ids
        final_mask = prefix_mask
        # Situation 1: Predict subtask
        if sub_task_ids is not None and sub_task_mask is not None:
            final_ids = torch.cat([final_ids, sub_task_ids], dim=0)
            final_mask = torch.cat([final_mask, sub_task_mask], dim=0)
            fast_action_indicator = torch.cat([fast_action_indicator, torch.zeros_like(sub_task_mask)], dim=0)
            subtask_indicator = torch.cat([subtask_indicator, torch.ones_like(sub_task_mask)], dim=0)
            loss_indicator = torch.cat([loss_indicator, sub_task_mask.to(dtype=torch.int32)], dim=0)

        # Situation 2: Predict discrete action
        if act_ids is not None and act_mask is not None:
            final_ids = torch.cat([final_ids, act_ids], dim=0)
            final_mask = torch.cat([final_mask, act_mask], dim=0)
            fast_action_indicator = torch.cat([fast_action_indicator, torch.ones_like(act_mask)], dim=0)
            subtask_indicator = torch.cat([subtask_indicator, torch.zeros_like(act_mask)], dim=0)
            action_loss_indicator = act_mask.to(dtype=torch.int32).clone()
            action_prefix = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')
            action_prefix_len = int(action_prefix['input_ids'].numel())
            action_loss_indicator[:action_prefix_len] = 0
            loss_indicator = torch.cat([loss_indicator, action_loss_indicator], dim=0)

        if final_ids.shape[0] > self.max_length and not self.autoregressive_inference_mode:
            print(
                '[prompt_tokenizer] truncating long token sequence: '
                f'sequence_length={int(final_ids.shape[0])}, max_length={self.max_length};'
                f'{_format_prompt_context(sample_context)} task_preview={_preview_text(task)!r}',
                flush=True,
            )
            final_ids = final_ids[: self.max_length]
            final_mask = final_mask[: self.max_length]
            fast_action_indicator = fast_action_indicator[: self.max_length]
            subtask_indicator = subtask_indicator[: self.max_length]
            loss_indicator = loss_indicator[: self.max_length]

        batch_inputs = {
            'input_ids': final_ids.tolist(),
            'attention_mask': final_mask.tolist(),
        }
        # Padding and set loss mask
        padding_side = 'left' if self.autoregressive_inference_mode else 'right'
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs, padding='max_length', padding_side=padding_side, max_length=self.max_length, return_tensors='pt'
        )
        final_ids = padded_output['input_ids']
        padded_mask = padded_output['attention_mask']

        att_mask = (padded_mask != 0).cumsum(dim=0) > prefix_length
        att_mask = att_mask & padded_mask

        pad_len = self.max_length - fast_action_indicator.shape[0]
        fast_action_indicator = F.pad(fast_action_indicator, (0, pad_len), mode='constant', value=0)
        subtask_indicator = F.pad(subtask_indicator, (0, pad_len), mode='constant', value=0)
        loss_indicator = F.pad(loss_indicator, (0, pad_len), mode='constant', value=0)
        loss_mask = loss_indicator.to(device=padded_mask.device, dtype=torch.bool) & padded_mask

        if getattr(self, 'state_input_mode', 'prompt') == 'proprio_anchor':
            counts = (final_ids == self.propri_token_id).sum()
            if int(counts.item()) != 1:
                raise ValueError(f'Expected one {PROPRI_TOKEN} after tokenization/padding, got {int(counts.item())}')

        return (
            final_ids.to(dtype=torch.int32, device=self.device),
            padded_mask.to(dtype=torch.bool, device=self.device),
            att_mask.to(dtype=torch.bool, device=self.device),
            loss_mask.to(dtype=torch.bool, device=self.device),
            fast_action_indicator.to(dtype=torch.bool, device=self.device),
            subtask_indicator.to(dtype=torch.bool, device=self.device),
            predict_subtask_only,
        )

    def _format_subtask_inference_text(
        self,
        task: Any,
        state: torch.Tensor | None,
        control_mode: str | None,
        end_effector_type: str | None,
        embodiment_id: int | None,
        sample_context: dict[str, Any] | None,
    ) -> str:
        task = resolve_task_text(task, sample_context)
        main_task, _ = split_task_and_subtask(task)
        control_mode_text = '' if control_mode is None else f', Control mode: {control_mode}'
        end_effector_text = '' if end_effector_type is None else f', End effector: {end_effector_type}'

        if getattr(self, 'state_input_mode', 'prompt') == 'proprio_anchor':
            return f'Task: {main_task}{control_mode_text}{end_effector_text}, ' f'State: {PROPRI_TOKEN};\n'

        if not self.discrete_state_input or self._skip_discrete_state_in_prompt(embodiment_id) or state is None:
            return f'Task: {main_task}{control_mode_text}{end_effector_text}\n'

        bins = torch.linspace(-1, 1, 256 + 1, device=self.device)[:-1]
        discretized = torch.bucketize(state, bins) - 1
        state_str = ' '.join(str(val.item()) for val in discretized)
        return f'Task: {main_task}{control_mode_text}{end_effector_text}, State: {state_str};\n'

    def create_subtask_inference_tokens(
        self, data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Build a prompt that asks the language head to generate the current
        subtask from the task/state context.

        This intentionally omits any ground-truth subtask from ``task`` and
        stops right before the training-time subtask target segment. The first
        generated tokens should therefore be the learned ``Subtask: ...`` text.
        """
        if 'task' not in data:
            raise ValueError('No task found in data')

        task = data['task']
        state = data.get('observation.state', None)
        embodiment_id = data.get('embodiment_id')
        control_mode = self._resolve_control_mode(data)
        end_effector_type = self._resolve_end_effector_type(data)
        text = self._format_subtask_inference_text(
            task,
            state,
            control_mode,
            end_effector_type,
            embodiment_id,
            data,
        )

        prefix_out = self.paligemma_tokenizer(
            [text],
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        prefix_ids = prefix_out['input_ids'][0]
        prefix_mask = prefix_out['attention_mask'][0]
        if prefix_ids.shape[0] > self.max_length:
            print(
                '[prompt_tokenizer] truncating long subtask-inference prefix: '
                f'prefix_length={int(prefix_ids.shape[0])}, max_length={self.max_length};'
                f'{_format_prompt_context(data)} task_preview={_preview_text(task)!r}',
                flush=True,
            )
            prefix_ids = prefix_ids[: self.max_length]
            prefix_mask = prefix_mask[: self.max_length]

        batch_inputs = {
            'input_ids': prefix_ids.tolist(),
            'attention_mask': prefix_mask.tolist(),
        }
        padding_side = 'left' if self.autoregressive_inference_mode else 'right'
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs,
            padding='max_length',
            padding_side=padding_side,
            max_length=self.max_length,
            return_tensors='pt',
        )
        final_ids = padded_output['input_ids'].squeeze(0)
        padded_mask = padded_output['attention_mask'].squeeze(0)
        zeros = torch.zeros(self.max_length, dtype=torch.bool)
        return (
            final_ids.to(dtype=torch.int32, device=self.device),
            padded_mask.to(dtype=torch.bool, device=self.device),
            zeros.to(device=self.device),
            zeros.to(device=self.device),
            zeros.to(device=self.device),
            zeros.to(device=self.device),
            True,
        )

    def __call__(self, data: dict) -> dict:
        """Applies the full prompt tokenization pipeline to a data dictionary.

        Args:
            data: A dictionary containing 'task', optionally 'observation.state' and 'action'.

        Returns:
            The output of `create_input_tokens`.
        """
        if 'task' not in data:
            raise ValueError('No task found in data')

        task = data['task']
        state = data.get('observation.state', None)
        action = data.get('action', None)
        embodiment_id = data.get('embodiment_id')
        control_mode = self._resolve_control_mode(data)
        end_effector_type = self._resolve_end_effector_type(data)
        image_grid_thw = data.get('image_grid_thw', None)
        action_fps = data.get('action_fps')

        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        return self.create_input_tokens(
            task,
            state,
            action,
            control_mode=control_mode,
            end_effector_type=end_effector_type,
            embodiment_id=embodiment_id,
            sample_context=data,
            image_grid_thw=image_grid_thw,
            action_fps=action_fps,
        )

    def extract_actions(self, tokens: list[list[int]], action_horizon: int, action_dim: int) -> torch.Tensor:
        """Extract continuous actions from predicted FAST action tokens.

        Args:
            tokens: The predicted FAST action tokens (PaliGemma token IDs).
            action_horizon: The time horizon (number of time steps) for the continuous action.
                This should match the horizon used during encoding.
            action_dim: The dimension of the continuous action vector.

        Returns:
            The extracted continuous actions as a tensor of shape (1, action_horizon, action_dim).

        Raises:
            ValueError: If the generated token sequence cannot be decoded.
        """
        assert len(tokens) == 1, 'Only support batch size 1'
        sequence = tokens[0].tolist() if hasattr(tokens[0], 'tolist') else list(tokens[0])

        bos_tokens = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0).tolist()
        eos_tokens = self.paligemma_tokenizer('|<eos>', add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0).tolist()

        def find_subsequence(sequence_ids: list[int], pattern: list[int], start: int = 0) -> int:
            if not pattern:
                return -1
            max_start = len(sequence_ids) - len(pattern)
            for idx in range(start, max_start + 1):
                if sequence_ids[idx : idx + len(pattern)] == pattern:
                    return idx
            return -1

        bos_idx = find_subsequence(sequence, bos_tokens)
        if bos_idx == -1:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        action_start = bos_idx + len(bos_tokens)
        eos_idx = find_subsequence(sequence, eos_tokens, start=action_start)
        if eos_idx == -1:
            eos_idx = len(sequence)

        paligemma_action_ids = sequence[action_start:eos_idx]

        if not paligemma_action_ids:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)
        base_token_id = vocab_size - 1 - self.fast_skip_tokens

        fast_tokens: list[int] = []
        for paligemma_id in paligemma_action_ids:
            if paligemma_id > base_token_id:
                continue
            fast_id = base_token_id - paligemma_id
            if fast_id < 0:
                continue
            fast_tokens.append(int(fast_id))

        if not fast_tokens:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        decoded_actions = self.fast_tokenizer.decode(
            [fast_tokens],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )
        actions = torch.tensor(decoded_actions, dtype=torch.float32)
        return actions


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Pick the next token id from per-step logits.

    Shared by the deployment pipeline (``generate_autoregressive_tokens``) and the
    offline rollout scripts so greedy / temperature / top-k / top-p decoding stay
    consistent.

    Args:
        logits: (B, vocab) next-token logits for a single decode step.
        temperature: <= 0 means greedy (argmax). Otherwise scales logits before sampling.
        top_k: keep only the top-k logits (0 = disabled).
        top_p: nucleus sampling — drop the tail beyond cumulative prob ``top_p``
            (1.0 = disabled). The top-1 token is always kept.

    Returns:
        (B,) long tensor of sampled token ids.
    """
    if not temperature or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # always keep the top-1 token
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class SampleGenerator:
    """Generates random prompt format samples based on given ratios."""

    # New names are explicit about which fields are input prompt context and
    # which fields are autoregressive language targets.
    SAMPLE_BEHAVIORS: dict[str, tuple[bool, bool, bool]] = {
        'input_task': (False, False, False),
        'input_subtask': (True, False, False),
        'input_task_target_subtask': (True, True, False),
        'input_task_target_action': (False, False, True),
        'input_subtask_target_action': (True, False, True),
        'input_task_target_subtask_action': (True, True, True),
        'identity': (False, False, False),
    }

    LEGACY_SAMPLE_NAME_ALIASES: dict[str, str] = {
        'task_only': 'input_task',
        'task_with_subtask': 'input_subtask',
        'task_only_using_subtask_regression': 'input_task_target_subtask',
        'task_only_using_fast_regression': 'input_task_target_action',
        'task_with_subtask_using_fast_regression': 'input_subtask_target_action',
        'input_task_subtask': 'input_subtask',
        'input_task_subtask_target_action': 'input_subtask_target_action',
        'subtask_and_fast_regression': 'input_task_target_subtask_action',
    }

    def __init__(self, sample_ratios: dict[str, float]):
        """Initializes the sample generator.

        Args:
            sample_ratios: A dictionary mapping prompt format names to their
                sampling ratios. The sum of ratios should be 1.0.
        """
        sample_ratios = self._normalize_sample_ratios(sample_ratios)

        valid_sample_names = list(self.SAMPLE_BEHAVIORS.keys())
        assert all(
            sample_name in valid_sample_names for sample_name in sample_ratios.keys()
        ), f'sample_name should be one of {valid_sample_names}, got {sample_ratios.keys()}'
        assert all(
            sample_ratio >= 0 for sample_ratio in sample_ratios.values()
        ), f'sample_ratio should be greater than or equal to 0, got {sample_ratios.values()}'
        assert all(
            sample_ratio <= 1 for sample_ratio in sample_ratios.values()
        ), f'sample_ratio should be less than or equal to 1, got {sample_ratios.values()}'
        # sum of sample_ratios should be 1
        if 'identity' not in sample_ratios:
            sample_ratios['identity'] = 1.0 - sum(sample_ratios.values())
        assert math.isclose(sum(sample_ratios.values()), 1.0, abs_tol=1e-6), f'sum of sample_ratios should be 1, got {sum(sample_ratios.values())}'
        self.sample_ratios = sample_ratios

    @classmethod
    def _normalize_sample_name(cls, sample_name: str) -> str:
        return cls.LEGACY_SAMPLE_NAME_ALIASES.get(sample_name, sample_name)

    @classmethod
    def _normalize_sample_ratios(cls, sample_ratios: dict[str, float]) -> dict[str, float]:
        normalized_ratios: dict[str, float] = {}
        for raw_name, raw_ratio in sample_ratios.items():
            sample_name = cls._normalize_sample_name(str(raw_name))
            normalized_ratios[sample_name] = normalized_ratios.get(sample_name, 0.0) + float(raw_ratio)
        return normalized_ratios

    def get_sample(self) -> tuple[bool, bool, bool]:
        """Randomly selects a prompt format based on the configured ratios.

        Returns:
            A tuple of booleans: (encode_sub_task_input, is_sub_task_train, encode_action_input).
        """
        sample_type = random.random()
        sample_name = None
        prob_acc = 0.0
        for name, sample_ratio in self.sample_ratios.items():
            prob_acc += sample_ratio
            if sample_type < prob_acc:
                sample_name = name
                break

        if sample_name is None:
            sample_name = 'identity'

        return self.SAMPLE_BEHAVIORS[sample_name]

    def subtask_sample_ratios(self) -> dict[str, float]:
        """Return nonzero ratios for modes that use subtask as input or
        target."""
        return {name: ratio for name, ratio in self.sample_ratios.items() if ratio > 0 and self.SAMPLE_BEHAVIORS[name][0]}
