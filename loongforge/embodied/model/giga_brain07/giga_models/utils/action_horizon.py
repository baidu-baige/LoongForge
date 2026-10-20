import torch
from torch import Tensor

FPS_15HZ_ATOL = 0.5


def resolve_flow_action_steps(n_action_steps: int, flow_action_horizon: int | None) -> tuple[int, int, int | None]:
    raw_action_steps = int(n_action_steps)
    if raw_action_steps < 1:
        raise ValueError(f'n_action_steps must be positive, got {n_action_steps}')

    if flow_action_horizon is None:
        return raw_action_steps, raw_action_steps, None

    flow_action_horizon = int(flow_action_horizon)
    if flow_action_horizon < 1:
        raise ValueError(f'flow_action_horizon must be positive, got {flow_action_horizon}')
    if flow_action_horizon > raw_action_steps:
        raise ValueError(f'flow_action_horizon must be <= n_action_steps, got {flow_action_horizon} > {raw_action_steps}')
    if raw_action_steps % flow_action_horizon != 0:
        raise ValueError(f'n_action_steps must be divisible by flow_action_horizon, got ' f'{raw_action_steps} and {flow_action_horizon}')
    return raw_action_steps, flow_action_horizon, flow_action_horizon


def _coerce_action_fps_tensor(
    action_fps: object,
    *,
    device: torch.device | str | None,
) -> Tensor | None:
    if action_fps is None:
        return None
    try:
        return torch.as_tensor(action_fps, dtype=torch.float32, device=device).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None


def _is_15hz_action_fps(action_fps: Tensor) -> Tensor:
    return torch.isfinite(action_fps) & (torch.abs(action_fps - 15.0) <= FPS_15HZ_ATOL)


def flow_action_horizon_indices(
    action_len: int,
    flow_action_horizon: int | None,
    *,
    action_fps: object = None,
    device: torch.device | str | None = None,
) -> Tensor | None:
    if flow_action_horizon is None:
        return None

    action_len = int(action_len)
    flow_action_horizon = int(flow_action_horizon)
    if flow_action_horizon < 1:
        raise ValueError(f'flow_action_horizon must be positive, got {flow_action_horizon}')
    if flow_action_horizon > action_len:
        raise ValueError(f'flow_action_horizon must be <= action length, got {flow_action_horizon} > {action_len}')
    if action_len % flow_action_horizon != 0:
        raise ValueError(f'action length must be divisible by flow_action_horizon, got {action_len} and {flow_action_horizon}')

    step = action_len // flow_action_horizon
    segment_end_indices = torch.arange(step - 1, action_len, step, device=device)

    fps_tensor = _coerce_action_fps_tensor(action_fps, device=device)
    if fps_tensor is None or fps_tensor.numel() == 0:
        return segment_end_indices

    first_indices = torch.arange(flow_action_horizon, device=device)
    is_15hz = _is_15hz_action_fps(fps_tensor)
    if fps_tensor.numel() == 1:
        return first_indices if bool(is_15hz.item()) else segment_end_indices

    return torch.where(
        is_15hz[:, None],
        first_indices[None, :].expand(fps_tensor.numel(), -1),
        segment_end_indices[None, :].expand(fps_tensor.numel(), -1),
    )


def _validate_action_mask_shapes(
    actions: Tensor,
    action_loss_mask: Tensor,
    action_dim_loss_mask: Tensor | None,
    original_time_len: int,
) -> None:
    if action_loss_mask.shape[1] != original_time_len:
        raise ValueError(
            f'action_loss_mask temporal length {action_loss_mask.shape[1]} does not match ' f'action temporal length {original_time_len}'
        )
    if action_loss_mask.shape[0] != actions.shape[0]:
        raise ValueError(f'action_loss_mask batch size {action_loss_mask.shape[0]} does not match ' f'action batch size {actions.shape[0]}')

    if action_dim_loss_mask is not None and action_dim_loss_mask.shape != actions.shape:
        raise ValueError(f'action_dim_loss_mask shape {tuple(action_dim_loss_mask.shape)} does not match ' f'action shape {tuple(actions.shape)}')


def downsample_flow_action_tensors(
    actions: Tensor,
    action_loss_mask: Tensor,
    flow_action_horizon: int | None,
    *,
    action_fps: object = None,
    action_dim_loss_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    original_time_len = actions.shape[1]
    indices = flow_action_horizon_indices(
        original_time_len,
        flow_action_horizon,
        action_fps=action_fps,
        device=actions.device,
    )
    if indices is None:
        if action_dim_loss_mask is not None:
            _validate_action_mask_shapes(actions, action_loss_mask, action_dim_loss_mask, original_time_len)
            return actions, action_loss_mask, action_dim_loss_mask
        return actions, action_loss_mask

    _validate_action_mask_shapes(actions, action_loss_mask, action_dim_loss_mask, original_time_len)

    if indices.ndim == 1:
        actions = actions.index_select(1, indices)
        action_loss_mask = action_loss_mask.index_select(1, indices.to(device=action_loss_mask.device))

        if action_dim_loss_mask is not None:
            action_dim_loss_mask = action_dim_loss_mask.index_select(1, indices.to(device=action_dim_loss_mask.device))
            return actions, action_loss_mask, action_dim_loss_mask
        return actions, action_loss_mask

    if indices.ndim != 2:
        raise ValueError(f'action horizon indices must have shape [T] or [B, T], got {tuple(indices.shape)}')
    if indices.shape[0] != actions.shape[0]:
        raise ValueError(f'action_fps batch size {indices.shape[0]} does not match action batch size {actions.shape[0]}')

    action_indices = indices[:, :, None].expand(-1, -1, actions.shape[2])
    actions = actions.gather(1, action_indices)
    action_loss_mask = action_loss_mask.gather(1, indices.to(device=action_loss_mask.device))

    if action_dim_loss_mask is not None:
        dim_indices = indices.to(device=action_dim_loss_mask.device)[:, :, None].expand(-1, -1, action_dim_loss_mask.shape[2])
        action_dim_loss_mask = action_dim_loss_mask.gather(1, dim_indices)
        return actions, action_loss_mask, action_dim_loss_mask
    return actions, action_loss_mask
