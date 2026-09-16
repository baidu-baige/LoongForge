"""Vendor module implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_module_with_gradient_checkpointing(module: nn.Module):
    """Vendor interface."""
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        """Vendor interface."""
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            """Vendor interface."""
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    """Vendor interface."""
    module.__class__ = module.__class__._restore_cls


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    """Vendor interface."""
    assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
    class _AttentionWrapper(module.__class__):
        """Vendor interface."""

        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            """Vendor interface."""
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                           self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    module.__class__ = _AttentionWrapper
    return module


def wrap_dinov3_attention_with_sdpa(module: nn.Module):
    """Vendor interface."""
    assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
    class _AttentionWrapper(module.__class__):
        """Vendor interface."""

        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            """Vendor interface."""
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                           self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)      # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    module.__class__ = _AttentionWrapper
    return module


def sync_ddp_hook(state, bucket: torch.distributed.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """Vendor interface."""
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    torch.distributed.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut


def depth_to_pointcloud(depth, intrinsic_normalized, depth_scale=1.0):
    """
    Convert depth map to point cloud (pure Tensor version, no point filtering)

    Args:
        depth: torch.Tensor, shape (H, W) or (B, H, W), depth map
        intrinsic_normalized: torch.Tensor, shape (3, 3) or (B, 3, 3), normalized intrinsic matrix
            Normalized intrinsics: fx' = fx/W, fy' = fy/H, cx' = cx/W, cy' = cy/H
        depth_scale: float, depth scale factor, default 1000.0

    Returns:
        points: torch.Tensor, shape (H, W, 3) or (B, H, W, 3), point cloud coordinates (x, y, z)
    """
    # Handle batch dimension
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # (1, H, W)
        intrinsic_normalized = intrinsic_normalized.unsqueeze(0)  # (1, 3, 3)
        squeeze_output = True
    else:
        squeeze_output = False

    B, H, W = depth.shape
    device = depth.device

    # Denormalize intrinsics
    fx = intrinsic_normalized[:, 0, 0] * W  # (B,)
    fy = intrinsic_normalized[:, 1, 1] * H
    cx = intrinsic_normalized[:, 0, 2] * W
    cy = intrinsic_normalized[:, 1, 2] * H

    # Create pixel coordinate grid (H, W)
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Expand to batch dimension (B, H, W)
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    # Backproject to 3D space
    z = depth / depth_scale  # (B, H, W)

    # Expand intrinsic dimensions for broadcasting (B, 1, 1)
    fx = fx.view(B, 1, 1)
    fy = fy.view(B, 1, 1)
    cx = cx.view(B, 1, 1)
    cy = cy.view(B, 1, 1)

    x = (u - cx) * z / fx  # (B, H, W)
    y = (v - cy) * z / fy  # (B, H, W)

    # Stack coordinates (B, H, W, 3)
    points = torch.stack([x, y, z], dim=-1)

    if squeeze_output:
        points = points.squeeze(0)  # (H, W, 3)

    return points