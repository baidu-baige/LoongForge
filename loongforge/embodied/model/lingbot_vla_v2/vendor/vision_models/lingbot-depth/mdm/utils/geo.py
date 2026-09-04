"""Vendor module implementation."""
import torch


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
     device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) "
    "and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / \
                       height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


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


# Usage example
if __name__ == "__main__":
    # Single image
    depth = torch.rand(480, 640) * 5000  # Depth values
    intrinsic_norm = torch.tensor([
        [525.0 / 640, 0, 319.5 / 640],
        [0, 525.0 / 480, 239.5 / 480],
        [0, 0, 1]
    ])

    points = depth_to_pointcloud(depth, intrinsic_norm)
    print(f"Point cloud shape: {points.shape}")  # (480, 640, 3)

    # Batch processing
    depth_batch = torch.rand(4, 480, 640) * 5000
    intrinsic_batch = intrinsic_norm.unsqueeze(0).expand(4, -1, -1)

    points_batch = depth_to_pointcloud(depth_batch, intrinsic_batch)
    print(f"Batch point cloud shape: {points_batch.shape}")  # (4, 480, 640, 3)

    # Flatten to (N, 3) format if needed
    points_flat = points.reshape(-1, 3)
    print(f"Flattened shape: {points_flat.shape}")  # (480*640, 3)

    # Batch flatten to (B, N, 3)
    points_batch_flat = points_batch.reshape(4, -1, 3)
    print(f"Batch flattened shape: {points_batch_flat.shape}")  # (4, 480*640, 3)