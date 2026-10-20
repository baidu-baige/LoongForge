import numpy as np
import torch

from . import points_utils


def get_width(boxes: np.ndarray | torch.Tensor, offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Width of boxes.

    Args:
        boxes: (..., 4) boxes [x1, y1, x2, y2].
        offset: Add-one offset for inclusive widths.

    Returns:
        Array/Tensor of widths.
    """
    return boxes[:, 2] - boxes[:, 0] + offset


def get_height(boxes: np.ndarray | torch.Tensor, offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Height of boxes.

    Args:
        boxes: (..., 4) [x1, y1, x2, y2].
        offset: Add-one offset for inclusive heights.

    Returns:
        Heights per box.
    """
    return boxes[:, 3] - boxes[:, 1] + offset


def get_size(boxes: np.ndarray | torch.Tensor, offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Area of boxes.

    Args:
        boxes: (..., 4) boxes.
        offset: Inclusive offset for width/height.

    Returns:
        Area per box.
    """
    return get_width(boxes, offset) * get_height(boxes, offset)


def get_dims(boxes: np.ndarray | torch.Tensor, offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Return (w, h) per box.

    Args:
        boxes: (..., 4) boxes.
        offset: Inclusive offset for width/height.

    Returns:
        (N, 2) dims.
    """
    return boxes[:, 2:4] - boxes[:, 0:2] + offset


def get_centers(boxes: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Centers (x, y) per box.

    Args:
        boxes: (..., 4) boxes.

    Returns:
        (N, 2) centers.
    """
    return (boxes[:, 0:2] + boxes[:, 2:4]) / 2


def dims_to_norm_corners(dims: np.ndarray | torch.Tensor, origin: float = 0.5) -> np.ndarray | torch.Tensor:
    """Corners from dimensions, centered at origin.

    Args:
        dims: (N, 2) widths and heights.
        origin: Center origin in [0, 1].

    Returns:
        (N, 4, 2) normalized corners scaled by dims.
    """
    norm_corners = np.unravel_index(np.arange(4), [2, 2])
    norm_corners = np.stack(norm_corners, axis=1)  # (4, 2)
    if isinstance(dims, torch.Tensor):
        norm_corners = dims.new_tensor(norm_corners)
    else:
        norm_corners = norm_corners.astype(dims.dtype)
    norm_corners = norm_corners[[0, 1, 3, 2]]  # (4, 2)
    norm_corners -= origin
    norm_corners = dims.reshape((-1, 1, 2)) * norm_corners.reshape((1, 4, 2))  # (N, 4, 2)
    return norm_corners


def boxes_to_corners(boxes: np.ndarray | torch.Tensor, offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Convert axis-aligned boxes to 4-corner representation.

    Args:
        boxes: (N, 4) boxes.
        offset: Inclusive offset for dims.

    Returns:
        (N, 4, 2) corners.
    """
    dims = get_dims(boxes, offset)  # (N, 2)
    centers = get_centers(boxes)  # (N, 2)
    corners = dims_to_norm_corners(dims)  # (N, 4, 2)
    corners += centers.reshape((-1, 1, 2))  # (N, 4, 2)
    return corners


def corners_to_boxes(corners: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Tight axis-aligned boxes from an arbitrary set of points.

    Args:
        corners: (N, K, 2) corners.

    Returns:
        (N, 4) minmax boxes.
    """
    if isinstance(corners, torch.Tensor):
        minxy = torch.min(corners, dim=1)[0]  # (N, 2)
        maxxy = torch.max(corners, dim=1)[0]  # (N, 2)
        boxes = torch.cat([minxy, maxxy], dim=1)  # (N, 4)
    else:
        minxy = np.min(corners, axis=1)  # (N, 2)
        maxxy = np.max(corners, axis=1)  # (N, 2)
        boxes = np.concatenate([minxy, maxxy], axis=1)  # (N, 4)
    return boxes


def boxes_to_masks(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize boxes into binary masks of shape (N, H, W).

    Args:
        boxes: (N, 4) boxes.
        height: Output mask height.
        width: Output mask width.

    Returns:
        (N, H, W) boolean masks.
    """
    masks = np.zeros((len(boxes), height, width), dtype=np.bool_)
    boxes = boxes.astype(np.int32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        masks[i, y1:y2, x1:x2] = True
    return masks


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Compute tight boxes around nonzero mask regions.

    Args:
        masks: (N, H, W) masks (bool/uint8).

    Returns:
        (N, 4) boxes.
    """
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    masks = masks.astype(np.uint8)
    for i, mask in enumerate(masks):
        if mask.max() > 0:
            x1x2 = np.where(mask.max(0) != 0)[0]
            y1y2 = np.where(mask.max(1) != 0)[0]
            y1, y2 = y1y2.min(), y1y2.max()
            x1, x2 = x1x2.min(), x1x2.max()
            boxes[i] = [x1, y1, x2, y2]
    return boxes


def corners_in_image(corners: np.ndarray | torch.Tensor, image_shape: tuple[int, int], method: int = 2) -> np.ndarray | torch.Tensor:
    """Check whether any corner center is inside the image.

    Args:
        corners: (N, 4, 2) corners.
        image_shape: (H, W) image size.
        method (int): 1 uses clipped box size>0; 2 checks any point in bounds.

    Returns:
        Bool mask indicating visibility.
    """
    if method == 1:
        boxes = corners_to_boxes(corners)
        boxes = clip_boxes(boxes, image_shape, offset=1)
        flag = get_size(boxes) > 0
        return flag.any(-1)
    elif method == 2:
        centers = corners.mean(1)
        image_h, image_w = image_shape[:2]
        if isinstance(corners, torch.Tensor):
            corners = torch.cat([corners, centers[:, None]], dim=1)
            visible = torch.logical_and(corners[..., 0] >= 0, corners[..., 1] >= 0)
            visible = torch.logical_and(visible, corners[..., 0] < image_w)
            visible = torch.logical_and(visible, corners[..., 1] < image_h)
        else:
            corners = np.concatenate([corners, centers[:, None]], axis=1)
            visible = np.logical_and(corners[..., 0] >= 0, corners[..., 1] >= 0)
            visible = np.logical_and(visible, corners[..., 0] < image_w)
            visible = np.logical_and(visible, corners[..., 1] < image_h)
        return visible.any(-1)
    else:
        assert False


def check_boxes(boxes: np.ndarray | torch.Tensor, image_shape: tuple[int, int], offset: int | float = 0) -> None:
    """Assert common validity constraints for boxes inside an image.

    Args:
        boxes: (N, 4) boxes.
        image_shape: (H, W).
        offset: Inclusive offset for max bounds.
    """
    if len(boxes) == 0:
        return
    image_h, image_w = image_shape[:2]
    if isinstance(boxes, torch.Tensor):
        flag = torch.logical_and(boxes[:, 0] >= 0, boxes[:, 2] >= 0)
        assert flag.all()
        flag = torch.logical_and(boxes[:, 1] >= 0, boxes[:, 3] >= 0)
        assert flag.all()
        flag = torch.logical_and(boxes[:, 0] < boxes[:, 2] + offset, boxes[:, 1] < boxes[:, 3] + offset)
        assert flag.all()
        flag = torch.logical_and(boxes[:, 0] <= image_w - 1, boxes[:, 2] <= image_w - offset)
        assert flag.all()
        flag = torch.logical_and(boxes[:, 1] <= image_h - 1, boxes[:, 3] <= image_h - offset)
        assert flag.all()
    else:
        flag = np.logical_and(boxes[:, 0] >= 0, boxes[:, 2] >= 0)
        assert flag.all()
        flag = np.logical_and(boxes[:, 1] >= 0, boxes[:, 3] >= 0)
        assert flag.all()
        flag = np.logical_and(boxes[:, 0] < boxes[:, 2] + offset, boxes[:, 1] < boxes[:, 3] + offset)
        assert flag.all()
        flag = np.logical_and(boxes[:, 0] <= image_w - 1, boxes[:, 2] <= image_w - offset)
        assert flag.all()
        flag = np.logical_and(boxes[:, 1] <= image_h - 1, boxes[:, 3] <= image_h - offset)
        assert flag.all()


def rotate_boxes(boxes: np.ndarray | torch.Tensor, rot_mat_t: np.ndarray | torch.Tensor, offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Rotate boxes by rotating their corners and re-enclosing.

    Args:
        boxes: (N, 4) boxes.
        rot_mat_t: (2,2) or (3,2) rotation matrix.
        offset: Inclusive offset when recomputing boxes.

    Returns:
        (N, 4) rotated boxes.
    """
    corners = boxes_to_corners(boxes, offset)  # (N, 4, 2)
    corners = points_utils.rotate_points(corners.reshape(-1, 2), rot_mat_t).reshape((-1, 4, 2))
    boxes = corners_to_boxes(corners)
    return boxes


def flip_boxes(
    boxes: np.ndarray | torch.Tensor,
    image_shape: tuple[int, int],
    direction: str = 'horizontal',
    offset: int | float = 0,
) -> np.ndarray | torch.Tensor:
    """Flip boxes horizontally or vertically inside an image frame."""
    image_h, image_w = image_shape[:2]
    if isinstance(boxes, torch.Tensor):
        new_boxes = boxes.clone()
    else:
        new_boxes = boxes.copy()
    if direction == 'horizontal':
        new_boxes[:, 0] = image_w - offset - boxes[:, 2]
        new_boxes[:, 2] = image_w - offset - boxes[:, 0]
    elif direction == 'vertical':
        new_boxes[:, 1] = image_h - offset - boxes[:, 3]
        new_boxes[:, 3] = image_h - offset - boxes[:, 1]
    else:
        assert False
    return new_boxes


def clip_boxes(boxes: np.ndarray | torch.Tensor, image_shape: tuple[int, int], offset: int | float = 0) -> np.ndarray | torch.Tensor:
    """Clamp boxes to image boundaries (inclusive with offset on max).

    Args:
        boxes: (N, 4) boxes.
        image_shape: (H, W).
        offset: Max side offset.

    Returns:
        Clipped boxes.
    """
    if len(boxes) == 0:
        return boxes
    image_h, image_w = image_shape[:2]
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=image_w - 1)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=image_h - 1)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=image_w - offset)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=image_h - offset)
    else:
        boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=image_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=image_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], a_min=0, a_max=image_w - offset)
        boxes[:, 3] = np.clip(boxes[:, 3], a_min=0, a_max=image_h - offset)
    return boxes


def crop_boxes(
    boxes: np.ndarray | torch.Tensor,
    crop_range: np.ndarray | torch.Tensor,
    keep_outside_center: bool = False,
    keep_outside_boxes: bool = False,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Crop boxes by a rectangle and optionally filter by center-inside policy.

    Args:
        boxes: (N, 4) boxes.
        crop_range: (x1,y1,x2,y2) rectangle.
        keep_outside_center: If True, do not filter by centers.
        keep_outside_boxes: If True, do not clamp boxes before filtering.

    Returns:
        (clipped_boxes, keep_indices)
    """
    if isinstance(boxes, torch.Tensor):
        crop_range = crop_range.type_as(boxes)
        if keep_outside_center:
            keep = torch.ones(boxes.shape[0], dtype=torch.bool)
        else:
            centers = get_centers(boxes)
            keep = torch.logical_and(crop_range[:2] <= centers, centers <= crop_range[2:]).all(axis=1)
        if not keep_outside_boxes:
            boxes[:, 0:2] = torch.max(boxes[:, 0:2], crop_range[0:2])
            boxes[:, 2:4] = torch.min(boxes[:, 2:4], crop_range[2:4])
        boxes[:, :2] -= crop_range[:2]
        boxes[:, 2:4] -= crop_range[:2]
        keep = torch.logical_and(keep, (boxes[:, :2] < boxes[:, 2:4]).all(axis=1))
        keep = torch.nonzero(keep, as_tuple=False).view(-1)
        return boxes[keep], keep
    else:
        if keep_outside_center:
            keep = np.ones(boxes.shape[0], dtype=bool)
        else:
            centers = get_centers(boxes)
            keep = np.logical_and(crop_range[:2] <= centers, centers <= crop_range[2:]).all(axis=1)
        if not keep_outside_boxes:
            boxes[:, 0:2] = np.maximum(boxes[:, 0:2], crop_range[0:2])
            boxes[:, 2:4] = np.minimum(boxes[:, 2:4], crop_range[2:4])
        boxes[:, :2] -= crop_range[:2]
        boxes[:, 2:4] -= crop_range[:2]
        keep = np.logical_and(keep, (boxes[:, :2] < boxes[:, 2:4]).all(axis=1))
        keep = np.where(keep == 1)[0]
        return boxes[keep], keep


def scale_boxes(
    boxes: np.ndarray | torch.Tensor,
    scale_hw: tuple[float, float],
    image_shape: tuple[int, int] | None = None,
    offset: int | float = 0,
) -> np.ndarray | torch.Tensor:
    """Scale boxes by (scale_h, scale_w) with optional clipping to image.

    Args:
        boxes: (N, 4) boxes.
        scale_hw: (scale_h, scale_w) factors.
        image_shape: Optional (H, W) for clipping.
        offset: Inclusive offset when clipping.

    Returns:
        Scaled (and possibly clipped) boxes.
    """
    scale_h, scale_w = scale_hw
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    if image_shape is not None:
        boxes = clip_boxes(boxes, image_shape, offset)
    return boxes


def expand_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    scale_hw: tuple[float, float] | None = None,
    aspect_ratio: float | None = None,
) -> np.ndarray:
    """Expand boxes around center to meet target scale and/or aspect ratio.

    Args:
        boxes: (N, 4) boxes.
        image_shape: (H, W) for boundary clamp.
        scale_hw: Optional (scale_h, scale_w).
        aspect_ratio: Optional target h/w.

    Returns:
        Expanded boxes (N, 4).
    """
    image_h, image_w = image_shape[:2]
    boxes = boxes.copy()
    centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    dims = boxes[:, 2:4] - boxes[:, 0:2]
    if scale_hw is not None:
        scale_h, scale_w = scale_hw
        dims[:, 0] *= scale_w
        dims[:, 1] *= scale_h
    new_boxes = []
    for i in range(len(boxes)):
        ctr_x, ctr_y = centers[i, 0], centers[i, 1]
        w, h = dims[i, 0], dims[i, 1]
        while True:
            if aspect_ratio is not None:
                if aspect_ratio * w > h:
                    h = aspect_ratio * w
                else:
                    w = h / aspect_ratio
            x1 = ctr_x - w / 2
            y1 = ctr_y - h / 2
            x2 = ctr_x + w / 2
            y2 = ctr_y + h / 2
            new_x1 = max(min(x1, image_w - 1), 0)
            new_y1 = max(min(y1, image_h - 1), 0)
            new_x2 = max(min(x2, image_w - 1), 0)
            new_y2 = max(min(y2, image_h - 1), 0)
            new_ctr_x = (new_x1 + new_x2) / 2
            new_ctr_y = (new_y1 + new_y2) / 2
            if new_ctr_x == ctr_x and new_ctr_y == ctr_y:
                new_boxes.append([x1, y1, x2, y2])
                break
            else:
                ctr_x = new_ctr_x
                ctr_y = new_ctr_y
                w = new_x2 - new_x1
                h = new_y2 - new_y1
    new_boxes = np.array(new_boxes, dtype=boxes.dtype)
    return new_boxes


def points_in_boxes(points: np.ndarray | torch.Tensor, boxes: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return indicator matrix whether each point lies inside each box.

    Args:
        points: (P, 2) points.
        boxes: (N, 4) boxes.

    Returns:
        (P, N) indicator matrix.
    """
    if isinstance(points, torch.Tensor):
        indices = torch.zeros((points.shape[0], boxes.shape[0]), dtype=torch.int32)
    else:
        indices = np.zeros((points.shape[0], boxes.shape[0]), dtype=np.int32)
    for i in range(points.shape[0]):
        for j in range(boxes.shape[0]):
            if boxes[j, 0] <= points[i, 0] <= boxes[j, 2] and boxes[j, 1] <= points[i, 1] <= boxes[j, 3]:
                indices[i, j] = 1
    return indices


def overlaps(
    boxes_a: np.ndarray | torch.Tensor,
    boxes_b: np.ndarray | torch.Tensor,
    mode: str = 'iou',
    is_aligned: bool = False,
    offset: int | float = 0,
    eps: float = 1e-6,
) -> np.ndarray | torch.Tensor:
    """Pairwise overlaps/IoU/IoA/IoB/GIoU between boxes.

    Args:
        boxes_a: (n, 4) boxes.
        boxes_b: (m, 4) boxes.
        mode (str): 'iou' | 'ioa' | 'iob' | 'giou' | None.
        is_aligned (bool): If True, compare element-wise and return (n,).
        offset (float): Inclusive offset for W/H.
        eps (float): Numerical stability epsilon.

    Returns:
        (n, m) matrix or (n,) vector if aligned.
    """
    if is_aligned:
        assert boxes_a.shape[0] == boxes_b.shape[0]
    if boxes_a.shape[0] * boxes_b.shape[0] == 0:
        if is_aligned:
            if isinstance(boxes_a, torch.Tensor):
                return boxes_a.new_zeros((boxes_a.shape[0],))
            else:
                return np.zeros((boxes_a.shape[0],))
        else:
            if isinstance(boxes_a, torch.Tensor):
                return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
            else:
                return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    if is_aligned:
        if isinstance(boxes_a, torch.Tensor):
            lt = torch.max(boxes_a[:, :2], boxes_b[:, :2])  # (n, 2)
            rb = torch.min(boxes_a[:, 2:4], boxes_b[:, 2:4])  # (n, 2)
            wh = (rb - lt + offset).clamp(min=0)  # (n, 2)
        else:
            lt = np.maximum(boxes_a[:, :2], boxes_b[:, :2])  # (n, 2)
            rb = np.minimum(boxes_a[:, 2:4], boxes_b[:, 2:4])  # (n, 2)
            wh = np.maximum(rb - lt + offset, 0)  # (n, 2)
        overlap = wh[:, 0] * wh[:, 1]  # (n,)
        if mode is None:
            return overlap
        if mode in ('iou', 'ioa', 'giou'):
            area_a = get_size(boxes_a, offset=offset)  # (n,)
        if mode in ('iou', 'iob', 'giou'):
            area_b = get_size(boxes_b, offset=offset)  # (n,)
        if mode in ('iou', 'giou'):  # intersection over union
            union = area_a + area_b - overlap  # (n,)
        elif mode == 'ioa':  # intersection over a
            union = area_a  # (n,)
        elif mode == 'iob':  # intersection over b
            union = area_b  # (n,)
        else:
            assert False
        if mode == 'giou':
            if isinstance(boxes_a, torch.Tensor):
                enclosed_lt = torch.min(boxes_a[:, :2], boxes_b[:, :2])  # (n, 2)
                enclosed_rb = torch.max(boxes_a[:, 2:4], boxes_b[:, 2:4])  # (n, 2)
                enclosed_wh = (enclosed_rb - enclosed_lt + offset).clamp(min=0)  # (n, 2)
            else:
                enclosed_lt = np.minimum(boxes_a[:, :2], boxes_b[:, :2])  # (n, 2)
                enclosed_rb = np.maximum(boxes_a[:, 2:4], boxes_b[:, 2:4])  # (n, 2)
                enclosed_wh = np.maximum(enclosed_rb - enclosed_lt + offset, 0)  # (n, 2)
            enclosed_area = enclosed_wh[:, 0] * enclosed_wh[:, 1]  # (n,)
    else:
        if isinstance(boxes_a, torch.Tensor):
            lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (n, m, 2)
            rb = torch.min(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])  # (n, m, 2)
            wh = (rb - lt + offset).clamp(min=0)  # (n, m, 2)
        else:
            lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (n, m, 2)
            rb = np.minimum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])  # (n, m, 2)
            wh = np.maximum(rb - lt + offset, 0)  # (n, m, 2)
        overlap = wh[:, :, 0] * wh[:, :, 1]  # (n, m)
        if mode is None:
            return overlap
        if mode in ('iou', 'ioa', 'giou'):
            area_a = get_size(boxes_a, offset=offset)  # (n, )
        if mode in ('iou', 'iob', 'giou'):
            area_b = get_size(boxes_b, offset=offset)  # (m, )
        if mode in ('iou', 'giou'):  # intersection over union
            union = area_a[:, None] + area_b[None, :] - overlap  # (n, m)
        elif mode == 'ioa':  # intersection over a
            union = area_a[:, None]  # (n, m)
        elif mode == 'iob':  # intersection over b
            union = area_b[None, :]  # (n, m)
        else:
            assert False
        if mode == 'giou':
            if isinstance(boxes_a, torch.Tensor):
                enclosed_lt = torch.min(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (n, m, 2)
                enclosed_rb = torch.max(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])  # (n, m, 2)
                enclosed_wh = (enclosed_rb - enclosed_lt + offset).clamp(min=0)
            else:
                enclosed_lt = np.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (n, m, 2)
                enclosed_rb = np.maximum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])  # (n, m, 2)
                enclosed_wh = np.maximum(enclosed_rb - enclosed_lt + offset, 0)  # (n, m, 2)
            enclosed_area = enclosed_wh[:, :, 0] * enclosed_wh[:, :, 1]  # (n, m)
    if isinstance(boxes_a, torch.Tensor):
        union = union.clamp(min=eps)
    else:
        union = np.maximum(union, eps)
    ios = overlap / union
    if mode == 'giou':
        if isinstance(boxes_a, torch.Tensor):
            enclosed_area = enclosed_area.clamp(min=eps)
        else:
            enclosed_area = np.maximum(enclosed_area, eps)
        giou = ios - (enclosed_area - union) / enclosed_area
        return giou
    else:
        return ios
