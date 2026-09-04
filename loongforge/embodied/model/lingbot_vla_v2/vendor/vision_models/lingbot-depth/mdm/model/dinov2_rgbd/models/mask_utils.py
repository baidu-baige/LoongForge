import torch


def depth_masking(
    x,
    patch_num_h,
    patch_num_w,
    depth_values,
    depth_mask_threshold_ratio=None,
    depth_mask_threshold_num=None,
    valid_depth_range=(0.1, 10.0),
):
    """
    Perform patch masking based on depth validity

    Args:
        x: [B, N, D] input features (after patch embedding)
        patch_num_h: int, height of the patch grid
        patch_num_w: int, width of the patch grid
        depth_values: [B, 1, H_img, W_img], raw depth map
        depth_mask_threshold_ratio: float or list, valid depth ratio threshold (0-1)
        depth_mask_threshold_num: int or list, valid depth pixel count threshold
        valid_depth_range: tuple, valid depth range (min, max)

    Returns:
        visible_list: list of [N_visible_i, D], visible patches for each sample
        mask_info: dict, containing masking information
    """
    B, N, D = x.shape
    device = x.device

    assert N == patch_num_h * patch_num_w, \
        f"N={N} must equal patch_num_h * patch_num_w = {patch_num_h * patch_num_w}"

    # Compute depth invalid mask
    depth_invalid_mask = _compute_depth_invalid_mask(
        depth_values,
        patch_num_h,
        patch_num_w,
        depth_mask_threshold_ratio,
        depth_mask_threshold_num,
        valid_depth_range
    )  # [B, N], True indicates this patch is invalid

    # Process each sample separately
    visible_list = []
    mask_info = {
        'visible_indices': [],
        'mask_indices': [],
        'num_visible': [],
    }

    for i in range(B):
        # Get valid patch indices
        valid_mask = ~depth_invalid_mask[i]  # [N]
        visible_indices = torch.where(valid_mask)[0]
        masked_indices = torch.where(depth_invalid_mask[i])[0]

        # Extract visible patches
        visible = x[i, visible_indices]  # [N_visible, D]
        visible_list.append(visible)

        # Record information
        mask_info['visible_indices'].append(visible_indices)
        mask_info['mask_indices'].append(masked_indices)
        mask_info['num_visible'].append(len(visible_indices))

    return visible_list, mask_info


def _compute_depth_invalid_mask(
    depth_values,
    H_patch,
    W_patch,
    threshold_ratio,
    threshold_num,
    valid_range
):
    """
    Compute depth validity for each patch

    Args:
        depth_values: [B, 1, H_img, W_img] raw depth map
        H_patch, W_patch: patch grid dimensions
        threshold_ratio: float or list, valid depth ratio threshold
        threshold_num: int or list, valid depth pixel count threshold
        valid_range: tuple, (min_depth, max_depth)

    Returns:
        invalid_mask: [B, N] bool tensor, True indicates this patch is invalid
    """
    B, _, H_img, W_img = depth_values.shape
    N = H_patch * W_patch
    device = depth_values.device

    min_depth, max_depth = valid_range

    # Calculate pixel size for each patch
    patch_h = H_img // H_patch
    patch_w = W_img // W_patch

    assert H_img % H_patch == 0 and W_img % W_patch == 0, \
        f"Image size ({H_img}, {W_img}) must be divisible by patch grid ({H_patch}, {W_patch})"

    # Reshape depth map into patches: [B, 1, H_img, W_img] -> [B, H_patch, patch_h, W_patch, patch_w]
    depth_reshaped = depth_values.view(B, 1, H_patch, patch_h, W_patch, patch_w)

    # Transpose and flatten: [B, H_patch, W_patch, patch_h, patch_w] -> [B, N, patch_h*patch_w]
    depth_reshaped = depth_reshaped.permute(0, 2, 4, 1, 3, 5).reshape(B, N, -1)

    # Calculate valid depth
    valid_depth = (depth_reshaped >= min_depth) & (depth_reshaped <= max_depth)
    valid_depth_ratio = valid_depth.float().mean(dim=-1)  # [B, N]
    valid_depth_num = valid_depth.float().sum(dim=-1)  # [B, N]

    # Handle list-form thresholds (different thresholds for each sample in batch)
    if isinstance(threshold_ratio, list) or isinstance(threshold_num, list):
        invalid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for i in range(B):
            tr = threshold_ratio[i] if isinstance(threshold_ratio, list) else threshold_ratio
            tn = threshold_num[i] if isinstance(threshold_num, list) else threshold_num

            sample_mask = torch.zeros(N, dtype=torch.bool, device=device)
            if tr is not None:
                sample_mask |= (valid_depth_ratio[i] < tr)
            if tn is not None:
                sample_mask |= (valid_depth_num[i] < tn)

            invalid_mask[i] = sample_mask
    else:
        # Uniform threshold
        invalid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        if threshold_ratio is not None:
            invalid_mask |= (valid_depth_ratio < threshold_ratio)
        if threshold_num is not None:
            invalid_mask |= (valid_depth_num < threshold_num)

    return invalid_mask