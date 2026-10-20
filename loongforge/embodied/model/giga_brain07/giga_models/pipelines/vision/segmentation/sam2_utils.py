import random

import matplotlib.colors as mcolors
import numpy as np

PREDEFINED_COLORS_SEGMENTATION = np.array(
    [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 140, 0],  # Dark Orange
        [255, 105, 180],  # Hot Pink
        [0, 0, 139],  # Dark Blue
        [0, 128, 128],  # Teal
        [75, 0, 130],  # Indigo
        [128, 0, 128],  # Purple
        [255, 69, 0],  # Red-Orange
        [34, 139, 34],  # Forest Green
        [128, 128, 0],  # Olive
        [70, 130, 180],  # Steel Blue
        [255, 215, 0],  # Gold
        [255, 222, 173],  # Navajo White
        [144, 238, 144],  # Light Green
        [255, 99, 71],  # Tomato
        [221, 160, 221],  # Plum
        [0, 255, 127],  # Spring Green
        [255, 255, 255],  # White
    ]
)


def generate_distinct_colors() -> np.ndarray:
    """Generate one visually distinguishable RGB color (uint8, shape (3,))."""
    # Randomize hue, saturation, and lightness within a range
    hue = random.uniform(0, 1)  # Full spectrum of hues
    saturation = random.uniform(0.1, 1)  # Vibrant colors
    lightness = random.uniform(0.2, 1.0)  # Avoid too dark

    r, g, b = mcolors.hsv_to_rgb((hue, saturation, lightness))
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def segmentation_color_mask(segmentation_mask: np.ndarray, use_fixed_color_list: bool = False) -> np.ndarray:
    """Convert stacked binary masks to an RGB color volume.

    Args:
        segmentation_mask: ndarray, shape (num_masks, T, H, W), binary masks.
        use_fixed_color_list: If True, use predefined color palette with random permutation.

    Returns:
        ndarray, shape (3, T, H, W) with values in [0, 255].
    """

    num_masks, T, H, W = segmentation_mask.shape
    segmentation_mask_sorted = [segmentation_mask[i] for i in range(num_masks)]
    # Sort the segmentation mask by the number of non-zero pixels, from most to least
    segmentation_mask_sorted = sorted(segmentation_mask_sorted, key=lambda x: np.count_nonzero(x), reverse=True)

    output = np.zeros((3, T, H, W), dtype=np.uint8)
    if use_fixed_color_list:
        predefined_colors_permuted = PREDEFINED_COLORS_SEGMENTATION[np.random.permutation(len(PREDEFINED_COLORS_SEGMENTATION))]
    else:
        predefined_colors_permuted = [generate_distinct_colors() for _ in range(num_masks)]
    # index the segmentation mask from last channel to first channel, i start from num_masks-1 to 0
    for i in range(num_masks):
        mask = segmentation_mask_sorted[i]
        color = predefined_colors_permuted[i % len(predefined_colors_permuted)]

        # Create boolean mask and use it for assignment
        bool_mask = mask > 0
        for c in range(3):
            output[c][bool_mask] = color[c]

    return output
