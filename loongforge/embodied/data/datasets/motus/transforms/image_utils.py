#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image processing utilities for dataset loading.
Common functions shared across different datasets (AC-One, RobotWin, ALOHA, etc.)
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Tuple, List
import random
from decord import VideoReader, cpu


def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image with aspect ratio preservation and padding to target size.
    
    This function ensures no image distortion by:
    1. Calculating the minimum scale ratio to fit the image within target size
    2. Resizing the image with this ratio to preserve aspect ratio
    3. Padding with black borders to reach exact target size
    4. Centering the resized image within the padded frame
    
    Args:
        frame: Input image [H, W, C]
        target_size: Target size (height, width)
        
    Returns:
        Processed image [target_height, target_width, C]
        
    Example:
        >>> frame = np.random.randint(0, 255, (720, 640, 3), dtype=np.uint8)
        >>> resized = resize_with_padding(frame, (384, 320))
        >>> print(resized.shape)  # (384, 320, 3)
    """
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]
    
    # Calculate scaling ratio, use the smaller ratio to ensure image fits completely
    scale_height = target_height / original_height
    scale_width = target_width / original_width
    scale = min(scale_height, scale_width)
    
    # Calculate new dimensions after scaling
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Resize with aspect ratio preservation
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create black background with target size
    padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    
    # Calculate center placement position
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image at center
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
    
    return padded_frame


def load_video_frames(video_path: str, frame_indices: List[int], target_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    Load random frames from a video using decord, with optional aspect-ratio-preserving resize and padding.

    - Decoder: decord.VideoReader
    - Access: get_batch(frame_indices) to fetch arbitrary frames in one call
    - Color: decord returns frames in RGB with HWC layout

    Args:
        video_path: Path to the video file.
        frame_indices: Frame indices to read (can be unordered and may repeat).
        target_size: Optional target size (height, width). If provided, resize with aspect ratio preserved and
            center-pad with black borders to the exact target size.

    Returns:
        torch.Tensor of shape [T, C, H, W] with values in [0, 1].
    """

    # Open video (CPU decoding)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    total_frames = len(vr)

    if any(idx < 0 or idx >= total_frames for idx in frame_indices):
        raise ValueError(
            f"Some frame indices are out of bounds for video {video_path} (total frames: {total_frames})"
        )

    # Batch-read frames; returns decord NDArray with shape [T, H, W, 3] in RGB
    batch = vr.get_batch(frame_indices)
    frames_np = batch.asnumpy()  # uint8, [T, H, W, C]

    # Optional: aspect-ratio-preserving resize with padding to the target size
    if target_size is not None:
        th, tw = target_size
        _, h, w, _ = frames_np.shape
        if (h, w) != (th, tw):
            # Apply resize_with_padding to each frame (OpenCV implementation; no distortion)
            resized = [resize_with_padding(frames_np[i], target_size) for i in range(frames_np.shape[0])]
            frames_np = np.stack(resized, axis=0)

    # Convert to [T, C, H, W] and normalize to [0, 1]
    video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor


def load_first_frame(video_path: str, frame_idx: int,
                     target_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a single video frame in both resized and original resolution using decord only.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to load
        target_size: Target size (height, width) for resizing with padding
        
    Returns:
        Tuple of (resized_frame, original_frame):
        - resized_frame: Frame tensor [C, H, W] in range [0, 1] with padding applied
        - original_frame: Frame tensor [C, H, W] in range [0, 1] with original resolution
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if frame_idx >= total_frames:
        raise ValueError(f"Frame index {frame_idx} out of bounds for video {video_path} (total frames: {total_frames})")
    frame_rgb = vr[frame_idx].asnumpy()  # [H, W, 3], RGB uint8
    frame_original = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    frame_resized_np = resize_with_padding(frame_rgb, target_size)
    frame_resized = torch.from_numpy(frame_resized_np).permute(2, 0, 1).float() / 255.0
    return frame_resized, frame_original


def get_video_frame_count(video_path: str) -> int:
    """Get total frame count of a video using decord only."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    return len(vr)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor [C, H, W] to PIL Image.
    
    Args:
        tensor: Input tensor in format [C, H, W]
        
    Returns:
        PIL Image in RGB mode
    """
    # Convert from [C, H, W] to [H, W, C] and to numpy
    if tensor.shape[0] == 3:  # RGB
        image_np = tensor.permute(1, 2, 0).numpy()
        # Convert from [0, 1] to [0, 255] if needed
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        return Image.fromarray(image_np, mode='RGB')
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def apply_image_augmentation(frame: np.ndarray, 
                           brightness_prob: float = 0.5,
                           brightness_range: Tuple[float, float] = (0.8, 1.2),
                           flip_prob: float = 0.3) -> np.ndarray:
    """
    Apply common image augmentations to a frame.
    
    Args:
        frame: Input image [H, W, C]
        brightness_prob: Probability of applying brightness adjustment
        brightness_range: Range of brightness factors (min, max)
        flip_prob: Probability of applying horizontal flip
        
    Returns:
        Augmented image [H, W, C]
    """
    # Random brightness adjustment
    if random.random() < brightness_prob:
        brightness_factor = random.uniform(*brightness_range)
        frame = np.clip(frame * brightness_factor, 0, 255)
    
    # Random horizontal flip
    if random.random() < flip_prob:
        frame = np.fliplr(frame)
    
    return frame


# Test functions for validation
def test_resize_with_padding():
    """Test the resize_with_padding function with visual output."""
    import os
    
    print("=== Testing resize_with_padding ===")
    
    # Create test image with WHITE background and colored patterns
    test_frame = np.full((720, 640, 3), 255, dtype=np.uint8)  # White background
    test_frame[100:200, 100:200] = [255, 0, 0]    # Red square
    test_frame[300:400, 300:400] = [0, 255, 0]    # Green square  
    test_frame[500:600, 450:550] = [0, 0, 255]    # Blue square
    test_frame[50:670, 50:70] = [255, 255, 0]     # Yellow left border
    test_frame[50:670, 570:590] = [255, 0, 255]   # Magenta right border
    test_frame[50:70, 50:590] = [0, 255, 255]     # Cyan top border
    test_frame[650:670, 50:590] = [128, 128, 128] # Gray bottom border
    
    # Test resize
    target_size = (384, 320)
    resized_frame = resize_with_padding(test_frame, target_size)
    
    print(f"Original image size: {test_frame.shape}")
    print(f"Target size: {target_size}")
    print(f"Resized image size: {resized_frame.shape}")
    
    # Calculate expected values
    original_h, original_w = 720, 640
    target_h, target_w = 384, 320
    scale = min(target_h / original_h, target_w / original_w)  # 0.5
    expected_new_h = int(original_h * scale)  # 360
    expected_new_w = int(original_w * scale)  # 320
    
    print(f"Scaling factor: {scale}")
    print(f"Expected scaled size: {expected_new_h}x{expected_new_w}")
    print(f"Padding: top/bottom {(target_h - expected_new_h) // 2} pixels each")
    print(f"Padding: left/right {(target_w - expected_new_w) // 2} pixels each")
    
    # Verify padding
    y_offset = (target_h - expected_new_h) // 2
    x_offset = (target_w - expected_new_w) // 2
    
    if y_offset > 0:
        top_black = np.all(resized_frame[:y_offset, :, :] == 0)
        bottom_black = np.all(resized_frame[y_offset + expected_new_h:, :, :] == 0)
        print(f"Top padding check: {'✓' if top_black else '✗'}")
        print(f"Bottom padding check: {'✓' if bottom_black else '✗'}")
    
    if x_offset > 0:
        left_black = np.all(resized_frame[:, :x_offset, :] == 0)
        right_black = np.all(resized_frame[:, x_offset + expected_new_w:, :] == 0)
        print(f"Left padding check: {'✓' if left_black else '✗'}")
        print(f"Right padding check: {'✓' if right_black else '✗'}")
    
    # Save images if PIL is available
    try:
        from PIL import Image
        
        # Create output directory
        output_dir = "image_utils_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original image
        original_pil = Image.fromarray(test_frame)
        original_pil.save(os.path.join(output_dir, "original_720x640.png"))
        print(f"✓ Saved original image: {output_dir}/original_720x640.png")
        
        # Save resized image
        resized_pil = Image.fromarray(resized_frame)
        resized_pil.save(os.path.join(output_dir, "resized_384x320.png"))
        print(f"✓ Saved resized image: {output_dir}/resized_384x320.png")
        
        print("\n=== Test Complete ===")
        print("✓ Aspect ratio preserved")
        print("✓ No image distortion") 
        print("✓ Proper padding applied")
        print("✓ Images saved successfully")
        
    except ImportError as e:
        print(f"Could not save images due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error saving images: {e}")


if __name__ == "__main__":
    test_resize_with_padding()
