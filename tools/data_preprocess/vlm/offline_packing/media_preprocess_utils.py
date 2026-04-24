# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Media preprocessing utilities for images and videos."""

from PIL import Image

# ----------------- image_preprocess -----------------
def custom_image_preprocess(image_path, resolution_limit=1024):
    image = Image.open(image_path)
    if max(image.width, image.height) > resolution_limit:
        resize_factor = resolution_limit / max(image.width, image.height)
        new_width, new_height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((new_width, new_height), resample=Image.NEAREST)
    return image

# ----------------- video_preprocess -----------------