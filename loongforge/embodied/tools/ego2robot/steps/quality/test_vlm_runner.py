# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded L3 video sampling."""

import base64
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

try:
    from . import vlm_runner
except ImportError:  # Support ``unittest discover -s steps/quality``.
    from steps.quality import vlm_runner


class _FakeFrame:
    def __init__(self, index: int, fps: float):
        self.index = index
        self.time = index / fps

    def to_image(self) -> Image.Image:
        return Image.new("RGB", (8, 8), (self.index, self.index, self.index))


class _FakeContainer:
    def __init__(self, frame_count: int, fps: float):
        self.frame_count = frame_count
        self.fps = fps
        self.streams = SimpleNamespace(
            video=[SimpleNamespace(average_rate=fps)])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def decode(self, _stream):
        return (_FakeFrame(index, self.fps)
                for index in range(self.frame_count))


class VideoSamplingTest(unittest.TestCase):
    def test_uniform_indices_cover_full_sequence(self):
        self.assertEqual(vlm_runner._uniform_indices(100, 5),
                         [0, 25, 50, 74, 99])
        self.assertEqual(vlm_runner._uniform_indices(3, 5), [0, 1, 2])

    def test_sample_video_caps_and_uniformly_covers_timeline(self):
        fake_av = SimpleNamespace(
            open=lambda _path: _FakeContainer(frame_count=100, fps=10.0))
        with mock.patch.dict(sys.modules, {"av": fake_av}):
            images = vlm_runner._sample_video(
                Path("unused.mp4"), fps=10.0, max_frames=5,
                max_edge=448, jpeg_quality=100)

        values = []
        for data_url in images:
            encoded = data_url.partition(",")[2]
            image = Image.open(BytesIO(base64.b64decode(encoded)))
            values.append(image.getpixel((0, 0))[0])
        self.assertEqual(len(values), 5)
        for actual, expected in zip(values, [0, 25, 50, 74, 99]):
            self.assertLessEqual(abs(actual - expected), 2)

    def test_none_uses_hard_default_cap(self):
        fake_av = SimpleNamespace(
            open=lambda _path: _FakeContainer(
                frame_count=vlm_runner.DEFAULT_MAX_FRAMES + 20, fps=10.0))
        with mock.patch.dict(sys.modules, {"av": fake_av}):
            images = vlm_runner._sample_video(
                Path("unused.mp4"), fps=10.0, max_frames=None,
                max_edge=448, jpeg_quality=85)
        self.assertEqual(len(images), vlm_runner.DEFAULT_MAX_FRAMES)


if __name__ == "__main__":
    unittest.main()
