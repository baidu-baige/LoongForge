# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""Utilities for reading and decoding video frames for DreamZero datasets."""

import json
import logging
import subprocess
from collections import OrderedDict

import cv2
import numpy as np
import torchvision

logger = logging.getLogger(__name__)

# Import decord with graceful fallback
try:
    import decord

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


_DECORD_READER_CACHE: OrderedDict[tuple[str, str], tuple[object, np.ndarray]] = OrderedDict()


def _split_video_backend_kwargs(video_backend_kwargs: dict | None) -> tuple[dict, int]:
    kwargs = dict(video_backend_kwargs or {})
    reader_cache_size = int(kwargs.pop("reader_cache_size", 0) or 0)
    return kwargs, reader_cache_size


def _decord_reader_cache_key(video_path: str, kwargs: dict) -> tuple[str, str]:
    return (
        str(video_path),
        json.dumps(kwargs, sort_keys=True, default=repr, separators=(",", ":")),
    )


def _get_decord_reader_and_timestamps(
    video_path: str,
    video_backend_kwargs: dict | None,
) -> tuple[object, np.ndarray]:
    kwargs, cache_size = _split_video_backend_kwargs(video_backend_kwargs)
    if cache_size <= 0:
        vr = decord.VideoReader(video_path, **kwargs)
        frame_ts = vr.get_frame_timestamp(range(len(vr)))
        return vr, frame_ts

    key = _decord_reader_cache_key(video_path, kwargs)
    cached = _DECORD_READER_CACHE.get(key)
    if cached is not None:
        _DECORD_READER_CACHE.move_to_end(key)
        return cached

    vr = decord.VideoReader(video_path, **kwargs)
    frame_ts = vr.get_frame_timestamp(range(len(vr)))
    while len(_DECORD_READER_CACHE) >= cache_size:
        _DECORD_READER_CACHE.popitem(last=False)
    _DECORD_READER_CACHE[key] = (vr, frame_ts)
    return vr, frame_ts


def _nearest_frame_indices_from_timestamps(
    frame_timestamps: np.ndarray,
    timestamps: list[float] | np.ndarray,
) -> np.ndarray:
    frame_start_seconds = np.asarray(frame_timestamps)[:, 0]
    requested = np.asarray(timestamps, dtype=frame_start_seconds.dtype).reshape(-1)
    if requested.size == 0:
        return np.empty((0,), dtype=np.int64)

    right = np.searchsorted(frame_start_seconds, requested, side="left")
    right = np.clip(right, 0, len(frame_start_seconds) - 1)
    left = np.clip(right - 1, 0, len(frame_start_seconds) - 1)
    use_right = (
        np.abs(frame_start_seconds[right] - requested)
        < np.abs(requested - frame_start_seconds[left])
    )
    return np.where(use_right, right, left).astype(np.int64, copy=False)

def _extract_frames_at_timestamps_ffmpeg(video_path: str, timestamps: list[float]) -> np.ndarray:
    """Extract frames at specific timestamps using ffmpeg."""
    frames = []

    for timestamp in timestamps:
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)

            # Check if output is empty (timestamp doesn't exist)
            if len(output) == 0:
                raise subprocess.CalledProcessError(1, cmd)

            # Get frame dimensions
            if len(frames) == 0:
                info_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    video_path,
                ]
                info_output = subprocess.check_output(info_cmd).decode("utf-8")
                info_data = json.loads(info_output)
                width = info_data["streams"][0]["width"]
                height = info_data["streams"][0]["height"]

            # Decode raw RGB data
            frame_data = np.frombuffer(output, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            frames.append(frame)

        except subprocess.CalledProcessError:
            # Timestamp might be out of bounds, use last frame or black frame
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

    return np.array(frames)


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "ffmpeg",
    video_backend_kwargs: dict = {},
    fps: None | float = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps.

    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "ffmpeg".
        fps (float, optional): FPS of the video. Defaults to 30.
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available. Install it with: pip install decord")
        vr, frame_ts = _get_decord_reader_and_timestamps(video_path, video_backend_kwargs)
        indices = _nearest_frame_indices_from_timestamps(frame_ts, timestamps)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )

        # https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata
        # Temporary fix: use 30 fps as the fps of the video (agibot)
        # TODO: get fps as parameter
        if fps is None:
            fps = decoder.metadata.average_fps
        interval = 1 / fps
        timestamps = np.array(timestamps).astype(np.float64)

        if np.all(timestamps == 0):
            timestamps = np.arange(len(timestamps)) / fps

        # Get video duration range from first and last frames
        # This is a robust way to get valid timestamp range without depending on specific metadata attributes
        first_frame = decoder.get_frames_at(indices=[0])
        last_frame = decoder.get_frames_at(indices=[len(decoder) - 1])
        min_pts = float(first_frame.pts_seconds[0])
        max_pts = float(last_frame.pts_seconds[0])

        # Clamp timestamps to valid range to avoid RuntimeError
        timestamps = np.clip(timestamps, min_pts, max_pts)

        # Correct float precision issues in timestamps
        # E.g. for 5fps video: [1.0, 1.20000005, 1.39999998] -> [1.0, 1.2, 1.4]
        # Without this, the torchcodec will read the delayed frame (e.g. 1.39999998 -> 1.2)
        # Round to nearest frame interval to prevent torchcodec from reading wrong frames
        # Allow max 1% error from expected interval
        if fps is None:
            closest_timestamps = np.round(timestamps / interval) * interval
            # Re-clamp after rounding to ensure still in valid range
            closest_timestamps = np.clip(closest_timestamps, min_pts, max_pts)
            timestamp_errors = np.abs(closest_timestamps - timestamps) / interval
            invalid_mask = timestamp_errors >= 0.01
            if np.any(invalid_mask):
                invalid_indices = np.where(invalid_mask)[0]
                invalid_timestamps = timestamps[invalid_indices]
                raise ValueError(
                    f"Try to read invalid timestamps {invalid_timestamps} from video {video_path} (FPS: {fps})"
                )

            timestamps = closest_timestamps

        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "ffmpeg":
        return _extract_frames_at_timestamps_ffmpeg(video_path, list(timestamps))
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames

    elif video_backend == "torchvision_av":
        # set backend
        torchvision.set_video_backend("pyav")

        # set a video stream reader
        reader = torchvision.io.VideoReader(video_path, "video")

        # set the first and last requested timestamps
        # Note: previous timestamps are usually loaded, since we need to access the previous key frame
        first_ts = timestamps[0]
        last_ts = timestamps[-1]

        # access closest key frame of the first requested frame
        # Note: closest key frame timestamp is usally smaller than `first_ts`
        # (e.g. key frame can be the first frame of the video)
        # for details on what `seek` is doing see:
        # https://pyav.basswood-io.com/docs/stable/api/container.html#av.container.InputContainer.seek
        reader.seek(first_ts, keyframes_only=True)

        # Decode frames sequentially, storing the ones we need in a dictionary
        # to map timestamps to frame data. This allows for easy re-ordering later.
        found_frames_map = {}
        tolerance = 0.001  # 1ms tolerance for timestamp matching

        for frame in reader:
            current_ts = frame["pts"]

            # Use tolerance-based matching instead of exact match
            for ts in timestamps:
                if ts not in found_frames_map and abs(current_ts - ts) < tolerance:
                    found_frames_map[ts] = frame["data"]
                    break

            if current_ts >= last_ts + tolerance or len(found_frames_map) == len(timestamps):
                break

        reader.container.close()
        reader = None

        logger.debug(
            "Requested %d timestamps: %s%s",
            len(timestamps),
            timestamps[:4],
            "..." if len(timestamps) > 4 else "",
        )
        logger.debug(
            "Found %d frames with tolerance=%ss",
            len(found_frames_map),
            tolerance,
        )
        if len(found_frames_map) < len(timestamps):
            missing = [ts for ts in timestamps if ts not in found_frames_map]
            logger.warning(
                "Missing timestamps: %s%s",
                missing[:4],
                "..." if len(missing) > 4 else "",
            )

        frames = np.array(list(found_frames_map.values()))
        return frames.transpose(0, 2, 3, 1)

    else:
        raise NotImplementedError
