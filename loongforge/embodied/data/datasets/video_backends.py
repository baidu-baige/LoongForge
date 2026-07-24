# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Video frame decoding backends.

Supports: opencv, decord, torchcodec, pyav, torchvision_av.
All backends return numpy arrays in (N, H, W, 3) RGB uint8 format.
"""

import numpy as np


def decode_video_frame(video_path: str, timestamp: float, backend: str = "torchcodec", **kwargs) -> np.ndarray:
    """Decode a single frame at the given timestamp (seconds).

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds to seek to.
        backend: One of "torchcodec", "pyav", "decord", "opencv", "torchvision_av".

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, RGB color order.
    """
    frames = decode_video_frames_by_timestamps(video_path, [timestamp], backend, **kwargs)
    return frames[0]


def decode_video_frames_by_timestamps(
    video_path: str,
    timestamps: list,
    backend: str = "torchcodec",
    **kwargs,
) -> np.ndarray:
    """Decode frames at given timestamps.

    Returns:
        np.ndarray of shape (N, H, W, 3), dtype uint8, RGB.
    """
    if backend == "torchcodec":
        return _decode_torchcodec(video_path, timestamps=timestamps)
    elif backend == "decord":
        return _decode_decord(video_path, timestamps=timestamps, **kwargs)
    elif backend == "opencv":
        return _decode_opencv(video_path, timestamps=timestamps)
    elif backend == "pyav":
        return _decode_pyav(video_path, timestamps=timestamps)
    elif backend == "torchvision_av":
        return _decode_torchvision_av(video_path, timestamps=timestamps)
    else:
        raise ValueError(f"Unknown video_backend: '{backend}'. "
                         f"Supported: torchcodec, decord, opencv, pyav, torchvision_av")


def decode_video_frames_by_indices(
    video_path: str,
    indices: list,
    backend: str = "torchcodec",
    fps: float = None,
    **kwargs,
) -> np.ndarray:
    """Decode frames at given frame indices.

    If the backend supports index-based access, uses it directly.
    Otherwise converts indices to timestamps using fps.
    """
    if backend == "decord":
        return _decode_decord(video_path, indices=indices, **kwargs)
    if fps is None:
        raise ValueError("fps is required for index-based decoding with this backend")
    timestamps = [i / fps for i in indices]
    return decode_video_frames_by_timestamps(video_path, timestamps, backend, **kwargs)


# ─── Backend Implementations ───────────────────────────────────────────────────
def _decode_torchcodec(video_path, timestamps=None):
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(video_path))
    frames = decoder.get_frames_played_at(seconds=timestamps).data
    return frames.permute(0, 2, 3, 1).numpy()


def _decode_decord(video_path, timestamps=None, indices=None, **kwargs):
    import decord
    decord.bridge.set_bridge("numpy")
    vr = decord.VideoReader(str(video_path), **kwargs)
    if indices is not None:
        frames = vr.get_batch(indices).asnumpy()
    else:
        fps = vr.get_avg_fps()
        frame_indices = [min(int(t * fps), len(vr) - 1) for t in timestamps]
        frames = vr.get_batch(frame_indices).asnumpy()
    return frames


def _decode_opencv(video_path, timestamps=None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for t in timestamps:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    cap.release()
    return np.stack(frames)

def _decode_pyav(video_path, timestamps=None):
    import av
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    seek_pts = int(first_ts / stream.time_base)
    container.seek(seek_pts, stream=stream)
    loaded_frames = []
    loaded_ts = []
    for frame in container.decode(video=0):
        loaded_frames.append(frame.to_ndarray(format="rgb24"))
        loaded_ts.append(float(frame.pts * stream.time_base))
        if loaded_ts[-1] >= last_ts:
            break
    container.close()
    query_ts = np.array(timestamps, dtype=np.float32)
    loaded_ts_arr = np.array(loaded_ts, dtype=np.float32)
    indices = np.argmin(np.abs(query_ts[:, None] - loaded_ts_arr[None, :]), axis=1)
    return np.stack([loaded_frames[i] for i in indices])


def _decode_torchvision_av(video_path, timestamps=None):
    import torchvision
    torchvision.set_video_backend("pyav")
    from torchvision.io import VideoReader
    reader = VideoReader(str(video_path), "video")
    frames = []
    for t in timestamps:
        reader.seek(t)
        frame_data = next(reader)
        frame = frame_data["data"].permute(1, 2, 0).numpy()
        frames.append(frame)
    return np.stack(frames)
