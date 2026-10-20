import math
import queue
import threading
from typing import Optional, Tuple

import cv2
import torch
from decord import VideoReader


class VideoReaderCV2:
    """Background-threaded OpenCV video reader with optional resize and length
    cap."""

    def __init__(self, video_path: str, video_length: Optional[int] = None, dst_size: Optional[Tuple[int, int]] = None, queue_size: int = 4):
        """Initialize the OpenCV-based reader and start decoding thread.

        Args:
            video_path (str): Path to input video file.
            video_length (Optional[int]): Optional max number of frames to read.
            dst_size (Optional[Tuple[int, int]]): Optional resize target (W, H).
            queue_size (int): Max buffered frames in the internal queue.
        """
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size
        self.queue_size = queue_size
        self.video = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self) -> None:
        """Open the video and spawn a background decoding thread.

        Raises:
            Exception: If the video cannot be opened by OpenCV.
        """
        if self.video is None:
            self.video = cv2.VideoCapture(self.video_path)
            if not self.video.isOpened():
                raise Exception('Ensure file is valid video and system dependencies are up to date.\n')
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.decode_thread, args=(self.video, self.frame_queue, self.stop), daemon=True)
            self.thread.start()

    def close(self) -> None:
        """Stop the decoding thread, release resources, and clear buffers."""
        if self.video is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            self.video.release()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.video = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self) -> None:
        """Re-open the video by closing and opening it again (resets state)."""
        self.close()
        self.open()

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return the (W, H) frame size of the opened video."""
        return (
            math.trunc(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.trunc(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> float:
        """Return the frames-per-second reported by OpenCV."""
        return self.video.get(cv2.CAP_PROP_FPS)

    def read(self):
        """Blocking read of next frame from the internal queue.

        Returns:
            np.ndarray | None: RGB frame (H, W, C) or None when stream ends.
        """
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self, video, frame_queue, stop) -> None:
        """Internal decoding loop that pushes frames into a queue.

        Args:
            video: OpenCV VideoCapture object.
            frame_queue: Thread-safe queue to push (index, frame) pairs.
            stop: threading.Event to signal termination.
        """
        frame_idx = 0
        try:
            while not stop.is_set():
                ret, frame = video.read()
                if not ret:
                    break
                if self.dst_size is not None:
                    frame = cv2.resize(frame, self.dst_size, interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
                if self.video_length is not None and frame_idx >= self.video_length:
                    break
        except Exception:
            stop.set()
        finally:
            frame_queue.put((None, None))


class VideoReaderDecord:
    """Background-threaded Decord video reader with optional resize and length
    cap."""

    def __init__(self, video_path: str, video_length: Optional[int] = None, dst_size: Optional[Tuple[int, int]] = None, queue_size: int = 4):
        """Initialize the Decord-based reader and start decoding thread.

        Args:
            video_path (str): Path to input video file.
            video_length (Optional[int]): Optional max number of frames to read.
            dst_size (Optional[Tuple[int, int]]): Optional resize target (W, H).
            queue_size (int): Max buffered frames in the internal queue.
        """
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size
        self.queue_size = queue_size
        self.video = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self) -> None:
        """Open the video and spawn a background decoding thread."""
        if self.video is None:
            self.video = VideoReader(self.video_path)
            if self.video_length is None:
                self.video_length = len(self.video)
            assert self.video_length <= len(self.video)
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.decode_thread, args=(self.video, self.frame_queue, self.stop), daemon=True)
            self.thread.start()

    def close(self) -> None:
        """Stop the decoding thread, release resources, and clear buffers."""
        if self.video is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.video = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self) -> None:
        """Re-open the video by closing and opening it again (resets state)."""
        self.close()
        self.open()

    @property
    def fps(self) -> float:
        """Return the frames-per-second reported by Decord."""
        return self.video.get_avg_fps()

    def read(self):
        """Blocking read of next frame from the internal queue.

        Returns:
            np.ndarray | None: RGB frame (H, W, C) or None when stream ends.
        """
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self, video, frame_queue, stop) -> None:
        """Internal decoding loop that pushes frames into a queue.

        Args:
            video: Decord VideoReader object.
            frame_queue: Thread-safe queue to push (index, frame) pairs.
            stop: threading.Event to signal termination.
        """
        frame_idx = 0
        try:
            while not stop.is_set():
                frame = video.next()
                frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame.asnumpy()
                if self.dst_size is not None:
                    frame = cv2.resize(frame, self.dst_size, interpolation=cv2.INTER_LINEAR)
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
                if frame_idx >= self.video_length:
                    break
        except Exception:
            stop.set()
        finally:
            frame_queue.put((None, None))
