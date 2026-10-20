import bisect
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ....utils import download_from_huggingface
from ...pipeline import BasePipeline


class FilmPipeline(BasePipeline):
    """Pipeline for FILM-based frame interpolation."""

    def __init__(self, model_path: str) -> None:
        """Initialize the pipeline and load the TorchScript model.

        Args:
            model_path: Local path to the TorchScript FILM model file. If the
                file does not exist, it will be downloaded from Hugging Face.
        """
        self.download(model_path)
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        self.device = 'cpu'

    def download(self, file_path: str) -> None:
        """Ensure the model file exists locally, otherwise download it."""
        if os.path.exists(file_path):
            return
        file_name = os.path.basename(file_path)
        download_from_huggingface(
            repo_id='jkawamoto/frame-interpolation-pytorch',
            filenames=[file_name],
            local_dir=os.path.dirname(file_path),
        )

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def pad_batch(self, batch: np.ndarray, align: int):
        """Pad a BCHW-like batch (B, H, W, C) so H and W are multiples of align.

        Returns both the padded batch and the crop region to recover original size.
        """
        height, width = batch.shape[1:3]
        height_to_pad = (align - height % align) if height % align != 0 else 0
        width_to_pad = (align - width % align) if width % align != 0 else 0
        crop_region = [
            height_to_pad >> 1,
            width_to_pad >> 1,
            height + (height_to_pad >> 1),
            width + (width_to_pad >> 1),
        ]
        batch = np.pad(
            batch,
            (
                (0, 0),
                (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)),
                (0, 0),
            ),
            mode='constant',
        )
        return batch, crop_region

    def __call__(self, frames: list[Image.Image], interval: int):
        """Interpolate frames between each consecutive pair.

        Args:
            frames: List of input PIL images ordered in time.
            interval: Number of interpolated frames to insert between each pair.

        Returns:
            A list of PIL images containing all interpolated frames in order.
        """
        output_frames = []
        for n in range(0, len(frames) - 1):
            img_1 = np.array(frames[n])[:, :, ::-1]
            img_2 = np.array(frames[n + 1])[:, :, ::-1]
            img_1 = img_1.astype(np.float32) / np.float32(255)
            img_2 = img_2.astype(np.float32) / np.float32(255)
            img_1 = np.expand_dims(img_1, axis=0)
            img_2 = np.expand_dims(img_2, axis=0)
            img_batch_1, crop_region_1 = self.pad_batch(img_1, align=64)
            img_batch_2, crop_region_2 = self.pad_batch(img_2, align=64)
            img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
            img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)
            results = [img_batch_1, img_batch_2]
            idxes = [0, interval + 1]
            remains = list(range(1, interval + 1))
            splits = torch.linspace(0, 1, interval + 2)
            inner_loop_progress = tqdm(range(len(remains)), leave=False, disable=True)
            for _ in inner_loop_progress:
                starts = splits[idxes[:-1]]
                ends = splits[idxes[1:]]
                distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - 0.5).abs()
                matrix = torch.argmin(distances).item()
                start_i, step = np.unravel_index(matrix, distances.shape)
                end_i = start_i + 1
                x0 = results[start_i]
                x1 = results[end_i]
                x0 = x0.half()
                x1 = x1.half()
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])
                with torch.no_grad():
                    prediction = self.model(x0, x1, dt)
                insert_position = bisect.bisect_left(idxes, remains[step])
                idxes.insert(insert_position, remains[step])
                results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
                inner_loop_progress.update(1)
                del remains[step]
            inner_loop_progress.close()
            y1, x1, y2, x2 = crop_region_1
            if n < len(frames) - 2:
                output_frames += [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results[:-1]]
            else:
                output_frames += [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
        output_frames = [Image.fromarray(frame) for frame in output_frames]
        return output_frames
