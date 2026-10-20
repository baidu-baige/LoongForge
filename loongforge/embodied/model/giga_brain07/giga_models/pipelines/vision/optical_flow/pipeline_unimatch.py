import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from unimatch.unimatch import UniMatch

from ....utils import download_from_url
from ...pipeline import BasePipeline


class UniMatchPipeline(BasePipeline):
    """Optical flow pipeline using UniMatch pretrained model.

    This pipeline downloads the checkpoint if missing, loads a UniMatch model, and exposes a callable interface that returns a list of optical flow
    scores over sampled video frames.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the UniMatch model from a checkpoint.

        Args:
            model_path: Local path where the UniMatch checkpoint is stored. Will
                be downloaded automatically if not present.
        """
        self.download(model_path)
        self.model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow',
        ).eval()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['model'])
        self.device = 'cpu'

    def download(self, file_path: str) -> None:
        """Download UniMatch checkpoint if not present.

        Args:
            file_path: Target path to store the checkpoint.
        """
        if os.path.exists(file_path):
            return
        file_name = os.path.basename(file_path)
        url = f'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/{file_name}'
        download_from_url(url, file_path)

    def to(self, device: str):
        self.device = device
        self.model.to(self.device)
        return self

    def __call__(self, video_path: str, video_valid_range: Optional[Tuple[int, int]] = None) -> List[float]:
        """Compute optical flow scores for a video.

        Args:
            video_path: Path to a video readable by ``decord.VideoReader``.
            video_valid_range: Optional (start, end) indices to sample frames
                from. If None, uses the full range.

        Returns:
            List[float]: Mean absolute flow magnitude per step for sampled
                consecutive frame pairs.
        """
        video = VideoReader(video_path)
        if video_valid_range is None:
            video_valid_range = (0, len(video))
        frames = read_video_frames(video, video_valid_range).to(self.device)
        optical_flow = get_unimatch(frames, self.model)
        return optical_flow


def read_video_frames(video: VideoReader, video_valid_range: Tuple[int, int]) -> torch.Tensor:
    """Sample and preprocess frames from a ``VideoReader``.

    Args:
        video: Decord VideoReader instance.
        video_valid_range: (start, end) indices to sample within.

    Returns:
        torch.Tensor: Float tensor of shape (N, C, H, W) normalized to [0, 255]
            scale (no mean/std normalization here). Frames are resized to
            320x576 with bilinear interpolation.
    """
    start_index = video_valid_range[0]
    end_index = video_valid_range[1]
    indexes = np.linspace(start_index, end_index, 10, dtype=int)[:8]
    frames = video.get_batch(indexes)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    frames = [Image.fromarray(x) for x in frames]
    # transform
    frames = torch.stack([pil_to_tensor(x) for x in frames])
    frames = frames.float()
    H, W = frames.shape[-2:]
    if H > W:
        frames = rearrange(frames, 'N C H W -> N C W H')
    frames = F.interpolate(frames, size=(320, 576), mode='bilinear', align_corners=True)
    return frames


def get_unimatch(frames: torch.Tensor, model: UniMatch) -> List[float]:
    """Run UniMatch inference and aggregate flow scores.

    Args:
        frames: Tensor of shape (N, C, H, W). Consecutive frames are paired
            internally as (frames[:-1], frames[1:]).
        model: UniMatch model in eval mode.

    Returns:
        List[float]: Mean absolute flow magnitude for the final prediction.
    """
    with torch.no_grad():
        res = model(
            frames[:-1],
            frames[1:],
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
        )
        flow_maps = res['flow_preds'][-1].cpu()
        flow_scores: List[float] = flow_maps.abs().mean().tolist()
    return flow_scores
