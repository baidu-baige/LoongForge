import os
from typing import Any, List, Optional, Sequence, Tuple

import grounding_dino.groundingdino.datasets.transforms as T
import numpy as np
import torch
from grounding_dino.groundingdino.util.inference import load_model, predict
from PIL import Image
from pycocotools import mask as mask_utils
from sam2 import sam2_video_predictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils import misc
from torchvision.ops import box_convert
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from ....utils import download_from_huggingface, download_from_url
from ...pipeline import BasePipeline
from .sam2_utils import segmentation_color_mask


class GroundedSAM2Pipeline(BasePipeline):
    """Grounded-SAM2 video segmentation pipeline with text prompts.

    Combines GroundingDINO for object detection using text captions with SAM2 for video mask propagation.
    """

    def __init__(self, gd_config_path: str, gd_model_path: str, sam_config_path: str, sam_model_path: str) -> None:
        """Initialize and load models and transforms.

        Args:
            gd_config_path: Path to GroundingDINO config file.
            gd_model_path: Path to GroundingDINO checkpoint.
            sam_config_path: Path to SAM2 config file.
            sam_model_path: Path to SAM2 checkpoint.
        """
        self.download_gd([gd_config_path, gd_model_path])
        self.download_sam(sam_model_path)
        self.grounding_model = load_model(
            model_config_path=gd_config_path,
            model_checkpoint_path=gd_model_path,
            device='cpu',
        )
        self.sam2_video_predictor = build_sam2_video_predictor(
            config_file=sam_config_path,
            ckpt_path=sam_model_path,
            device='cpu',
        )
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.device = 'cpu'

    def download_gd(self, file_paths: Sequence[str]) -> None:
        for file_path in file_paths:
            if os.path.exists(file_path):
                continue
            file_name = os.path.basename(file_path)
            if file_name == 'GroundingDINO_SwinT_OGC.py':
                file_name = 'GroundingDINO_SwinT_OGC.cfg.py'
            local_path = download_from_huggingface(
                repo_id='ShilongLiu/GroundingDINO',
                filenames=[file_name],
                local_dir=os.path.dirname(file_path),
            )
            if local_path != file_path:
                os.rename(local_path, file_path)

    def download_sam(self, file_path: str) -> None:
        if os.path.exists(file_path):
            return
        file_name = os.path.basename(file_path)
        url = f'https://dl.fbaipublicfiles.com/segment_anything_2/092824/{file_name}'
        download_from_url(url, file_path)

    def to(self, device: str):
        """Move models to device and return self."""
        self.device = device
        self.grounding_model.to(device)
        self.sam2_video_predictor.to(device)
        return self

    def detect(
        self,
        frames: Sequence[Image.Image],
        det_labels: Sequence[str] | str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, List[str], int]:
        """Detect boxes with GroundingDINO from frames and text labels.

        Returns boxes (xyxy, float32), matched labels, and the index of first frame where detection occurs, or -1 if not found.
        """
        if not isinstance(det_labels, list):
            det_labels = [det_labels]
        caption = ''
        for det_label in det_labels:
            det_label = det_label.lower().strip()
            caption += '{}.'.format(det_label)
        frame_idx = -1
        for i in range(len(frames)):
            boxes, scores, labels = predict(
                model=self.grounding_model,
                image=self.transform(frames[i], None)[0],
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )
            if len(boxes) > 0:
                h, w = frames[i].height, frames[i].width
                boxes = boxes * torch.Tensor([w, h, w, h])
                boxes = box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt='xyxy').numpy()
                frame_idx = i
                break
        if frame_idx == -1:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = []
        return boxes, labels, frame_idx

    def segment(
        self,
        frames: Sequence[Image.Image],
        boxes: np.ndarray,
        labels: Sequence[str],
        frame_idx: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Propagate SAM2 masks across frames from initial boxes and labels."""
        frames = [np.array(frame) for frame in frames]
        frames = np.stack(frames, axis=0)
        inference_state = self.sam2_video_predictor.init_state(frames)
        for object_id, (label, box) in enumerate(zip(labels, boxes), start=1):
            self.sam2_video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=box,
            )
        video_segments = []
        if frame_idx > 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_video_predictor.propagate_in_video(
                inference_state, start_frame_idx=frame_idx, reverse=True
            ):
                assert frame_idx - out_frame_idx == len(video_segments)
                assert len(out_obj_ids) == len(labels) == len(out_mask_logits)
                assert out_mask_logits.shape[1] == 1
                out_mask_logits = out_mask_logits[:, 0]
                out_masks = (out_mask_logits > 0.0).cpu().numpy()
                out_obj_ids = np.array(out_obj_ids, dtype=int)
                video_segments.append((out_masks, out_obj_ids))
            video_segments = list(reversed(video_segments[1:]))
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_video_predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx):
            assert out_frame_idx == len(video_segments)
            assert len(out_obj_ids) == len(labels) == len(out_mask_logits)
            assert out_mask_logits.shape[1] == 1
            out_mask_logits = out_mask_logits[:, 0]
            out_masks = (out_mask_logits > 0.0).cpu().numpy()
            out_obj_ids = np.array(out_obj_ids, dtype=int)
            video_segments.append([out_masks, out_obj_ids])
        assert len(video_segments) == len(frames)
        return video_segments

    def __call__(
        self,
        frames: Sequence[Image.Image],
        det_labels: Sequence[str] | str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        return_mode: Optional[str] = None,
    ) -> Tuple[Any, np.ndarray, List[str]]:
        """Run detection + segmentation; return segments plus boxes and
        labels."""
        boxes, labels, frame_idx = self.detect(frames, det_labels, box_threshold, text_threshold)
        if frame_idx == -1:
            return None, boxes, labels
        video_segments = self.segment(frames, boxes, labels, frame_idx)
        video_segments = convert_frames(video_segments, return_mode)
        return video_segments, boxes, labels


def convert_frames(video_segments: List[Tuple[np.ndarray, np.ndarray]], mode: Optional[str] = None):
    if mode is None:
        return video_segments
    if mode == 'rle_encode':
        for i in range(len(video_segments)):
            segment = video_segments[i][0]
            if len(segment) > 0:
                segment = np.array(segment, order='F')
                encoded = mask_utils.encode(np.array(segment.reshape(-1, 1), order='F'))
                encoded = {'data': encoded, 'data_shape': segment.shape}
                video_segments[i][0] = encoded

    elif mode == 'rle_decode':
        for i in range(len(video_segments)):
            encoded = video_segments[i][0]
            if isinstance(encoded, dict):
                segment = mask_utils.decode(encoded['data'])
                segment = segment.reshape(encoded['data_shape'])
                video_segments[i][0] = segment

    elif mode == 'box':
        for i in range(len(video_segments)):
            segment = video_segments[i][0]
            if len(segment) > 0:
                boxes = [get_box_from_mask(segment[j]) for j in range(len(segment))]
                boxes = np.concatenate(boxes, axis=0)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
            video_segments[i][0] = boxes

    elif mode == 'image':
        segments = []
        for i in range(len(video_segments)):
            segment = video_segments[i][0]
            segments.append(segment)
        masks = np.stack(segments, axis=1)
        all_masks = segmentation_color_mask(masks)
        all_masks = all_masks.transpose((1, 2, 3, 0))
        assert len(all_masks) == len(video_segments)
        for i in range(len(video_segments)):
            video_segments[i][0] = Image.fromarray(all_masks[i])
        return video_segments

    else:
        assert False
    return video_segments


def get_box_from_mask(mask: np.ndarray) -> np.ndarray:
    position = np.where(mask == 1)
    if len(position[0]) == 0:
        return np.zeros((1, 4), dtype=np.float32)
    if len(position[1]) == 0:
        return np.zeros((1, 4), dtype=np.float32)
    x1 = np.min(position[1])
    x2 = np.max(position[1]) + 1
    y1 = np.min(position[0])
    y2 = np.max(position[0]) + 1
    box = np.array([x1, y1, x2, y2], dtype=np.float32).reshape((1, 4))
    return box


def load_video_frames(
    video_path: np.ndarray,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    async_loading_frames: bool = False,
    compute_device: torch.device = torch.device('cuda'),
):
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    # Get the original video height and width
    frames = video_path
    video_height, video_width = frames.shape[1], frames.shape[2]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    frames = F.resize(frames, (image_size, image_size), InterpolationMode.BILINEAR)
    frames = frames.float() / 255.0
    if not offload_video_to_cpu:
        frames = frames.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    frames -= img_mean
    frames /= img_std
    return frames, video_height, video_width


setattr(sam2_video_predictor, 'load_video_frames', load_video_frames)
setattr(misc, 'load_video_frames', load_video_frames)
