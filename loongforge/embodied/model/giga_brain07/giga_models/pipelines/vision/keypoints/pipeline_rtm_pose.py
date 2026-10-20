import os

import cv2
import numpy as np
import torch
from PIL import Image
from rtmlib import Wholebody
from rtmlib.tools.file import download_checkpoint
from rtmlib.visualization import coco133, openpose134

from ...pipeline import BasePipeline


class RTMPosePipeline(BasePipeline):
    """Pipeline for RTM Wholebody pose estimation."""

    def __init__(
        self,
        det_path: str,
        pose_path: str,
        mode: str,
        to_openpose: bool = True,
        conf_thres: float = 0.6,
    ) -> None:
        self.download(det_path, pose_path, mode)
        self.model = Wholebody(
            det=det_path,
            det_input_size=Wholebody.MODE[mode]['det_input_size'],
            pose=pose_path,
            pose_input_size=Wholebody.MODE[mode]['pose_input_size'],
            to_openpose=to_openpose,
            mode=mode,
        )
        self.to_openpose = to_openpose
        self.conf_thres = conf_thres
        self.device = 'cpu'

    def download(self, det_path: str, pose_path: str, mode: str) -> None:
        """Ensure RTM Wholebody weights exist locally, download if missing.

        Args:
            det_path: Local path for the detector checkpoint file.
            pose_path: Local path for the pose checkpoint file.
            mode: RTM mode key used to resolve remote URLs and input sizes.
        """
        if not os.path.exists(det_path):
            url_path = Wholebody.MODE[mode]['det']
            local_path = download_checkpoint(url_path, dst_dir=os.path.dirname(det_path))
            if local_path != det_path:
                os.rename(local_path, det_path)
        if not os.path.exists(pose_path):
            url_path = Wholebody.MODE[mode]['pose']
            local_path = download_checkpoint(url_path, dst_dir=os.path.dirname(pose_path))
            if local_path != pose_path:
                os.rename(local_path, pose_path)

    def to(self, device: str):
        if device == 'cuda':
            device = f'cuda:{torch.cuda.current_device()}'
        self.device = device
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
            provider_options = None
        elif device.startswith('cuda'):
            providers = ['CUDAExecutionProvider']
            provider_options = [{'device_id': int(device[-1])}]
        else:
            assert False
        self.model.det_model.session.set_providers(providers, provider_options)
        self.model.pose_model.session.set_providers(providers, provider_options)
        return self

    def __call__(self, input_image: Image.Image, return_image: bool = True):
        """Run RTM Wholebody keypoint estimation.

        Args:
            input_image: PIL image in RGB; will be internally converted to BGR.
            return_image: If True, return a rendered skeleton image; otherwise
                return raw keypoints and scores.

        Returns:
            If return_image is True: PIL.Image with drawn skeleton.
            Else: Tuple (keypoints: np.ndarray, scores: np.ndarray).
        """
        input_image = np.array(input_image, dtype=np.uint8)
        input_image = input_image[:, :, ::-1]  # rgb --> bgr
        keypoints, scores = self.model(input_image)
        if return_image:
            black_frame = np.zeros_like(input_image)
            skeleton_frame = plot_person_kpts(
                black_frame,
                keypoints,
                scores=scores,
                kpt_thr=self.conf_thres,
                openpose_format=self.to_openpose,
                line_width=4,
            )
            output_image = Image.fromarray(skeleton_frame)
            return output_image
        else:
            return keypoints, scores


def plot_person_kpts(
    pose_vis_img: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.6,
    openpose_format: bool = True,
    line_width: int = 4,
) -> np.ndarray:
    """Plot a single person in-place update the pose image."""
    for kpts, ss in zip(keypoints, scores):
        try:
            pose_vis_img = draw_skeleton(pose_vis_img, kpts, ss, kpt_thr=kpt_thr, openpose_format=openpose_format, line_width=line_width)
        except ValueError as e:
            print(f'Error in draw_skeleton func, {e}')

    return pose_vis_img


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.6,
    openpose_format: bool = True,
    radius: int = 2,
    line_width: int = 4,
) -> np.ndarray:
    """Draw a person skeleton onto an image.

    Args:
        img: BGR image array to draw on (modified in place).
        keypoints: Array of shape (K, 2) with pixel xy coordinates.
        scores: Array of shape (K,) with per-keypoint confidence scores.
        kpt_thr: Visibility threshold for showing a keypoint/connection.
        openpose_format: If True, use OpenPose 134 topology; else COCO 133.
        radius: Circle radius for keypoints.
        line_width: Line thickness for skeleton links.

    Returns:
        np.ndarray: Image with skeleton rendered.
    """
    skeleton_topology = openpose134 if openpose_format else coco133
    assert len(keypoints.shape) == 2
    keypoint_info, skeleton_info = (
        skeleton_topology['keypoint_info'],
        skeleton_topology['skeleton_info'],
    )
    vis_kpt = [s >= kpt_thr for s in scores]
    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]

        if vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])), (int(kpt1[0]), int(kpt1[1])), link_color, thickness=line_width)

    return img
