
"""latent dataset"""

import csv
import numpy as np
import pandas as pd
import torch
import math
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import torch.nn.functional as F
import json
from pathlib import Path
from . import video_transforms
from megatron.training import get_args
from transformers import AutoProcessor


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape)


def get_transforms_latent():
    """
    Transformes for latent
    the tensor is in shape: [T, C, H, W]

    For latent input, we only need `RandomHorizontalFlipVideo`.
    Other transformes: `ToTensorVideo`, `UCFCenterCropVideo`, `Normalize`
    are processed in offline scripts
    """
    transform_latent = transforms.Compose(
        [
            video_transforms.RandomHorizontalFlipVideo(),
        ]
    )
    return transform_latent


def pad_video(video, max_T, max_H, max_W):
    """pad video to the max_T, max_H and max_W"""

    C, T, H, W = video.size()
    assert T <= max_T and H <= max_H and W <= max_W
    mask = torch.ones_like(video, dtype=torch.bool)
    padding_shape = (0, max_W - W, 0, max_H - H, 0, max_T - T)
    video = F.pad(video, padding_shape)
    mask = F.pad(mask, padding_shape)
    return video, mask


def pad_text(text, mask, max_length):
    """pad text to the max_length"""
    assert text.shape[1] <= max_length
    text = F.pad(text, (0, 0, 0, max_length - text.shape[1]))
    mask = F.pad(mask, (0, max_length - mask.shape[0]))
    return text, mask


class LatentDatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        diffusion,
        num_frames=16,
        max_height=32,
        max_width=32,
        max_text_length=120,
        frame_interval=3,
    ):
        self.csv_path = csv_path
        self.diffusion = diffusion
        with open(csv_path, "r", encoding="UTF-8") as f:
            reader = csv.reader(f)
            self.samples = list(reader)

        ext = self.samples[0][0].split(".")[-2]
        assert ext.lower() in (
            "mp4",
            "avi",
            "mov",
            "mkv",
        ), f"Unsupported file format: {ext}"

        self.transform = get_transforms_latent()

        self.num_frames = num_frames
        self.max_height = max_height
        self.max_width = max_width
        self.max_text_length = max_text_length
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(
            num_frames * frame_interval
        )

    def getitem(self, index):
        """
        get single item

        the item is a torch.Tensor with shape: [C, T, H, W]
        """
        video_path, text_path = self.samples[index]
        vframes = torch.load(video_path, map_location="cpu").permute(1, 0, 2, 3)
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        if end_frame_ind - start_frame_ind >= self.num_frames:
            frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
            )
            video = vframes[frame_indice]
        else:
            video = vframes
        video = self.transform(video)  # T C H W

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        video, padding_mask = pad_video(
            video, self.num_frames, self.max_height, self.max_width
        )

        # Other data
        text = torch.load(text_path, map_location="cpu")
        text_enc, text_mask = pad_text(text["y"], text["mask"], self.max_text_length)
        timestep = torch.randint(0, self.diffusion.num_timesteps, ())
        noise = torch.randn_like(video)
        return {
            "video": video,
            "video_noised": self.diffusion.add_noise(video, timestep, noise=noise),
            "text_enc": text_enc,
            "video_mask": padding_mask,
            "text_mask": text_mask.bool(),
            "labels": noise,
            "position_ids": torch.rand_like(video),
            "timestep": timestep,
            "fps": torch.tensor(24 // self.frame_interval),
        }

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                raise e
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


class VariableLatentDataset(torch.utils.data.Dataset):
    """load video according to the csv file."""

    def __init__(
        self,
        data_path,
        frame_interval=1,
        micro_frame_size=17,
        patch_size=(4, 8, 8),
        frame_size_divisible_by=8,
        max_text_length=300,
    ):
        self.data = pd.read_csv(data_path)
        self.frame_interval = frame_interval
        self.data["id"] = np.arange(len(self.data))
        self.micro_frame_size = micro_frame_size
        self.patch_size = patch_size
        self.frame_size_divisible_by = frame_size_divisible_by
        self.max_text_length = max_text_length

    def getitem(self, index):
        """get single item"""
        # T H W of Bucket
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        video_path = sample["video"]
        text_path = sample["text"]

        # loading
        video = torch.load(video_path, map_location="cpu")  # CTHW
        text = torch.load(text_path, map_location="cpu")  # 1,S,C

        # padding
        pad_t = math.ceil(num_frames / self.micro_frame_size) * math.ceil(
            self.micro_frame_size / self.patch_size[0]
        )
        pad_h = height // self.patch_size[1]
        pad_w = width // self.patch_size[2]
        pad_t = self.make_divisible(pad_t)
        video, padding_mask = pad_video(video, pad_t, pad_h, pad_w)
        text["y"], text["mask"] = pad_text(
            text["y"], text["mask"], self.max_text_length
        )

        ret = {
            "video": video,
            "video_mask": padding_mask,
            "num_frames": sample["num_frames"],
            "height": sample["height"],
            "width": sample["width"],
            "ar": sample["aspect_ratio"],
            "fps": sample["fps"],
            "text_enc": text["y"],
            "text_mask": text["mask"].bool(),
        }
        return ret

    def make_divisible(self, orig_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""
        after = orig_size
        while (after % self.frame_size_divisible_by) != 0:
            after += 1
        return after

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, steps_per_epoch=0):
        args = get_args()
        self.metadata = []
        self.load_metadata(metadata_path)
        base_path = Path(args.data_path[0])
        if args.model_name == "wan2_1_i2v":
            self.path = [base_path / Path(data["video_path"]).name for data in self.metadata]
        if args.model_name == "wan2_2_i2v":
            self.path = [base_path / data["video"] for data in self.metadata]

        self.path = [
            p.with_suffix(p.suffix + ".tensors.pth")
            for p in self.path
            if (p.with_suffix(p.suffix + ".tensors.pth")).exists()
        ]
        self.steps_per_epoch = steps_per_epoch
        print(
            f"self.steps_per_epoch: {self.steps_per_epoch}, total_samples: {len(self.metadata)}"
        )
        assert len(self.path) > 0
        self.manual_seed = args.seed

    def load_metadata(self, metadata_path):
        """load metadata from different types of files"""
        if metadata_path is None:
            print("No metadata_path. Please provide metadata_path.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.metadata = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.metadata = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.metadata = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, index):
        seed = (self.manual_seed + index) % 2**32
        numpy_random_state = np.random.RandomState(seed=seed)
        data_id = numpy_random_state.randint(0, self.steps_per_epoch)
        data_id = data_id % len(self.path)
        # data_id = 0
        path = self.path[data_id]
        data = torch.load(path, weights_only=False, map_location="cpu")
        # used for generate timestep
        data["seed"] = seed
        return data

    def __len__(self):
        return self.steps_per_epoch


class ErnieImageDataset(torch.utils.data.Dataset):
    """Dataset for ernie-vl"""
    def __init__(self, args, metadata_path, steps_per_epoch=0):
        self.manual_seed = args.seed
        self.steps_per_epoch = steps_per_epoch
        self.processor = AutoProcessor.from_pretrained(args.hf_tokenizer_path,  trust_remote_code=True)
        self.file_names = []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.file_names.append(line.strip())

    def __getitem__(self, index):
        seed = (self.manual_seed + index) % 2**32
        numpy_random_state = np.random.RandomState(seed=seed)
        data_id = numpy_random_state.randint(0, self.steps_per_epoch)
        data_id = data_id % len(self.file_names)
        data_name = self.file_names[data_id]
        data = np.load(data_name)
        data_item = {
            "images": torch.from_numpy(data["images"]),
            "input_ids": torch.from_numpy(data["input_ids"]),
            "token_type_ids": torch.from_numpy(data["token_type_ids"])[:, :-1],
            "position_ids": torch.from_numpy(data["position_ids"]),
            "grid_thw": torch.from_numpy(data["grid_thw"]),
            "image_type_ids": torch.from_numpy(data["image_type_ids"]),
            "labels": torch.from_numpy(data["labels"])
        }
        return data_item

    def __len__(self):
        return self.steps_per_epoch