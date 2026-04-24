# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLMTaskEncoder class."""

import torch
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import override
from dataclasses import dataclass

from megatron.energon import (
    CaptioningSample,
    VQASample,
)
from importlib.metadata import version
if version('megatron-energon') < "7.0.0":
    from megatron.energon.flavors.webdataset import VideoData as AVData
    _ENERGON_NEEDS_SUBFLAVOR = True
else:
    from megatron.energon.flavors.webdataset import AVData
    _ENERGON_NEEDS_SUBFLAVOR = False

from megatron.energon.task_encoder.base import stateless
from transformers import AutoProcessor
from loongforge.utils import constants, get_chat_template
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .base.task_encoder import (
    BaseTaskEncoder,
    BaseTaskSample,
    BaseTaskSamplePacked,
    BaseTaskBatchPacked,
)
from loongforge.data.multimodal import (
    MultiMixQASample,
    PackedCaptioningSample,
    PackedVQASample,
    PackedMultiMixQASample,
)

IGNORE_INDEX = -100  # ID for labels that should be ignored.
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]
VIDEO_TOKEN_WITH_TAGS = VISION_TAGS[0] + VIDEO_TOKEN + VISION_TAGS[1]


@dataclass
class VLMTaskSample(BaseTaskSample):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None

    def __init__(self, image_grid_thw: str, video_grid_thw=None, **kwargs):
        super().__init__(**kwargs)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class VLMTaskSamplePacked(BaseTaskSamplePacked):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None

    def __init__(
        self, sample: BaseTaskSample, image_grid_thw: str, video_grid_thw=None
    ):
        init_args = vars(sample).copy()
        init_args.update({
            '__key__': sample.__key__,
            '__restore_key__': sample.__restore_key__,
            '__subflavors__': sample.__subflavors__
        })
        super().__init__(**init_args)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw

    def __repr__(self):
        base = super().__repr__() if hasattr(super(), "__repr__") else ""
        grid_str = ""
        if self.image_grid_thw is not None:
            grid_str += (
                f", image_grid_thw“="
                f"{tuple(self.image_grid_thw.shape) if hasattr(self.image_grid_thw, 'shape') else self.image_grid_thw}"
            )
        if self.video_grid_thw is not None:
            grid_str += (
                f", video_grid_thw="
                f"{tuple(self.video_grid_thw.shape) if hasattr(self.video_grid_thw, 'shape') else self.video_grid_thw}"
            )
        return base[:-1] + grid_str + ")"


@dataclass
class VLMTaskBatchPacked(BaseTaskBatchPacked):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None

    def __init__(
        self, sample: BaseTaskSample, image_grid_thw: str, video_grid_thw=None
    ):
        init_args = vars(sample).copy()
        init_args.update({
            '__key__': sample.__key__,
            '__restore_key__': sample.__restore_key__,
            '__subflavors__': sample.__subflavors__
        })
        super().__init__(**init_args)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


class VLMTaskEncoder(BaseTaskEncoder):
    """A simple task encoder for VLMs."""

    def __init__(self, args):
        super().__init__()
        if args.training_phase in ['sft']:
            self.chat_template = get_chat_template()
        self.processor = AutoProcessor.from_pretrained(self.args.hf_tokenizer_path, trust_remote_code=True)
        if args.image_resolution:
            setattr(self.processor, "image_resolution", args.image_resolution)
        # video
        self.frame_min_pixels = args.frame_min_pixels
        self.frame_max_pixels = args.frame_max_pixels
        self.video_max_pixels = args.video_max_pixels
        self.fps = args.fps
        self.fps_min_frames = args.fps_min_frames
        self.fps_max_frames = args.fps_max_frames
        # image
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels

    def _resize_video(self, vision: AVData, image_factor=28, frame_factor=2):
        """Resize video: frame number, height, width"""
        if _ENERGON_NEEDS_SUBFLAVOR:
            total_frames = len(vision.frames)                     
            video_fps = vision.info["video_fps"]                  
            vision.info["fps"] = self.fps                         
            vision.info["min_frames"] = self.fps_min_frames       
            vision.info["max_frames"] = self.fps_max_frames      

            nframes = smart_nframes(                              
                vision.info, total_frames=total_frames, video_fps=video_fps         
            )
            idx = torch.linspace(0, total_frames - 1, nframes).round().long()   
            video = vision.frames[idx]                                  
        else:
            _, total_frames = vision.get_video_duration(get_frame_count=True)
            video_fps = vision.get_video_fps()
            if not hasattr(vision, "info") or vision.info is None:
                vision.info = {}

            vision.info["video_fps"] = video_fps
            vision.info["fps"] = self.fps
            vision.info["min_frames"] = self.fps_min_frames
            vision.info["max_frames"] = self.fps_max_frames

            # resize frame
            nframes = smart_nframes(
                vision.info, total_frames=total_frames, video_fps=video_fps
            )
            idx = torch.linspace(0, total_frames - 1, nframes).round().long()
            frame_ranges = [(int(i), int(i) + 1) for i in idx.tolist()]
            clips = vision.get_clips(video_clip_ranges=frame_ranges, video_unit="frames")
            video = torch.stack([clip[0] for clip in clips.video_clips], dim=0)
        # resize height, width
        nframes, _, height, width = video.shape                       
        resized_height, resized_width = smart_resize(                 
            height,
            width,
            factor=image_factor,
            min_pixels=int(self.frame_min_pixels * 1.05),
            max_pixels=min(
                self.frame_max_pixels, self.video_max_pixels / nframes * frame_factor
            ),
        )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        return video

    def _resize_image(self, image, size_factor=28):
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=size_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((resized_width, resized_height))

        return image

    def _process(self, image, text):
        """ " Process the data to get the model's input"""
        inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0].logical_not()
        image_grid_thw = None
        pixel = []
        if image is not None:
            image_grid_thw = inputs["image_grid_thw"]  # [t,h,w]
            pixel = [inputs["pixel_values"]]  # [hw, 2*3*14*14]

        target = input_ids.clone()
        vision_start_id, img_pad_id, vision_end_id = (
            self.tokenizer.convert_tokens_to_ids(
                [VISION_TAGS[0], IMAGE_TOKEN, VISION_TAGS[1]]
            )
        )
        target[target == vision_start_id] = IGNORE_INDEX
        target[target == img_pad_id] = IGNORE_INDEX
        target[target == vision_end_id] = IGNORE_INDEX

        return input_ids, target, pixel, image_grid_thw, attn_mask

    def process_sft_vqa(self, context, answer, image):
        """process the data for sft vqa"""
        text = self.processor.apply_chat_template(
            [
                {"role": "user", "content": context},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
        if text[-1] == "\n":
            text = text[:-1]
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answer = self.tokenizer.tokenize(answer)
        target[-len(answer) - 1 : -1] = torch.tensor(answer)

        return input_ids, target, attn_mask, imgs, image_grid_thw

    def process_sft_qa(
        self, messages: list, system: str, raw_video: list, raw_image: list
    ):
        """process the data for sft qa"""
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        pixel_values_images = []
        video = []
        image = []

        if raw_image is not None:
            for i in raw_image:
                image.append(self._resize_image(i))

        if raw_video is not None:
            for v in raw_video:
                video.append(self._resize_video(v))

        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image if image is not None else [],
            video if raw_video is not None else [],
            self.processor,
        )
        if raw_video is not None:
            video_grid_thw = mm_inputs["video_grid_thw"]
            pixel_values_videos = [mm_inputs["pixel_values_videos"]]
        if raw_image is not None:
            image_grid_thw = mm_inputs["image_grid_thw"]
            pixel_values_images = [mm_inputs["pixel_values"]]

        encode_pairs = self.chat_template.encode_multiturn(
            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )

        input_ids, target = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
            input_ids += source_ids + target_ids
            target += [IGNORE_INDEX] * len(source_ids) + target_ids
        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        attn_mask = torch.zeros_like(input_ids).bool()

        return (
            input_ids,
            target,
            attn_mask,
            pixel_values_images,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )

    def encode_captioning(self, sample: CaptioningSample) -> BaseTaskSample:
        """Encode CaptioningSample."""
        """Preprocessing function for datasets like COCO, containing image-caption pairs.
        See Energon codebase for more details on CaptioningSample.
        https://github.com/NVIDIA/Megatron-Energon/blob/develop/src/megatron/energon/flavors/captioning.py
        """

        assert self.args.training_phase == constants.TrainingPhase.PRETRAIN, "Only support PRETRAIN phase"

        text = (
            IMAGE_TOKEN_WITH_TAGS + sample.caption + self.tokenizer.tokenizer.eos_token
        )

        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(
            sample.image, text
        )
        num_tiles = [len(image_grid_thw)]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} thw {image_grid_thw}"
        
        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def encode_vqa(self, sample: VQASample) -> BaseTaskSample:
        """Encode pretrain sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                text = (sample.context + sample.answers).replace(
                    "<image>", IMAGE_TOKEN_WITH_TAGS
                )
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            text = text + self.tokenizer.tokenizer.eos_token
            input_ids, target, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        elif self.args.training_phase == constants.TrainingPhase.SFT:
            input_ids, target, attn_mask, imgs, image_grid_thw = self.process_sft_vqa(sample.context, \
                                        sample.answers, sample.image)
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        num_tiles = [len(image_grid_thw)]

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def encode_multi_vid_qa(self, sample: VQASample) -> BaseTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                video,
                video_grid_thw,
            ) = self.process_sft_qa(sample.messages, sample.system, sample.video, None)
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert (
                video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {video_grid_thw}"


        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=video,
                video_grid_thw=video_grid_thw,
                num_tiles=[len(video_grid_thw)],
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=video,
                video_grid_thw=video_grid_thw,
                num_tiles=[len(video_grid_thw)],
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )


    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> BaseTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            num_tiles = []

            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            ) = self.process_sft_qa(
                sample.messages, sample.system, sample.video, sample.image
            )
            if sample.video is not None:
                num_tiles = [len(video_grid_thw)]
            elif sample.image is not None:
                num_tiles = [len(image_grid_thw)]
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        elif sample.video is not None:
            assert (
                video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {video_grid_thw}"
        elif sample.image is not None:
            assert (
                image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"


        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def encode_packed_captioning(
        self, sample: PackedCaptioningSample
    ) -> BaseTaskSample:
        """Generates an encoded multimodal packed captioning sample from a raw sample."""
        n_orig_sample = len(sample.images)
        l_VLMTaskSample = []
        for idx in range(n_orig_sample):
            if _ENERGON_NEEDS_SUBFLAVOR:
                cur_capsample = CaptioningSample(
                    __key__=f"{sample.__key__}.img{idx:03d}_jpg",
                    __restore_key__=sample.__restore_key__,
                    __subflavor__=None,
                    __subflavors__=sample.__subflavors__,
                    image=sample.images[idx],
                    caption=sample.captions[idx],
                )
            else:
                cur_capsample = CaptioningSample(
                    __key__=f"{sample.__key__}.img{idx:03d}_jpg",
                    __restore_key__=sample.__restore_key__,
                    __subflavors__=sample.__subflavors__,
                    image=sample.images[idx],
                    caption=sample.captions[idx],
                )
            l_VLMTaskSample.append(self.encode_captioning(cur_capsample))
        l_sample_packed = self.pack_selected_samples(l_VLMTaskSample)
        self.is_packing_enabled = True
        return l_sample_packed

    def encode_packed_vqa(self, sample: PackedVQASample) -> BaseTaskSample:
        """Generates an encoded multimodal packed vqa sample from a raw sample."""
        n_orig_sample = len(sample.images)
        l_VLMTaskSample = []
        for idx in range(n_orig_sample):
            if _ENERGON_NEEDS_SUBFLAVOR:
                cur_capsample = VQASample(
                    __key__=f"{sample.__key__}.img{idx:03d}_jpg",
                    __restore_key__=sample.__restore_key__,
                    __subflavor__=None,
                    __subflavors__=sample.__subflavors__,
                    image=sample.images[idx],
                    answers=sample.answers[idx],
                    context=sample.contexts[idx],
                )
            else:
                cur_capsample = VQASample(
                    __key__=f"{sample.__key__}.img{idx:03d}_jpg",
                    __restore_key__=sample.__restore_key__,
                    __subflavors__=sample.__subflavors__,
                    image=sample.images[idx],
                    answers=sample.answers[idx],
                    context=sample.contexts[idx],
                )
            l_VLMTaskSample.append(self.encode_vqa4packing(cur_capsample))
        l_sample_packed = self.pack_selected_samples(l_VLMTaskSample)
        self.is_packing_enabled = True
        return l_sample_packed

    def encode_packed_multi_mix_qa(
        self, sample: PackedMultiMixQASample
    ) -> BaseTaskSample:
        """Generates an encoded multimodal packed multi mix qa sample from a raw sample."""
        n_orig_sample = len(sample.contexts)
        l_VLMTaskSample = []
        images = sample.images if sample.images is not None else []
        videos = sample.videos if sample.videos is not None else []

        has_images = len(images) > 0
        has_videos = len(videos) > 0
        if has_images and has_videos:
            raise ValueError(
                f"encode_packed_multi_mix_qa: cannot mix images and videos in same sample for key={sample.__key__}"
            )
        media_list = images if has_images else videos
        media_type = "image" if has_images else "video"

        if len(media_list) != n_orig_sample:
            raise ValueError(
                f"encode_packed_multi_mix_qa: media count ({len(media_list)}) "
                f"!= context count ({n_orig_sample}) for key={sample.__key__}"
            )
        for idx in range(n_orig_sample):
            context = sample.contexts[idx]  # str
            media_group = media_list[idx]  # List[Tensor] or List[AVData]
            answer_group = sample.answers[idx] if sample.answers else []  # List[str]

            if isinstance(answer_group, list):
                answer = "\n\n".join(answer_group) if answer_group else ""
            else:
                answer = answer_group or ""

            messages = [
                {"role": "user", "content": context},
                {"role": "assistant", "content": answer},
            ]
            if has_images:
                init_kwargs = {
                    "__key__": f"{sample.__key__}.q{idx:03d}",
                    "__restore_key__": sample.__restore_key__,
                    "__subflavors__": sample.__subflavors__,
                    "messages": messages,
                    "image": media_group,
                    "video": None,
                    "system": None,
                }
                if _ENERGON_NEEDS_SUBFLAVOR:
                    init_kwargs["__subflavor__"] = None
                cur_sample = MultiMixQASample(**init_kwargs)
            else:  # has_videos
                init_kwargs = {
                    "__key__": f"{sample.__key__}.q{idx:03d}",
                    "__restore_key__": sample.__restore_key__,
                    "__subflavors__": sample.__subflavors__,
                    "messages": messages,
                    "image": None,
                    "video": media_group,  # List[AVData]
                    "system": None,
                }
                if _ENERGON_NEEDS_SUBFLAVOR:
                    init_kwargs["__subflavor__"] = None
                cur_sample = MultiMixQASample(**init_kwargs)
            l_VLMTaskSample.append(self.encode_multi_mix_qa4packing(cur_sample))
        l_sample_packed = self.pack_selected_samples(l_VLMTaskSample)
        self.is_packing_enabled = True
        return l_sample_packed

    def encode_multi_mix_qa4packing(self, sample: MultiMixQASample) -> BaseTaskSample:
        """Encode MultiMixQASample in Qwen2VL style."""

        if self.args.training_phase == constants.TrainingPhase.SFT:
            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            ) = self.process_sft_qa(
                sample.messages, sample.system, sample.video, sample.image
            )

            num_tiles = []
            if sample.video is not None:
                num_tiles = [len(video_grid_thw)]
            elif sample.image is not None:
                num_tiles = [len(image_grid_thw)]
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        elif sample.video is not None:
            assert (
                video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {video_grid_thw}"
        elif sample.image is not None:
            assert (
                image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
    

    def encode_vqa4packing(self, sample: VQASample) -> BaseTaskSample:
        """Encode VQASample in Qwen2VL style."""

        text = self.processor.apply_chat_template(
            [
                {"role": "user", "content": sample.context},
                {"role": "assistant", "content": sample.answers},
            ],
            tokenize=False,
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)

        if text[-1] == "\n":
            text = text[:-1]
            pass

        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(
            sample.image, text
        )
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answers = self.tokenizer.tokenize(sample.answers)
        target[-len(answers) - 1 : -1] = torch.tensor(answers)
        target[-1] = input_ids[-1]

        num_tiles = [len(image_grid_thw)]
        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert (
                image_grid_thw.prod() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def process_samples_grid(self, samples):
        """concat grid_thw for image and video"""
        image_grid_thw = [
            x.image_grid_thw for x in samples if x.image_grid_thw is not None
        ]
        video_grid_thw = [
            x.video_grid_thw for x in samples if x.video_grid_thw is not None
        ]

        if len(image_grid_thw) > 0:
            image_grid_thw = torch.cat(image_grid_thw).to(dtype=torch.int32)
        else:
            image_grid_thw = None

        if len(video_grid_thw) > 0:
            video_grid_thw = torch.cat(video_grid_thw).to(dtype=torch.int32)
        else:
            video_grid_thw = None

        return image_grid_thw, video_grid_thw

    @override
    @stateless
    def pack_selected_samples(
        self, samples: List[VLMTaskSample]
    ) -> List[VLMTaskSamplePacked]:
        """Pack selected samples into one big sample."""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return VLMTaskSamplePacked(
            super().pack_selected_samples(samples),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

    @override
    def batch(
        self, samples: List[Union[VLMTaskSample, VLMTaskSamplePacked]]
    ) -> VLMTaskBatchPacked:
        """Batch samples together"""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return VLMTaskBatchPacked(
            super().batch(samples),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

    @override
    def process_images(
        self, samples: List[Union[VLMTaskSample, VLMTaskSamplePacked]]
    ) -> torch.Tensor:
        """ " Process the data to get the model's input"""
        imgs = [img for s in samples if s.imgs is not None for img in s.imgs]
        if len(imgs) > 0:
            return torch.cat(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def process_videos(
        self, samples: List[Union[VLMTaskSample, VLMTaskSamplePacked]]
    ) -> torch.Tensor:
        """ " Process the data to get the model's input"""
        pixel_values_videos = [
            pixel_values_video
            for s in samples
            if s.pixel_values_videos is not None
            for pixel_values_video in s.pixel_values_videos
        ]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)
