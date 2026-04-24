# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the definition of the VisionProcessor class.
Uses transformers.image_utils.ChannelDimension for channel dimension handling.
"""

import copy
import json
import logging
import random
from collections import OrderedDict, namedtuple
from itertools import groupby

import numpy as np

from .utils.constant import DATATYPE_2_ID, IDTYPES_2_ID, IMAGETYPES_2_ID
from .utils.image_enhance import ImageEnhance
import logging

logger = logging.getLogger(__name__)
from .utils.processor_base import ProcessorBase
from .tokenizer_vl import special_tokens_info

try:
    from .utils.io_utils import get_downloadable_image
except Exception as e:
    logger.warning(f" decord not found: {e}")
    get_downloadable_image = None

# ── pure-transformers replacement ────────────────────────────────────────────
from transformers.image_utils import ChannelDimension
# ─────────────────────────────────────────────────────────────────────────────

VisionExample = namedtuple(
    "Example",
    [
        "meta",
        "ids",
        "sids",
        "task",
        "lossmask",
        "src",
        "part",
        "info",
        "name",
        "data_type",
        "token_type_ids",
        "image_type_ids",
    ],
)

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


class ImageModificationProcessor(ProcessorBase):
    """
    ImageModificationProcessor
    """

    def __init__(self, args, tokenizer, image_preprocess):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.image_token_len = args.image_token_len
        self.image_preprocess = image_preprocess
        self.image_dtype = args.image_dtype
        self.variable_resolution = True
        self.rope_3d = True

        vocab = self.tokenizer.get_vocab()
        self.im_patch_id = vocab[special_tokens_info["image_placeholder"]]
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>"
        )
        self.eos_token_id = vocab[self.eos_token]
        self.cls_token_id = vocab[self.cls_token]
        self.sep_token_id = vocab[self.sep_token]
        self.sft_shift_by_one = args.sft_shift_by_one
        self.chat_template = "ernie_vl"
        self.should_shift_by_one = self.is_training and (
            self.is_pretraining or self.sft_shift_by_one
        )
        self.sft_replace_ids = args.sft_replace_ids
        self.sft_image_rescale = args.sft_image_rescale
        self.sft_image_normalize = args.sft_image_normalize

    def get_rope_index(
        self,
        spatial_merge_size,
        temporal_merge_size,
        image_token_id,
        video_token_id,
        vision_start_indices,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        """Calculate the 3D rope index based on image and video's temporal, height and width in LLM."""
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = np.ones(
                [3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = np.array(input_ids[attention_mask[i] == 1])
                image_nums, video_nums = 0, 0
                vision_start_indices_tmp = vision_start_indices[i]
                vision_tokens = input_ids[vision_start_indices_tmp]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item() if t.item() == 1 else t.item() // temporal_merge_size,
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )

                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape([1, -1]).repeat(3, axis=0) + st_idx
                    )

                    t_index = np.tile(
                        np.arange(llm_grid_t).reshape([-1, 1]),
                        ([1, llm_grid_h * llm_grid_w]),
                    ).flatten()
                    h_index = np.tile(
                        np.arange(llm_grid_h).reshape([1, -1, 1]),
                        ([llm_grid_t, 1, llm_grid_w]),
                    ).flatten()
                    w_index = np.tile(
                        np.arange(llm_grid_w).reshape([1, 1, -1]),
                        ([llm_grid_t, llm_grid_h, 1]),
                    ).flatten()

                    llm_pos_ids_list.append(
                        np.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape([1, -1]).repeat(3, axis=0) + st_idx
                    )

                llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape([3, -1])
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = np.expand_dims(np.array(mrope_position_deltas), 1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = np.asarray(attention_mask, dtype="int64").cumsum(-1) - 1
                position_ids.masked_fill_(mask=attention_mask == 0, value=1)
                position_ids = position_ids.unsqueeze(0).tile([3, 1, 1])
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    np.arange(input_ids.shape[1])
                    .reshape([1, 1, -1])
                    .tile([3, input_ids.shape[0], 1])
                )
                mrope_position_deltas = np.zeros(
                    [input_ids.shape[0], 1], dtype=input_ids.dtype
                )
            return position_ids, mrope_position_deltas

    def position_ids_for_rope_3d(self, feature):
        """
        position_ids_for_rope_3d
        """
        if feature.get("images", None) is None or len(feature["images"]) == 0:
            position_ids = np.repeat(
                np.arange(feature["input_ids"].shape[0])[:, np.newaxis], 3, axis=1
            )
            feature["position_ids"] = position_ids
            return feature

        input_ids = copy.deepcopy(feature["input_ids"])
        grid_thw = feature["grid_thw"]
        token_type_ids = (
            feature["token_type_ids"][:-1]
            if self.should_shift_by_one
            else feature["token_type_ids"]
        )
        image_type_ids = feature["image_type_ids"]

        fake_image_token_id = -10000
        fake_video_token_id = -20000

        input_ids[
            np.bitwise_and(
                token_type_ids == IDTYPES_2_ID["image"], input_ids == self.im_patch_id
            )
        ] = fake_image_token_id
        input_ids[
            np.bitwise_and(
                token_type_ids == IDTYPES_2_ID["video"], input_ids == self.im_patch_id
            )
        ] = fake_video_token_id

        visual_token_indices = np.nonzero(feature["input_ids"] == self.im_patch_id)
        visual_token_indices = np.stack(visual_token_indices, axis=0).flatten()

        vision_start_indices = []
        image_grid_thw = []
        video_grid_thw = []
        index_of_visual_token_indices = 0
        index_of_image_type_ids = 0

        for cur_grid_thw in grid_thw:
            vision_start_indices.append(
                visual_token_indices[index_of_visual_token_indices]
            )
            index_of_visual_token_indices += (
                cur_grid_thw[0]
                * cur_grid_thw[1]
                * cur_grid_thw[2]
                // (self.image_preprocess.merge_size**2)
            )
            if image_type_ids[index_of_image_type_ids] == IMAGETYPES_2_ID["image"]:
                image_grid_thw.append(cur_grid_thw)
            else:
                video_grid_thw.append(cur_grid_thw)
                index_of_visual_token_indices //= (
                    self.image_preprocess.temporal_conv_size
                )

        position_ids, position_ids_delta = self.get_rope_index(
            self.image_preprocess.merge_size,
            self.image_preprocess.temporal_conv_size,
            fake_image_token_id,
            fake_video_token_id,
            np.array([vision_start_indices]),
            np.array([input_ids]),
            image_grid_thw,
            video_grid_thw,
            attention_mask=np.ones([1, input_ids.shape[-1]]),
        )
        position_ids = np.squeeze(position_ids, axis=1).transpose([1, 0])

        feature["position_ids"] = position_ids
        return feature

    def image_handling_for_adaptive(self, example, download_fn):
        """
        image_handling_for_adaptive
        """
        pixel_values_list = []
        grid_thw_list = []

        image_type_ids = np.array(example.image_type_ids)
        image_type_ids[image_type_ids == IMAGETYPES_2_ID["padded_image"]] = (
            IMAGETYPES_2_ID["video"]
        )
        image_type_ids = image_type_ids.tolist()

        metas = []
        for meta in example.meta:
            if isinstance(meta, np.ndarray):
                meta = json.loads(meta.tobytes().decode())
            metas.extend(meta)

        assert len(example.image_type_ids) == len(metas), (
            f"len(image_type_ids) {len(example.image_type_ids)} != len(metas) {len(metas)}"
            f", image_type_ids: {example.image_type_ids}, metas: {metas}"
        )

        for key, group in groupby(zip(image_type_ids, metas), key=lambda x: x[0]):
            imgs = []
            predetermined_grid_thw = []
            uids = []
            for img_one in group:
                img_one = img_one[1]

                img = download_fn(
                    img_one["image_url"],
                    need_exif_info=False,
                )[0]

                random_resize_factor = img_one.get("random_resize_factor", 1)
                image_enhance_augs = img_one.get("image_enhance_augs", None)
                img = ImageEnhance.apply_effect(img, image_enhance_augs, random_resize_factor)

                imgs.append(img.convert("RGB"))
                predetermined_grid_thw.append(
                    [img_one.get("grid_h", -1), img_one.get("grid_w", -1)]
                )
                uids.append(img_one.get("video_uid", random.random()))

            predetermined_grid_thw = np.array(predetermined_grid_thw)
            if predetermined_grid_thw[0][0] < -1:
                predetermined_grid_thw = None
            if key == IMAGETYPES_2_ID["image"]:
                ret = self.image_preprocess.preprocess(
                    images=imgs,
                    videos=None,
                    do_normalize=self.sft_image_normalize,
                    do_rescale=self.sft_image_rescale,
                    predetermined_grid_thw=predetermined_grid_thw,
                    do_convert_rgb=True,
                    input_data_format=ChannelDimension.LAST,
                )
                pixel_values = ret["pixel_values"]
                grid_thw = ret["image_grid_thw"]

                pixel_values_list.append(pixel_values)
                grid_thw_list.append(grid_thw)

            elif key == IMAGETYPES_2_ID["video"]:
                cnt = 0
                for uid, group in groupby(zip(uids, imgs), key=lambda x: x[0]):
                    grouped_imgs = [i[1] for i in group]
                    if predetermined_grid_thw is not None:
                        cur_predetermined_grid_thw = predetermined_grid_thw[
                            cnt : cnt + len(grouped_imgs)
                        ]
                    else:
                        cur_predetermined_grid_thw = None
                    cnt += len(grouped_imgs)
                    ret = self.image_preprocess.preprocess(
                        images=None,
                        videos=np.stack(
                            [np.array(img.convert("RGB")) for img in grouped_imgs],
                            axis=0,
                        ),
                        do_normalize=self.sft_image_normalize,
                        do_rescale=self.sft_image_rescale,
                        predetermined_grid_thw=cur_predetermined_grid_thw,
                        do_convert_rgb=True,
                        input_data_format=ChannelDimension.LAST,
                    )
                    pixel_values = ret["pixel_values_videos"]
                    grid_thw = ret["video_grid_thw"]

                    pixel_values_list.append(pixel_values)
                    grid_thw_list.append(grid_thw)
            else:
                raise ValueError(f"encounter unsupported image type! {key}")

        pixel_values_list = np.concatenate(pixel_values_list, axis=0)
        grid_thw_list = np.concatenate(grid_thw_list, axis=0)

        return pixel_values_list, grid_thw_list

    def mm_example_to_feature(self, example, download_fn=None):
        """
        mm_example_to_feature
        """
        download_fn = download_fn or get_downloadable_image
        try:
            assert isinstance(example, VisionExample), " only support VisionExample"
            images, grid_thw = self.image_handling_for_adaptive(
                example, download_fn=download_fn
            )
            input_ids = np.array(example.ids, dtype=np.int64)
            token_type_ids = np.array(example.token_type_ids, dtype=np.int64)
            image_type_ids = np.array(example.image_type_ids, dtype=np.int64)
            image_type_ids[image_type_ids == IMAGETYPES_2_ID["padded_image"]] = (
                IMAGETYPES_2_ID["video"]
            )

            if example.lossmask is not None:
                labels = np.array(
                    [
                        self.tokenizer.ignored_index if j == 0 else i
                        for i, j in zip(example.ids, example.lossmask)
                    ],
                    dtype=np.int64,
                )
            else:
                labels = input_ids
                print("example.lossmask is not None, labels=input_ids")

            if not self.is_pretraining:
                replace_token_id = self.cls_token_id
                if self.chat_template == "ernie":
                    replace_token_id = self.cls_token_id
                elif (
                    self.chat_template == "ernie_vl"
                    or self.chat_template == "ernie_vl_thinking"
                ):
                    replace_token_id = self.sep_token_id
                else:
                    raise NotImplementedError(
                        f"{self.chat_template} is not supported now."
                    )
                labels[labels == replace_token_id] = self.eos_token_id
                print(
                    "labels[labels == replace_token_id] = self.eos_token_id",
                    f"self.should_shift_by_one: {self.should_shift_by_one}",
                )
                if self.sft_replace_ids:
                    input_ids[input_ids == replace_token_id] = self.eos_token_id

            features = OrderedDict(
                src_id=example.src,
                images=images,
                input_ids=input_ids[:-1] if self.should_shift_by_one else input_ids,
                labels=labels[1:] if self.should_shift_by_one else labels,
                data_type=(
                    DATATYPE_2_ID["mm"] if images is not None else DATATYPE_2_ID["lm"]
                ),
                token_type_ids=token_type_ids,
                image_type_ids=image_type_ids,
                data_not_valid=0,
                grid_thw=grid_thw,
            )
        except Exception as e:
            logger.exception(e)
            if not self.is_training:
                raise e
            if self.variable_resolution:
                images = np.zeros(
                    [4, 3 * (self.image_preprocess.patch_size**2)],
                    dtype=self.image_dtype,
                )
                grid_thw = np.array([[1, 2, 2]])
                input_ids = np.array([self.im_patch_id] * 1 + [1])
                labels = (
                    np.ones_like([self.im_patch_id] * 1 + [1])
                    * self.tokenizer.ignored_index
                )
                print(f"Exception create: labels")
                token_type_ids = np.array(
                    1 * [IDTYPES_2_ID["image"]] + 1 * [IDTYPES_2_ID["text"]],
                    dtype="int64",
                )
                image_type_ids = np.array(1 * [IMAGETYPES_2_ID["image"]])
                features = OrderedDict(
                    src_id=example.src,
                    images=images,
                    input_ids=input_ids,
                    labels=labels,
                    data_type=DATATYPE_2_ID["mm"],
                    token_type_ids=token_type_ids,
                    image_type_ids=image_type_ids,
                    data_not_valid=1,
                    grid_thw=grid_thw,
                )
        finally:
            pass

        return features

    def fill_empty_field_in_features(self, features):
        """make sure all fields are filled"""
        new_features = OrderedDict(
            src_id=features["src_id"],
            input_ids=features["input_ids"],
            labels=features["labels"],
            data_type=features["data_type"],
            token_type_ids=features["token_type_ids"],
            images=features.get("images", None),
            image_type_ids=features.get("image_type_ids", None),
            data_not_valid=features.get(
                "data_not_valid", np.array([1], dtype="float32")
            ),
            grid_thw=features.get("grid_thw", None),
            position_ids=features.get("position_ids", None),
        )
        return new_features

    def json_2_example(self, data):
        """Convert json format data to VisionExample"""
        def _vision_key_formatting(data):
            ret = {}
            ret["meta"] = data["meta"]
            ret["ids"] = data["ids"] if "ids" in data else data["ds16"]
            ret["sids"] = None
            ret["task"] = "mm"
            ret["src"] = data.get("part", -1)
            ret["part"] = data.get("part", -1)
            ret["lossmask"] = (
                data["lossmask"] if "lossmask" in data else data["ds16_lossmask"]
            )
            ret["info"] = -1
            ret["name"] = "dummy"
            ret["data_type"] = DATATYPE_2_ID["mm"]
            ret["token_type_ids"] = (
                data["token_type_ids"]
                if "token_type_ids" in data
                else data["ds16_tokenwise_type_id"]
            )
            ret["image_type_ids"] = (
                data["image_type_ids"]
                if "image_type_ids" in data
                else data["ds16_imagewise_type_id"]
            )
            return ret

        assert isinstance(data, dict)
        data = _vision_key_formatting(data)
        return VisionExample(**data)

    def process(self, data, **kwargs):
        """Convert json format data to VisionExample, and then to features"""
        self.should_shift_by_one = self.is_training and (
            self.is_pretraining or self.sft_shift_by_one
        )
        if isinstance(data, dict):
            example = self.json_2_example(data)
        else:
            example = data
        features = self.mm_example_to_feature(example, kwargs.get("download_fn", None))

        if self.rope_3d:
            features = self.position_ids_for_rope_3d(features)

        features = self.fill_empty_field_in_features(features)
        return features
