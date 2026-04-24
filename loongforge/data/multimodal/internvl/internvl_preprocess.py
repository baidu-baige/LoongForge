# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from InternVL.
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License
# --------------------------------------------------------


"""Internvl preprocessing"""

from typing import Dict, Tuple, List, Union
from copy import deepcopy
import dataclasses
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import transformers
import av
import sys
import cv2

IGNORE_TOKEN_ID = -100
IGNORE_INDEX = -100

from .internvl_constants import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CLIP_MEAN,
    CLIP_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
)


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + message + self.sep
            else:
                ret += role
        return ret

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        """Copy the current instance"""
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        """Convert the conversation to a dictionary"""
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


class InternvlPreprocess:
    """Internvl Preprocess"""

    def __init__(self, args, tokenizer):
        self.args = args
        super().__init__()
        self.image_size = args.force_image_size
        self.is_train = False
        self.use_thumbnail = args.use_thumbnail
        self.pad2square = args.pad2square
        self.normalize_type = args.normalize_type
        self.dynamic_image_size = args.dynamic_image_size
        self.min_dynamic_patch = args.min_dynamic_patch
        self.template_name = args.conv_style
        self.num_image_token = int(
            (args.force_image_size // args.patch_size) ** 2
            * (args.down_sample_ratio**2)
        )
        self.group_by_length = args.group_by_length and not args.packing_sft_data
        self.packing_sft_data = args.packing_sft_data
        self.tokenizer = tokenizer
        self.num_images_expected = args.num_images_expected
        self.max_num_frame = args.max_num_frame
        self.min_num_frame = args.min_num_frame
        self.bos_client = None
        self.sampling_method = "rand"
        self.max_dynamic_patch = args.max_dynamic_patch

    def video_get_item(self, data_item):
        """Process videos + texts"""
        # # Build transformation function
        # Ensure the first conversation contains a video placeholder
        first_turn_idx = 1 if data_item["texts"][0]["value"] == "system" else 0
        if "<video>" not in data_item["texts"][first_turn_idx]["value"]:
            data_item["texts"][first_turn_idx]["value"] = (
                "<video>\n" + data_item["texts"][first_turn_idx]["value"]
            )

        # Get the video file path
        video_file = data_item["videos"]
        if isinstance(video_file, list):
            vison = video_file[0]

        image_list = self.read_frames_decord_opencv(
            vison,
            num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            client=self.bos_client,
            sample=self.sampling_method,
            clip=data_item.get("clip", None),
        )

        # Generate special tokens for each video frame
        special_tokens = "\n".join(
            ["Frame-{}: <image>".format(i + 1) for i in range(len(image_list))]
        )
        data_item["texts"][first_turn_idx]["value"] = data_item["texts"][
            first_turn_idx
        ]["value"].replace("<video>\n", special_tokens + "\n")

        # Transform each frame image and stack them into a tensor
        transform = self.build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )

        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        # Select the appropriate preprocessing function based on the template name
        use_pretrain = data_item["texts"][first_turn_idx]["from"] == "pretrain"
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["texts"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            is_packing_enabled=self.packing_sft_data,
            ds_name=None,
            num_image=num_patches,
        )
        # Calculate position_ids for packed dataset
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        # print(f"video encode data: {ret}")
        return ret

    def multi_image_get_item(self, data_item):
        """Process images + texts"""
        # # Build transformation function
        transform = self.get_transform()
        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item["texts"][0]["value"] == "system" else 0
        if "<image>" not in data_item["texts"][first_turn_idx]["value"]:
            data_item["texts"][first_turn_idx]["value"] = (
                "<image>\n" * len(data_item["image"])
                + data_item["texts"][first_turn_idx]["value"]
            )

        image_tiles, num_tiles = [], []
        num_image = len(data_item["image"])
        for image in data_item["image"]:
            # If dynamic image size is enabled, preprocess the image dynamically
            if self.dynamic_image_size:
                images = self.dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // num_image),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                image_tiles += images
                num_tiles.append(len(images))
            else:  # Otherwise, use the original image as a single patch
                image_tiles.append(image)
                num_tiles.append(1)

        pixel_values = [transform(image) for image in image_tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = data_item["texts"][first_turn_idx]["from"] == "pretrain"
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["texts"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            is_packing_enabled=self.packing_sft_data,
            ds_name=None,
            num_image=num_image,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1

        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (
            ret["input_ids"][0] == image_end_token_id
        ).sum() == num_image, (
            f"image tokens are truncated, this dataset is {self.ds_name}"
        )
        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def pure_text_get_item(self, data_item):
        """Process texts only"""
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new("RGB", (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = self.dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=1,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert (
            num_patches == 1
        ), f"The number of patches should be 1, but got {num_patches}."
        # Select the appropriate preprocessing function based on the template name
        use_pretrain = data_item["texts"][0]["from"] == "pretrain"
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)
        preprocess_function = self.get_preprocess_function()

        # Preprocess the texts and generate the return dictionary
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["texts"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=self.group_by_length,
            is_packing_enabled=self.packing_sft_data,
            ds_name=None,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        )
        return ret

    def get_preprocess_function(self, use_pretrain=False):
        """Select the appropriate preprocessing function based on the template name"""

        assert use_pretrain is False, "only support sft for now"

        if self.template_name == "internvl2_5":
            preprocess_function = self.preprocess_internvl2_5
        elif self.template_name == "internvl3_5_gpt_oss":
            preprocess_function = self.preprocess_internvl3_5_gpt_oss
        else:
            preprocess_function = self.preprocess
        return preprocess_function

    def get_transform(self):
        """Build transformation function"""
        transform = self.build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def build_transform(
        self, is_train, input_size, pad2square=False, normalize_type="imagenet"
    ):
        """Build transform for image"""
        if normalize_type == "imagenet":
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        elif normalize_type == "clip":
            MEAN, STD = CLIP_MEAN, CLIP_STD
        elif normalize_type == "siglip":
            MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
        else:
            raise NotImplementedError
        if is_train:  # use data augumentation
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.RandomChoice(
                        [
                            T.Lambda(jpeg_degrade_functions[quality])
                            for quality in qualities
                        ]
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            if pad2square is False:  # now we use this transform function by default
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.Resize(
                            (input_size, input_size),
                            interpolation=InterpolationMode.BICUBIC,
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )
            else:
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.Lambda(
                            lambda img: expand2square(
                                img, tuple(int(x * 255) for x in MEAN)
                            )
                        ),
                        T.Resize(
                            (input_size, input_size),
                            interpolation=InterpolationMode.BICUBIC,
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )

        return transform

    def get_conv_template(self, template_name):
        """Get conv template"""
        return Conversation(
            name="internvl2_5",
            system_template="<|im_start|>system\n{system_message}",
            system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            sep="<|im_end|>\n",
        )

    def preprocess_internvl2_5(
        self,
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        is_packing_enabled: bool = False,
        ds_name: str = None,
        num_image: int = 1,
    ) -> Dict:
        """Process internvl2.5"""
        assert len(sources) == 1, "process only the first conversations"
        conversations = sources[0]

        if conversations[0]["from"] == "system":
            system_prompt = conversations[0]["value"]
            conversations = conversations[1:]  # remove system prompt
        else:
            conv = self.get_conv_template(template_name)
            system_prompt = conv.system_message

        if not text_only:
            new_conversations = []
            current_image_idx = 0
            for conversation in conversations:
                if conversation["from"] == "human":
                    image_cnt = conversation["value"].count("<image>")
                    for i in range(image_cnt):
                        if current_image_idx == num_image:
                            break
                        image_tokens = (
                            f"{IMG_START_TOKEN}"
                            f"{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}"
                            f"{IMG_END_TOKEN}"
                        )
                        conversation["value"] = conversation["value"].replace(
                            "<image>", image_tokens, 1
                        )
                        current_image_idx += 1
                new_conversations.append(conversation)
            conversations = new_conversations
            assert current_image_idx == num_image, f"{current_image_idx} != {num_image}"

        batches, roles = [], []
        if system_prompt is not None:
            batches.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
            roles.append("system")
        for conversation in conversations:
            if conversation["from"] == "human":
                batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
                roles.append("human")
            elif conversation["from"] == "gpt":
                batches.append(
                    f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n'
                )
                roles.append("gpt")
            else:
                raise NotImplementedError

        add_bos_token = getattr(tokenizer, "add_bos_token", False)
        if add_bos_token:  # for InternLM series
            batches[0] = tokenizer.bos_token + batches[0]
        # Tokenize conversations
        input_ids = tokenizer(
            batches,
            return_tensors="np",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=False,
        ).input_ids

        if add_bos_token:  # for InternLM series
            input_ids = [item[1:] for item in input_ids]

        final_input_ids, final_targets = [], []
        ignore_ids = tokenizer(
            "<|im_start|>assistant\n", return_tensors="np"
        ).input_ids[0]
        ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
        for role, input_id in zip(roles, input_ids):
            final_input_ids.append(input_id)
            if role == "system" or role == "human":
                final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
            elif role == "gpt":
                target = input_id.copy()
                target[:ignore_len] = (
                    IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
                )
                target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
                final_targets.append(target)
            else:
                raise NotImplementedError
        input_ids = torch.tensor(np.concatenate(final_input_ids))[
            : tokenizer.model_max_length
        ]
        targets = torch.tensor(np.concatenate(final_targets))[
            : tokenizer.model_max_length
        ]

        padding = False if group_by_length or is_packing_enabled else True
        if padding:
            current_length = input_ids.size(0)
            padding_length = tokenizer.model_max_length - current_length
            input_ids = F.pad(
                input_ids, (0, padding_length), value=tokenizer.pad_token_id
            )
            targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

        input_ids = input_ids.unsqueeze(0)
        targets = targets.unsqueeze(0)
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def preprocess_internvl3_5_gpt_oss(
        self,
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        is_packing_enabled: bool = False,
        ds_name: str = None,
        num_image: int = 1,
    ) -> Dict:
        """Preprocess internvl3_5_gpt_oss"""
        assert len(sources) == 1, "process only the first conversations"
        conversations = sources[0]

        if conversations[0]["from"] == "system":
            system_prompt = conversations[0]["value"]
            conversations = conversations[1:]  # remove system prompt
        else:
            conv = self.get_conv_template(template_name)
            system_prompt = conv.system_message
            # system_prompt = None

        if not text_only:
            new_conversations = []
            current_image_idx = 0
            for conversation in conversations:
                if conversation["from"] == "human":
                    image_cnt = conversation["value"].count("<image>")
                    for i in range(image_cnt):
                        if current_image_idx == num_image:
                            break
                        image_tokens = (
                            f"{IMG_START_TOKEN}"
                            f"{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}"
                        )
                        conversation["value"] = conversation["value"].replace(
                            "<image>", image_tokens, 1
                        )
                        current_image_idx += 1
                new_conversations.append(conversation)
            conversations = new_conversations
            assert current_image_idx == num_image, f"{current_image_idx} != {num_image}"

        batches, roles = [], []
        if system_prompt is not None:
            batches.append(f"<|start|>system<|message|>{system_prompt}<|end|>")
            roles.append("system")
        for conversation in conversations:
            if conversation["from"] == "human":
                batches.append(
                    f'<|start|>user<|message|>{conversation["value"]}<|end|>'
                )
                roles.append("human")
            elif conversation["from"] == "gpt":
                batches.append(
                    f'<|start|>assistant<|channel|>final<|message|>{conversation["value"]}<|return|>'
                )
                roles.append("gpt")
            elif conversation["from"] == "function":
                batches.append(
                    f'<|start|>tool<|message|>{conversation["value"]}<|end|>'
                )
                roles.append("function")
            else:
                raise NotImplementedError(f"Invalid role: {conversation['from']}")

        add_bos_token = getattr(tokenizer, "add_bos_token", False)
        if add_bos_token:  # for InternLM series
            batches[0] = tokenizer.bos_token + batches[0]

        # Tokenize conversations
        input_ids = tokenizer(
            batches,
            return_tensors="np",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=False,
        ).input_ids

        if add_bos_token:  # for InternLM series
            input_ids = [item[1:] for item in input_ids]

        final_input_ids, final_targets = [], []
        ignore_ids = tokenizer("<|start|>assistant", return_tensors="np").input_ids[0]
        ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
        for role, input_id in zip(roles, input_ids):
            final_input_ids.append(input_id)
            if role == "system" or role == "human" or role == "function":
                final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
            elif role == "gpt":
                target = input_id.copy()
                target[:ignore_len] = (
                    IGNORE_TOKEN_ID  # ignore loss for `<|start|>assistant`
                )
                final_targets.append(target)
            else:
                raise NotImplementedError
        input_ids = torch.tensor(np.concatenate(final_input_ids))[
            : tokenizer.model_max_length
        ]
        targets = torch.tensor(np.concatenate(final_targets))[
            : tokenizer.model_max_length
        ]

        # padding = False if group_by_length or is_packing_enabled else True
        padding = False
        if padding:
            current_length = input_ids.size(0)
            padding_length = tokenizer.model_max_length - current_length
            input_ids = F.pad(
                input_ids, (0, padding_length), value=tokenizer.pad_token_id
            )
            targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

        input_ids = input_ids.unsqueeze(0)
        targets = targets.unsqueeze(0)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def preprocess(
        self,
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        is_packing_enabled: bool = False,
        ds_name: str = None,
        num_image: int = 1,
    ) -> Dict:
        """Default preprocess"""
        conv = self.get_conv_template(template_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        if not text_only:
            new_conversations = []
            for conversation in conversations:
                for i in range(num_image):
                    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                    conversation = conversation.replace("<image>", image_tokens, 1)
                new_conversations.append(conversation)
            conversations = new_conversations

        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding=False if group_by_length or is_packing_enabled else "max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                logger.info(tokenizer.decode(z))
                exit()

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}."
                    )
                    sys.stdout.flush()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        """Find closest aspect ratio"""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
    ):
        """Dynamic preprocess"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def get_frame_indices(
        self,
        num_frames,
        vlen,
        sample="rand",
        fix_start=None,
        input_fps=1,
        max_num_frames=-1,
    ):
        """Get frame indices"""

        if sample in ["rand", "middle"]:  # uniform sampling
            acc_samples = min(num_frames, vlen)
            # split the video into `acc_samples` intervals, and sample from each interval.
            intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
            ranges = []
            for idx, interv in enumerate(intervals[:-1]):
                ranges.append((interv, intervals[idx + 1] - 1))
            if sample == "rand":
                try:
                    frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
                except:
                    frame_indices = np.random.permutation(vlen)[:acc_samples]
                    frame_indices.sort()
                    frame_indices = list(frame_indices)
            elif fix_start is not None:
                frame_indices = [x[0] + fix_start for x in ranges]
            elif sample == "middle":
                frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
            else:
                raise NotImplementedError

            if len(frame_indices) < num_frames:  # padded with last frame
                padded_frame_indices = [frame_indices[-1]] * num_frames
                padded_frame_indices[: len(frame_indices)] = frame_indices
                frame_indices = padded_frame_indices
        elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
            output_fps = float(sample[3:])
            duration = float(vlen) / input_fps
            delta = (
                1 / output_fps
            )  # gap between frames, this is also the clip length each frame represents
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            frame_indices = np.around(frame_seconds * input_fps).astype(int)
            frame_indices = [e for e in frame_indices if e < vlen]
            if max_num_frames > 0 and len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
                # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
        else:
            raise ValueError
        return frame_indices

    def read_frames_decord_opencv(self, av_decoder, num_frames, sample='rand',
        fix_start=None, client=None, clip=None, min_num_frames=4):
        """ read_frames_decord """
        cap = cv2.VideoCapture(av_decoder.stream, cv2.CAP_FFMPEG, [])

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = vlen / float(fps)

        if clip:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)

        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
        frame_indices = self.get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
        if clip:
            frame_indices = [f + start_index for f in frame_indices]

        # Determine final resolution
        target_width = original_width
        target_height = original_height
        frame_list = []
        for idx in sorted(frame_indices):  # Sort indices to improve reading efficiency
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                # Generate a placeholder black frame
                frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            else:
                # Convert color space and resize if necessary
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (target_width, target_height) != (original_width, original_height):
                    frame = cv2.resize(frame, (target_width, target_height))
            
            # Convert numpy array to PIL Image
            pil_frame = Image.fromarray(frame)
            frame_list.append(pil_frame)

        cap.release()
        
        return frame_list

    def get_cu_seqlens_and_indexes(
        self,
        data_index: torch.LongTensor,  # (seq_len,)
        input_ids: torch.LongTensor,  # (seq_len,)
        labels: torch.LongTensor,  # (seq_len,)
        len2weight: callable,
    ):
        """Get cu_seqlens and indexes"""
        indexes = []
        cu_seqlens = [0]
        loss_weight = []

        start = data_index.min()
        end = data_index.max() + 1
        for i in range(start, end):
            num_tokens = (data_index == i).sum().item()
            indexes.extend(list(range(num_tokens)))
            cu_seqlens.append(cu_seqlens[-1] + num_tokens)
            assert num_tokens > 0

            curr_data_index = data_index[cu_seqlens[-2] : cu_seqlens[-2] + num_tokens]
            assert (curr_data_index == i).all(), data_index

            curr_labels = labels[cu_seqlens[-2] : cu_seqlens[-2] + num_tokens]
            num_effective_tokens = (curr_labels != IGNORE_TOKEN_ID).sum().item()
            loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)

        assert len(indexes) == data_index.size(
            0
        ), f"{len(indexes)=}, {data_index.size(0)=}"

        loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
        return cu_seqlens, indexes, loss_weight

    def pad_imgs(self, packed_sample, num_images_expected):
        """Pad images"""
        if len(packed_sample.imgs) == num_images_expected:
            return packed_sample

        num_pad_images = num_images_expected - len(packed_sample.imgs)
        pad_images = [torch.zeros_like(packed_sample.imgs[0]) for _ in range(num_pad_images)]
        pad_image_flags = torch.tensor([0] * num_pad_images, dtype=torch.long)

        packed_sample.imgs = packed_sample.imgs + pad_images
        packed_sample.image_flags = torch.cat(
            [packed_sample.image_flags, pad_image_flags]
        )

        return packed_sample

    def len2weight(self, x, loss_reduction):
        """Loss weight multiplier according to loss_reduction"""
        if x == 0:
            return x
        if loss_reduction == "token":
            return 1
        if loss_reduction == "sample":
            return 1 / x
        if loss_reduction == "square":
            return 1 / (x**0.5)
        raise NotImplementedError(loss_reduction)
