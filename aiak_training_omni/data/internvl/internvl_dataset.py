# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
""" dataset for internvl """
import dataclasses

import io
import sys
import tempfile
import time

#from transformers.trainer_pt_utils import LabelSmoother

#IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_TOKEN_ID = -100
import logging
import torch.nn.functional as F
import torchvision.transforms as T
import os
import random
from megatron.core import mpu
import traceback
import warnings
from copy import deepcopy
from typing import Dict, Tuple, List, Union

import numpy as np
from torchvision.transforms import InterpolationMode

from .dataset_packed import IGNORE_TOKEN_ID

import json

import torch
import transformers
from .constants import (IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN,
                        CLIP_STD, SIGLIP_MEAN, SIGLIP_STD)

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
import cv2
# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False
# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger('internvl')

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def bos_retry(func, *args, max_retries=3, retry_delay=1, **kwargs):
    """
    Retry wrapper for BOS requests
    Args:
        func: function to retry
        max_retries: maximum number of retry attempts (default: 3)
        retry_delay: delay between retries in seconds (default: 1)
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt, re-raise the exception
                raise e
            else:
                logger.warning(f"BOS request failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
    return None


def simulate_jpeg_degradation(quality):
    """ simulate jpeg degradation """

    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """ find closest aspect ratio """
    best_ratio_diff = float('inf')
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
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    sep: str = '\n'
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
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        """ Copy the current instance """
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
        """ Convert the conversation to a dictionary """
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }

def get_conv_template(template_name):
    """ get conv template """
    return Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep='<|im_end|>\n',
    )

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """ dynamic preprocess """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    """ get frame indices """
    if sample in ['rand', 'middle']:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None, client=None, clip=None, min_num_frames=4):
    """ read_frames_decord """

    cap = cv2.VideoCapture(video_path)
    temp_video_path = None
    
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
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
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
    
    # Clean up temporary file immediately after reading
    if temp_video_path is not None:
        os.unlink(temp_video_path)
    
    return frame_list


def expand2square(pil_img, background_color):
    """ expand pil image to square """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    """ build transform """
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform


def preprocess(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    """ preprocess """
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
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
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
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
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_pretrain(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    """ preprocess_pretrain """
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]
    assert len(conversations) == 1
    assert conversations[0]['from'] == 'pretrain'

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            image_cnt = conversation['value'].count('<image>')
            for i in range(image_cnt):
                if current_image_idx == num_image:
                    break
                image_tokens = (f'{IMG_START_TOKEN}'
                                f'{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}')
                conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    for conversation in conversations:
        batches.append(conversation["value"])
        roles.append('pretrain')

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        final_targets.append(input_id)

    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]
    targets = torch.where(targets == tokenizer.convert_tokens_to_ids(IMG_START_TOKEN), IGNORE_TOKEN_ID, targets)
    targets = torch.where(targets == tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN), IGNORE_TOKEN_ID, targets)
    targets = torch.where(targets == tokenizer.convert_tokens_to_ids(IMG_END_TOKEN), IGNORE_TOKEN_ID, targets)

    # padding = False if group_by_length or use_packed_ds else True
    padding = False
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    """ preprocess_internvl2_5 """
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = (f'{IMG_START_TOKEN}'
                                    f'{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}'
                                    f'{IMG_END_TOKEN}')
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_internvl3_5_gpt_oss(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
) -> Dict:
    """ preprocess_internvl3_5_gpt_oss """
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = (f'{IMG_START_TOKEN}'
                                    f'{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}')
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|start|>system<|message|>{system_prompt}<|end|>')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|start|>user<|message|>{conversation["value"]}<|end|>')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|start|>assistant<|channel|>final<|message|>{conversation["value"]}<|return|>')
            roles.append('gpt')
        elif conversation['from'] == 'function':
            batches.append(f'<|start|>tool<|message|>{conversation["value"]}<|end|>')
            roles.append('function')
        else:
            raise NotImplementedError(f"Invalid role: {conversation['from']}")

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|start|>assistant', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human' or role == 'function':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|start|>assistant`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    # padding = False if group_by_length or use_packed_ds else True
    padding = False
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        ckpt_path='/workspace',
        split_annotations=True
    ):
        super(LazySupervisedDataset, self).__init__()
        if not ckpt_path:
            ckpt_path = '/workspace'
        self.dataset_log_path = os.path.join(ckpt_path, 'dataset_log')
        self.video_log = os.path.join(self.dataset_log_path, 'video_log')
        os.makedirs(self.video_log, exist_ok=True)
        self.current_state = None
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.bos_client = None

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.split_annotations = split_annotations
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')

        total_ranks = mpu.get_data_parallel_world_size()
        self.total_ranks = total_ranks
        current_rank = mpu.get_data_parallel_rank()
        """
        This section of the code is used to read hundreds of millions of data entries.
        By using caching and splitting the data according to rank, it ensures fast reading
        speed and prevents out-of-memory.
        """
        # Create a cache directory path
        basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
        data_dir = os.path.join(os.path.dirname(meta['annotation']), f'{basename}_temp')
        os.makedirs(data_dir, exist_ok=True)  # Create the cache directory if it does not exist
        # Create a temporary path for the current rank
        temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')
        # Check if the temporary file for the current rank already exists
        if self.split_annotations and os.path.exists(temp_path):
            # If it exists, read the raw data from the file
            with open(temp_path, 'r') as f:
                self.raw_data = f.readlines()
        else:
            # If it does not exist, read the raw data from the original annotation file
            with open(meta['annotation'], 'r') as f:
                self.raw_data = f.readlines()

            # Adjust the raw data based on the repeat_time parameter
            if repeat_time < 1:
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            else:
                self.raw_data = self.raw_data * int(repeat_time)
            if self.split_annotations:
                # Calculate the total number of lines and distribute lines to each rank
                total_lines = len(self.raw_data)
                logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
                lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
                lines_per_rank = max(1, lines_per_rank)

                # Calculate the start and end line numbers for the current rank
                start_line = lines_per_rank * current_rank  # Starting line for the current rank
                end_line = start_line + lines_per_rank  # Ending line for the current rank

                # Assign the appropriate lines to the current rank
                self.raw_data = self.raw_data[start_line:end_line]

                # Write the raw data for the current rank to the temporary file
                with open(temp_path, 'w') as f:
                    f.writelines(self.raw_data)

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.num_fake_dump = 0

        assert not group_by_length
        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations,
                            return_tensors='pt',
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (max_dynamic_patch +
                                                                                         use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        if not self.use_packed_ds:
            return len(self.raw_data) * self.total_ranks
        else:
            return len(self.raw_data)

    def get_preprocess_function(self, use_pretrain=False):
        """ Select the appropriate preprocessing function based on the template name """
        if use_pretrain:
            return preprocess_pretrain

        if self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        elif self.template_name == 'internvl3_5_gpt_oss':
            preprocess_function = preprocess_internvl3_5_gpt_oss
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        """Load the image using bos_client if available, otherwise use PIL"""
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        """ get image path """
        # If image_path is already an absolute bos path, use it directly
        if image_path.startswith('bos:/'):
            return image_path
        
        # Handle different root types
        if self.root.startswith('bos:/'):
            return f"{self.root.rstrip('/')}/{image_path.lstrip('/')}"
        else:
            return os.path.join(self.root, image_path)

    def get_transform(self):
        """Build transformation function"""
        transform = build_transform(is_train=self.is_train,
                                    input_size=self.image_size,
                                    pad2square=self.pad2square,
                                    normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        """ multi_modal_get_item """
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item['conversations'][0]['value'] == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = (
                    '<image>\n' + data_item['conversations'][first_turn_idx]['value'])

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image,
                                        min_num=self.min_dynamic_patch,
                                        max_num=self.max_dynamic_patch,
                                        image_size=self.image_size,
                                        use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        assert (ret['input_ids'][0] == image_end_token_id
                ).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(input_ids=ret['input_ids'][0],
                   labels=ret['labels'][0],
                   attention_mask=ret['attention_mask'][0],
                   position_ids=position_ids[0],
                   pixel_values=pixel_values,
                   image_flags=torch.tensor([1] * num_patches, dtype=torch.long))
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        """ multi_modal_multi_image_get_item """
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item['conversations'][0]['value'] == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = (
                    '<image>\n' * len(data_item['image']) + data_item['conversations'][first_turn_idx]['value'])

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image,
                                           min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size,
                                           use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer,
                                  num_image_tokens,
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name,
                                  num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id
                ).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'
        # Create the final return dictionary
        ret = dict(input_ids=ret['input_ids'][0],
                   labels=ret['labels'][0],
                   attention_mask=ret['attention_mask'][0],
                   position_ids=position_ids[0],
                   pixel_values=pixel_values,
                   image_flags=torch.tensor([1] * num_patches, dtype=torch.long))
        return ret

    def video_get_item(self, data_item):
        """ video get item """
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        first_turn_idx = 1 if data_item['conversations'][0]['value'] == 'system' else 0
        if '<video>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = (
                    '<video>\n' + data_item['conversations'][first_turn_idx]['value'])

        # Get the video file path
        video_file = data_item['video']
        if isinstance(video_file, list):
            video_file = video_file[0]
        
        # Reuse get_image_path method for video path processing
        video_path = self.get_image_path(video_file)
        # Load the video frames using decord
        with open(os.path.join(self.video_log, f'worker{self.worker_id}.txt'), 'w') as f:
            f.write(f"worker_id={self.worker_id}[BEGIN], {data_item=}, {video_path=}\n")
            image_list = read_frames_decord(video_path,
                                            num_frames=self.max_num_frame,
                                            min_num_frames=self.min_num_frame,
                                            client=self.bos_client,
                                            sample=self.sampling_method,
                                            clip=data_item.get('clip', None))
            f.write(f"{self.worker_id=}[END], {data_item=}, {video_path=}")
        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][first_turn_idx]['value'] = (
            data_item['conversations'][first_turn_idx]['value'].replace('<video>\n', special_tokens + '\n'))

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer,
                                  num_image_tokens,
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name,
                                  num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(input_ids=ret['input_ids'][0],
                   labels=ret['labels'][0],
                   attention_mask=ret['attention_mask'][0],
                   position_ids=position_ids[0],
                   pixel_values=pixel_values,
                   image_flags=torch.tensor([1] * num_patches, dtype=torch.long))
        return ret

    def pure_text_get_item(self, data_item):
        """ pure_text_get_item """
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image,
                                    min_num=self.min_dynamic_patch,
                                    max_num=1,
                                    image_size=self.image_size,
                                    use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][0]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  text_only=True,
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(input_ids=ret['input_ids'][0],
                   labels=ret['labels'][0],
                   attention_mask=ret['attention_mask'][0],
                   position_ids=position_ids[0],
                   pixel_values=pixel_values,
                   image_flags=torch.tensor([0] * num_patches, dtype=torch.long))
        return ret

    def fake_data_get_item(self):
        """ fake_data_get_item """
        # Build transformation function
        self.num_fake_dump += 1
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function(use_pretrain=True)

        conversations = [
            {"from": "pretrain", "value": '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'},
        ]

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(conversations)],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=torch.ones_like(ret['input_ids'][0]) * -100,
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        logger.warning(f'Dumping a fake data, the dataset is: {self.ds_name} ({self.num_fake_dump})')
        return ret

    def _enable_worker_distributed(self):
        if (self.distributed_mode and not self.worker_distributed and self.worker_id is not None):
            self.worker_distributed = True
            if self.split_annotations:
                num_worker_per_rank = self.num_workers // self.total_ranks
                self.raw_data = self.raw_data[self.worker_id % num_worker_per_rank::num_worker_per_rank]
            else:
                self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def save_state(self):
        """Saves the rng state for torch, numpy and random."""
        return dict({
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        })

    def restore_state(self, state) -> None:
        """Restores the rng state for torch, numpy and random."""
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['numpy'])
        random.setstate(state['random'])

    def set_rng_state(self, state) -> None:
        """Restores the rng state for torch, numpy and random."""
        self.current_state = state

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        next_sample_idx = i + 1

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                if self.use_packed_ds:
                    raise StopIteration
                return self.fake_data_get_item()
            try:
                data = self.raw_data[i]
                if self.current_state is not None:
                    # print(f"worker{self.worker_id} i_{i}, restore_state")
                    self.restore_state(self.current_state)
                    self.current_state = None
                rng_state = self.save_state()
                # data = self.raw_data[0]
                # print(f"i:{i},data:{data}"):%d
                data_item = json.loads(data)
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and isinstance(data_item['image'], (list, str)) \
                                and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                error_type = type(e).__name__
                error_msg = str(e)

                # 打印更详细的错误信息，包括数据集名称、索引和错误类型
                print(
                    f"[ERROR] Dataset: {self.ds_name} | Index: {i} | "
                    f"Try: {try_cnt}/{max_try} | Error Type: {error_type}"
                )
                print(f"[ERROR] Error Message: {error_msg}")

                # 根据错误类型不同打印不同的堆栈信息
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    print(f"[ERROR] Full traceback for dataset {self.ds_name}:")
                    traceback.print_exc()

                # 打印数据项的详细信息以帮助调试
                try:
                    data_item = json.loads(self.raw_data[i])
                    print(f"[ERROR] Data item structure: {list(data_item.keys())}")

                    # 根据数据类型打印不同的错误信息
                    if 'image' in data_item:
                        if type(data_item['image']) == list:
                            images = [self.get_image_path(item) for item in data_item['image']]
                            print(f"[ERROR] Failed to load images: {images}")
                            print(f"[ERROR] Number of images: {len(images)}")
                        else:
                            data_path = self.get_image_path(data_item['image'])
                            print(f"[ERROR] Failed to load image: {data_path}")
                    elif 'video' in data_item:
                        if isinstance(data_item['video'], list):
                            video_paths = [self.get_image_path(v) for v in data_item['video']]
                            print(f"[ERROR] Failed to load videos: {video_paths}")
                        else:
                            data_path = self.get_image_path(data_item['video'])
                            print(f"[ERROR] Failed to load video: {data_path}")

                    # 打印会话内容的摘要以帮助诊断
                    if 'conversations' in data_item:
                        conv_count = len(data_item['conversations'])
                        print(f"[ERROR] Conversations count: {conv_count}")
                        if conv_count > 0:
                            first_conv = data_item['conversations'][0]
                            print(f"[ERROR] First conversation from: {first_conv.get('from', 'unknown')}")
                            # 只打印前100个字符的内容摘要
                            value = first_conv.get('value', '')
                            print(f"[ERROR] Content preview: {value[:100]}{'...' if len(value) > 100 else ''}")

                except Exception as json_err:
                    print(f"[ERROR] Error parsing data item: {json_err}")
                    print(f"[ERROR] Raw data: {self.raw_data[i][:200]}...")

                # 随机选择新的索引以继续处理
                print(f"[INFO] Skipping this item and trying with a random index")
                i = random.randint(0, len(self.raw_data) - 1)
        ret['data_idx'] = torch.zeros_like(ret['input_ids']) + next_sample_idx - 1
        ret['real_idx'] = torch.zeros_like(ret['input_ids']) + i
        ret['meta_info'] = dict(current_idx=next_sample_idx)
        ret['rng_state'] = rng_state
        return ret

    def load_state_dict(self, state_dict):
        """ load state_dict """
        self._state_dict.update(state_dict)
        self._enable_worker_distributed()

    def init_dataset(self):
        """ init_dataset """
        self._enable_worker_distributed()

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None, f'worker_state_key is not set.'
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']
            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(f'[{self.ds_name}] [Worker id {self.worker_id}] '
                        f'begin to iter with {start_idx=}')

        for i in range(start_idx, len(self)):
            yield self[i]
