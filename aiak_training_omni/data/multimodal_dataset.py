""" https://github.com/THUDM/CogVLM2/blob/main/finetune_demo/peft_lora.py """

import json
import os
import random
from typing import Optional, Tuple, List, Union, Literal, Dict, Any
from abc import ABC, abstractmethod

from PIL import Image
from torchvision import transforms
import torch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

def _history_to_prompt(signal_type, history, query):
    """ Build a prompt from the conversation history and current question. """
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt


def _build_position_ids(x, attention_mask=None):
    """  Build position ids based on the input tokens. """
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[1:] |= (tmp[1:] == VISION_TOKEN_TYPE) & (tmp[:-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[0] |= (tmp[0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:-1] |= (tmp[:-1] == VISION_TOKEN_TYPE) & (tmp[1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[-1] |= (tmp[-1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[1:] = (tmp[1:] == LANGUAGE_TOKEN_TYPE) | ((tmp[1:] == VISION_TOKEN_TYPE) & (tmp[:-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y


class BaseDataset(ABC, torch.utils.data.Dataset):
    """ Conversation dataset class. """
    def __init__(self,
                 root_dir,
                 tokenizer,
                 torch_type,
                 max_length,
                 patch_size=14,
                 image_size=(1344, 1344),
                 template_version: Optional[Literal["base", "chat", "vqa"]] = None,
                 ):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.filenames = os.listdir(self.image_dir)
        self.torch_type = torch_type
        self.patch_size = patch_size
        self.image_size = image_size
        self.template_version = template_version
        self.max_length = max_length
        self.tokenizer.pad = 128002 # llama3 adapt for cogvlm

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def custom_collate_fn(batch):
        """ Collate function for batching data. """
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")

        return batched_data

    @abstractmethod
    def get_texts(self, label_data):
        """ query, history, response """
        pass

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        label_name = os.path.join(self.label_dir, self.filenames[idx].replace('.jpg', '.json'))

        image = Image.open(img_name).convert('RGB')
        with open(label_name, 'r') as f:
            label_data = json.load(f)

        query, history, response = self.get_texts(label_data)

        input_data = self.build_conversation_input_ids(
            query=query,
            history=history,
            images=[image],
            answer=response
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            """ Pad tensor to given length. """
            current_length = len(unpadded_tensor)
            if current_length >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full([pad_to_length - current_length],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device)), dim=0)

        input_ids_len = len(input_data['input_ids'])

        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            self.max_length,
            pad_value=self.tokenizer.pad,
        )

        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            self.max_length,
            pad_value=0
        )
        input_data['token_type_ids'] = pad_to_len(
            input_data['token_type_ids'],
            self.max_length,
            pad_value=0
        )

        input_data['labels'] = pad_to_len(
            input_data['labels'][1:], # for cross entropy
            self.max_length,
            pad_value=-100
        )

        input_data['loss_mask'] = (input_data['labels'] != -100).to(torch.float32)
        input_data['position_ids'] = _build_position_ids(input_data['token_type_ids'], input_data['attention_mask'])
        input_data['attention_mask'] = input_data['attention_mask'].logical_not().unsqueeze(0).unsqueeze(0) # [1, 1, s]

        return input_data


    def build_conversation_input_ids(
            self,
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List["PIL.Image"]] = None,
            answer: str = None,
    ):
        """ Build input ids for conversation task. """
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []
        text = _history_to_prompt(self.template_version, history, query)
        input_ids = [self.tokenizer.bos]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.image_size, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            images = transform(images[0]).to(self.torch_type)
            # language
            vision_token_num = \
                (self.image_size[0] // self.patch_size // 2) * (self.image_size[1] // self.patch_size // 2) + 2

            input_ids += [self.tokenizer.pad] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = self.tokenizer.tokenize(text, add_special_tokens=False)

        if answer is not None:
            answer_ids = self.tokenizer.tokenize(answer, add_special_tokens=False)
            answer_ids += [self.tokenizer.eos]
            text_ids += answer_ids


        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        
        if answer is not None:
            labels = [-100 for _ in range(len(input_ids) - len(answer_ids))] + answer_ids            
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None

        return {
            'input_ids': torch.tensor(input_ids),
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'images': images,
            'labels': labels,
        }


class CaptionDataset(BaseDataset):
    """ Caption dataset class. """
    def get_texts(self, label_data):
        """ caption, None, None """
        return "", None, label_data["captions"][0]["content"]


class ConversationDataset(BaseDataset):
    """ Conversation dataset class. """
    def get_texts(self, label_data):
        """ query, history, response """
        num_rounds = len(label_data["conversations"]) // 2
        sampled_round_id = random.randint(0, num_rounds - 1)
        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None
        query = label_data["conversations"][sampled_round_id * 2]["content"]
        response = label_data["conversations"][sampled_round_id * 2 + 1]["content"]

        return query, history, response