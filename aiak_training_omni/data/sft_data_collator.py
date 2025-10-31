"""sft datacollator"""

from typing import TYPE_CHECKING, Optional, Union, Any, Sequence, Dict
from dataclasses import dataclass

import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """
    tokenizer: "PreTrainedTokenizerBase"
    model: Optional[Any] = None
    padding: Optional[Union[bool, str, "PaddingStrategy"]] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    collator: object = None

    def __post_init__(self):

        self.collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
            padding=self.padding,
            max_length=self.max_length,
        )

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # padding loss mask here
        loss_mask = [feature["loss_mask"] for feature in features] if "loss_mask" in features[0].keys() else None

        if loss_mask is not None and self.padding and self.padding != PaddingStrategy.DO_NOT_PAD:
            max_loss_length = max(len(l) for l in loss_mask)
            if self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None:
                max_loss_length = self.max_length
            if self.pad_to_multiple_of is not None:
                max_loss_length = (
                    (max_loss_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                if len(feature["loss_mask"]) < max_loss_length:
                    remainder = [0] * (max_loss_length - len(feature["loss_mask"]))
                    if isinstance(feature["loss_mask"], list):
                        if padding_side == "right":
                            feature["loss_mask"] += remainder
                        elif padding_side == "left":
                            feature["loss_mask"] = remainder + feature["loss_mask"]
                    else:
                        if padding_side == "right":
                            feature["loss_mask"] = np.concatenate([feature["loss_mask"], remainder]).astype(np.int64)
                        else:
                            feature["loss_mask"] = np.concatenate([remainder, feature["loss_mask"]]).astype(np.int64)

        # default only padding labels
        return self.collator(features, return_tensors)


@dataclass
class MultiModalDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """ Multi-modal data collator  """
    processor: "ProcessorMixin" = None
    plugin: "MMPlugin" = None
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens = [], [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_seqlens.append(len(feature["input_ids"]))

        mm_inputs = self.plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens, self.processor
        )

        features: Dict[str, "torch.Tensor"] = super().__call__(features)

        # keys are named in transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
        if "pixel_values" in mm_inputs:
            features['images'] = mm_inputs.get('pixel_values')
            features['image_grid_thw'] = mm_inputs.get('image_grid_thw')
        if "pixel_values_videos" in mm_inputs:
            features['videos'] = mm_inputs.get('pixel_values_videos')
            features['video_grid_thw'] = mm_inputs.get('video_grid_thw')

        return features