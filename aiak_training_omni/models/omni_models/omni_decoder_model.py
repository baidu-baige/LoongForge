"""OmniDecoderModel:解码器模型基类。
为图像/视频/音频生成任务提供统一接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .configuration import OmniDecoderConfig
from ..common.base_model_mixins import BaseDecoderModelMixin


class OmniDecoderModel(BaseDecoderModelMixin):
    """Omni 多模态解码器模型。"""
    config_class = OmniDecoderConfig

    def __init__(self, config: OmniDecoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.modality: List[str] = []

        # 解码器
        # if getattr(config.image_config, "model_type", ""):
        #     self.image_decoder = build_component_from_config(config.image_config)
        #     self.modality.append("image")

        # if getattr(config.video_config, "model_type", ""):
        #     self.video_decoder = build_component_from_config(config.video_config)
        #     self.modality.append("video")

        # if getattr(config.audio_config, "model_type", ""):
        #     self.audio_decoder = build_component_from_config(config.audio_config)
        #     self.modality.append("audio")

    def forward_loss(
        self,
        decoder_inputs: Dict[str, torch.Tensor],
        foundation_outputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """计算解码器损失。"""
        losses = {}
        
        # 图像解码器
        if "image" in self.modality and "image" in decoder_inputs:
            image_input = decoder_inputs["image"]
            if hasattr(self.image_decoder, 'forward_loss'):
                image_loss = self.image_decoder.forward_loss(
                    image_input, foundation_outputs, **kwargs
                )
                losses["image_loss"] = image_loss
            else:
                # 占位实现：返回零损失
                losses["image_loss"] = torch.tensor(0.0, device=foundation_outputs.device)

        # 视频解码器
        if "video" in self.modality and "video" in decoder_inputs:
            video_input = decoder_inputs["video"]
            if hasattr(self.video_decoder, 'forward_loss'):
                video_loss = self.video_decoder.forward_loss(
                    video_input, foundation_outputs, **kwargs
                )
                losses["video_loss"] = video_loss
            else:
                # 占位实现：返回零损失
                losses["video_loss"] = torch.tensor(0.0, device=foundation_outputs.device)

        # 音频解码器
        if "audio" in self.modality and "audio" in decoder_inputs:
            audio_input = decoder_inputs["audio"]
            if hasattr(self.audio_decoder, 'forward_loss'):
                audio_loss = self.audio_decoder.forward_loss(
                    audio_input, foundation_outputs, **kwargs
                )
                losses["audio_loss"] = audio_loss
            else:
                # 占位实现：返回零损失
                losses["audio_loss"] = torch.tensor(0.0, device=foundation_outputs.device)

        return losses

    def freeze(self, modality: Optional[str] = None) -> None:
        """冻结解码器模型参数。
        
        参数:
        ----------
        modality : str, 可选
            要冻结的特定模态 ('image', 'video', 'audio')。如果为 None，则冻结所有模态解码器。
        """
        if modality is not None:
            # 冻结特定模态的解码器
            if modality not in self.modality:
                raise ValueError(f"模态 '{modality}' 不存在。可用模态: {self.modality}")
            
            decoder_name = f"{modality}_decoder"
            if hasattr(self, decoder_name):
                decoder = getattr(self, decoder_name)
                for param in decoder.parameters():
                    param.requires_grad = False
                print(f"已冻结 {modality} 解码器")
        else:
            # 冻结所有模态解码器
            for mod in self.modality:
                decoder_name = f"{mod}_decoder"
                if hasattr(self, decoder_name):
                    decoder = getattr(self, decoder_name)
                    for param in decoder.parameters():
                        param.requires_grad = False
                    print(f"已冻结 {mod} 解码器")

