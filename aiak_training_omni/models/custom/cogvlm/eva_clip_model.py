"""CLIP ViT model."""

from typing import Optional, Union

import torch
from einops import rearrange
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron.core.transformer.transformer_config import TransformerConfig


class PatchEmbedding(torch.nn.Module):
    """
    image to patch embedding

    refer to:
    https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B/blob/main/visual.py#L8
    """

    def __init__(
        self,
        in_channels: int,
        visual_hidden_size: int,
        seq_length: int,
        class_token_len: int = 0,
        patch_dim: int = 14,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.add_class_token = class_token_len > 0
        self.proj = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=visual_hidden_size,
            kernel_size=patch_dim,
            stride=patch_dim,
        )

        if self.add_class_token:
            self.class_token = torch.nn.Parameter(
                torch.zeros(class_token_len, visual_hidden_size)
            )

        self.position_ids = torch.arange(seq_length).expand(1, -1).cuda()
        self.position_embedding = torch.nn.Embedding(seq_length, visual_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function"""
        x = self.proj(x)
        x = rearrange(x, "B C h w -> B (h w) C")

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]
            x = torch.cat(
                [class_token, x], dim=1
            )  # [batch, grid ** 2 + class_token_len, hidden_size]

        assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"
        x = x + self.position_embedding(self.position_ids)

        return x


class EVA2CLIPModel(VisionModule):
    """EVA2 CLIP vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.class_token_len = config.class_token_len
        self.visual_hidden_size = config.hidden_size
        self.patch_dim = config.patch_size
        self.img_h = config.image_size[0]
        self.img_w = config.image_size[1]
        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0

        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + self.class_token_len

        self.patch_embedding = PatchEmbedding(
            in_channels=config.in_channels,
            visual_hidden_size=self.visual_hidden_size,
            seq_length=self.seq_length,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
        )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Follow-up changes will make pre and post_process configurable. They are needed for supporting
        # pipeline parallelism.

        # Note: a final layer norm and/or linear layer present in some implementations are omitted here.
        # They can be added separately where needed.
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            vp_stage=vp_stage,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        x = self.patch_embedding(x)

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        x = (
            x.contiguous()
        )  # contiguous() call required as `permute` can sparsify the tensor and this breaks pipelining

        x = self.decoder(x, attention_mask=None)
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        return x
