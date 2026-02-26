"""ErnieVisionModel"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from omni_training.models.common import BaseMegatronVisionModule
from megatron.training import get_args
from transformers import AutoProcessor

from .ernie_vision_block import DFNRopeVisionBlock
from .ernie_adapter import UniqueNameGuard

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """PatchEmbed"""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        """
        Args:
            patch_size (int, optional): patch size. Defaults to 14.
            in_channels (int, optional): number of channels. Defaults to 3.
            embed_dim (int, optional): embedding dimension. Defaults to 1152.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): hidden states

        Returns:
            torch.Tensor: output tensor
        """
        target_dtype = self.proj.weight.dtype

        hidden_states = self.proj(hidden_states.to(target_dtype))

        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    """VisionRotaryEmbedding"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Args:
            dim (int): the dimension of each token.
            theta (float, optional): the frequency factor. Defaults to 10000.0.
        """
        super().__init__()
        self.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Args:
            seqlen (int): length of sequence.

        Returns:
            torch.Tensor: rotary position embedding
        """
        seq = torch.arange(seqlen).to(self.inv_freq.dtype)
        freqs = torch.outer(input=seq, vec2=self.inv_freq)
        return freqs


class VariableResolutionResamplerModel(nn.Module):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.in_dim = input_size
        self.out_dim = output_size
        self.config = config
        self.spatial_conv_size = config.temporal_merge_size
        self.temporal_conv_size = config.temporal_merge_size
        self.use_temporal_conv = config.use_temporal_conv

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress 3d conv(video) to 1d
        self.temporal_dim = (
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        # using unique name space start with "mm_resampler_"
        with UniqueNameGuard("mm_resampler_") as guard:

            self.spatial_linear = nn.Sequential(
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.GELU(),
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.LayerNorm(self.spatial_dim, eps=1e-6),
            )

            if self.use_temporal_conv:
                self.temporal_linear = nn.Sequential(
                    nn.Linear(self.temporal_dim, self.spatial_dim),
                    nn.GELU(),
                    nn.Linear(self.spatial_dim, self.spatial_dim),
                    nn.LayerNorm(self.spatial_dim, eps=1e-6),
                )


    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        reshape before linear to imitation conv
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x,  grid_thw):
        """
        x: image_features
        grid_thw: [B_image, 3]
        """

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x = self.spatial_linear(x)

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                the second dimension: [t, h, w]
            """

            grid_thw_cpu = grid_thw.cpu().numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = torch.tensor(np.concatenate(slice_offsets, axis=-1)).to(
                x.device
            )

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = torch.tensor(np.concatenate(slice_offsets2, axis=-1)).to(
                x.device
            )

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets)
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2)
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1)
            return x

        def fwd_temporal(x):
            x = self.temporal_linear(x)
            return x

        # def fwd_mlp(x):
        #     x = self.mlp(x)
        #     x = self.after_norm(x)
        #     return x

        x = fwd_spatial(x)
        if self.use_temporal_conv:
            x = fwd_placeholder(x, grid_thw)
            x = fwd_temporal(x)
        # x = fwd_mlp(x)
        return x


class ErnieVisionModel(BaseMegatronVisionModule):
    """Base on DFNRopeVisionTransformerPreTrainedModel"""

    def __init__(self, config, vp_stage: int = None) -> None:
        """
        Args:
            config (dict): model configuration
        """
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [DFNRopeVisionBlock(config) for _ in range(config.num_layers)]
        )

        self.ln = nn.LayerNorm(config.embed_dim, eps=1e-6)

        self.resampler = VariableResolutionResamplerModel(config,
            config.resampler_hidden_in, config.resampler_hidden_out)

        class DummyFunc:
            """dummy_func"""
            def __init__(self):
                self.set_input_tensor = lambda tensor: tensor
        self.decoder = DummyFunc()
        args = get_args()
        self.image_preprocess = AutoProcessor.from_pretrained(args.hf_tokenizer_path,  trust_remote_code=True)
        self.image_preprocess.eval()
        self.image_preprocess = self.add_image_preprocess(self.image_preprocess)
        if hasattr(config, 'freeze') and config.freeze:
            for name, param in self.named_parameters():
                # resampler_model is the only part that needs to be updated in encoder
                if 'resampler_model' not in name:
                    param.requires_grad = False


    def add_image_preprocess(self, processor):
        """add image preprocess"""
        logger.info("image preprocess is set")

        image_preprocess = processor.image_processor
        image_preprocess.image_mean_tensor = torch.tensor(
            image_preprocess.image_mean, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_preprocess.image_std_tensor = torch.tensor(
            image_preprocess.image_std, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_preprocess.rescale_factor = torch.tensor(
            image_preprocess.rescale_factor, dtype=torch.float32
        )
        image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze(
            [-2, -1]
        ).repeat_interleave(self.config.patch_size**2 * 1, -1)
        image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze(
            [-2, -1]
        ).repeat_interleave(self.config.patch_size**2 * 1, -1)

        return image_preprocess

    def rot_pos_emb(self, grid_thw, num_pad=0):
        """rot_pos_emb

        Args:
            grid_thw (torch.Tensor): grid thw of input

        Returns:
            torch.Tensor: rotary position embedding
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw.cpu(), dtype=np.int64)
        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape([-1, 1])
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape([1, -1])
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate(
                [pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)]
            )
        max_grid_size = np.amax(grid_hw_array[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_dim=1)
        return rotary_pos_emb

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1'
        self.decoder.set_input_tensor(input_tensor[0])

    def preprocess(self, images, grid_thw):
        """Preprocess"""
        assert images.dtype == torch.uint8, images.dtype
        current_device = images.device
        self.image_preprocess.image_mean_tensor = (
            self.image_preprocess.image_mean_tensor.to(current_device)
        )
        self.image_preprocess.image_std_tensor = (
            self.image_preprocess.image_std_tensor.to(current_device)
        )
        images = self.image_preprocess.rescale_factor * images.to(torch.float32)
        images = (
            images - self.image_preprocess.image_mean_tensor
        ) / self.image_preprocess.image_std_tensor
        images = images.to(torch.bfloat16)
        # process grid thw
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        return images, grid_thw

    def forward(
        self, hidden_states: torch.Tensor, image_grid_thw: torch.Tensor, num_pad=0
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): image input tensor
            image_grid_thw (torch.Tensor): grid thw of input
            num_pad (int): number of padding tokens

        Returns:
            torch.Tensor: output tensor
        """
        # preprocess images and grid_thw
        hidden_states, image_grid_thw = self.preprocess(hidden_states, image_grid_thw)
        # vit
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(image_grid_thw, num_pad=num_pad)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        vision_out = self.ln(hidden_states)  # add norm
        # resample
        ret = self.resampler(vision_out, image_grid_thw)
        return ret, None, [None]
