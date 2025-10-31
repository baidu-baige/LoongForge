""" VisionTransformer module """

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from aiak_training_omni.models.qwen_vl.vision_transformer_block import TransformerBlock
from aiak_training_omni.models.qwen_vl.vision_model import _rotate_half, apply_rotary_pos_emb_vision
from aiak_training_omni.models.llavaov_1_5.llavaov_1_5_config import VisionConfig
from megatron.training import print_rank_0
from aiak_training_omni.models.custom.common.local_norm import FusedLayerNorm


class PatchEmbed(torch.nn.Module):
    """" Patch Embedding """
    def __init__(
        self,
        patch_size: int = 14,
        # temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        # self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # kernel_size = [temporal_patch_size, patch_size, patch_size]
        # self.proj = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """" Forward pass """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            # -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(torch.nn.Module):
    """" Rotary Position Embedding """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = inv_freq.to(torch.cuda.current_device())

    def forward(self, seqlen: int) -> torch.Tensor:
        """ Forward Pass """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionModel(VisionModule):
    """VisionTransformer model. """
    def __init__(self, 
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2
    ) -> None:
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.spatial_merge_size = spatial_merge_size

        self.rotary_pos_emb = VisionRotaryEmbedding(config.kv_channels // 2)

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        """ rotation position embedding """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """ forward function """
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(2).float()

        x = self.patch_embed(x)
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x = self.decoder(x, rotary_pos_emb=rotary_pos_emb, attention_mask=None, attn_mask_type=AttnMaskType.no_mask)
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        return x, None


class RiceViTModel(VisionModel):
    """"""
    def __init__(self,
        config: VisionConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
        # window_size: int = 112,
    ) -> None:
        super().__init__(config, transformer_layer_spec, spatial_merge_size)
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = list(range(config.num_layers))
        # self.window_size = window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.register_buffer('class_embedding', torch.randn(config.hidden_size))
        self.register_buffer('class_pos_emb', torch.randn(1, config.kv_channels // 2))
        # self.class_embedding = torch.nn.Parameter(torch.randn(config.hidden_size))
        # self.class_pos_emb = torch.nn.Parameter(torch.randn(1, config.kv_channels // 2))

        self.pre_layernorm = torch.nn.LayerNorm(
            config.hidden_size,
            eps=1e-4)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):
            new_x.append(class_tokens[i : i + 1])
            new_x.append(x[start_idx:start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)

        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i : i + 1])
            new_rotary_pos_emb.append(rotary_pos_emb[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)

        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # 起始为0
        for length in tokens_per_sample:

            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)


        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)

        x = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [原始seq_len, hidden_size]
        return patch_output, None


    def forward_debug(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """forward debug"""
        output = {}

        x = self.patch_embed(x)
        output["after_patch_embed"] = x.clone()

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):

            new_x.append(class_tokens[i : i + 1])

            new_x.append(x[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)
        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i : i + 1])
            new_rotary_pos_emb.append(rotary_pos_emb[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)
        output["rotary_pos_emb"] = rotary_pos_emb.clone()
        output["class_embedding"] = self.class_embedding.clone()
        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # 起始为0
        for length in tokens_per_sample:

            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)


        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)
        output["after_pre_layernorm"] = x.clone()
        x = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [原始seq_len, hidden_size]
        output["before_adapter"] = patch_output.clone()
        return output
