"""stdit model"""

from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn
from einops import rearrange
import numpy as np

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from aiak_training_omni.models.custom.transformer.vision.stdit_model_embedding import (
    CaptionEmbedder,
    PatchEmbed3D,
    TimestepEmbedder,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    T2IFinalLayer,
)
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
)

from aiak_training_omni.models.stdit.communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)


class STDiTModel(VisionModule):
    """STDiT Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to True.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights
            are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional): Position embedding type, Defaults to 'rope'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type
            is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer
            sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: StditTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        self.num_temporal = config.num_latent_frames // config.latent_patch_size[0]
        self.num_height = config.max_latent_height // config.latent_patch_size[1]
        self.num_width = config.max_latent_width // config.latent_patch_size[2]
        self.num_spatial = self.num_height * self.num_width

        self.spatial_pos_embed = self.get_spatial_pos_embed()
        self.temporal_pos_embed = self.get_temporal_pos_embed()

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.x_embedder = PatchEmbed3D(
                patch_size=config.latent_patch_size,
                in_chans=config.latent_in_channels,
                embed_dim=config.hidden_size,
            )
            self.t_embedder = TimestepEmbedder(config.hidden_size)
            self.t_block = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
            )
            self.y_embedder = CaptionEmbedder(
                in_channels=config.caption_channels,
                hidden_size=config.hidden_size,
                uncond_prob=0.1,
                act_layer=approx_gelu,
                token_num=config.max_text_length,
            )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = T2IFinalLayer(
                config.hidden_size,
                np.prod(config.latent_patch_size),
                config.latent_out_channels,
            )

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config,
                self.state_dict(),
                prefix=f"{type(self).__name__}_init_ckpt",
            )

    def get_spatial_pos_embed(self):
        """Spatial position embedding."""
        pos_embed = get_2d_sincos_pos_embed(
            self.config.hidden_size,
            (self.num_height, self.num_width),
            scale=self.config.latent_space_scale,
        )  # S H
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )  # 1 S H
        return pos_embed

    def get_temporal_pos_embed(self):
        """Temporal position embedding."""
        config = self.config
        pos_embed = get_1d_sincos_pos_embed(
            config.hidden_size,
            self.num_temporal,
            scale=config.latent_time_scale,
        )  # T H
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )  # 1 T H
        return pos_embed

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        videos: Tensor,
        videos_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        timestep: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = {},
    ) -> Tensor:
        """Forward function of the LLaMA Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then video and position_ids are ignored.
        # Otherwise, apply embedding layer on video and position_ids to get decoder_input.

        # TODO: support PP
        assert decoder_input is None and self.pre_process and self.post_process
        assert videos_mask is not None, "TODO"

        decoder_input = self.x_embedder(videos)

        decoder_input = rearrange(
            decoder_input,
            "B (T S) C -> B T S C",
            T=self.num_temporal,
            S=self.num_spatial,
        )
        decoder_input += self.spatial_pos_embed.to(
            device=decoder_input.device, dtype=decoder_input.dtype
        )
        decoder_input = rearrange(decoder_input, "B T S C -> (T S) B C")
        t = self.t_embedder(timestep, dtype=videos.dtype)  # [B, C]
        text = self.y_embedder(text, True)  # [B, 1, N_token, C]
        text = text.squeeze(1).transpose(0, 1).contiguous()  # [N_token, B, C]

        # attn mask
        B, _, T, H, W = videos_mask.size()
        T_indice = np.linspace(0, T, self.num_temporal + 1, dtype=int)[:-1]
        H_indice = np.linspace(0, H, self.num_height + 1, dtype=int)[:-1]
        W_indice = np.linspace(0, W, self.num_width + 1, dtype=int)[:-1]

        t_padding = videos_mask[:, 0, T_indice, 0, 0]  # B T
        t_attn_mask = t_padding.repeat(self.num_spatial, 1)  # B T -> (BS) T

        s_padding = videos_mask[:, 0, 0, H_indice, :][:, :, W_indice].flatten(1)  # B S
        s_attn_mask = (
            s_padding[:, None, :] * t_padding[:, :, None]
        )  # B 1 S * B T 1->  B T S
        s_attn_mask = s_attn_mask.flatten(0, 1)  # B T S -> (BT) S

        # B C T H W -> B 1 1 THW
        videos_mask = videos_mask[:, 0, T_indice, ...][..., H_indice, :][
            ..., W_indice
        ].view(B, 1, 1, -1)
        text_mask = text_mask[:, None, ...]  # B 1 1 N_token

        extra_block_kwargs["s_attn_mask"] = s_attn_mask[:, None, None, :].logical_not()
        extra_block_kwargs["t_attn_mask"] = t_attn_mask[:, None, None, :].logical_not()
        extra_block_kwargs["timestep"] = self.t_block(t)  # [B, C]
        extra_block_kwargs["temporal_pos_embed"] = self.temporal_pos_embed.transpose(
            0, 1
        )  # T 1 C

        # Run decoder.

        # TODO: Fuse in embeddings
        if self.config.sequence_parallel:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input
            )
            text = tensor_parallel.scatter_to_sequence_parallel_region(text)
            extra_block_kwargs["temporal_pos_embed"] = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    extra_block_kwargs["temporal_pos_embed"]
                )
            )
            if self.config.clone_scatter_output_in_embedding:
                decoder_input = decoder_input.clone()
                text = text.clone()
                extra_block_kwargs["temporal_pos_embed"] = extra_block_kwargs[
                    "temporal_pos_embed"
                ].clone()

        if self.config.context_parallel_size > 1:
            decoder_input = split_forward_gather_backward(
                decoder_input, get_context_parallel_group(), dim=0, grad_scale="down"
            )
            text = split_forward_gather_backward(
                text, get_context_parallel_group(), dim=0, grad_scale="down"
            )
            extra_block_kwargs["temporal_pos_embed"] = torch.chunk(
                extra_block_kwargs["temporal_pos_embed"],
                get_context_parallel_world_size(),
                dim=0,
            )[get_context_parallel_rank()].contiguous()

        videos_mask = videos_mask.logical_not()
        text_mask = text_mask.logical_not()
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=None,
            context=text,
            context_mask=(videos_mask, text_mask),
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        if self.config.context_parallel_size > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states, get_context_parallel_group(), dim=0, grad_scale="up"
            )

        # TODO Fuse in final layer
        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)

        # logits and loss #
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        logits = self.output_layer(hidden_states, t)  # B（t s）C -> B (t s) p*C_out
        logits = self.unpatchify(logits)  # [B, C_out, T, H, W]

        return logits

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        config = self.config
        N_t, N_h, N_w = [self.num_temporal, self.num_height, self.num_width]
        T_p, H_p, W_p = config.latent_patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=config.latent_out_channels,
        )
        return x

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        output_layer_extra_state_key = f"{prefix}output_layer._extra_state"

        # Old GPT checkpoints only stored the output layer weight key. So we remove the _extra_state key
        # but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f"Expected output layer extra state to be empty, got: {output_extra_state}"

        return sharded_state_dict

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the emedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing
            it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def tie_embeddings_and_output_weights_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if not self.post_process:
            # No output layer
            assert (
                output_layer_weight_key not in sharded_state_dict
            ), sharded_state_dict.keys()
            return

        if self.pre_process:
            # Output layer is equivalent to the embedding already
            return

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        tensor = self.shared_embedding_or_output_weight()
        last_stage_word_emb_replica_id = (
            1,  # copy of first stage embedding
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = (
            make_tp_sharded_tensor_for_checkpoint(
                tensor=tensor,
                key=first_stage_word_emb_key,
                replica_id=last_stage_word_emb_replica_id,
                allow_shape_mismatch=True,
            )
        )
