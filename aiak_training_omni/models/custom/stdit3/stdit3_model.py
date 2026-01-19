"""stdit3 model"""

from typing import Dict, Literal, Optional

import torch
import logging
from collections import namedtuple
from torch import Tensor, nn
from einops import rearrange
import numpy as np

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from aiak_training_omni.models.custom.transformer.vision.stdit_model_embedding import (
    SizeEmbedder,
    CaptionEmbedder,
    PatchEmbed3D,
    TimestepEmbedder,
    approx_gelu,
    T2IFinalLayer,
    PositionEmbedding2D,
)
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, log_single_rank
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
)

from aiak_training_omni.models.stdit3.communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)


def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for name, keys in incompatible_keys._asdict().items():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning(
                    f"_extra_state key {key} being removed from {name}"
                )
                keys.remove(key)


class STDiT3Model(VisionModule):
    """STDiT3 Transformer language model.

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
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "stdit3 is work in progress. Features are missing and methods can change.",
        )

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.patch_size = config.latent_patch_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)

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
            self.fps_embedder = SizeEmbedder(config.hidden_size)
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

        if position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )
        else:
            self.rotary_pos_emb = None

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
            vp_stage=vp_stage,
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

        self.decoder.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_extra_state
        )

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

    def get_dynamic_size(self, x):
        """Get dynamic size of the input tensor"""
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def forward(
        self,
        videos: Tensor,
        videos_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        timestep: Tensor,
        fps: Tensor = None,
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

        T, H, W = self.get_dynamic_size(videos)
        decoder_input = self.x_embedder(videos)

        decoder_input = rearrange(decoder_input, "B (T S) C -> B T S C", T=T, S=(H * W))
        decoder_input += self.pos_embed(
            decoder_input,
            H,
            W,
            scale=self.config.latent_space_scale,
            base_size=round((H * W) ** 0.5),
        )
        decoder_input = rearrange(decoder_input, "B T S C -> (T S) B C")
        t = self.t_embedder(timestep, dtype=videos.dtype)  # [B, C]
        if fps is not None:
            t = t + self.fps_embedder(fps.unsqueeze(1), timestep.shape[0])
        text = self.y_embedder(text, train=True)  # [B, 1, N_token, C]
        text = text.squeeze(1).transpose(0, 1).contiguous()  # [N_token, B, C]

        # attn mask
        B, _, _t, _h, _w = videos_mask.size()
        T_indice = np.linspace(0, _t, T + 1, dtype=int)[:-1]
        H_indice = np.linspace(0, _h, H + 1, dtype=int)[:-1]
        W_indice = np.linspace(0, _w, W + 1, dtype=int)[:-1]

        t_padding = videos_mask[:, 0, T_indice, 0, 0]  # B T
        t_attn_mask = t_padding.repeat(H * W, 1)  # B T -> (BS) T

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

        # Run decoder.

        if self.rotary_pos_emb is not None:
            pos_emb = self.rotary_pos_emb(
                T, packed_seq=True
            )  # trick to pass chunk in rope
            cp_size = self.config.context_parallel_size
            cp_rank = get_context_parallel_rank()
            extra_block_kwargs["rotary_pos_emb"] = pos_emb.chunk(cp_size, dim=0)[
                cp_rank
            ]

        # TODO: Fuse in embeddings
        if self.config.sequence_parallel:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input
            )
            text = tensor_parallel.scatter_to_sequence_parallel_region(text)
            if self.config.clone_scatter_output_in_embedding:
                decoder_input = decoder_input.clone()
                text = text.clone()
            T = T // self.config.sequence_parallel_size

        if self.config.context_parallel_size > 1:
            decoder_input = split_forward_gather_backward(
                decoder_input, get_context_parallel_group(), dim=0, grad_scale="down"
            )
            text = split_forward_gather_backward(
                text, get_context_parallel_group(), dim=0, grad_scale="down"
            )
            T = T // self.config.context_parallel_size

        extra_block_kwargs["T"] = T
        extra_block_kwargs["S"] = H * W

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
            T = T * self.config.context_parallel_size

        # TODO Fuse in final layer
        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
            T = T * self.config.sequence_parallel_size

        # logits and loss #
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        logits = self.output_layer(hidden_states, t)  # B（t s）C -> B (t s) p*C_out
        logits = self.unpatchify(logits, T, H, W)  # [B, C_out, T, H, W]

        return logits

    def unpatchify(self, x, T, H, W):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        config = self.config
        N_t, N_h, N_w = [T, H, W]
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
