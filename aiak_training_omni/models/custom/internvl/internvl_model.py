# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""intern-vl clip model"""
import warnings
from collections import namedtuple

import logging

from functools import partial
from typing import List, Literal

import torch
from typing import Optional
from einops import rearrange

from megatron.core import InferenceParams, parallel_state
from aiak_training_omni.models.custom.internvl.qwen import QwenModel
from aiak_training_omni.models.custom.internvl.internlm import InternLMModel, DynamicRotaryEmbedding
from aiak_training_omni.utils import get_args
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core import parallel_state, tensor_parallel, mpu

from .intern_vision_model import InternVisionModel
from .adapter import Adapter

VISION_TOKEN_TYPE = 1


class InternVLModel(MegatronModule):
    """InternVL Clip multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
            language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for positional
            embedding.
        vision_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        adapter_config (TransformerConfig): Config for the projection from vision model outputs to language
            model inputs.
        adapter_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        adapter_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        allow_missing_adapter_checkpoint (bool): Allow vision projection weights to be missing when
            loading a checkpoint. Default False.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks.
            This is typically True for training and False for inference.
        language_position_embedding_type (str): Position embedding type to use in the language model.
            Default learned absolute.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings in the
            language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder
            (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder will live on only a subset of the pipeline stages
            (specifically, every stage after the first one).
        img_h (int): The height of each image that the ViT will see.
        img_w (int): The width of each image that the ViT will see.
        patch_dim (int): The size of each patch side.
    """

    def __init__(self,
                 language_config: TransformerConfig,
                 vision_config: TransformerConfig,
                 adapter_config: TransformerConfig,
                 language_layer_spec: ModuleSpec,
                 vision_layer_spec: ModuleSpec,
                 adapter_layer_spec: ModuleSpec,
                 language_vocab_size: int,
                 language_max_sequence_length: int,
                 allow_missing_adapter_checkpoint: bool = True,
                 parallel_output: bool = True,
                 language_position_embedding_type: Literal['learned_absolute', 'rope'] = 'rope',
                 language_rotary_percent: float = 1.0,
                 pre_process: bool = True,
                 post_process: bool = True,
                 add_encoder: bool = True,
                 add_decoder: bool = True,
                 language_rotary_base: int = 10000,
                 language_rope_scaling: bool = False,
                 language_rope_scaling_factor: float = 8.0,
                 language_rotary_dtype: torch.dtype = torch.float32,
                 language_seq_len_interpolation_factor: float = None,
                 scatter_embedding_sequence_parallel: bool = True,
                 fp16_lm_cross_entropy: bool = False,
                 share_embeddings_and_output_weights: bool = True,
                 img_context_token_id: int = None) -> None:
        super().__init__(config=language_config)
        self.language_config = language_config
        args = get_args()
        self.attention_mask = None
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.language_config = language_config
        self.encoder_hidden_state = None
        self.vision_model = None
        self.adapter = None
        self.language_model = None
        self.select_layer = vision_config.select_layer
        self.ps_version = vision_config.ps_version
        self.img_context_token_id = img_context_token_id
        self.downsample_ratio = vision_config.downsample_ratio

        #  define the vision model and the projection from vision model outputs to language model inputs.
        if self.add_encoder:
            self.vision_model = InternVisionModel(
                vision_config,
                vision_layer_spec,
            )

            if vision_config.image_size != args.force_image_size:
                self.vision_model.resize_pos_embeddings(old_size=vision_config.image_size,
                                                        new_size=args.force_image_size,
                                                        patch_size=vision_config.patch_size)
                vision_config.image_size = args.force_image_size

            # FIXME: this is a hack to avoid empty param list in optimizer
            # Map (intermediate) vision model outputs to the language model input dimension.
            self.adapter = Adapter(config=adapter_config,
                                   submodules=adapter_layer_spec,
                                   input_size=vision_config.hidden_size * int(1 / self.downsample_ratio) ** 2)

            # This allows ignoring missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains pretrained
            # vision and language models but not the projection from vision model outputs to language model inputs.
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [f"adapter.{name}" for name in self.adapter.state_dict().keys()]
                self.adapter.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names))

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        if self.add_decoder:
            if args.model_family in ['internvl2.5-8b', 'internvl2.5-26b']:

                self.language_model = InternLMModel(
                    config=language_config,
                    transformer_layer_spec=language_layer_spec,
                    vocab_size=language_vocab_size,
                    max_sequence_length=language_max_sequence_length,
                    parallel_output=parallel_output,
                    position_embedding_type=language_position_embedding_type,
                    rotary_percent=language_rotary_percent,
                    pre_process=self.pre_process,
                    post_process=self.post_process,
                    rotary_base=language_rotary_base,
                    rotary_dtype=language_rotary_dtype,
                    rope_scaling=language_rope_scaling,
                    rope_scaling_factor=language_rope_scaling_factor,
                    seq_len_interpolation_factor=language_seq_len_interpolation_factor,
                    scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                )
            else:
                self.language_model = QwenModel(
                    config=language_config,
                    transformer_layer_spec=language_layer_spec,
                    vocab_size=language_vocab_size,
                    max_sequence_length=language_max_sequence_length,
                    parallel_output=parallel_output,
                    position_embedding_type=language_position_embedding_type,
                    rotary_percent=language_rotary_percent,
                    pre_process=self.pre_process,
                    post_process=self.post_process,
                    rotary_base=language_rotary_base,
                    rope_scaling=language_rope_scaling,
                    rope_scaling_factor=language_rope_scaling_factor,
                    seq_len_interpolation_factor=language_seq_len_interpolation_factor,
                    scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                )
                if args.model_family in ['internvl3-8b', 'internvl3-14b', 'internvl3-38b', 'internvl3-78b']:
                    self.language_model.rotary_pos_emb = DynamicRotaryEmbedding(
                        kv_channels=self.language_config.kv_channels,
                        rotary_percent=language_rotary_percent,
                        rotary_interleaved=self.language_config.rotary_interleaved,
                        seq_len_interpolation_factor=language_seq_len_interpolation_factor,
                        rotary_base=language_rotary_base,
                        dtype=language_rotary_dtype,
                        rope_scaling=language_rope_scaling,
                        rope_scaling_factor=language_rope_scaling_factor,
                        use_cpu_initialization=self.language_config.use_cpu_initialization,
                    )
            self.share_embeddings_and_output_weights = (self.language_model.share_embeddings_and_output_weights)

            #self.language_model.compute_language_model_loss = self.compute_language_model_loss

        def remove_extra_states_check(self, incompatible_keys):
            """
            Temporarily remove ._extra_state as a unexpected_keys key
            when loading checkpoints on xpu.
            """
            for key in incompatible_keys.unexpected_keys.copy():
                if "._extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)
            for key in incompatible_keys.missing_keys.copy():
                if "._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)


    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """set input tensor"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        # assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        # TODO to be fixed once ViT support PP
        # if self.add_encoder and self.add_decoder:
        #     self.vision_model.set_input_tensor(input_tensor[0])
        # elif self.add_encoder:
        #     self.vision_model.set_input_tensor(input_tensor[0])
        # if self.pre_process:
        #     self.encoder_hidden_state = input_tensor[0]
        # else:
        self.language_model.set_input_tensor(input_tensor[0])  # Only LLM support PP
        if len(input_tensor) > 1:
            self.attention_mask = input_tensor[1]
            self.labels = input_tensor[2]
            self.loss_weights = input_tensor[3]
            self.ignore = input_tensor[4]
            if self.attention_mask is not None:
                def recover(tensor):
                    trimmed_tensor = []
                    for row in tensor:
                        # 找到最后一个不等于 -1 的位置
                        last_non_neg_one = torch.where(row != -1)[0].max() + 1 if (row != -1).any() else 0
                        # 切片并保留非 -1 的部分
                        trimmed_tensor.append(row[:last_non_neg_one])
                    return torch.stack(trimmed_tensor)

                self.attention_mask = self.attention_mask[:torch.where(self.attention_mask != -1)[0].max() + 1]
                self.labels = recover(self.labels)
                self.loss_weights = recover(self.loss_weights)

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_adapter: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_adapter (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_adapter and self.adapter is not None:
            modules.append(self.adapter)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                image_flags: torch.LongTensor,
                attn_mask_type: Optional[AttnMaskType] = None,
                decoder_input: torch.Tensor = None,
                labels: torch.Tensor = None,
                inference_params: InferenceParams = None,
                packed_seq_params: PackedSeqParams = None,
                rotary_pos_emb=None,
                statistics=None) -> torch.Tensor:
        """Forward function of the LLaMA Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional)."""

        input_embeds = None
        ignore_flag = False
        if self.pre_process:
            image_flags = image_flags.squeeze(-1)
            input_embeds = self.language_model.embedding(input_ids=input_ids,
                                                         position_ids=position_ids).clone().transpose(0, 1)  # [b s h]
            
            vit_embeds = self.extract_feature(pixel_values)  # [b ,s, h]  s=HW*scale**2
            vit_embeds = vit_embeds[image_flags == 1]  # [b s h]
            vit_batch_size = pixel_values.shape[0]
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            flatten_vit_embeds = vit_embeds.reshape(-1, C)
            if not self.language_config.sequence_parallel:
                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                new_vit_embeds = flatten_vit_embeds
            else:
                b, n = input_ids.shape
                input_ids = input_ids.reshape(b * n)
                tp_rank = mpu.get_tensor_model_parallel_rank()
                tp_size = mpu.get_tensor_model_parallel_world_size()
                part_size = b * n // tp_size
                start_idx = part_size * tp_rank
                end_idx = min(part_size * (tp_rank + 1), b * n)
                ignore_imgs = (input_ids[0:start_idx] == self.img_context_token_id)
                selected = (input_ids[start_idx:end_idx] == self.img_context_token_id)
                ignore_imgs_num = ignore_imgs.sum().item()
                selected_imgs_num = selected.sum().item()
                new_vit_embeds = flatten_vit_embeds[ignore_imgs_num:(ignore_imgs_num + selected_imgs_num), :]
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + new_vit_embeds
                ignore_flag = False
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}, skip...')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                ignore_flag = True
            self.ignore = torch.tensor([ignore_flag], dtype=torch.bool)
            input_embeds = input_embeds.reshape(B, N, C).transpose(0, 1)  # .contiguous()  # [s b h]

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=input_embeds,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            rotary_pos_emb=rotary_pos_emb
            # TODO
            # output_attentions=False,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )  # [b s]
        
        return output

    def extract_feature(self, pixel_values):
        """ extract feature """
        if self.select_layer == -1:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=False,
                                           return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=True,
                                           return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # [b, s, h]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])  # B, HW*scale**2, C/scale**2
        vit_embeds = self.adapter(vit_embeds)
        return vit_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        """ pixel shuffle """
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward_cogvlm(
            self,
            images: torch.Tensor,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            attn_mask_type: Optional[AttnMaskType] = None,
            token_type_ids: torch.Tensor = None,
            labels: torch.Tensor = None,
            inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [batch, img_h, img_w].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """

        use_inference_kv_cache = (inference_params is not None
                                  and "image_tokens_count" in inference_params.key_value_memory_dict)
        # If running inference, we can skip image token computation if they were computed already
        # earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder:
            image_embeddings = self.vision_model(images)  # [b, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len:, :]
            # contiguous() call required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, b, h_vision]
            # map vision model output size to language model input size.
            image_embeddings = self.adapter(image_embeddings)  # [img_seq_len, b, h_vision]

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (image_embeddings.shape[0])
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings

        if self.pre_process:
            language_embeddings = self.language_model.embedding(input_ids=input_ids,
                                                                position_ids=None)  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already
            # earlier for this sample.
            if use_inference_kv_cache:
                combined_embeddings = language_embeddings
            else:
                combined_embeddings = language_embeddings.transpose(
                    0, 1).index_put([token_type_ids == VISION_TOKEN_TYPE],
                                    rearrange(image_embeddings, "s b h -> (b s) h")).transpose(0, 1).contiguous()
                # [combined_seq_len, b, h_language]
        else:
            combined_embeddings = None

        rotary_pos_emb = self.language_model.rotary_pos_emb(position_ids.max() + 1)

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            extra_block_kwargs={
                "token_type_ids": token_type_ids.transpose(0, 1).contiguous(),
                "position_ids": position_ids.transpose(0, 1).contiguous(),
            },
        )

        return output


def _load_state_dict_hook_ignore_param_names(param_names: List[str], module: torch.nn.Module,
                                             incompatible_keys: namedtuple):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and
    language model weights but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the
            missing and unexpected keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in InternVL_MLP_LLama3 Model")
            incompatible_keys.missing_keys.remove(param_name)
