"""cogvlm model"""

import logging
from collections import namedtuple
from functools import partial
from typing import List

import torch
from typing import Optional
from einops import rearrange

from megatron.core import InferenceParams, parallel_state
from aiak_training_omni.models.foundation import LLaMAModel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import make_viewless_tensor

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import log_single_rank

from .eva_clip_model import EVA2CLIPModel
from .adapter import Adapter

VISION_TOKEN_TYPE = 1


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


class CogVLMModel(MegatronModule):
    """CogVLM multi-modal model.

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

    def __init__(
        self,
        language_config: TransformerConfig,
        vision_config: TransformerConfig,
        adapter_config: TransformerConfig,
        language_layer_spec: ModuleSpec,
        vision_layer_spec: ModuleSpec,
        adapter_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        allow_missing_adapter_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = "rope",
        language_rotary_percent: float = 1.0,
        language_rotary_dtype: torch.dtype = torch.float32,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        share_embeddings_and_output_weights: bool = True,
        seq_len_interpolation_factor: float = None,
    ) -> None:
        super().__init__(config=language_config)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "clip_model is work in progress. Features are missing and methods can change.",
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.adapter = None
        self.language_model = None

        #  define the vision model and the projection from vision model outputs to language model inputs.
        if self.add_encoder:
            self.vision_model = EVA2CLIPModel(
                vision_config,
                vision_layer_spec,
            )
            self.vision_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )
            self._drop_vision_class_token = self.vision_model.class_token_len > 0
            # Map (intermediate) vision model outputs to the language model input dimension.
            self.adapter = Adapter(
                adapter_config,
                adapter_layer_spec,
                vision_config.hidden_size,  # input size to the projection.
            )
            # This allows ignoring missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains pretrained
            # vision and language models but not the projection from vision model outputs to language model inputs.
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"adapter.{name}" for name in self.adapter.state_dict().keys()
                ]
                self.adapter.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names, adapter_param_names
                    )
                )

        # # This attribute is needed to check if an all-reduce is required
        # # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        if self.add_decoder:
            self.language_model = LLaMAModel(
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
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            )
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )
            self.language_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

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
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_adapter: bool,
    ):
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

    def unfreeze_expert_linear(
        self, unfreeze_language_expert_linear, unfreeze_vision_expert_linear
    ):
        """Unfreeze model modules."""
        for param in self.language_model.parameters():
            if (
                getattr(param, "is_language_expert_parameter", False)
                and unfreeze_language_expert_linear
            ):
                param.requires_grad = True
            if (
                getattr(param, "is_vision_expert_parameter", False)
                and unfreeze_vision_expert_linear
            ):
                param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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

        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        # If running inference, we can skip image token computation if they were computed already
        # earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder:
            image_embeddings = self.vision_model(images)  # [b, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[
                    :, self.vision_model.class_token_len :, :
                ]
            # contiguous() call required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [img_seq_len, b, h_vision]
            # map vision model output size to language model input size.
            image_embeddings = self.adapter(
                image_embeddings
            )  # [img_seq_len, b, h_vision]

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings

        if self.pre_process:
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None
            )  # [text_seq_len, b, h_language]

            # If running inference, we can skip image token computation if they were computed already
            # earlier for this sample.
            if use_inference_kv_cache:
                combined_embeddings = language_embeddings
            else:
                combined_embeddings = (
                    language_embeddings.transpose(0, 1)
                    .index_put(
                        [token_type_ids == VISION_TOKEN_TYPE],
                        rearrange(image_embeddings, "s b h -> (b s) h"),
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
                # [combined_seq_len, b, h_language]
        else:
            combined_embeddings = None

        rotary_pos_emb = self.language_model.rotary_pos_emb(position_ids.max() + 1)

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
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


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
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
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
