# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL backbone helpers for GR00T-N1.7."""

import logging
import os
import inspect

from huggingface_hub.errors import GatedRepoError
import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


_GATED_BACKBONE_HINT = (
    "Cannot download the VLM backbone '{model_name}', which is a gated Hugging "
    "Face repo. Every GR00T checkpoint loads this backbone, so both inference "
    "and finetuning require access to it."
)

_GATED_MARKERS = ("gated repo", "is restricted", "access to model", "401 client error")


def _coerce_torch_dtype(value) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.lower().replace("torch.", "")
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "float"}:
            return torch.float32
    return None


def _is_qwen3vl_flash_attention_module(module: torch.nn.Module) -> bool:
    try:
        config = module.config
    except AttributeError:
        return False
    if config._attn_implementation not in {"flash_attention_2", "flash_attention_3"}:
        return False

    module_type = type(module)
    return "qwen3_vl" in module_type.__module__ and module_type.__name__.startswith("Qwen3VL")


def _qwen3vl_flash_attention_target_dtype(module: torch.nn.Module) -> torch.dtype:
    config = module.config
    for dtype_value in (config._pre_quantization_dtype, config.dtype):
        dtype = _coerce_torch_dtype(dtype_value)
        if dtype in {torch.float16, torch.bfloat16}:
            return dtype

    # Cosmos-Reason2 advertises bf16 in text_config.dtype.  In transformers
    # 4.57 this was also the dtype used to cast fp32 q/k/v before FA2.
    return torch.bfloat16


def _patch_qwen3vl_flash_attention_target_dtype() -> None:
    try:
        from transformers.integrations import flash_attention
    except ImportError:
        return

    try:
        original_get_target_dtype = flash_attention.get_target_dtype
    except AttributeError:
        return
    if original_get_target_dtype.__dict__.get("_loongforge_qwen3vl_compat", False):
        return

    def get_target_dtype_compat(query: torch.Tensor, module: torch.nn.Module) -> torch.dtype | None:
        if query.dtype == torch.float32 and _is_qwen3vl_flash_attention_module(module):
            return _qwen3vl_flash_attention_target_dtype(module)
        return original_get_target_dtype(query, module)

    get_target_dtype_compat._loongforge_qwen3vl_compat = True
    get_target_dtype_compat._loongforge_original = original_get_target_dtype
    flash_attention.get_target_dtype = get_target_dtype_compat


def _patch_qwen3vl_output_projection_dtypes(model: torch.nn.Module) -> int:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return 0

    patched = 0
    try:
        text_attention_cls = qwen_mod.Qwen3VLTextAttention
    except AttributeError:
        text_attention_cls = ()
    try:
        vision_attention_cls = qwen_mod.Qwen3VLVisionAttention
    except AttributeError:
        vision_attention_cls = ()

    for module in model.modules():
        proj = None
        if isinstance(module, text_attention_cls):
            proj = module.o_proj
        elif isinstance(module, vision_attention_cls):
            proj = module.proj

        if proj is None or not isinstance(proj, torch.nn.Linear):
            continue
        if proj.__dict__.get("_loongforge_qwen3vl_dtype_patched", False):
            continue

        original_forward = proj.forward

        def forward_compat(input, *args, _original_forward=original_forward, _proj=proj, **kwargs):
            target_dtype = _proj.weight.dtype
            if input.dtype != target_dtype:
                input = input.to(target_dtype)
            return _original_forward(input, *args, **kwargs)

        forward_compat._loongforge_qwen3vl_dtype_patched = True
        proj.forward = forward_compat
        proj._loongforge_qwen3vl_dtype_patched = True
        patched += 1

    if patched:
        logger.info("Patched %d Qwen3-VL output projection(s) to preserve fp32 residual dtype.", patched)
    return patched


def _patch_qwen3vl_skip_unused_lm_head() -> bool:
    """Allow GR00T to call Qwen top-level forward without unused logits projection."""
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen_mod
    except ImportError:
        return False

    cls = getattr(qwen_mod, "Qwen3VLForConditionalGeneration", None)
    output_cls = getattr(qwen_mod, "Qwen3VLCausalLMOutputWithPast", None)
    if cls is None or output_cls is None:
        return False
    if cls.forward.__dict__.get("_loongforge_skip_lm_head_compat", False):
        return False

    original_forward = cls.forward

    def forward_compat(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep=0,
        loongforge_skip_lm_head: bool = False,
        **kwargs,
    ):
        if not loongforge_skip_lm_head or labels is not None:
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = hidden_states.new_empty((*hidden_states.shape[:-1], 0))
        return output_cls(
            loss=None,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=(hidden_states,),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )

    forward_compat._loongforge_skip_lm_head_compat = True
    forward_compat._loongforge_original = original_forward
    cls.forward = forward_compat
    return True


def _is_gated_repo_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    for _ in range(10):
        if current is None:
            break
        if isinstance(current, GatedRepoError) or any(marker in str(current).lower() for marker in _GATED_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _real_inference_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def _force_tie_qwen_lm_head(model: torch.nn.Module) -> bool:
    """Restore the Qwen3-VL tied lm_head behavior used by transformers 4.57.x."""
    try:
        lm_head = model.lm_head
    except AttributeError:
        return False
    try:
        lm_head.weight
    except AttributeError:
        return False

    backbone = _unwrap_qwen_backbone(model)
    language_model = backbone.language_model
    try:
        embed_tokens = language_model.embed_tokens
    except AttributeError:
        return False

    try:
        embed_tokens.weight
    except AttributeError:
        return False

    if lm_head.weight is embed_tokens.weight:
        return False

    if lm_head.weight.shape != embed_tokens.weight.shape:
        logger.warning(
            "Cannot tie Qwen3-VL lm_head to input embeddings: shape mismatch %s vs %s.",
            tuple(lm_head.weight.shape),
            tuple(embed_tokens.weight.shape),
        )
        return False

    lm_head.weight = embed_tokens.weight
    tied_keys = model.__dict__.get("all_tied_weights_keys")
    if isinstance(tied_keys, dict):
        tied_keys["lm_head.weight"] = "model.language_model.embed_tokens.weight"
    return True


def _unwrap_qwen_backbone(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.model
    except AttributeError:
        return model


def _forward_has_explicit_arg(model: torch.nn.Module, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _build_mm_token_type_ids(config, input_ids: torch.Tensor) -> torch.Tensor | None:
    token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int32, device=input_ids.device)

    image_token_id = config.image_token_id
    if image_token_id is not None:
        image_mask = input_ids == image_token_id
        token_type_ids = torch.where(
            image_mask,
            torch.ones((), dtype=token_type_ids.dtype, device=token_type_ids.device),
            token_type_ids,
        )

    video_token_id = config.video_token_id
    if video_token_id is not None:
        video_mask = input_ids == video_token_id
        token_type_ids = torch.where(
            video_mask,
            torch.full((), 2, dtype=token_type_ids.dtype, device=token_type_ids.device),
            token_type_ids,
        )

    return token_type_ids


def _build_qwen3vl_compat_position_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor | None:
    qwen_model = _unwrap_qwen_backbone(model)
    try:
        get_rope_index = qwen_model.get_rope_index
    except AttributeError:
        return None

    try:
        position_ids, rope_deltas = get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
    except TypeError:
        return None

    if "rope_deltas" in qwen_model.__dict__:
        qwen_model.rope_deltas = rope_deltas
    return torch.cat((position_ids[:1], position_ids), dim=0)



def recompute_vision_rotary_inv_freq(
    rotary: torch.nn.Module,
    head_dim_half: int,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild the vision rotary inverse frequency tensor on the target device."""
    with torch.device(device):
        fresh = type(rotary)(head_dim_half)
    return fresh.inv_freq.detach().to(device=device, dtype=torch.float32)


def recompute_text_rotary_inv_freq(
    rotary: torch.nn.Module,
    config,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Rebuild the text rotary inverse frequency tensor and scaling value."""
    with torch.device(device):
        fresh = type(rotary)(config=config, device=device)
    inv_freq = fresh.inv_freq.detach().to(device=device, dtype=torch.float32)
    try:
        attention_scaling = float(fresh.attention_scaling)
    except AttributeError:
        attention_scaling = 1.0
    return inv_freq, attention_scaling


def _assign_inv_freq(
    rotary: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    *,
    persistent: bool,
) -> bool:
    current = rotary._buffers.get(name, vars(rotary).get(name))
    if (
        isinstance(current, torch.Tensor)
        and current.device.type != "meta"
        and current.device == value.device
        and current.shape == value.shape
        and current.dtype == value.dtype
        and torch.equal(current, value)
    ):
        return False
    if name in rotary._buffers:
        rotary.register_buffer(name, value, persistent=persistent)
    else:
        setattr(rotary, name, value)
    return True


class Qwen3Backbone(torch.nn.Module):
    """Qwen3-VL backbone with GR00T-specific compatibility patches."""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict | None = None,
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()
        transformers_loading_kwargs = dict(transformers_loading_kwargs or {})

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                _patch_qwen3vl_flash_attention_target_dtype()
                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["dtype"] = torch.bfloat16

        if (
            str(os.environ.get("HF_HUB_OFFLINE", "")).lower() in {"1", "true", "yes"}
            or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).lower() in {"1", "true", "yes"}
        ):
            transformers_loading_kwargs = {
                **transformers_loading_kwargs,
                "local_files_only": True,
            }

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **extra_kwargs,
                **transformers_loading_kwargs,
            ).eval()
        except Exception as exc:
            if _is_gated_repo_error(exc):
                raise RuntimeError(_GATED_BACKBONE_HINT.format(model_name=model_name)) from exc
            raise
        if _force_tie_qwen_lm_head(self.model):
            logger.info(
                "Tied Qwen3-VL lm_head.weight to language_model.embed_tokens.weight "
                "for transformers 4.57.x compatibility."
            )
        _patch_qwen3vl_output_projection_dtypes(self.model)
        self._supports_mm_token_type_ids = _forward_has_explicit_arg(self.model, "mm_token_type_ids")

        # needed since we don't use these layers. Also saves compute
        while len(self.language_model.layers) > select_layer:
            self.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

        self._skip_lm_head_enabled = str(os.environ.get("GROOT_QWEN_SKIP_LM_HEAD", "")).lower() in {
            "1",
            "true",
            "yes",
        }
        if self._skip_lm_head_enabled and _patch_qwen3vl_skip_unused_lm_head():
            logger.info("Patched Qwen3-VL top-level forward to skip unused lm_head logits in GR00T training.")
        self._reset_rotary_inv_freq()
        if str(os.environ.get("GROOT_QWEN_CHANNELS_LAST_3D", "")).lower() in {"1", "true", "yes"}:
            self._apply_vision_patch_embed_channels_last()

    def _apply_vision_patch_embed_channels_last(self) -> None:
        visual = self.visual
        if visual is None:
            return

        def _to_channels_last_3d(_module, inputs):
            if not inputs or torch.jit.is_tracing():
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and x.dim() == 5:
                return (x.contiguous(memory_format=torch.channels_last_3d), *inputs[1:])
            return inputs

        patched = 0
        for module in visual.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
                module.register_forward_pre_hook(_to_channels_last_3d)
                patched += 1
        if patched:
            logger.debug(
                "Applied channels_last_3d to %d vision patch-embed Conv3d "
                "module(s) (torch>=2.9 cuDNN Conv3d perf workaround).",
                patched,
            )

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        """Enable the requested trainable parameter groups for the backbone."""
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.language_model.requires_grad_(False)
            try:
                self.model.lm_head.requires_grad_(False)
            except AttributeError:
                pass
        if not tune_visual:
            self.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.language_model and not self.tune_llm:
                self.language_model.eval()
            if self.visual and not self.tune_visual:
                self.visual.eval()

    @property
    def language_model(self) -> torch.nn.Module:
        """Return the wrapped language model module."""
        return _unwrap_qwen_backbone(self.model).language_model

    @property
    def visual(self) -> torch.nn.Module:
        """Return the wrapped visual encoder module."""
        return _unwrap_qwen_backbone(self.model).visual

    def _reset_rotary_inv_freq(self) -> None:
        config = self.model.config
        vision_changed = self._reset_vision_rotary_inv_freq(config)
        text_changed = self._reset_language_rotary_inv_freq()
        logger.debug(
            "Qwen3-VL RoPE inv_freq reset (vision_rewritten=%s, text_rewritten=%s).",
            vision_changed,
            text_changed,
        )

    def _reset_vision_rotary_inv_freq(self, config) -> bool:
        visual = self.visual
        try:
            rotary = visual.rotary_pos_emb
        except AttributeError:
            rotary = None
        if rotary is None:
            raise RuntimeError("Qwen3-VL visual rotary_pos_emb/inv_freq not found.")
        try:
            rotary.inv_freq
        except AttributeError:
            raise RuntimeError("Qwen3-VL visual rotary_pos_emb/inv_freq not found.")

        vision_config = config.vision_config
        if vision_config is None:
            raise RuntimeError("Qwen3-VL vision_config missing hidden_size/num_heads.")

        head_dim = vision_config.hidden_size // vision_config.num_heads
        device = _real_inference_device(visual)
        inv_freq = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, device)
        return _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)

    def _reset_language_rotary_inv_freq(self) -> bool:
        language_model = self.language_model
        try:
            rotary = language_model.rotary_emb
        except AttributeError:
            rotary = None
        if rotary is None:
            raise RuntimeError("Qwen3-VL language rotary_emb/inv_freq/config not found.")
        try:
            rotary.inv_freq
        except AttributeError:
            raise RuntimeError("Qwen3-VL language rotary_emb/inv_freq/config not found.")
        try:
            text_config = rotary.config
        except AttributeError:
            text_config = language_model.config
        if text_config is None:
            raise RuntimeError("Qwen3-VL language rotary_emb/inv_freq/config not found.")

        device = _real_inference_device(language_model)
        inv_freq, _attention_scaling = recompute_text_rotary_inv_freq(rotary, text_config, device)
        changed = _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)
        if "original_inv_freq" in rotary.__dict__:
            changed = _assign_inv_freq(rotary, "original_inv_freq", inv_freq.clone(), persistent=False) or changed
        return changed

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Wrap a raw batch in a BatchFeature for the backbone."""
        return BatchFeature(data=batch)

    def prepare_cuda_graph_batch(self, batch) -> None:
        """Precompute Qwen dynamic position metadata before CUDA graph capture."""
        if not self._supports_mm_token_type_ids:
            return
        if getattr(batch, "position_ids", None) is not None:
            return
        if getattr(batch, "mm_token_type_ids", None) is not None:
            return

        input_ids = getattr(batch, "input_ids", None)
        image_grid_thw = getattr(batch, "image_grid_thw", None)
        attention_mask = getattr(batch, "attention_mask", None)
        if input_ids is None or image_grid_thw is None or attention_mask is None:
            return

        with torch.no_grad():
            mm_token_type_ids = _build_mm_token_type_ids(self.model.config, input_ids)
            position_ids = _build_qwen3vl_compat_position_ids(
                self.model,
                input_ids,
                mm_token_type_ids,
                image_grid_thw,
                attention_mask,
            )
        if position_ids is None:
            batch.mm_token_type_ids = mm_token_type_ids
        else:
            batch.position_ids = position_ids

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """Run the backbone and return the last hidden state plus masks."""
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        source_input = vl_input
        vl_input = {k: source_input[k] for k in keys_to_use}
        for optional_key in ("position_ids", "mm_token_type_ids"):
            if optional_key in source_input and source_input[optional_key] is not None:
                vl_input[optional_key] = source_input[optional_key]
        if self._supports_mm_token_type_ids:
            if "position_ids" not in vl_input and "mm_token_type_ids" not in vl_input:
                mm_token_type_ids = _build_mm_token_type_ids(self.model.config, vl_input["input_ids"])
                position_ids = _build_qwen3vl_compat_position_ids(
                    self.model,
                    vl_input["input_ids"],
                    mm_token_type_ids,
                    vl_input["image_grid_thw"],
                    vl_input["attention_mask"],
                )
                if position_ids is None:
                    vl_input["mm_token_type_ids"] = mm_token_type_ids
                else:
                    vl_input["position_ids"] = position_ids
        if self._skip_lm_head_enabled:
            outputs = self.model(
                **vl_input,
                output_hidden_states=False,
                logits_to_keep=0,
                loongforge_skip_lm_head=True,
            )
        else:
            outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1].to(torch.float32)
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
