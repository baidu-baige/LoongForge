# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from VLDA and openpi; preserves the VLDA world/action training objectives.

"""Three PiGemma streams with differentiable world-to-action conditioning."""

import logging
from typing import Any, Literal

import torch
from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma as PaliGemmaForConditionalGeneration,
)
from lerobot.policies.pi_gemma import (
    PiGemmaForCausalLM as GemmaForCausalLM,
)
from lerobot.policies.pi_gemma import (
    _gated_residual,
    layernorm_forward,
)
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from loongforge.embodied.model.pi05.modeling_pi05 import _flex_block_mask, _joint_attention_flex


def _attention(query, key, value, mask, layer):
    if isinstance(mask, BlockMask):
        return _joint_attention_flex(query, key, value, mask, layer.scaling)
    return modeling_gemma.eager_attention_forward(layer, query, key, value, mask, layer.scaling)[0]


def _final_norm(hidden, norm, cond):
    return layernorm_forward(norm, hidden, cond)[0]


def _compute_qkv(hidden_states, layer):
    """Compute Q, K, V for a single model from its hidden_states (after layernorm)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return (q, k, v)


def _compute_kv(hidden_states, layer):
    """Compute K, V for a single model from its hidden_states (after layernorm)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return (k, v)


def _apply_rotary_and_attention(
    query_states, key_states, value_states, position_ids, rotary_emb, attention_layer, attention_mask, batch_size
):
    """Apply RoPE and joint attention. Returns reshaped attention output."""
    dummy = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    (cos, sin) = rotary_emb(dummy, position_ids)
    (query_states, key_states) = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    att_output = _attention(query_states, key_states, value_states, attention_mask, attention_layer)
    head_dim = attention_layer.head_dim
    att_output = att_output.reshape(batch_size, -1, query_states.shape[1] * head_dim)
    return att_output


def _apply_rotary_and_attention_action_only(
    query_states,
    key_states,
    value_states,
    position_ids_q,
    position_ids_k,
    rotary_emb,
    attention_layer,
    attention_mask,
    batch_size,
):
    """Apply separate query/key RoPE and attention over the cached prefix."""
    dummy_q = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    (cos_q, sin_q) = rotary_emb(dummy_q, position_ids_q)
    (query_states, _) = modeling_gemma.apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q, unsqueeze_dim=1)
    dummy_k = torch.zeros(
        key_states.shape[0], key_states.shape[2], key_states.shape[-1], device=key_states.device, dtype=key_states.dtype
    )
    (cos_k, sin_k) = rotary_emb(dummy_k, position_ids_k)
    (_, key_states) = modeling_gemma.apply_rotary_pos_emb(key_states, key_states, cos_k, sin_k, unsqueeze_dim=1)
    att_output = _attention(query_states, key_states, value_states, attention_mask, attention_layer)
    return att_output.reshape(batch_size, query_states.shape[2], -1)


def _apply_mlp_residual(hidden_states, attn_out, gate, layer, cond=None):
    """Apply o_proj, gated residual, post-attention-layernorm, MLP, second residual."""
    if attn_out.dtype != layer.self_attn.o_proj.weight.dtype:
        attn_out = attn_out.to(layer.self_attn.o_proj.weight.dtype)
    out = layer.self_attn.o_proj(attn_out)
    out = _gated_residual(hidden_states, out, gate)
    after_first_residual = out.clone()
    (out, gate) = layernorm_forward(layer.post_attention_layernorm, out, cond)
    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        out = out.to(dtype=torch.bfloat16)
    out = layer.mlp(out)
    out = _gated_residual(after_first_residual, out, gate)
    return out


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        world_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        attention_backend: str = "eager",
        compile_model: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
    ):
        if use_adarms is None:
            use_adarms = [False, False, False]
        super().__init__()
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )
        world_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=world_expert_config.head_dim,
            hidden_size=world_expert_config.width,
            intermediate_size=world_expert_config.mlp_dim,
            num_attention_heads=world_expert_config.num_heads,
            num_hidden_layers=world_expert_config.depth,
            num_key_value_heads=world_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[2],
            adarms_cond_dim=world_expert_config.width if use_adarms[2] else None,
        )
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.action_gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.world_gemma_expert = GemmaForCausalLM(config=world_expert_config_hf)
        self.action_gemma_expert.model.embed_tokens = None
        self.world_gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)
        self.attention_backend = attention_backend
        self._layer_fn = self._compute_layer_2way
        self._action_layer_fn = self._compute_action_expert_layer
        self._norm_fn = _final_norm
        if compile_model:
            # Compile tensor operations, keeping parameters and checkpoint keys intact.
            for name in ("_layer_fn", "_action_layer_fn", "_norm_fn"):
                compiled = torch.compile(getattr(self, name), mode=compile_mode, fullgraph=True, dynamic=False)
                setattr(self, name, compiled)
        logging.info("VLDA Gemma: attention=%s, compile=%s, mode=%s", attention_backend, compile_model, compile_mode)

    def _attention_mask(self, mask):
        # The training masks contain only 0 (visible) and a large negative value.
        # Build once per pass, then share the BlockMask across all transformer layers.
        return _flex_block_mask(mask[:, 0] == 0) if self.attention_backend == "flex" else mask

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any((selector in name for selector in params_to_keep_float32)):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """Preserve openpi's FP32 embeddings and unscaled projected features.

        Native SigLIP does not insert the embedding-to-encoder dtype cast from
        openpi's patch. Native PaliGemma also divides projected features by
        sqrt(hidden_size), so use its constituent modules directly here.
        """
        vision = self.paligemma.model.vision_tower.vision_model
        hidden = vision.embeddings(image)
        hidden = hidden.to(vision.encoder.layers[0].self_attn.q_proj.weight.dtype)
        hidden = vision.encoder(inputs_embeds=hidden).last_hidden_state
        hidden = vision.post_layernorm(hidden)
        return self.paligemma.model.multi_modal_projector(hidden)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None, None]
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            action_output = None
            world_output = None
        elif inputs_embeds[0] is None and inputs_embeds[2] is None:
            action_output = self.action_gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            action_output = action_output.last_hidden_state
            prefix_output = None
            world_output = None
            prefix_past_key_values = None
        elif inputs_embeds[0] is None and inputs_embeds[1] is None:
            world_output = self.world_gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[2] if adarms_cond is not None else None,
            )
            world_output = world_output.last_hidden_state
            prefix_output = None
            action_output = None
            prefix_past_key_values = None
        else:
            raise ValueError("PaliGemmaWithExpertModel.forward only supports single-branch inference.")
        return ([prefix_output, action_output, world_output], prefix_past_key_values)

    @staticmethod
    def _compute_layer_2way(
        inputs_embeds, attention_mask, position_ids, adarms_cond, llm_layer, expert_layer, rotary_emb
    ):
        """One MOT layer: LLM + one expert — identical logic to the original
        ``compute_layer_complete`` but parameterised by the expert model."""
        (query_states, key_states, value_states, gates) = ([], [], [], [])
        layers = [llm_layer, expert_layer]
        for i, hidden_states in enumerate(inputs_embeds):
            layer = layers[i]
            (hidden_states, gate) = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
            gates.append(gate)
            (q, k, v) = _compute_qkv(hidden_states, layer)
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)
        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
        batch_size = query_states.shape[0]
        attn_layer = llm_layer.self_attn
        att_output = _apply_rotary_and_attention(
            query_states, key_states, value_states, position_ids, rotary_emb, attn_layer, attention_mask, batch_size
        )
        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = layers[i]
            end_pos = start_pos + hidden_states.shape[1]
            out = _apply_mlp_residual(
                hidden_states, att_output[:, start_pos:end_pos], gates[i], layer, cond=adarms_cond[i]
            )
            outputs_embeds.append(out)
            start_pos = end_pos
        return outputs_embeds

    def forward_stage1(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        inputs_embeds: list[torch.Tensor],
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        """Joint LLM + World Expert MOT forward (training).

        Thin wrapper around the shared layer implementation so that
        Stage\u202f1 and Stage\u202f2 Pass\u202f1 share exactly the same layer code.

        Returns
        -------
        [prefix_output, suffix_output] : list[Tensor]
        """
        models = [self.paligemma.model.language_model, self.world_gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        if adarms_cond is None:
            adarms_cond = [None, None]
        use_gc = self._should_use_gradient_checkpointing(self.world_gemma_expert.model)
        llm_model = models[0]
        expert_model = models[1]
        attention_mask = self._attention_mask(attention_mask)
        for layer_idx in range(num_layers):
            if use_gc:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    self._layer_fn,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    llm_model.layers[layer_idx],
                    expert_model.layers[layer_idx],
                    llm_model.rotary_emb,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = self._layer_fn(
                    inputs_embeds, attention_mask, position_ids, adarms_cond,
                    llm_model.layers[layer_idx], expert_model.layers[layer_idx], llm_model.rotary_emb,
                )

        if use_gc:
            prefix_output = torch.utils.checkpoint.checkpoint(
                self._norm_fn, inputs_embeds[0], models[0].norm, adarms_cond[0],
                use_reentrant=False, preserve_rng_state=False,
            )
            suffix_output = torch.utils.checkpoint.checkpoint(
                self._norm_fn, inputs_embeds[1], models[1].norm, adarms_cond[1],
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            prefix_output = self._norm_fn(inputs_embeds[0], models[0].norm, adarms_cond[0])
            suffix_output = self._norm_fn(inputs_embeds[1], models[1].norm, adarms_cond[1])
        return ([prefix_output, suffix_output], None)

    def _should_use_gradient_checkpointing(self, model) -> bool:
        """Check whether gradient checkpointing should be used for *model*."""
        return (
            hasattr(model, "gradient_checkpointing")
            and model.gradient_checkpointing
            and self.training
            or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)
        )

    def _enable_expert_gradient_checkpointing(self, expert_model):
        """Force‑enable gradient checkpointing on an expert if we're training."""
        if self.training and hasattr(expert_model, "gradient_checkpointing"):
            if not expert_model.gradient_checkpointing:
                expert_model.gradient_checkpointing = True

    def _forward_llm_world(self, llm_embeds, world_embeds, attention_mask, position_ids, world_adarms_cond):
        """Run LLM + World Expert MOT and **return per‑layer LLM hidden states**.

        Returns
        -------
        world_output : Tensor   [B, K, D_world]   final world expert output (after norm)
        llm_states   : list[Tensor]   len = num_layers+1
            ``llm_states[i]`` is the LLM hidden state **entering** layer *i*;
            ``llm_states[-1]`` is the LLM output after the last layer.
        """
        llm_model = self.paligemma.model.language_model
        world_model = self.world_gemma_expert.model
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        use_gc = self._should_use_gradient_checkpointing(world_model)
        inputs_embeds = [llm_embeds, world_embeds]
        adarms_cond = [None, world_adarms_cond]
        llm_states = []
        attention_mask = self._attention_mask(attention_mask)
        for layer_idx in range(num_layers):
            llm_states.append(inputs_embeds[0])
            if use_gc:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    self._layer_fn,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    llm_model.layers[layer_idx],
                    world_model.layers[layer_idx],
                    llm_model.rotary_emb,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = self._layer_fn(
                    inputs_embeds, attention_mask, position_ids, adarms_cond,
                    llm_model.layers[layer_idx], world_model.layers[layer_idx], llm_model.rotary_emb,
                )
        llm_states.append(inputs_embeds[0])

        if use_gc:
            world_output = torch.utils.checkpoint.checkpoint(
                self._norm_fn, inputs_embeds[1], world_model.norm, world_adarms_cond,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            world_output = self._norm_fn(inputs_embeds[1], world_model.norm, world_adarms_cond)
        return (world_output, llm_states)

    @staticmethod
    def _compute_action_expert_layer(
        llm_hidden, action_hidden, attention_mask, position_ids, action_adarms_cond,
        llm_layer, action_layer, rotary_emb,
    ):
        """One layer for the Action Expert using a **pre‑computed** LLM hidden state.

        The LLM hidden state acts as read‑only context: its K,V are computed
        so the action expert can attend to them, but the LLM state is NOT
        updated. Gradients still flow through the cached prefix hidden states.
        """
        (llm_normed, _) = layernorm_forward(llm_layer.input_layernorm, llm_hidden)
        (action_normed, action_gate) = layernorm_forward(
            action_layer.input_layernorm, action_hidden, action_adarms_cond
        )
        (llm_k, llm_v) = _compute_kv(llm_normed, llm_layer)
        (action_q, action_k, action_v) = _compute_qkv(action_normed, action_layer)
        llm_len = llm_hidden.shape[1]
        action_len = action_hidden.shape[1]
        total_len = action_len + llm_len
        assert total_len == position_ids.shape[1], (
            f"lenth of position_ids {position_ids.shape[1]} dosen't equal to total_len {total_len}"
        )
        query_states = action_q
        batch_size = query_states.shape[0]
        key_states = torch.cat([llm_k, action_k], dim=2)
        value_states = torch.cat([llm_v, action_v], dim=2)
        attn_layer = llm_layer.self_attn
        position_ids_q = position_ids[:, llm_len:total_len]
        position_ids_k = position_ids[:, :total_len]
        assert query_states.shape[2] == action_len
        assert key_states.shape[2] == total_len
        assert value_states.shape[2] == total_len
        assert position_ids_q.shape[1] == action_len
        assert position_ids_k.shape[1] == total_len
        att_output = _apply_rotary_and_attention_action_only(
            query_states,
            key_states,
            value_states,
            position_ids_q,
            position_ids_k,
            rotary_emb,
            attn_layer,
            attention_mask,
            batch_size,
        )
        action_out = _apply_mlp_residual(action_hidden, att_output, action_gate, action_layer, cond=action_adarms_cond)
        return action_out

    def _forward_action_with_cached_llm(
        self, llm_states, action_embeds, attention_mask, position_ids, action_adarms_cond
    ):
        """Run Action Expert MOT reusing the LLM hidden states from Pass 1.

        ``llm_states[i]`` is the LLM hidden state entering layer *i*.
        """
        llm_model = self.paligemma.model.language_model
        action_model = self.action_gemma_expert.model
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        use_gc = self._should_use_gradient_checkpointing(action_model)
        action_hidden = action_embeds
        llm_len = llm_states[0].shape[1]
        attention_mask = self._attention_mask(attention_mask[:, :, llm_len:, :])
        for layer_idx in range(num_layers):
            llm_hidden = llm_states[layer_idx]
            if use_gc:
                action_hidden = torch.utils.checkpoint.checkpoint(
                    self._action_layer_fn,
                    llm_hidden,
                    action_hidden,
                    attention_mask,
                    position_ids,
                    action_adarms_cond,
                    llm_model.layers[layer_idx],
                    action_model.layers[layer_idx],
                    llm_model.rotary_emb,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                action_hidden = self._action_layer_fn(
                    llm_hidden,
                    action_hidden,
                    attention_mask,
                    position_ids,
                    action_adarms_cond,
                    llm_model.layers[layer_idx],
                    action_model.layers[layer_idx],
                    llm_model.rotary_emb,
                )

        if use_gc:
            action_output = torch.utils.checkpoint.checkpoint(
                self._norm_fn, action_hidden, action_model.norm, action_adarms_cond,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            action_output = self._norm_fn(action_hidden, action_model.norm, action_adarms_cond)
        return action_output

    def forward_stage2_pass1(
        self,
        attention_mask_world: torch.Tensor,
        position_ids_world: torch.LongTensor,
        llm_embeds: torch.Tensor,
        world_embeds: torch.Tensor,
        world_adarms_cond: torch.Tensor | None,
    ):
        """Pass\u202f1: LLM + World Expert MOT → world_output + LLM cached states."""
        return self._forward_llm_world(
            llm_embeds, world_embeds, attention_mask_world, position_ids_world, world_adarms_cond
        )

    def forward_stage2_pass2(
        self,
        llm_states: list[torch.Tensor],
        action_embeds: torch.Tensor,
        attention_mask_action: torch.Tensor,
        position_ids_action: torch.LongTensor,
        action_adarms_cond: torch.Tensor | None,
    ):
        """Pass\u202f2: Cached LLM + Action Expert MOT → action_output."""
        return self._forward_action_with_cached_llm(
            llm_states, action_embeds, attention_mask_action, position_ids_action, action_adarms_cond
        )
