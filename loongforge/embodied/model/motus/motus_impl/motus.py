# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motus - Modular Architecture.

Three-modal UniDiffuser: Video Model (WAN) + Action Expert + Understanding Expert.
Implements a MoT (Mixture of Tokens) architecture with unified attention.
"""

import sys
import json
import os
import torch
import logging
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# Vendored into loongforge: wan/ and motus_utils are package-local siblings, imported
# relatively. (The original sys.path-insert of the source repo's ``bak/`` root is
# removed on purpose so we never accidentally pick up the un-vendored source copy.)
from .motus_utils import get_t_distribution
from .wan.modules.model import sinusoidal_embedding_1d
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig
# Add Flow-Matching schedulers
from .wan.utils.fm import FlowMatchScheduler
from .wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


@dataclass
class MotusConfig:
    """Configuration for Motus."""
    # Video model settings
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    wan_config_path: str = ""
    video_precision: str = "bfloat16"

    # VLM settings
    vlm_checkpoint_path: str = ""
    
    # Understanding Expert settings - configurable from yaml
    und_expert_hidden_size: int = 512        # Understanding expert hidden dimension
    und_expert_ffn_dim_multiplier: int = 4   # Understanding expert FFN dimension multiplier
    und_expert_norm_eps: float = 1e-5        # Understanding expert layer norm epsilon
    und_layers_to_extract: List[int] = None  # Which VLM layers to extract from
    
    # VLM adapter settings for understanding expert
    vlm_adapter_input_dim: int = 2048        # VLM feature dimension (input)
    vlm_adapter_projector_type: str = "mlp3x_silu"  # VLM adapter type

    # Action expert settings  
    num_layers: int = 30 
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict
    
    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width
    
    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Control whether to load pretrained WAN/VLM backbones.
    # None = default behavior (load), False = skip loading (init from config only)
    load_pretrained_backbones: Optional[bool] = None

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size is determined by global downsample rate and frequency ratio
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        
        # Default understanding layers to extract from (if not specified)
        if self.und_layers_to_extract is None:
            # Extract from all layers for comprehensive understanding
            self.und_layers_to_extract = list(range(self.num_layers))


def to_channels_last_3d_if_needed(x: torch.Tensor) -> torch.Tensor:
      """Return ``x`` in channels-last-3d layout for 5D tensors, otherwise unchanged."""
      if x.ndim == 5 and not x.is_contiguous(memory_format=torch.channels_last_3d):
          return x.contiguous(memory_format=torch.channels_last_3d)
      return x


class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(self, video_model, dtype, device, grid_sizes):
        """Store the WAN video model plus dtype/device and the video-latent grid sizes."""
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video tokens from pre-processed noisy latent."""
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)

        # Derive grid_sizes from the ACTUAL patched grid + runtime batch, instead of
        # relying on a build-time value pre-expanded to config.batch_size. The Conv3d
        # patch_embedding yields [B, C, F, H, W]; (F, H, W) is exactly the per-sample
        # spatio-temporal grid and B is the true per-device batch. This keeps
        # grid_sizes' row count in lock-step with the video batch so downstream
        # unpatchify (zip(x, grid_sizes)) and rope never truncate the batch. Values
        # are identical across rows (single video geometry), so this is numerically
        # equivalent to the old pre-expanded tensor whenever the batch matched.
        b, _c, f, h, w = video_patched.shape
        self.grid_sizes = torch.tensor(
            [f, h, w], dtype=torch.long, device=video_patched.device
        ).unsqueeze(0).expand(b, -1)

        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)

        # Calculate sequence length and padding
        # seq_lens = torch.tensor([u.size(1) for u in video_tokens_list], dtype=torch.long, device=self.device)
        # seq_len = seq_lens.max().item()

        # Concatenate with padding
        # video_tokens = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) 
        #     for u in video_tokens_list
        # ])

        # return video_tokens

        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        # Handle both old format (List[torch.Tensor]) and new format (torch.Tensor)
        if isinstance(language_embeddings, list):
            # Old format: List[torch.Tensor] - do padding
            text_len = self.video_model.wan_model.text_len  # 512
            padded_embeddings = []

            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)

            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            # New format: torch.Tensor [B, seq_len, dim] - already padded by collate_fn
            t5_context_raw = language_embeddings
        
        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)

        return t5_context

    def pad_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Frozen/no-param half of preprocess_t5_embeddings: pad/stack the raw T5
        embeddings to [B, text_len, dim] WITHOUT the trainable text_embedding.

        Used by the DeepCompile split path so the data-dependent list padding runs
        eagerly (avoids the varlen graph break) while text_embedding is applied
        inside the compiled forward. Numerically identical to the padding branch
        of preprocess_t5_embeddings.
        """
        if isinstance(language_embeddings, list):
            text_len = self.video_model.wan_model.text_len  # 512
            padded_embeddings = []
            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)
            return torch.stack(padded_embeddings, dim=0)
        return language_embeddings

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()
            
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32
            
        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor, 
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None  # WAN uses None for fixed-length context
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out
    
    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        
        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        und_tokens: torch.Tensor,
        und_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trimodal joint self-attention: WAN + Action + Understanding via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Action heads for WAN space (1024 -> 24*128)
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # Understanding Expert processing
        norm_und = und_block.norm1(und_tokens)
        L_u = norm_und.shape[1]
        
        # Understanding Expert heads for WAN space (2048 -> 24*128)
        u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
        u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
        u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
        u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
        u_v = u_v_h.view(B, L_u, n, d)

        # Meta info for WAN attention. Fixed-resolution training uses equal
        # sequence lengths across the batch, so on the static dense-flash compile
        # path (graph_fhw set) pass None to use flash_attention's graph-safe flatten
        # path instead of dynamic Python slicing with tensor lengths.
        use_graph_safe_attention = getattr(wan_layer.self_attn, "graph_fhw", None) is not None
        seq_lens = None if use_graph_safe_attention else torch.full(
            (B,), L_v + L_a + L_u, dtype=torch.long, device=self.device
        )
        freqs = self.video_model.wan_model.freqs
        #if freqs.device != self.device:
        #    freqs = freqs.to(self.device)

        # Call WAN self-attn with trimodal MoT
        y, action_out_h, und_out_h = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            und_q=u_q, und_k=u_k, und_v=u_v
        )
        
        # Project Understanding Expert output
        und_out = und_block.wan_und_o(und_out_h.flatten(2))

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out  # Regular residual connection

        return video_tokens, action_tokens, und_tokens


class UndModule(nn.Module):
    """Understanding module - handles VLM with understanding queries and Understanding Expert."""

    def __init__(self, vlm_model, und_expert, config, dtype, device):
        """Store the VLM backbone and Understanding Expert plus config/dtype/device."""
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        
        # VLM model reference
        self.vlm_model = vlm_model
        
        # Understanding Expert reference
        self.und_expert = und_expert
        
    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Extract understanding features from VLM last layer."""
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        with torch.no_grad():
            (
                inputs_embeds,
                attention_mask,
                visual_pos_masks,
                deepstack_image_embeds,
                position_ids,
            ) = self._process_vlm_inputs_to_tokens(vlm_inputs, B)
                
            # Forward through VLM with proper attention_mask and DeepStack features
            vlm_kwargs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_values': None,
                'use_cache': False,
                'output_attentions': False,
                'output_hidden_states': True,
                'return_dict': True
            }
            
            # Add DeepStack parameters for Qwen3-VL
            if visual_pos_masks is not None:
                vlm_kwargs['visual_pos_masks'] = visual_pos_masks
            if deepstack_image_embeds is not None:
                vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1].detach()  # [B, seq_len, vlm_dim]

        # [B, seq_len, vlm_dim] -> [B, seq_len, und_dim]
        adapted_features = self.und_expert.vlm_adapter(last_layer_features)

        return adapted_features

    def extract_vlm_last_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Frozen/no-param half of extract_und_features: run the frozen VLM and
        return its last-layer hidden states WITHOUT the trainable vlm_adapter.

        Used by the DeepCompile path: the frozen VLM forward (dispatch-heavy,
        non-trainable) runs eager in _deepcompile_preprocess, while the trainable
        vlm_adapter projection runs inside the compiled _deepcompile_core region.
        The un-adapted features are exactly extract_und_features up to the adapter.
        """
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        with torch.no_grad():
            (
                inputs_embeds,
                attention_mask,
                visual_pos_masks,
                deepstack_image_embeds,
                position_ids,
            ) = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

            vlm_kwargs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_values': None,
                'use_cache': False,
                'output_attentions': False,
                'output_hidden_states': True,
                'return_dict': True
            }

            if visual_pos_masks is not None:
                vlm_kwargs['visual_pos_masks'] = visual_pos_masks
            if deepstack_image_embeds is not None:
                vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Last-layer features only; adapter is applied later inside the graph.
        last_layer_features = vlm_output.hidden_states[-1].detach()  # [B, seq_len, vlm_dim]

        return last_layer_features

    def _process_vlm_inputs_to_tokens(
        self, vlm_inputs, B: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        # image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(
        #     pixel_values_batch, image_grid_thw_batch)
        vision_output = self.vlm_model.get_image_features(
             pixel_values_batch, image_grid_thw_batch
        )

        image_embeds = vision_output.pooler_output
        deepstack_image_embeds = vision_output.deepstack_features

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)

        # Insert image embeddings
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        mm_token_type_ids = torch.zeros_like(input_ids_batch, dtype=torch.long)
        mm_token_type_ids[visual_pos_masks] = 1  # 图像部分标1

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch,
            mm_token_type_ids=mm_token_type_ids
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    
    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Process Understanding Expert FFN with regular LayerNorm."""
        block = self.und_expert.blocks[layer_idx]
        
        # Pre-norm for FFN (regular LayerNorm)
        ffn_input = block.norm2(und_tokens)
        ffn_output = block.ffn(ffn_input)
        
        # FFN residual connection
        und_tokens = und_tokens + ffn_output
        
        return und_tokens


class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""
    
    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        """Store the Action Expert and references to WAN/VLM weights plus config/dtype/device."""
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.vlm_model = vlm_model      # For accessing VLM weights
        self.dtype = dtype
        self.device = device
    
    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            
            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]
            
            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(
                2, (6, self.config.action_expert_dim)
            )  # [B, seq_len, 6, dim]
            
            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class Motus(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM, VLM, and Action modules.
    """

    def __init__(self, config: MotusConfig):
        """Build the video/VLM backbones, action/understanding experts and the MoT modules."""
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Initialize video/VLM backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        self._action_slice_start = 0 if self.config.training_mode == 'pretrain' else 1

        # Initialize video model (WAN)
        logger.info("Initializing WAN video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        #for block in self.video_model.wan_model.blocks:
        #    block.ffn = torch.compile(block.ffn, backend="inductor", mode="max-autotune", dynamic=False)
        #
        #self.video_model.wan_model.head = torch.compile(
        #    self.video_model.wan_model.head,
        #    backend="inductor",
        #    mode="max-autotune",
        #    dynamic=False,
        #)

        #self.video_model.gradient_checkpointing_enable()

        self.video_model.wan_model.freqs = self.video_model.wan_model.freqs.cuda()

        # Initialize VLM (frozen)
        logger.info("Initializing VLM (frozen)...")
        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )
            logger.info("Load pretrained VLM...")
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            # Move to CUDA and dtype explicitly
            self.vlm_model.to(device="cuda", dtype=self.dtype)
            logger.info("Initializing VLM from config...")

        # Freeze VLM parameters
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        logger.info("VLM parameters frozen")

        # Keep VLM complete (do not truncate)
        logger.info(f"VLM kept complete with {len(self.vlm_model.model.language_model.layers)} layers")

        # Get WAN and VLM configurations directly
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_text_config = (
            self.vlm_model.config.text_config
            if hasattr(self.vlm_model.config, 'text_config')
            else self.vlm_model.config
        )
        vlm_num_kv_heads = getattr(vlm_text_config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers  = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D")
        logger.info(
            f"  VLM: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × "
            f"{vlm_head_dim} head_dim = {vlm_dim}D"
        )

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads, 
            'head_dim': wan_head_dim
        }
        vlm_config = {
            'hidden_size': vlm_dim,
            'num_attention_heads': vlm_num_heads,
            'num_key_value_heads': vlm_num_kv_heads,
            'head_dim': vlm_head_dim,
            'num_hidden_layers': vlm_num_hidden_layers,
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        self.action_expert = ActionExpert(action_config, wan_config)

        #self.action_expert.input_encoder = torch.compile(
        #    self.action_expert.input_encoder,
        #    backend="inductor",
        #    mode="reduce-overhead",
        #    dynamic=False,
        #    fullgraph=False,
        #)

        #self.action_expert.decoder = torch.compile(
        #    self.action_expert.decoder,
        #    backend="inductor",
        #    mode="reduce-overhead",
        #    dynamic=False,
        #    fullgraph=False,
        #)

        #for block in self.action_expert.blocks:
        #    block.ffn = torch.compile(block.ffn, backend="inductor", mode="reduce-overhead", dynamic=False)

        #self.core_forward_static = torch.compile(
        #    self.core_forward_static,
        #    backend="inductor",
        #    mode="max-autotune",
        #    fullgraph=False,
        #    dynamic=False,
        #)
        

        # Initialize Understanding Expert
        logger.info("Initializing Understanding Expert...")
        und_config = UndExpertConfig(
            dim=config.und_expert_hidden_size,
            ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            vlm_input_dim=config.vlm_adapter_input_dim,
            vlm_projector_type=config.vlm_adapter_projector_type,
            eps=config.und_expert_norm_eps,
        )
        
        self.und_expert = UndExpert(und_config, wan_config, vlm_config)

        #for block in self.und_expert.blocks:
        #    block.ffn = torch.compile(block.ffn, backend="inductor", mode="reduce-overhead", dynamic=False)

        # Move models to device
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.und_expert.to(device=self.device, dtype=self.dtype)
        
        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W], 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded
        
        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device)
        self.action_module = ActionModule(
            self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device
        )

        # Initialize t distributions from config
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {
            'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'),
            'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0),
            'min_t': time_dist_config.get('min_t', 0.0),
            'max_t': time_dist_config.get('max_t', 1.0)
        }

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (video)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (action)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        # Log parameter counts
        self.log_parameter_counts()

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        und_params = sum(p.numel() for p in self.und_expert.parameters())

        logger.info(f"Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        logger.info(f"  VLM (frozen): {vlm_params / 1e9:.2f}B")
        logger.info(f"  Und Expert: {und_params / 1e6:.1f}M")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)
    
        # Load state dict
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        state_dict = checkpoint['module']  
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        
        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items() 
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")
        
        # Handle directory path - try two possible locations
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            # Try two possible paths
            possible_paths = [
                checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
                checkpoint_path / "mp_rank_00_model_states.pt",                   
            ]
            
            checkpoint_file = None
            for p in possible_paths:
                if p.exists():
                    checkpoint_file = p
                    logger.info(f"Found checkpoint: {checkpoint_file}")
                    break
            
            if checkpoint_file is None:
                raise FileNotFoundError(
                    f"Checkpoint not found. Tried:\n"
                    f"  - {possible_paths[0]}\n"
                    f"  - {possible_paths[1]}"
                )
            path = str(checkpoint_file)
        else:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading pretrain weights from {path}")
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        state_dict = checkpoint.get('module', checkpoint)
        
        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v
        
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    @torch.no_grad()
    def encode_video_latents(
        self,
        first_frame: torch.Tensor,   # [B, C, H, W]
        video_frames: torch.Tensor,  # [B, num_frames, C, H, W]
    ):
        """VAE-encode the (first_frame, video_frames) pair into latents.

        Factored out of training_step so it can be launched on a dedicated CUDA
        side-stream and overlapped with the previous batch's graphed compute.
        Runs under no_grad (VAE is frozen / not part of autograd), so it is safe
        to execute on a stream other than the main compute stream.
        """
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)               # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)    # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)     # [B, C, frames+1, H, W]
        # clean_full_latent: [B, 48, latent_frames, H', W']
        clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))
        # The WAN VAE encoder is causal and encodes the first frame as its own
        # chunk (encode() runs chunk0 = x[:, :, :1] with a freshly cleared cache),
        # and conv1 is a 1x1x1 CausalConv3d (no temporal mixing). So the first
        # latent frame of the full encode is bit-identical to separately encoding
        # first_frame_norm. Slice it out instead of paying for a second encode.
        condition_frame_latent = clean_full_latent[:, :, 0:1, :, :]             # [B, 48, 1, H', W']
        return clean_full_latent, condition_frame_latent

    def training_step(
        self,
        clean_full_latent: torch.Tensor,        # [B, 48, latent_frames, H', W'] - pre-encoded VAE latent
        condition_frame_latent: torch.Tensor,   # [B, 48, 1, H', W'] - pre-encoded condition-frame latent
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities.

        Args:
            clean_full_latent: Pre-encoded VAE latent for the full video (first frame + target frames)
            condition_frame_latent: Pre-encoded VAE latent for the condition (first) frame
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            return_dict: Whether to return detailed outputs

        Returns:
            Dictionary containing losses and metrics
        """
        B = clean_full_latent.shape[0]

        # PARITY_BATCH_CKSUM (default unset -> inert): byte-checksum every batch
        # tensor fed to the model on rank0 (eager path mirror of the probe in
        # _deepcompile_preprocess) so base<->transplant get_batch parity can be
        # verified step-by-step. Runs before any (re)seeding -> raw pipeline output.
        if os.environ.get("PARITY_BATCH_CKSUM") is not None:
            import hashlib as _hl
            import torch.distributed as _dist
            _rank = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else 0
            if _rank == 0:
                def _sha(t):
                    if t is None:
                        return "none"
                    if isinstance(t, torch.Tensor):
                        b = t.detach().float().cpu().contiguous().numpy().tobytes()
                        return f"{tuple(t.shape)}:{_hl.sha1(b).hexdigest()[:12]}"
                    if isinstance(t, (list, tuple)):
                        return "[" + ",".join(_sha(x) for x in t) + "]"
                    if isinstance(t, dict):
                        return "{" + ",".join(f"{k}={_sha(v)}" for k, v in sorted(t.items())) + "}"
                    return type(t).__name__
                _ck = getattr(self, "_parity_ck_step", 0)
                print(
                    f"[BATCH_CKSUM] step={_ck} clean={_sha(clean_full_latent)} "
                    f"cond={_sha(condition_frame_latent)} state={_sha(state)} "
                    f"actions={_sha(actions)} lang={_sha(language_embeddings)} "
                    f"vlm={_sha(vlm_inputs)}",
                    flush=True,
                )
            self._parity_ck_step = getattr(self, "_parity_ck_step", 0) + 1

        # PARITY_FM_SEED (default unset -> inert): reseed global RNG per training
        # step so the 4 flow-matching draws below (video randint/randn_like, action
        # randint/randn_like) are identical across base<->transplant. Shares the same
        # per-call counter as _deepcompile_preprocess (both start at 0, +1 per step);
        # the eager path only calls training_step, so no double increment.
        _pf = os.environ.get("PARITY_FM_SEED")
        if _pf is not None:
            _s = getattr(self, "_parity_fm_step", 0)
            torch.manual_seed(int(_pf) + _s)
            self._parity_fm_step = _s + 1

        # Flow-Matching noise mixture
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        # Scalar timesteps (0..num_train_timesteps) for time embedding
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for noise mixture
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # Teacher Forcing on the first frame
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # 2. Action pipeline 
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        # Discrete timesteps for time embedding (0..num_train_timesteps)
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(
            dtype=self.dtype, device=self.device
        )  # [B]
        # Sigma for action noise mixture
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(
            dtype=self.dtype, device=self.device
        ).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions

        # PARITY_FM_SEED probe (mirrors _deepcompile_preprocess): print the FM draws
        # per step so base<->transplant eager runs can be diffed.
        if _pf is not None:
            import torch.distributed as _dist2
            _rk = _dist2.get_rank() if _dist2.is_available() and _dist2.is_initialized() else 0
            if _rk == 0:
                import hashlib as _hl2
                def _h(t):
                    return _hl2.sha1(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
                print(
                    f"[FM_PROBE] fm_step={_s} "
                    f"vid_tid={timestep_id.tolist()} act_tid={timestep_id_action.tolist()} "
                    f"vnoise={_h(video_noise)} anoise={_h(action_noise)}",
                    flush=True,
                )

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, seq_len, und_dim]

        # Time embeddings
        # Use scheduler-provided timesteps (0..num_train_timesteps) for WAN/action time embeddings
        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
            video_t_embed, video_tokens.shape[1]
        )
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
            action_t_embed, action_tokens.shape[1]
        )

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        #with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
        #    # Process through 30 layers - modality-grouped execution
        #    for layer_idx in range(self.config.num_layers):
        #        # Compute AdaLN modulation once per layer using pre-computed parameters
        #        video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
        #        action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
        #        
        #        # Trimodal MoT: WAN + Action + Understanding Expert joint attention
        #        video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
        #            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
        #            self.action_expert.blocks[layer_idx],
        #            und_tokens, self.und_expert.blocks[layer_idx]
        #        )
        #
        #        # WAN cross
        #        video_tokens = self.video_module.process_cross_attention(video_tokens,
        #                           video_adaln_params, layer_idx, processed_t5_context)
        #
        #        # FFNs: WAN, Action, Understanding (each processes their own FFN)
        #        video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
        #        action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
        #        und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
        #        
        #
        #    # 4. Heads + Losses
        #    video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
        #    action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
        #    up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
        #    # Slice predicted actions depending on mode
        #    if self.config.training_mode == 'pretrain':
        #        action_pred = action_pred_full[:, :up_len, :]
        #    else:
        #        action_pred = action_pred_full[:, 1:up_len, :]

            # Video loss (mask the first frame)
        video_pred, action_pred = self.core_forward_static(video_tokens=video_tokens,
                                                               action_tokens=action_tokens,
                                                               und_tokens=und_tokens,
                                                               video_adaln_params=video_adaln_params,
                                                               action_adaln_params=action_adaln_params,
                                                               video_head_time_emb=video_head_time_emb,
                                                               action_head_time_emb=action_head_time_emb,
                                                               processed_t5_context=processed_t5_context)
        video_pred_masked = video_pred.clone()
        video_pred_masked[:, :, 0:1] = 0
        # Compute losses in fp32 to match origin. The graph capture forced the
        # loss out of core_forward_static's autocast block, so mse_loss would
        # otherwise run in bf16 (action branch quantized to the bf16 grid) and
        # feed coarse gradients back on the very first backward.
        video_loss = torch.nn.functional.mse_loss(video_pred_masked.float(), video_target.float(), reduction='mean')

        # Action loss
        action_loss = torch.nn.functional.mse_loss(action_pred.float(), action_target.float(), reduction='mean')

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )
        
        if return_dict:
            return {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'video_timestep_mean': sigma.float().mean().detach(),
                'action_timestep_mean': sigma_action.float().mean().detach(),
            }

    def core_forward_static(
        self,
        video_tokens,
        action_tokens,
        und_tokens,
        video_adaln_params,
        action_adaln_params,
        video_head_time_emb,
        action_head_time_emb,
        processed_t5_context,
    ):
        """Run the trimodal MoT layer stack and output heads; returns (video_pred, action_pred)."""
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx
                )
                action_adaln_modulation = self.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx
                )
        
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                    video_tokens,
                    action_tokens,
                    video_adaln_modulation,
                    action_adaln_modulation,
                    layer_idx,
                    self.action_expert.blocks[layer_idx],
                    und_tokens,
                    self.und_expert.blocks[layer_idx],
                )
        
                video_tokens = self.video_module.process_cross_attention(
                    video_tokens,
                    video_adaln_params,
                    layer_idx,
                    processed_t5_context,
                )
        
                video_tokens = self.video_module.process_ffn(
                    video_tokens,
                    video_adaln_modulation,
                    layer_idx,
                )
                action_tokens = self.action_module.process_ffn(
                    action_tokens,
                    action_adaln_modulation,
                    layer_idx,
                )
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
        
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
        
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers

            action_pred = action_pred_full[:, self._action_slice_start:up_len, :]

        return video_pred, action_pred

    def forward(self, **kwargs):
        """Gated entry point for DeepCompile (self._deepcompile_enabled).

        DeepSpeed DeepCompile only inserts its ZeRO-aware fx passes into the graph
        produced by engine.module.__call__ -> forward, and REQUIRES every
        nn.Parameter that appears in that graph to be ZeRO-optimizer-managed
        (trainable). Compiling the whole training_step therefore fails: it pulls in
        the frozen VLM/VAE/T5 backbone params (no param_id -> assertion in
        deepspeed/compile/backend.py). So under the gate this forward runs ONLY the
        all-trainable region (_deepcompile_core): every param in it (WAN patch/text/
        time embeddings + blocks + head, action expert, und expert incl. vlm_adapter)
        is trainable. The frozen preprocessing (VAE latents, frozen VLM, T5 padding,
        sampling) and the fp32 loss run eagerly OUTSIDE the engine (see
        _deepcompile_preprocess / _deepcompile_loss in the trainer).

        The default (flag off) path calls training_step() directly and never touches
        this method, so behaviour is byte-identical when DeepCompile is disabled.
        """
        if getattr(self, "_deepcompile_enabled", False):
            return self._deepcompile_core(**kwargs)
        return self.training_step(**kwargs)

    def _deepcompile_preprocess(
        self,
        clean_full_latent: torch.Tensor,
        condition_frame_latent: torch.Tensor,
        state: torch.Tensor = None,
        actions: torch.Tensor = None,
        language_embeddings=None,
        vlm_inputs=None,
    ):
        """EAGER half of the DeepCompile split (runs OUTSIDE the compiled engine).

        Does everything that must not be compiled: flow-matching sampling
        (randint on CPU + randn), the frozen VLM forward, and the data-dependent T5
        padding. Produces pure tensors (no trainable-param dependency) that feed the
        compiled _deepcompile_core, plus the targets/sigmas needed for the eager loss.
        Mirrors training_step's sampling exactly (same op order -> RNG lockstep).
        """
        B = clean_full_latent.shape[0]

        # PARITY_BATCH_CKSUM (default unset -> inert): byte-checksum every batch
        # tensor fed to the model on rank0 so base<->transplant get_batch parity can
        # be verified step-by-step. Runs before any (re)seeding so it reflects the
        # raw data pipeline output only.
        if os.environ.get("PARITY_BATCH_CKSUM") is not None:
            import hashlib as _hl
            import torch.distributed as _dist
            _rank = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else 0
            if _rank == 0:
                def _sha(t):
                    if t is None:
                        return "none"
                    if isinstance(t, torch.Tensor):
                        b = t.detach().float().cpu().contiguous().numpy().tobytes()
                        return f"{tuple(t.shape)}:{_hl.sha1(b).hexdigest()[:12]}"
                    if isinstance(t, (list, tuple)):
                        return "[" + ",".join(_sha(x) for x in t) + "]"
                    if isinstance(t, dict):
                        return "{" + ",".join(f"{k}={_sha(v)}" for k, v in sorted(t.items())) + "}"
                    return type(t).__name__
                _ck = getattr(self, "_parity_ck_step", 0)
                print(
                    f"[BATCH_CKSUM] step={_ck} clean={_sha(clean_full_latent)} "
                    f"cond={_sha(condition_frame_latent)} state={_sha(state)} "
                    f"actions={_sha(actions)} lang={_sha(language_embeddings)} "
                    f"vlm={_sha(vlm_inputs)}",
                    flush=True,
                )
            self._parity_ck_step = getattr(self, "_parity_ck_step", 0) + 1

        # PARITY_FM_SEED (default unset -> inert): reseed global RNG per training
        # step so the 4 flow-matching draws below (video randint/randn_like, action
        # randint/randn_like) are identical across base<->transplant. Keyed by an
        # internal per-call counter (both start at 0, incremented once per step).
        _pf = os.environ.get("PARITY_FM_SEED")
        if _pf is not None:
            _s = getattr(self, "_parity_fm_step", 0)
            torch.manual_seed(int(_pf) + _s)
            self._parity_fm_step = _s + 1

        # Video flow-matching noise mixture
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Action flow-matching noise mixture
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions

        # PARITY_FM_SEED probe: print the actual FM draws per step so base<->
        # transplant can be diffed. If step2 diverges while these match, the
        # cause is grad/optimizer (not FM noise); if these differ, the per-call
        # _parity_fm_step counter drifted at runtime.
        if _pf is not None:
            import torch.distributed as _dist2
            _rk = _dist2.get_rank() if _dist2.is_available() and _dist2.is_initialized() else 0
            if _rk == 0:
                import hashlib as _hl2
                def _h(t):
                    return _hl2.sha1(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
                print(
                    f"[FM_PROBE] fm_step={_s} "
                    f"vid_tid={timestep_id.tolist()} act_tid={timestep_id_action.tolist()} "
                    f"vnoise={_h(video_noise)} anoise={_h(action_noise)}",
                    flush=True,
                )

        # Frozen VLM last-layer features (adapter applied inside the compiled core).
        vlm_last_features = self.und_module.extract_vlm_last_features(vlm_inputs)
        # Data-dependent T5 padding (text_embedding applied inside the compiled core)
        t5_context_raw = self.video_module.pad_t5_embeddings(language_embeddings)

        core_inputs = {
            'noisy_video_latent': noisy_video_latent,
            'noisy_actions': noisy_actions,
            'state': state,
            'vlm_last_features': vlm_last_features,
            't5_context_raw': t5_context_raw,
            'video_t_embed': video_t_embed,
            'action_t_embed': action_t_embed,
        }
        loss_ctx = {
            'video_target': video_target,
            'action_target': action_target,
            'sigma': sigma,
            'sigma_action': sigma_action,
        }
        return core_inputs, loss_ctx

    def _deepcompile_core(
        self,
        noisy_video_latent: torch.Tensor,
        noisy_actions: torch.Tensor,
        state: torch.Tensor,
        vlm_last_features: torch.Tensor,
        t5_context_raw: torch.Tensor,
        video_t_embed: torch.Tensor,
        action_t_embed: torch.Tensor,
    ):
        """COMPILED forward for DeepCompile: the ALL-TRAINABLE region only.

        Every nn.Parameter touched here is ZeRO-managed (trainable), so DeepCompile's
        add_z1_reduce pass tags them and schedules their gradient reduce inside the
        compiled backward graph. Includes the trainable preprocessing that the frozen
        _deepcompile_preprocess deliberately left out: WAN patch_embedding,
        text_embedding, video/action time embeddings, action input_encoder + registers,
        und_expert.vlm_adapter -- then core_forward_static (30 MoT layers + heads).
        """
        B = noisy_video_latent.shape[0]

        # Video tokens (WAN patch_embedding, trainable)
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # Understanding tokens: trainable vlm_adapter on the frozen VLM features
        und_tokens = self.und_expert.vlm_adapter(vlm_last_features)

        # Action tokens (trainable input_encoder + registers)
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        # Time embeddings (trainable time_embedding + time_projection)
        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])

        # T5 context (trainable text_embedding on the pre-padded raw embeddings)
        processed_t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)

        # MoT core (30 layers + output heads)
        return self.core_forward_static(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            und_tokens=und_tokens,
            video_adaln_params=video_adaln_params,
            action_adaln_params=action_adaln_params,
            video_head_time_emb=video_head_time_emb,
            action_head_time_emb=action_head_time_emb,
            processed_t5_context=processed_t5_context,
        )

    def _deepcompile_loss(self, video_pred, action_pred, loss_ctx):
        """EAGER fp32 loss for the DeepCompile split (runs OUTSIDE the engine).

        Mirrors training_step's loss. fp32 to match origin (the loss is outside
        core_forward_static's autocast, so bf16 mse would quantize the action branch
        to the bf16 grid). Backward through this eager loss reaches the compiled
        outputs (video_pred/action_pred) -> compiled backward -> DeepCompile reduces
        fire for every trainable param.
        """
        video_target = loss_ctx['video_target']
        action_target = loss_ctx['action_target']
        video_pred_masked = video_pred.clone()
        video_pred_masked[:, :, 0:1] = 0
        video_loss = torch.nn.functional.mse_loss(video_pred_masked.float(), video_target.float(), reduction='mean')
        action_loss = torch.nn.functional.mse_loss(action_pred.float(), action_target.float(), reduction='mean')
        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )
        return {
            'total_loss': total_loss,
            'video_loss': video_loss,
            'action_loss': action_loss,
            'video_timestep_mean': loss_ctx['sigma'].float().mean().detach(),
            'action_timestep_mean': loss_ctx['sigma_action'].float().mean().detach(),
        }

    def enable_deepcompile_static_path(self):
        """Route attention through the fixed-shape dense-flash + fp64-rope path.

        Configures the WAN blocks so the whole-module compile performed by
        DeepCompile does not hit the varlen flash_attn graph break (data-dependent
        cu_seqlens/max_seqlen -> unbacked symint -> recompile blow-up -> silent
        eager fallback). grid_sizes is fixed from config, so this is set eagerly at
        build time and the very first compiled forward already traces the static
        path.
        """
        gs = self.grid_sizes  # [B, 3], all rows identical
        fhw = (int(gs[0, 0].item()), int(gs[0, 1].item()), int(gs[0, 2].item()))
        self.video_model.wan_model.graph_fhw = fhw
        for block in self.video_model.wan_model.blocks:
            block.self_attn.graph_fhw = fhw
            block.self_attn.use_sdpa = True
            block.cross_attn.use_sdpa = True
        logger.info(f"DeepCompile: enabled static dense-flash attn path (fhw={fhw}).")

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(
                first_frame_norm.to(self.dtype)
            )   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (B, C_latent, num_total_latent_frames, H_latent, W_latent),
            device=self.device,
            dtype=self.dtype,
        )
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: from noise (t=1) to clean (t=0)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(num_inference_steps):
            # Timesteps
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)

            # Tokens with Registers
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            state_tokens = state.unsqueeze(1).to(self.dtype)
            # Expand registers for batch
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Note: Understanding tokens already extracted before the loop, will be updated in joint attention
            und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, num_queries * num_layers, und_dim]

            
            # Trimodal MoT forward - joint denoising for WAN, Action, Understanding
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                    video_t_scaled, video_tokens.shape[1]
                )
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                    action_t_scaled, action_tokens.shape[1]
                )

                # Process through all layers - trimodal denoising of WAN, Action, Understanding
                for layer_idx in range(self.config.num_layers):
                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(
                        video_adaln_params, layer_idx
                    )
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(
                        action_adaln_params, layer_idx
                    )
                    
                    # Trimodal joint attention: WAN + Action + Understanding
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )

                    # WAN cross-attention with T5 embeddings 
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # FFNs: WAN, Action, Understanding
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Heads (velocities)
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                # Use decoder with all tokens (including registers)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                # Extract middle action chunk (skip first state token and last register tokens)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]

                # Euler integration
                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent

        # 4. Decode outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
        
        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        return predicted_frames, predicted_actions

    # Alternative inference (DPM++ solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        solver: Optional[str] = None,
        shift: Optional[float] = None,
        seed: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction with dpm++ solver.
        
        Args:
            first_frame: Initial frame [1, C, H, W] - batch size must be 1 for inference
            state: Initial robot state [1, state_dim]
            num_inference_steps: Number of denoising steps (default: 50)
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: VLM inputs for understanding expert
            solver: Solver type ("dpm++"), defaults to config.inference_solver
            shift: Noise schedule shift, defaults to config.inference_shift
            seed: Random seed for reproducible generation (-1 for random)
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        # Move inputs to device
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)
        
        # Use config defaults if not specified
        if solver is None:
            solver = self.config.inference_solver
        if shift is None:
            shift = self.config.inference_shift

        # Set random seed if specified
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 1. Encode condition frame and initialize latents
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [1, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [1, 48, 1, H', W']

        # Initialize video latent with noise - squeeze batch dimension for WAN format
        _, C_latent, _, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (C_latent, num_total_latent_frames, H_latent, W_latent), 
            device=self.device, 
            dtype=torch.float32,  # Use float32 for sampling numerical stability
            generator=generator
        )
        # Set first frame as condition (teacher forcing)
        video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()
        
        # Initialize action latent with noise
        action_latent = torch.randn(
            (1, self.config.action_chunk_size, self.config.action_dim), 
            device=self.device, 
            dtype=torch.float32,
            generator=generator
        )

        # 2. Prepare understanding features and T5 context (compute once, reuse for all steps)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Setup flow-matching schedulers (separate for video and action due to different tensor shapes)
        if solver == "dpm++":
            # Video scheduler
            video_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,  # Base shift is 1.0
                use_dynamic_shifting=False
            )
            # Action scheduler (independent instance)
            action_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False
            )
            # Get custom sigmas with shift parameter
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                video_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
            # Set same timesteps for action scheduler
            _, _ = retrieve_timesteps(
                action_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Solver '{solver}' not implemented. Currently only 'dpm++' is supported.")

        # 4. Denoising loop with flow-matching solver
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                # Prepare model inputs (add batch dimension back)
                video_latent_input = video_latent.unsqueeze(0).to(self.dtype)  # [1, 48, T, H, W]
                action_latent_input = action_latent.to(self.dtype)  # [1, chunk_size, action_dim]
                
                # Prepare tokens
                video_tokens = self.video_module.prepare_input(video_latent_input)
                state_tokens = state.unsqueeze(1)
                registers = self.action_expert.registers.expand(1, -1, -1)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent_input, registers)
                und_tokens = self.und_module.extract_und_features(vlm_inputs)
                
                # Model forward pass
                with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                    # Time embeddings (t is in [0, 1000] from scheduler)
                    video_t_scaled = t.expand(1).to(self.dtype)
                    action_t_scaled = t.expand(1).to(self.dtype)
                    video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                        video_t_scaled, video_tokens.shape[1]
                    )
                    action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                        action_t_scaled, action_tokens.shape[1]
                    )

                    # Process through all layers - trimodal joint denoising
                    for layer_idx in range(self.config.num_layers):
                        video_adaln_modulation = self.video_module.compute_adaln_modulation(
                        video_adaln_params, layer_idx)
                        action_adaln_modulation = self.action_module.compute_adaln_modulation(
                        action_adaln_params, layer_idx)

                        # Trimodal joint attention
                        video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                            self.action_expert.blocks[layer_idx],
                            und_tokens, self.und_expert.blocks[layer_idx]
                        )

                        # WAN cross-attention with T5
                        video_tokens = self.video_module.process_cross_attention(
                            video_tokens, video_adaln_params, layer_idx, processed_t5_context
                        )

                        # FFNs for each modality
                        video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                        action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                        und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                    # Prediction heads (predict velocity for flow-matching)
                    video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)  # [1, 48, T, H, W]
                    action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                    action_pred = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]  # [1, chunk_size, action_dim]

                # Update latents using separate schedulers (video and action have different tensor shapes)
                # Video: squeeze batch dim, call video_scheduler, squeeze back
                video_latent = video_scheduler.step(
                    video_pred.squeeze(0).unsqueeze(0),  # Add dummy batch dim for scheduler
                    t,
                    video_latent.unsqueeze(0),  # Add dummy batch dim for scheduler
                    return_dict=False,
                    generator=generator
                )[0].squeeze(0)  # Remove dummy batch dim
                
                # Action: directly use 3D tensor [1, chunk_size, action_dim] with action_scheduler
                # DPM-Solver doesn't require specific dimensions, just consistency
                action_latent = action_scheduler.step(
                    action_pred,  # [1, chunk_size, action_dim]
                    t,
                    action_latent,  # [1, chunk_size, action_dim]
                    return_dict=False,
                    generator=generator
                )[0]
                
                # Teacher forcing: keep first frame as condition
                video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()

        # 5. Decode final outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent.unsqueeze(0).to(self.dtype))  # Add batch dim back
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [1, chunk_size, action_dim]

        return predicted_frames, predicted_actions
    '''

    # Alternative inference (UniPC solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        if self.config.training_mode != 'pretrain':
            state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(
                                        first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, 
                            num_total_latent_frames, H_latent, W_latent), 
                            device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: use FlowUniPCMultistepScheduler for video and action latents
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        # Use a separate scheduler instance for the action branch to avoid shared internal state
        action_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        action_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        timesteps = scheduler.timesteps  # int64 on device

        for t in timesteps:
            # Tokens (with optional registers)
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
                registers = self.action_expert.registers.expand(B, -1, -1)
            else:
                registers = None
            if self.config.training_mode == 'pretrain':
                action_tokens = self.action_expert.input_encoder(None, action_latent, registers)
            else:
                state_tokens = state.unsqueeze(1).to(self.dtype)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Re-extract understanding features per step (keeps alignment with current pipeline)
            und_tokens = self.und_module.extract_und_features(vlm_inputs)
            
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings: use the current discrete t (0..num_train_timesteps)
                t_scalar = t.to(self.dtype).repeat(B)
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                                        t_scalar, 
                                        video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                                        t_scalar, 
                                        action_tokens.shape[1])

                # Layer stack for joint denoising
                for layer_idx in range(self.config.num_layers):
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx)
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx,
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )
                    # WAN cross
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Predict velocities (video and action) and take scheduler steps
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_velocity_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                up_len = action_velocity_full.shape[1] - self.action_expert.config.num_registers
                if self.config.training_mode == 'pretrain':
                    action_velocity = action_velocity_full[:, :up_len, :]
                else:
                    action_velocity = action_velocity_full[:, 1:up_len, :]

                # Scheduler steps
                video_latent = scheduler.step(model_output=video_velocity, timestep=t, 
                                        sample=video_latent, 
                                        return_dict=False)[0]
                # Teacher Forcing on the first frame (video)
                video_latent[:, :, 0:1] = condition_frame_latent
                action_latent = action_scheduler.step(model_output=action_velocity, 
                                        timestep=t, 
                                        sample=action_latent, return_dict=False)[0]

        # 4. Decode outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        return predicted_frames, predicted_actions
    '''


def test_motus():
    """Test the complete model."""
    print("Testing Motus...")

    config = MotusConfig()

    try:
        model = Motus(config)
        print("Model created successfully")

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"Model creation failed: {e}")
        print("This is expected without actual pretrained weights")


if __name__ == "__main__":
    test_motus()
