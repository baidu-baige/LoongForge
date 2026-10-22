import torch
import torch.nn as nn
from einops import rearrange

from ..modeling_giga_brain_0 import make_att_2d_masks
from .triton_conversion import convert_weights
from .triton_utils import (
    AttnGroupKeyMask,
    AttnMultiKey,
    conv2d_embed_n256_1152_res,
    layer_norm_matmul_n256_1152_2304_bias,
    layer_norm_matmul_n256_1152_4304_bias_gelu,
    layer_norm_QKV_matmul_n256_1152_3456_bias,
    matmul_k_32_1024_bias,
    matmul_k_2048_1024,
    matmul_n256_1152_1152_bias_res,
    matmul_n256_4304_1152_bias_res,
    matmul_n_2048_2304,
    matmul_n_9216_2304,
    rms_matmul_k_1024_32_bias_res,
    rms_matmul_k_1024_2048_gate,
    rms_matmul_k_1024_4096_qkv_rope,
    rms_matmul_n_2304_4096_qkv_rope,
    rms_matmul_n_2304_9216_gate,
)


class GigaBrain07Triton:
    def __init__(
        self,
        weights,
        chunk_size=50,
        max_action_dim=32,
        num_views=3,
        **kwargs,
    ):
        if int(max_action_dim) != 32:
            raise ValueError(f'GigaBrain action heads require max_action_dim=32, got {max_action_dim}')
        self.weights = weights
        self.chunk_size = chunk_size
        self.max_action_dim = 32
        self.num_views = num_views
        self.buffers = dict()

        embed_weight = self.weights.pop('text_embedding_w')
        self.embed_tokens = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], padding_idx=0)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight)
        self.lang_tokens = None
        self.lang_embeds = None

        self.infer_graph = None
        self.to(dtype=torch.bfloat16)

    @classmethod
    def from_torch_model(cls, torch_model, device, emb_id=0):
        state_dict = torch_model.state_dict()
        weights = convert_weights(
            state_dict,
            device=device,
            emb_id=emb_id,
        )
        triton_model = cls(
            weights=weights,
            chunk_size=torch_model.n_action_steps,
        )
        triton_model.to(device=device)
        return triton_model

    def to(self, device=None, dtype=None):
        for key in self.weights:
            self.weights[key] = self.weights[key].to(device=device, dtype=dtype)
        for key in self.buffers:
            if self.buffers[key].dtype != torch.bool:
                self.buffers[key] = self.buffers[key].to(dtype=dtype)
            self.buffers[key] = self.buffers[key].to(device=device)
        self.embed_tokens.to(device=device, dtype=dtype)
        return self

    def init_lang_embeds(self, lang_tokens, lang_masks):
        lang_len = lang_masks.sum().item()
        cur_lang_tokens = lang_tokens[:lang_len]
        if self.lang_tokens is not None:
            if self.lang_tokens.shape == cur_lang_tokens.shape:
                diff = torch.sum(torch.abs(self.lang_tokens - cur_lang_tokens))
                if diff == 0:
                    return False
        self.lang_tokens = cur_lang_tokens
        lang_embeds = self.embed_tokens(lang_tokens)
        lang_embeds *= lang_embeds.shape[-1] ** 0.5
        self.lang_embeds = lang_embeds
        return True

    def init_buffers(self, img_masks, lang_tokens, lang_masks):
        init_lang = self.init_lang_embeds(lang_tokens, lang_masks)

        # Calculate sequence lengths
        encoder_seq_len = self.num_views * 256 + len(self.lang_embeds)
        decoder_seq_len = self.chunk_size

        if len(self.buffers) == 0:
            self.buffers = {
                'observation_images_normalized': torch.empty(self.num_views, 224, 224, 3),
                'diffusion_noise': torch.empty(self.chunk_size, self.max_action_dim),
                # Vision Buffers
                'vision_x': torch.empty(self.num_views, 256, 1152),
                'vision_x_norm': torch.empty(self.num_views, 256, 1152),
                'vision_QKV': torch.empty(self.num_views, 256, 3 * 1152),
                'vision_hidden': torch.empty(self.num_views, 256, 4304),
                'vision_x_split_k_buf': torch.empty(self.num_views * 256 * 1152 * 4),
                # Encoder Buffers (PaliGemma2)
                'encoder_rope_weights': torch.empty(encoder_seq_len, 256),
                'encoder_x': torch.empty(encoder_seq_len, 2304),  # Hidden size 2304
                'encoder_x_norm': torch.empty(encoder_seq_len, 2304),
                # KV Cache: 26 layers, 4 KV heads * 256 dim = 1024
                'encoder_K': torch.empty(26, encoder_seq_len + decoder_seq_len, 1024),
                'encoder_V': torch.empty(26, encoder_seq_len + decoder_seq_len, 1024),
                'encoder_Q': torch.empty(encoder_seq_len * 8, 256),  # 8 Heads
                'encoder_hidden': torch.empty(encoder_seq_len, 9216),  # Interm size 9216
                'encoder_mask': torch.empty(encoder_seq_len, encoder_seq_len, dtype=torch.bool),
                # Decoder Buffers (Expert)
                'decoder_rope_weights': torch.empty(decoder_seq_len, 256),
                'decoder_x': torch.empty(decoder_seq_len, 1024),  # Hidden 1024
                'decoder_x_norm_factor': torch.empty(decoder_seq_len),
                'decoder_x_norm': torch.empty(decoder_seq_len, 1024),
                'decoder_Q': torch.empty(decoder_seq_len * 8, 256),
                'decoder_hidden': torch.empty(decoder_seq_len, 2048),  # Interm 2048
                'decoder_mask': torch.empty(decoder_seq_len, encoder_seq_len + decoder_seq_len, dtype=torch.bool),
                'decoder_attn': torch.empty(decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),
            }

            # RoPE Initialization
            real_seq_len = self.num_views * 256 + len(self.lang_tokens)
            position_ids = torch.arange(encoder_seq_len)
            position_ids[real_seq_len:] = position_ids[real_seq_len - 1]
            inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2) / 256))
            k_phase = inv_freq[None, :] * position_ids[:, None]
            k_cos = torch.cos(k_phase)
            k_sin = torch.sin(k_phase)
            self.buffers['encoder_rope_weights'].copy_(torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256))

            position_ids = torch.arange(decoder_seq_len) + real_seq_len
            inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2) / 256))
            k_phase = inv_freq[None, :] * position_ids[:, None]
            k_cos = torch.cos(k_phase)
            k_sin = torch.sin(k_phase)
            self.buffers['decoder_rope_weights'].copy_(torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256))

            device = self.embed_tokens.weight.device
            dtype = self.embed_tokens.weight.dtype
            for key in self.buffers:
                if self.buffers[key].dtype != torch.bool:
                    self.buffers[key] = self.buffers[key].to(dtype=dtype)
                self.buffers[key] = self.buffers[key].to(device=device)

        if init_lang:
            img_masks = torch.stack(img_masks, dim=0)
            img_masks = img_masks.expand(self.num_views, 256).reshape(-1)

            pad_masks = torch.cat([img_masks, lang_masks], dim=0).cpu()
            att_masks = torch.zeros(encoder_seq_len, dtype=torch.bool)
            att_2d_masks = make_att_2d_masks(pad_masks[None], att_masks[None])[0]
            self.buffers['encoder_mask'].copy_(att_2d_masks)

            pad_2d_masks = pad_masks[None, :].expand(decoder_seq_len, encoder_seq_len)
            pad_masks_dec = torch.ones(decoder_seq_len, dtype=torch.bool)
            att_masks_dec = torch.tensor([1] + ([0] * (self.chunk_size - 1)), dtype=torch.bool)

            att_2d_masks_dec = make_att_2d_masks(pad_masks_dec[None], att_masks_dec[None])[0]
            full_att_2d_masks = torch.cat([pad_2d_masks, att_2d_masks_dec], dim=1)
            self.buffers['decoder_mask'].copy_(full_att_2d_masks)

    def init_infer_graph(self):
        if self.infer_graph is not None:
            return
        self.infer_graph = torch.cuda.CUDAGraph()
        for i in range(3):
            self.forward_model()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.forward_model()
            self.infer_graph.capture_end()

    def forward_model(self):
        giga_model(self.weights, self.buffers)

    @torch.no_grad()
    def __call__(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        emb_ids: torch.Tensor,  # Ignored in Triton (baked into weights or assumed 0)
        noise: torch.Tensor | None = None,
        **kwargs,
    ):
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        assert bsize == 1 and len(images) == self.num_views
        self.init_buffers(img_masks, lang_tokens[0], lang_masks[0])
        self.init_infer_graph()

        if noise is None:
            actions_shape = (bsize, self.chunk_size, self.max_action_dim)
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=actions_shape,
                dtype=torch.float32,
                device=device,
            )

        images = torch.stack(images, dim=0)
        images = rearrange(images, 'v b c h w -> b v h w c', v=self.num_views, b=bsize)
        self.buffers['observation_images_normalized'].copy_(images[0])
        self.buffers['diffusion_noise'].copy_(noise[0])

        self.buffers['encoder_x'][self.num_views * 256 :].copy_(self.lang_embeds)

        if self.infer_graph is not None:
            self.infer_graph.replay()
        else:
            self.forward_model()

        return self.buffers['diffusion_noise'][None]


def giga_model(weights, buffers):
    encoder_seq_len = buffers['encoder_x'].shape[0]
    vision_encoder(weights, buffers)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len)


def vision_encoder(weights, buffers):
    conv2d_embed_n256_1152_res(
        buffers['observation_images_normalized'],
        weights['vision_patch_embedding_w'],
        weights['vision_patch_embedding_b'],
        weights['vision_position_embedding'],
        buffers['vision_x'],
    )
    for i in range(27):
        layer_norm_QKV_matmul_n256_1152_3456_bias(
            buffers['vision_x'],
            weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],
            weights['vision_attn_qkv_w'][i],
            weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'],
            buffers['vision_x_norm'],
        )
        attn = AttnMultiKey(buffers['vision_QKV'])
        matmul_n256_1152_1152_bias_res(
            attn,
            weights['vision_attn_o_w'][i],
            weights['vision_attn_o_b'][i],
            buffers['vision_x'],
            buffers['vision_x'],
            buffers['vision_x_split_k_buf'],
        )
        layer_norm_matmul_n256_1152_4304_bias_gelu(
            buffers['vision_x'],
            weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i],
            weights['vision_ffn_up_w'][i],
            weights['vision_ffn_up_b'][i],
            buffers['vision_hidden'],
            buffers['vision_x_norm'],
        )
        matmul_n256_4304_1152_bias_res(
            buffers['vision_hidden'],
            weights['vision_ffn_down_w'][i],
            weights['vision_ffn_down_b'][i],
            buffers['vision_x'],
            buffers['vision_x'],
            buffers['vision_x_split_k_buf'],
        )


def transformer_encoder(weights, buffers, encoder_seq_len):
    # Projector 1152 -> 2304
    layer_norm_matmul_n256_1152_2304_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm'],
    )

    # PaliGemma2 Layers (26)
    for i in range(26):
        # Attention: QKV
        # Input Norm (Pre-Attn) is baked into QKV weights in convert_weights
        rms_matmul_n_2304_4096_qkv_rope(
            buffers['encoder_x'],
            weights['encoder_attn_qkv_w'][i],
            buffers['encoder_rope_weights'],
            buffers['encoder_Q'],
            buffers['encoder_K'][i, :encoder_seq_len],
            buffers['encoder_V'][i, :encoder_seq_len],
            buffers['encoder_x_norm'],  # Scratch
        )

        if i != 25:
            attn_out = AttnGroupKeyMask(
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_mask'],
            )
            # Output Proj
            matmul_n_2048_2304(
                attn_out,
                weights['encoder_attn_o_w'][i],
                buffers['encoder_x_norm'],  # Scratch for attn_proj
            )
            attn_proj = buffers['encoder_x_norm']

            norm_w = weights['encoder_post_attn_norm_w'][i]
            attn_proj = attn_proj * torch.rsqrt(attn_proj.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + norm_w)

            buffers['encoder_x'] += attn_proj

            # FFN
            # Pre-FFN Norm baked into Gate/Up
            rms_matmul_n_2304_9216_gate(
                buffers['encoder_x'],
                weights['encoder_ffn_gate_w'][i],
                weights['encoder_ffn_up_w'][i],
                buffers['encoder_hidden'],
                buffers['encoder_x_norm'],  # Scratch
            )
            # Post-FFN Norm
            matmul_n_9216_2304(
                buffers['encoder_hidden'],
                weights['encoder_ffn_down_w'][i],
                buffers['encoder_x_norm'],
            )
            ffn_out = buffers['encoder_x_norm']

            norm_ffn_w = weights['encoder_post_ffn_norm_w'][i]
            ffn_out = ffn_out * torch.rsqrt(ffn_out.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + norm_ffn_w)

            buffers['encoder_x'] += ffn_out


def transformer_decoder(weights, buffers, encoder_seq_len):
    for step in range(10):
        matmul_k_32_1024_bias(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x'],
        )
        for i in range(26):
            rms_matmul_k_1024_4096_qkv_rope(
                buffers['decoder_x'],
                weights['decoder_attn_qkv_w'][step, i],
                buffers['decoder_rope_weights'],
                buffers['decoder_Q'],
                buffers['encoder_K'][i][encoder_seq_len:],
                buffers['encoder_V'][i][encoder_seq_len:],
                buffers['decoder_x_norm_factor'],
                weights['decoder_attn_qkv_b'][step, i],
            )

            attn_out = AttnGroupKeyMask(
                buffers['decoder_Q'],
                buffers['encoder_K'][i],
                buffers['encoder_V'][i],
                buffers['decoder_mask'],
            )

            matmul_k_2048_1024(attn_out, weights['decoder_attn_o_w'][step, i], buffers['decoder_x_norm'])

            attn_proj = buffers['decoder_x_norm']

            norm_w = weights['decoder_post_attn_norm_w'][step, i]
            attn_proj = attn_proj * torch.rsqrt(attn_proj.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_w

            buffers['decoder_x'] += attn_proj

            # FFN (AdaRMS Pre/Post FFW fused)
            rms_matmul_k_1024_2048_gate(
                buffers['decoder_x'],
                weights['decoder_ffn_gate_w'][step, i],
                weights['decoder_ffn_up_w'][step, i],
                buffers['decoder_hidden'],
                buffers['decoder_x_norm_factor'],
                weights['decoder_ffn_gate_b'][step, i],
                weights['decoder_ffn_up_b'][step, i],
            )

            matmul_k_2048_1024(buffers['decoder_hidden'], weights['decoder_ffn_down_w'][step, i], buffers['decoder_x_norm'])
            ffn_out = buffers['decoder_x_norm']

            norm_ffn_w = weights['decoder_post_ffn_norm_w'][step, i]
            attn_proj = ffn_out * torch.rsqrt(ffn_out.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_ffn_w

            buffers['decoder_x'] += attn_proj

        # Output Projection (AdaRMS Final fused)
        rms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'],
            weights['decoder_action_out_proj_w'][step],
            weights['decoder_action_out_proj_b'][step],
            buffers['diffusion_noise'],
            buffers['decoder_x_norm_factor'],
        )
