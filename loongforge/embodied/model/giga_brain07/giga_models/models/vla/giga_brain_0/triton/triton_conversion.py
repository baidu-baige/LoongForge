import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modeling_giga_brain_0 import create_sinusoidal_pos_embedding


def convert_weights(state_dict, device, emb_id=0):
    # Basic dimensions for PaliGemma2 / GigaBrain07
    hidden_size = 2304
    interm_size = 9216
    expert_hidden = 1024
    expert_interm = 2048
    num_layers = 26  # PaliGemma2-3b has 26 layers

    weights = {
        'text_embedding_w': torch.zeros(257216, hidden_size),  # PaliGemma2 vocab size
        # Vision weights (SigLIP)
        'vision_patch_embedding_w': torch.zeros(14, 14, 3, 1152),
        'vision_patch_embedding_b': torch.zeros(1152),
        'vision_position_embedding': torch.zeros(256, 1152),
        'vision_attn_qkv_w': torch.zeros(27, 1152, 3 * 1152),
        'vision_attn_qkv_b': torch.zeros(27, 3 * 1152),
        'vision_attn_o_w': torch.zeros(27, 1152, 1152),
        'vision_attn_o_b': torch.zeros(27, 1152),
        'vision_ffn_up_w': torch.zeros(27, 1152, 4304),
        'vision_ffn_up_b': torch.zeros(27, 4304),
        'vision_ffn_down_w': torch.zeros(27, 4304, 1152),
        'vision_ffn_down_b': torch.zeros(27, 1152),
        'vision_pre_attn_norm_w': torch.zeros(27, 1152),
        'vision_pre_attn_norm_b': torch.zeros(27, 1152),
        'vision_pre_ffn_norm_w': torch.zeros(27, 1152),
        'vision_pre_ffn_norm_b': torch.zeros(27, 1152),
        'vision_final_norm_w': torch.zeros(1152),
        'vision_final_norm_b': torch.zeros(1152),
        'encoder_multi_modal_projector_w': torch.zeros(1152, hidden_size),
        'encoder_multi_modal_projector_b': torch.zeros(hidden_size),
        # PaliGemma2 Transformer Layers
        # QKV: 2304 -> 4096 (8*256 + 4*256 + 4*256)
        'encoder_attn_qkv_w': torch.zeros(num_layers, hidden_size, 4096),
        'encoder_attn_o_w': torch.zeros(num_layers, 2048, hidden_size),  # O: 2048 (8*256) -> 2304
        'encoder_ffn_gate_w': torch.zeros(num_layers, hidden_size, interm_size),
        'encoder_ffn_up_w': torch.zeros(num_layers, hidden_size, interm_size),
        'encoder_ffn_down_w': torch.zeros(num_layers, interm_size, hidden_size),
        'encoder_post_attn_norm_w': torch.zeros(num_layers, hidden_size),
        'encoder_post_ffn_norm_w': torch.zeros(num_layers, hidden_size),
        # Decoder (Expert) weights - Time fused (10 steps)
        'decoder_action_in_proj_w': torch.zeros(32, 1024),
        'decoder_action_in_proj_b': torch.zeros(1024),  # 1024
        # Expert Layers (10 steps * 26 layers)
        # QKV: 1024 -> 4096 (8*256 + 4*256 + 4*256)
        'decoder_attn_qkv_w': torch.zeros(10, num_layers, expert_hidden, 4096),
        'decoder_attn_qkv_b': torch.zeros(10, num_layers, 4096),
        'decoder_attn_o_w': torch.zeros(10, num_layers, 2048, expert_hidden),  # 2048 -> 1024
        'decoder_ffn_gate_w': torch.zeros(10, num_layers, expert_hidden, expert_interm),
        'decoder_ffn_gate_b': torch.zeros(10, num_layers, expert_interm),
        'decoder_ffn_up_w': torch.zeros(10, num_layers, expert_hidden, expert_interm),
        'decoder_ffn_up_b': torch.zeros(10, num_layers, expert_interm),
        'decoder_ffn_down_w': torch.zeros(10, num_layers, expert_interm, expert_hidden),
        'decoder_post_attn_norm_w': torch.zeros(10, num_layers, expert_hidden),
        'decoder_post_ffn_norm_w': torch.zeros(10, num_layers, expert_hidden),
        'decoder_action_out_proj_w': torch.zeros(10, expert_hidden, 32),
        'decoder_action_out_proj_b': torch.zeros(10, 32),
    }

    # Extract text embeddings
    weights['text_embedding_w'].copy_(state_dict.pop('paligemma_with_expert.embed_tokens.weight'))

    # Extract Vision Weights (SigLIP)
    patch_embedding_weight = state_dict.pop('paligemma_with_expert.vision_tower.embeddings.patch_embedding.weight')
    patch_embedding_bias = state_dict.pop('paligemma_with_expert.vision_tower.embeddings.patch_embedding.bias')
    position_embedding_weight = state_dict.pop('paligemma_with_expert.vision_tower.embeddings.position_embedding.weight')
    patch_embedding_weight = rearrange(patch_embedding_weight, 'o i k1 k2 -> k1 k2 i o')
    weights['vision_patch_embedding_w'].copy_(patch_embedding_weight)
    weights['vision_patch_embedding_b'].copy_(patch_embedding_bias)
    weights['vision_position_embedding'].copy_(position_embedding_weight)

    for i in range(27):
        q_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.q_proj.weight').transpose(0, 1)
        q_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.q_proj.bias')
        k_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.k_proj.weight').transpose(0, 1)
        k_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.k_proj.bias')
        v_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.v_proj.weight').transpose(0, 1)
        v_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.v_proj.bias')
        o_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.out_proj.weight').transpose(0, 1)
        o_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.self_attn.out_proj.bias')

        norm1_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.layer_norm1.weight')
        norm1_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.layer_norm1.bias')
        norm2_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.layer_norm2.weight')
        norm2_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.layer_norm2.bias')

        fc1_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.mlp.fc1.weight').transpose(0, 1)
        fc1_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.mlp.fc1.bias')
        fc2_weight = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.mlp.fc2.weight').transpose(0, 1)
        fc2_bias = state_dict.pop(f'paligemma_with_expert.vision_tower.encoder.layers.{i}.mlp.fc2.bias')

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1)

        weights['vision_attn_qkv_w'][i] = qkv_weight
        weights['vision_attn_qkv_b'][i] = qkv_bias
        weights['vision_attn_o_w'][i] = o_weight
        weights['vision_attn_o_b'][i] = o_bias
        weights['vision_pre_attn_norm_w'][i] = norm1_weight
        weights['vision_pre_attn_norm_b'][i] = norm1_bias
        weights['vision_pre_ffn_norm_w'][i] = norm2_weight
        weights['vision_pre_ffn_norm_b'][i] = norm2_bias
        weights['vision_ffn_up_w'][i] = fc1_weight
        weights['vision_ffn_up_b'][i] = fc1_bias
        weights['vision_ffn_down_w'][i] = fc2_weight
        weights['vision_ffn_down_b'][i] = fc2_bias

    norm_weight = state_dict.pop('paligemma_with_expert.vision_tower.post_layernorm.weight')
    norm_bias = state_dict.pop('paligemma_with_expert.vision_tower.post_layernorm.bias')
    weights['vision_final_norm_w'].copy_(norm_weight)
    weights['vision_final_norm_b'].copy_(norm_bias)

    # Encoder Projector
    proj_weight = state_dict.pop('paligemma_with_expert.multi_modal_projector.linear.weight').transpose(0, 1)
    proj_bias = state_dict.pop('paligemma_with_expert.multi_modal_projector.linear.bias')
    weights['encoder_multi_modal_projector_w'].copy_(proj_weight)
    weights['encoder_multi_modal_projector_b'].copy_(proj_bias)

    # PaliGemma2 Transformer Weights
    for i in range(num_layers):
        # QKV
        q_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.q_proj.0.weight').transpose(0, 1)
        k_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.k_proj.0.weight').transpose(0, 1)
        v_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.v_proj.0.weight').transpose(0, 1)
        o_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.o_proj.0.weight').transpose(0, 1)

        # Norms
        norm_input_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.input_layernorms.0.weight')
        norm_post_attn_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.post_attention_layernorms.0.weight')
        norm_pre_ffn_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.pre_feedforward_layernorms.0.weight')
        norm_post_ffn_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.post_feedforward_layernorms.0.weight')

        eff_norm_input = 1.0 + norm_input_weight
        q_weight *= eff_norm_input[:, None]
        k_weight *= eff_norm_input[:, None]
        v_weight *= eff_norm_input[:, None]

        q_weight = rearrange(q_weight, 'c (h t d) -> c (h d t)', h=8, t=2)
        k_weight = rearrange(k_weight, 'c (h t d) -> c (h d t)', h=4, t=2)

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)

        gate_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.mlps.0.gate_proj.weight').transpose(0, 1)
        up_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.mlps.0.up_proj.weight').transpose(0, 1)
        down_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.mlps.0.down_proj.weight').transpose(0, 1)

        eff_norm_pre_ffn = 1.0 + norm_pre_ffn_weight
        gate_weight *= eff_norm_pre_ffn[:, None]
        up_weight *= eff_norm_pre_ffn[:, None]

        weights['encoder_attn_qkv_w'][i] = qkv_weight
        weights['encoder_attn_o_w'][i] = o_weight
        weights['encoder_post_attn_norm_w'][i] = norm_post_attn_weight
        weights['encoder_ffn_gate_w'][i] = gate_weight
        weights['encoder_ffn_up_w'][i] = up_weight
        weights['encoder_ffn_down_w'][i] = down_weight
        weights['encoder_post_ffn_norm_w'][i] = norm_post_ffn_weight

    # Decoder (Expert) - Time Fused
    # Action/Time Inputs
    # EmbodimentSpecificLinear: weights (NumEmb, In, Out).
    # We pick `emb_id` slice.
    action_in_proj_weight_all = state_dict.pop('action_in_proj.weight')  # (NumEmb, In, Out)
    action_in_proj_bias_all = state_dict.pop('action_in_proj.bias')  # (NumEmb, Out)

    # Slice for specific embodiment
    action_in_proj_weight = action_in_proj_weight_all[emb_id]  # (32, 1024)
    action_in_proj_bias = action_in_proj_bias_all[emb_id]  # (1024)

    weights['decoder_action_in_proj_w'].copy_(action_in_proj_weight)
    weights['decoder_action_in_proj_b'].copy_(action_in_proj_bias)

    # Time MLP
    time_mlp_in_weight = state_dict.pop('time_mlp_in.weight')
    time_mlp_in_bias = state_dict.pop('time_mlp_in.bias')
    time_mlp_out_weight = state_dict.pop('time_mlp_out.weight')
    time_mlp_out_bias = state_dict.pop('time_mlp_out.bias')

    time_mlp_in = nn.Linear(time_mlp_in_weight.shape[1], time_mlp_in_weight.shape[0], device=device)
    time_mlp_out = nn.Linear(time_mlp_out_weight.shape[1], time_mlp_out_weight.shape[0], device=device)
    with torch.no_grad():
        time_mlp_in.weight.copy_(time_mlp_in_weight)
        time_mlp_in.bias.copy_(time_mlp_in_bias)
        time_mlp_out.weight.copy_(time_mlp_out_weight)
        time_mlp_out.bias.copy_(time_mlp_out_bias)

    # Precompute Time Embeddings
    n_decode_steps = 10
    time_embs = torch.zeros(n_decode_steps, 1024, device=device)
    for t in range(n_decode_steps):
        time_val = 1.0 - t / n_decode_steps
        time_tensor = torch.tensor([time_val], device=device)
        time_emb = create_sinusoidal_pos_embedding(time_tensor, 1024, min_period=4e-3, max_period=4.0, device=device)
        time_emb = time_emb.to(torch.float32)
        with torch.no_grad():
            time_emb = time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            time_embs[t] = time_emb[0]

    # Expert Layers per timestep (AdaRMS fusion)
    for i in range(num_layers):
        q_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.q_proj.1.weight').transpose(0, 1)
        k_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.k_proj.1.weight').transpose(0, 1)
        v_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.v_proj.1.weight').transpose(0, 1)
        o_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.self_attn.o_proj.1.weight').transpose(0, 1)
        norm_in_dense_w = state_dict.pop(f'paligemma_with_expert.layers.{i}.input_layernorms.1.dense.weight')
        norm_in_dense_b = state_dict.pop(f'paligemma_with_expert.layers.{i}.input_layernorms.1.dense.bias')
        norm_in_dense = nn.Linear(norm_in_dense_w.shape[1], norm_in_dense_w.shape[0], bias=True, device=device)
        with torch.no_grad():
            norm_in_dense.weight.copy_(norm_in_dense_w)
            norm_in_dense.bias.copy_(norm_in_dense_b)
        norm_post_attn_w = state_dict.pop(f'paligemma_with_expert.layers.{i}.post_attention_layernorms.1.weight')

        for t in range(n_decode_steps):
            with torch.no_grad():
                modulation = norm_in_dense(time_embs[t, None])[0].cpu()
                scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
            q_w_t = q_weight * (1 + scale[:, None])
            q_b_t = torch.matmul(q_weight.T, shift)
            k_w_t = k_weight * (1 + scale[:, None])
            k_b_t = torch.matmul(k_weight.T, shift)
            v_w_t = v_weight * (1 + scale[:, None])
            v_b_t = torch.matmul(v_weight.T, shift)

            q_w_t = rearrange(q_w_t, 'c (h t d) -> c (h d t)', h=8, t=2)
            q_b_t = rearrange(q_b_t, '(h t d) -> (h d t)', h=8, t=2)
            k_w_t = rearrange(k_w_t, 'c (h t d) -> c (h d t)', h=4, t=2)
            k_b_t = rearrange(k_b_t, '(h t d) -> (h d t)', h=4, t=2)

            qkv_w_t = torch.cat([q_w_t, k_w_t, v_w_t], dim=-1)
            qkv_b_t = torch.cat([q_b_t, k_b_t, v_b_t], dim=-1)
            weights['decoder_attn_qkv_w'][t, i] = qkv_w_t
            weights['decoder_attn_qkv_b'][t, i] = qkv_b_t

            weights['decoder_attn_o_w'][t, i] = o_weight
            weights['decoder_post_attn_norm_w'][t, i] = (1.0 + norm_post_attn_w) * gate

        gate_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.mlps.1.gate_proj.weight').transpose(0, 1)
        up_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.mlps.1.up_proj.weight').transpose(0, 1)
        down_weight = state_dict.pop(f'paligemma_with_expert.layers.{i}.mlps.1.down_proj.weight').transpose(0, 1)

        norm_pre_ffw_dense_w = state_dict.pop(f'paligemma_with_expert.layers.{i}.pre_feedforward_layernorms.1.dense.weight')
        norm_pre_ffw_dense_b = state_dict.pop(f'paligemma_with_expert.layers.{i}.pre_feedforward_layernorms.1.dense.bias')
        norm_pre_ffw_dense = nn.Linear(norm_pre_ffw_dense_w.shape[1], norm_pre_ffw_dense_w.shape[0], bias=True, device=device)

        with torch.no_grad():
            norm_pre_ffw_dense.weight.copy_(norm_pre_ffw_dense_w)
            norm_pre_ffw_dense.bias.copy_(norm_pre_ffw_dense_b)
        norm_post_ffw_w = state_dict.pop(f'paligemma_with_expert.layers.{i}.post_feedforward_layernorms.1.weight')

        for t in range(n_decode_steps):
            with torch.no_grad():
                mod_pre_ffw = norm_pre_ffw_dense(time_embs[t, None])[0].cpu()
                scale, shift, gate = torch.chunk(mod_pre_ffw, 3, dim=-1)

            gate_w_t = gate_weight * (1 + scale[:, None])
            gate_b_t = torch.matmul(gate_weight.T, shift)
            up_w_t = up_weight * (1 + scale[:, None])
            up_b_t = torch.matmul(up_weight.T, shift)

            weights['decoder_ffn_gate_w'][t, i] = gate_w_t
            weights['decoder_ffn_gate_b'][t, i] = gate_b_t
            weights['decoder_ffn_up_w'][t, i] = up_w_t
            weights['decoder_ffn_up_b'][t, i] = up_b_t

            weights['decoder_ffn_down_w'][t, i] = down_weight

            weights['decoder_post_ffn_norm_w'][t, i] = (1.0 + norm_post_ffw_w) * gate

    # Decoder Output Projection
    action_out_proj_weight_all = state_dict.pop('action_out_proj.weight')
    action_out_proj_bias_all = state_dict.pop('action_out_proj.bias')
    action_out_proj_weight = action_out_proj_weight_all[emb_id]
    action_out_proj_bias = action_out_proj_bias_all[emb_id]

    # Norm Final (Expert Stream)
    norm_final_dense_w = state_dict.pop('paligemma_with_expert.norms.1.dense.weight')
    norm_final_dense_b = state_dict.pop('paligemma_with_expert.norms.1.dense.bias')
    norm_final_dense = nn.Linear(norm_final_dense_w.shape[1], norm_final_dense_w.shape[0], bias=True, device=device)
    with torch.no_grad():
        norm_final_dense.weight.copy_(norm_final_dense_w)
        norm_final_dense.bias.copy_(norm_final_dense_b)

    for t in range(n_decode_steps):
        with torch.no_grad():
            modulation = norm_final_dense(time_embs[t, None])[0].cpu()
            scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        w_t = action_out_proj_weight * (1 + scale[:, None])
        b_t = action_out_proj_bias + torch.matmul(action_out_proj_weight.T, shift)

        w_t *= -0.1
        b_t *= -0.1

        weights['decoder_action_out_proj_w'][t] = w_t
        weights['decoder_action_out_proj_b'][t] = b_t

    return weights
