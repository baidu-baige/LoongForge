# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GLM-5.3-Flash text adapter."""

import sys
import unittest
from pathlib import Path

import torch
import torch.distributed as dist


MODEL_DIR = Path(__file__).resolve().parents[3] / "loongforge/models/foundation/glm5_next"
sys.path.insert(0, str(MODEL_DIR))

from glm5_next_config import Glm5NextConfig, Glm5NextVisionConfig
from glm5_next_attention import KPoolDSAIndexer, KPoolDSAttention
from glm5_next_layer_spec import get_glm5_next_decoder_block_spec
from glm5_next_model import Glm5NextModel
from megatron.core.transformer.multi_latent_attention import MLASelfAttention


class Glm5NextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            dist.init_process_group("gloo", init_method="file:///tmp/glm5_next_unit_pg", rank=0, world_size=1)
        from megatron.core import parallel_state

        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

    @classmethod
    def tearDownClass(cls):
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()

    @unittest.skipUnless(torch.cuda.is_available(), "native Megatron MoE requires CUDA")
    def test_four_layer_forward_and_backward(self):
        config = Glm5NextConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=0,
            v_head_dim=8,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=6,
            index_kpool=3,
            linear_num_heads=2,
            linear_head_dim=8,
            linear_conv_kernel_dim=2,
            layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            mlp_layer_types=["dense"] * 3 + ["sparse"],
            indexer_types=["full"] * 4,
            pad_token_id=0,
            image_token_id=7,
            vision_config=Glm5NextVisionConfig(
                depth=2,
                hidden_size=24,
                intermediate_size=32,
                out_hidden_size=24,
                num_heads=2,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=1,
                projection_intermediate_size=48,
            ),
        )
        model = Glm5NextModel(config).cuda()
        input_ids = torch.randint(1, config.vocab_size, (1, 6), device="cuda")
        output = model(input_ids, labels=input_ids)
        self.assertEqual(output.logits.shape, (1, 6, config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()

    def test_vision_projector_forward_and_backward(self):
        config = Glm5NextConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=0,
            v_head_dim=8,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=6,
            index_kpool=3,
            linear_num_heads=2,
            linear_head_dim=8,
            linear_conv_kernel_dim=2,
            layer_types=["linear_attention"],
            mlp_layer_types=["dense"],
            indexer_types=["full"],
            pad_token_id=0,
            image_token_id=7,
            vision_config=Glm5NextVisionConfig(
                depth=1,
                hidden_size=24,
                intermediate_size=32,
                out_hidden_size=24,
                num_heads=2,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=1,
                projection_intermediate_size=48,
            ),
        )
        model = Glm5NextModel(config)
        input_ids = torch.tensor([[9, 7, 7, 7, 7, 10]])
        pixels = torch.randn(4, 3 * 2 * 2 * 2)
        output = model(
            input_ids,
            labels=input_ids,
            pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        self.assertEqual(output.logits.shape, (1, 6, config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.model.visual.patch_embed.proj.weight.grad)

    def test_sparse_attention_uses_native_mla_and_dsa(self):
        config = Glm5NextConfig(
            hidden_size=24,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=0,
            v_head_dim=8,
            index_head_dim=8,
            index_n_heads=2,
            index_topk=6,
            index_kpool=3,
            linear_num_heads=2,
            linear_head_dim=8,
            layer_types=["deepseek_sparse_attention"],
            mlp_layer_types=["dense"],
            indexer_types=["full"],
        )
        layer_spec = get_glm5_next_decoder_block_spec(config).layer_specs[0]
        attention_spec = layer_spec.submodules.self_attention
        core_attention_spec = attention_spec.submodules.core_attention
        indexer_spec = core_attention_spec.submodules.indexer

        self.assertIs(attention_spec.module, MLASelfAttention)
        self.assertIs(core_attention_spec.module, KPoolDSAttention)
        self.assertIs(indexer_spec.module, KPoolDSAIndexer)


if __name__ == "__main__":
    unittest.main()
