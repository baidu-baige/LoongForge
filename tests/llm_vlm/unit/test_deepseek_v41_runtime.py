# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from loongforge.models.foundation.deepseek_v41.deepseek_v41_engram import (
    DeepseekV41ParallelEmbedding,
)
from loongforge.models.foundation.deepseek_v41.deepseek_v41_model import (
    IMAGE,
    IMAGE_END,
    IMAGE_START,
    DeepseekV41Model,
)


def test_meta_materialization_preserves_parameter_attributes():
    from loongforge.train import training_utils

    class ParameterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4, device="meta"))
            self.weight.allreduce = False
            self.weight.tensor_model_parallel = True
            self.weight.partition_dim = 0
            self.weight.partition_stride = 1

        def parameter_method(self):
            return self.weight

    module = ParameterModule()
    original_parameter = module.weight
    materialized = training_utils._to_empty_preserving_parameter_attributes(module, device=torch.device("cpu"))

    assert materialized.weight.device.type == "cpu"
    assert materialized.weight is not original_parameter
    assert getattr(materialized.weight, "allreduce") is False
    assert getattr(materialized.weight, "tensor_model_parallel") is True
    assert getattr(materialized.weight, "partition_dim") == 0
    assert getattr(materialized.weight, "partition_stride") == 1
    assert materialized.parameter_method() is materialized.weight


def test_config_defaults_keep_vision_and_shared_expert():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_config import DeepseekV41Config
    from loongforge.utils.constants import LanguageModelFamilies, VisionLanguageModelFamilies

    config = DeepseekV41Config(num_layers=40, hidden_size=5120, ffn_hidden_size=2304, num_attention_heads=64)
    assert config.model_type in VisionLanguageModelFamilies.names()
    assert config.model_type not in LanguageModelFamilies.names()
    assert config.vision_config["num_hidden_layers"] == 32
    assert config.moe_shared_expert_intermediate_size == 2304
    assert config.moe_router_topk == config.num_experts_per_tok == 6


def test_config_accepts_virtual_pipeline_parallelism():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_config import DeepseekV41Config

    config = DeepseekV41Config(
        num_layers=8,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        csa_compress_ratios=[0, 0, 2, 2, 2, 2, 2, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
        engram_layer_ids=[1, 6],
        engram_num_embeddings=[1024, 2048],
    )
    assert config.pipeline_model_parallel_size == 2
    assert config.virtual_pipeline_model_parallel_size == 2
    assert config.variable_seq_lengths is True


def test_moe_spec_reuses_mcore_and_keeps_config_unchanged(monkeypatch):
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.transformer.moe.experts import SequentialMLP

    from loongforge.models.common.clamped_swiglu import ClampedSwiGLUSharedExpertMLP
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_layer_spec import get_deepseek_v41_moe_spec
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_layers import (
        DeepseekV41MoELayer,
        DeepseekV41SequentialMLP,
        DeepseekV41SwiGLU,
    )

    config = SimpleNamespace(
        num_moe_experts=8,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
        use_te_activation_func=False,
        activation_func_clamp_value=10.0,
    )
    spec = get_deepseek_v41_moe_spec(config, LocalSpecProvider())
    assert spec.module is DeepseekV41MoELayer
    assert spec.submodules.shared_experts.module is ClampedSwiGLUSharedExpertMLP
    assert spec.submodules.experts.module is DeepseekV41SequentialMLP
    captured = {}

    def initialize(self, num_local_experts, config, submodules, pg_collection):
        captured.update(config=config, activation=submodules.activation_func)

    monkeypatch.setattr(SequentialMLP, "__init__", initialize)
    DeepseekV41SequentialMLP(1, config, spec.submodules.experts.submodules)
    assert config.use_te_activation_func is False
    assert captured["config"] is not config
    assert captured["config"].use_te_activation_func is True
    assert captured["activation"] is DeepseekV41SwiGLU


@pytest.fixture(scope="module")
def single_rank_mcore(tmp_path_factory):
    from megatron.core import parallel_state
    from torch import distributed as dist

    rendezvous = tmp_path_factory.mktemp("shared_expert") / "rendezvous"
    dist.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel()
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ("deepseek_v4", "deepseek_v41"))
@pytest.mark.parametrize("fusion", (False, True))
def test_shared_experts_round_once(single_rank_mcore, family, fusion):
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch.nn.functional import silu

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        num_moe_experts=2,
        moe_ffn_hidden_size=4,
        moe_shared_expert_intermediate_size=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=silu,
        activation_func_clamp_value=10.0,
        bias_activation_fusion=fusion,
    )
    backend = LocalSpecProvider()
    if family == "deepseek_v4":
        for module_name in ("deep_gemm", "flashinfer", "lightning_indexer_bwd"):
            pytest.importorskip(module_name)
        from loongforge.models.foundation.deepseek_v4.deepseek_v4_layer_spec import (
            _get_mlp_module_spec_for_backend,
        )

        spec = _get_mlp_module_spec_for_backend(backend, num_experts=config.num_moe_experts)
    else:
        from loongforge.models.foundation.deepseek_v41.deepseek_v41_layer_spec import get_deepseek_v41_moe_spec

        spec = get_deepseek_v41_moe_spec(config, backend)
    shared = spec.submodules.shared_experts
    expert = build_module(shared, config=config)
    with torch.no_grad():
        expert.linear_fc1.weight.zero_()
        expert.linear_fc1.weight[:, 0] = torch.tensor([12, -12, 0.5, -0.5, 12, -12, 3, -3])
        expert.linear_fc2.weight.copy_(torch.eye(4))

    values = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bfloat16, requires_grad=True)
    gate, up = expert.linear_fc1(values)[0].chunk(2, dim=-1)
    expected = (silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)).to(values.dtype)
    assert isinstance(expert, SharedExpertMLP)
    baseline = SharedExpertMLP(config, shared.submodules, **shared.params)
    baseline.load_state_dict(expert.state_dict())
    assert torch.equal(baseline(values), expected) is fusion
    assert not config.use_te_activation_func
    actual = expert(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), values)[0]
    expected_grad = torch.autograd.grad(expected.sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_clamped_activation_reuses_mcore_forward_and_backward(dtype):
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_layers import DeepseekV41SwiGLU

    activation = DeepseekV41SwiGLU(SimpleNamespace(activation_func_clamp_value=10.0))
    values = torch.tensor([[12.0, -12.0, 0.5, -0.5, 12.0, -12.0, 2.0, -2.0]], dtype=dtype, requires_grad=True)
    expected_input = values.detach().clone().requires_grad_()
    gate, up = expected_input.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.float().clamp(max=10.0)) * up.float().clamp(-10.0, 10.0)).to(dtype)
    actual = activation(values)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(values.grad, expected_input.grad)


def test_hyper_connection_keeps_fp32_projection_under_autocast():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_layers import DeepseekV41HyperConnection

    config = SimpleNamespace(
        hc_mult=4,
        hidden_size=16,
        hc_eps=1e-6,
        layernorm_epsilon=1e-20,
        hc_sinkhorn_iters=20,
        init_method=lambda weight: torch.nn.init.normal_(weight, std=0.02),
    )
    module = DeepseekV41HyperConnection(config)
    values = torch.randn(2, 3, 4, 16)
    expected = module(values)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module(values)
    for key in ("pre", "post", "residual"):
        assert getattr(actual, key).dtype == torch.float32
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)


def test_native_vlm_and_composed_vlm_config_dispatch(monkeypatch):
    from omegaconf import OmegaConf

    from loongforge.models import utils

    monkeypatch.setattr(utils, "convert_megatron_transformer_config_args", lambda args: {})
    monkeypatch.setattr(utils, "instantiate", lambda config, **kwargs: SimpleNamespace(**kwargs))
    native = OmegaConf.create(
        {"_target_": "DeepseekV41Config", "model_type": "deepseek_v41", "vision_config": {"num_hidden_layers": 32}}
    )
    actual = utils.build_model_config(SimpleNamespace(), native)
    assert actual.model_type == "deepseek_v41"
    assert actual.vision_config["num_hidden_layers"] == 32

    composed = OmegaConf.create(
        {"model": {"model_type": "qwen3_vl", "foundation": {"_target_": "Qwen3Config", "hidden_size": 32}}}
    )
    monkeypatch.setattr(utils, "get_args_dict", lambda: {"foundation": SimpleNamespace()})
    monkeypatch.setattr(utils, "VLMModelConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    actual = utils.build_model_config(SimpleNamespace(), composed)
    assert actual.model_type == "qwen3_vl"
    assert actual.foundation.hidden_size == 32


def test_native_image_fusion_keeps_sequence_and_batch_axes():
    embedding = torch.nn.Embedding(100, 3)
    model = SimpleNamespace(
        vision=lambda pixels, grids: torch.tensor([[10.0] * 3, [20.0] * 3]),
        aligner=lambda features, grids: features,
        embedding=SimpleNamespace(
            word_embeddings=embedding,
            reduce_scatter_embeddings=False,
            embedding_dropout=torch.nn.Identity(),
        ),
        config=SimpleNamespace(image_token_id=99, sequence_parallel=False, fp32_residual_connection=False),
        image_start=torch.ones(3),
        image_newline=torch.full((3,), 2.0),
        image_end=torch.full((3,), 3.0),
    )
    ids = torch.tensor([[1, 99, 99, 99], [5, 99, 99, 99]])
    types = torch.tensor([[-1, IMAGE_START, IMAGE, IMAGE_END]] * 2)
    fused = DeepseekV41Model._fuse_vision(
        model,
        ids,
        torch.arange(4).expand(2, 4),
        torch.zeros(2, 3, 2, 2),
        torch.tensor([[1, 1], [1, 1]]),
        types,
    )
    assert fused.shape == (4, 2, 3)
    torch.testing.assert_close(fused[0], embedding(ids)[:, 0])
    torch.testing.assert_close(fused[2], torch.tensor([[10.0] * 3, [20.0] * 3]))
    torch.testing.assert_close(fused[1], torch.ones(2, 3))
    torch.testing.assert_close(fused[3], torch.full((2, 3), 3.0))


def test_native_vision_casts_image_patches_to_projection_dtype():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_vision import (
        DeepseekV41VisionConfig,
        DeepseekV41VisionTransformer,
    )

    config = DeepseekV41VisionConfig(
        num_hidden_layers=0,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        patch_size=2,
    )
    vision = DeepseekV41VisionTransformer(config)
    patches = torch.randn(4, 3, 2, 2, dtype=torch.bfloat16)
    output = vision(patches, torch.tensor([[1, 4]]))

    assert output.dtype == torch.float32
    assert output.shape == (4, 16)


def test_engram_lookup_preserves_dtype_and_repeated_row_gradients():
    table = DeepseekV41ParallelEmbedding(11, 32, dtype=torch.float32)
    ids = torch.tensor([[0, 7, 7, 10]])
    values = table(ids)
    assert table.weight.tensor_model_parallel
    assert table.weight.partition_dim == 0
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, table.weight[ids])
    values.sum().backward()
    torch.testing.assert_close(table.weight.grad[7], torch.full((32,), 2.0))
    assert table.weight.grad[1].count_nonzero() == 0


def test_engram_hash_restores_derived_buffers_after_meta_loading(monkeypatch):
    from loongforge.models.foundation.deepseek_v41 import deepseek_v41_engram

    monkeypatch.setattr(deepseek_v41_engram, "_compressed_token_map", lambda tokenizer: (tuple(range(8)), 8))
    config = SimpleNamespace(
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_layer_ids=[1, 14],
        engram_compressed_vocab_size=8,
        engram_pad_token_id=2,
        engram_vocab_size=17,
        engram_num_embeddings=[17, 19],
    )
    expected = deepseek_v41_engram.DeepseekV41NgramHash(config, None)
    with torch.device("meta"):
        actual = deepseek_v41_engram.DeepseekV41NgramHash(config, None)
    actual.to_empty(device="cpu")
    for buffer in actual.buffers():
        buffer.zero_()
    actual.load_state_dict({}, strict=True)
    ids = torch.tensor([[1, 4, 5, 2]])
    torch.testing.assert_close(actual(ids), expected(ids), rtol=0, atol=0)
    assert actual.state_dict() == {}


def test_engram_fp8_converter_pads_tail_without_changing_decoded_rows(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from mcore_checkpoint_convert.common.common_checkpoint import ENGRAM_EMBED
    from mcore_checkpoint_convert.mcore.mcore_base import McoreBase

    converter = McoreBase.__new__(McoreBase)
    converter.dtype = torch.bfloat16
    weight = torch.arange(11 * 64).reshape(11, 64).remainder(32).to(torch.float8_e4m3fn)
    scale = torch.full((11, 2), 0.5)
    shards, _ = converter.get_chunked_weight(
        ENGRAM_EMBED,
        8,
        "weight",
        None,
        weight,
        weight_scale=scale,
        clear_source=False,
    )
    assert len(shards) == 8
    assert all(shard.shape == (2, 64) for shard in shards)
    restored, _, restored_scale = converter.get_cat_weight(
        ENGRAM_EMBED,
        8,
        shards,
        None,
        None,
        False,
        False,
    )
    torch.testing.assert_close(restored[:11], (weight.float() * 0.5).bfloat16())
    assert restored[11:].count_nonzero() == 0
    assert restored_scale is None
    selected, _ = converter.get_chunked_weight(
        ENGRAM_EMBED,
        8,
        "weight",
        None,
        weight,
        weight_scale=scale,
        clear_source=False,
        tp_ranks=[5],
    )
    assert all(value is None for rank, value in enumerate(selected) if rank != 5)
    torch.testing.assert_close(selected[5], shards[5])


def test_fp8_conversion_uses_explicit_blocks_and_ignores_scale_padding(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from mcore_checkpoint_convert.utils.utils import convert_fp8_to_bf16

    weight = torch.ones(130, 257, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.full((4, 8), 99.0)
    scales[:2, :3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = convert_fp8_to_bf16(weight, scales, dtype=torch.float32)
    assert actual[127, 127] == 1
    assert actual[128, 128] == 5
    assert actual[129, 256] == 6

    scales = torch.arange(1, 46).reshape(5, 9).float()
    actual = convert_fp8_to_bf16(weight, scales, dtype=torch.float32, block_size=32)
    assert actual[31, 31] == 1
    assert actual[32, 32] == 11
    assert actual[129, 256] == 45


def test_csa2_unavailable_topk_slots_do_not_erase_the_first_visible_key():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_attention import (
        DeepseekV41Attention,
        DeepseekV41AttentionState,
    )
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_config import DeepseekV41Config

    config = DeepseekV41Config(
        num_layers=1,
        hidden_size=32,
        ffn_hidden_size=32,
        num_attention_heads=2,
        q_lora_rank=16,
        head_dim=16,
        v_head_dim=16,
        qk_rope_head_dim=8,
        qk_pos_emb_head_dim=8,
        o_groups=1,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        csa_compress_ratios=[2],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        engram_enabled=False,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        transformer_impl="local",
        params_dtype=torch.float32,
    )
    module = DeepseekV41Attention(config, 0, None, pg_collection=SimpleNamespace(tp=None))
    inputs = torch.randn(1, 16, 32)
    positions = torch.arange(16).unsqueeze(0)
    with torch.no_grad():
        wide, _ = module(
            inputs,
            state=DeepseekV41AttentionState(),
            position_ids=positions,
            attention_mask=None,
        )
        config.index_topk = 1
        narrow, _ = module(
            inputs,
            state=DeepseekV41AttentionState(),
            position_ids=positions,
            attention_mask=None,
        )
    torch.testing.assert_close(wide[:, :3], narrow[:, :3], atol=1e-7, rtol=1e-7)


def test_pipeline_carry_keeps_padding_metadata_and_source_gradients():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_attention import DeepseekV41AttentionState
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_layers import (
        _pack_v41_carry,
        _unpack_v41_carry,
    )

    hidden = torch.randn(8, 2, 16, dtype=torch.bfloat16, requires_grad=True)
    pre = torch.randn(2, 8, 4, requires_grad=True)
    compressed = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(16).reshape(2, 8)
    positions = torch.arange(8).expand(2, -1)
    valid = positions < torch.tensor([[8], [6]])
    state = DeepseekV41AttentionState(compressed_kv=compressed, compression_ratio=2)
    packed = _pack_v41_carry(
        hidden,
        pre,
        ids,
        positions,
        None,
        None,
        None,
        state,
        valid_tokens=valid,
    )
    restored = _unpack_v41_carry(packed, hidden.shape[-1])
    torch.testing.assert_close(restored[0], hidden, atol=0, rtol=0)
    torch.testing.assert_close(restored[2], ids, atol=0, rtol=0)
    torch.testing.assert_close(restored[-2], valid, atol=0, rtol=0)
    torch.testing.assert_close(restored[1], pre, atol=2e-5, rtol=2e-5)
    (restored[0].float().sum() + restored[1].sum() + restored[-1].compressed_kv.float().sum()).backward()
    for source in (hidden, pre, compressed):
        torch.testing.assert_close(source.grad, torch.ones_like(source))


def test_carry_survives_two_virtual_chunk_boundaries():
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_attention import DeepseekV41AttentionState
    from loongforge.models.foundation.deepseek_v41.deepseek_v41_layers import (
        _pack_v41_carry,
        _unpack_v41_carry,
    )

    hidden = torch.randn(8, 2, 16, dtype=torch.bfloat16)
    pre = torch.randn(2, 8, 4)
    compressed = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    ids = torch.arange(16).reshape(2, 8)
    positions = torch.arange(8).expand(2, -1)
    valid = positions < torch.tensor([[8], [6]])
    state = DeepseekV41AttentionState(compressed_kv=compressed, compression_ratio=2)

    first = _unpack_v41_carry(
        _pack_v41_carry(hidden, pre, ids, positions, None, None, None, state, valid_tokens=valid),
        hidden.shape[-1],
    )
    second = _unpack_v41_carry(
        _pack_v41_carry(first[0], first[1], first[2], first[3], None, None, None, first[-1], valid_tokens=first[-2]),
        hidden.shape[-1],
    )
    torch.testing.assert_close(second[0], hidden, atol=0, rtol=0)
    torch.testing.assert_close(second[2], ids, atol=0, rtol=0)
    torch.testing.assert_close(second[-2], valid, atol=0, rtol=0)
    torch.testing.assert_close(second[1], pre, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(second[-1].compressed_kv, compressed, atol=0, rtol=0)


def test_native_pretrain_passes_its_loader_through_megatron(monkeypatch):
    from loongforge.train.pretrain import pretrain_deepseek_v41 as training

    monkeypatch.setattr(training, "MegatronTrainer", lambda **kwargs: SimpleNamespace(**kwargs))
    args = SimpleNamespace(
        task_encoder="deepseekv41taskencoder",
        training_phase="pretrain",
        packing_pretrain_data=False,
        packing_sft_data=False,
        packing_buffer_size=10000,
    )
    trainer = training._build_trainer(args)
    assert args.dataloader_type == "external"
    assert args.packing_buffer_size is None
    assert trainer.train_valid_test_dataset_provider is training._native_dataset_provider
    assert trainer.forward_step_func is training._native_forward_step
    args.task_encoder = "vlmtaskencoder"
    with pytest.raises(ValueError, match="require --task-encoder deepseekv41taskencoder"):
        training._build_trainer(args)
    args.task_encoder = "deepseekv41taskencoder"
    args.packing_pretrain_data = True
    with pytest.raises(ValueError, match="packing is not supported"):
        training._build_trainer(args)


def test_native_get_batch_respects_vp_stage(monkeypatch):
    from loongforge.train.pretrain import pretrain_deepseek_v41 as training

    class ReachedDataPath(Exception):
        pass

    stages = []

    def record(vp_stage):
        stages.append(vp_stage)
        return False

    monkeypatch.setattr(training, "is_first_or_last_pipeline_stage", record)
    batch = training._native_get_batch(None, vp_stage=1)
    training._native_get_batch(None)
    assert stages == [1, None]
    assert set(batch) == {
        "tokens",
        "labels",
        "position_ids",
        "loss_mask",
        "attn_mask",
        "pixel_values",
        "image_grid_hws",
        "vision_token_types",
    }
    assert all(value is None for value in batch.values())

    def reach_data_path(*args, **kwargs):
        raise ReachedDataPath

    monkeypatch.setattr(training.tensor_parallel, "broadcast_data", reach_data_path)
    monkeypatch.setattr(training, "is_first_or_last_pipeline_stage", lambda vp_stage: True)
    with pytest.raises(ReachedDataPath):
        training._native_get_batch(None, vp_stage=0)


def test_native_collator_masks_image_targets_padding_and_sample_wrap(monkeypatch):
    from loongforge.data.multimodal import deepseek_v41_task_encoder as encoding

    monkeypatch.setattr(
        encoding,
        "get_position_ids",
        lambda batch: (torch.arange(batch["tokens"].shape[1]).expand_as(batch["tokens"]), None),
    )
    collator = encoding.DeepseekV41PretrainCollator(tokenizer=SimpleNamespace(pad_token_id=2, eos_token_id=1))
    batch = collator.collate_energon(
        {
            "tokens": torch.tensor([[5, 99, 99, 99, 99, 6, 1, 2], [10, 11, 1, 2, 2, 2, 2, 2]]),
            "labels": torch.tensor(
                [[5, -100, -100, -100, -100, 6, 1, -100], [10, 11, 1, -100, -100, -100, -100, -100]]
            ),
            "attn_mask": torch.tensor([[False] * 7 + [True], [False] * 3 + [True] * 5]),
            "vision_token_types": torch.tensor([[-1, 0, 1, 2, 3, -1, -1, -1], [-1] * 8]),
        }
    )
    torch.testing.assert_close(batch["loss_mask"], torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]))
    assert batch["attn_mask"][0].sum() == 7
    assert batch["attn_mask"][1].sum() == 3


def test_native_chat_keeps_system_prompt_and_trains_only_assistant(monkeypatch):
    from loongforge.data.multimodal.deepseek_v41_task_encoder import DeepseekV41TaskEncoder

    encoder = DeepseekV41TaskEncoder.__new__(DeepseekV41TaskEncoder)
    encoder.args = SimpleNamespace(training_phase="sft")
    encoder.eos_token = "<eos>"
    segments = []
    monkeypatch.setattr(
        encoder,
        "_encode_segments",
        lambda values: (segments.extend(values), None, None, None, None),
    )
    monkeypatch.setattr(encoder, "_make_sample", lambda sample, **kwargs: kwargs)
    sample = SimpleNamespace(
        system="Answer briefly.",
        image=[],
        messages=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
    )
    encoder.encode_chat_mix(sample)
    assert segments[0] == ("Answer briefly.", [], False)
    assert [item[2] for item in segments] == [False, False, True]
    assert segments[-1][0] == "Hi<eos>"


def test_native_sample_and_batch_keep_installed_energon_metadata():
    from loongforge.data.multimodal.deepseek_v41_task_encoder import DeepseekV41TaskEncoder

    encoder = DeepseekV41TaskEncoder.__new__(DeepseekV41TaskEncoder)
    encoder.args = SimpleNamespace(seq_length=8)
    encoder.tokenizer = SimpleNamespace(pad=2)
    source = SimpleNamespace(__key__="sample", __restore_key__=(), __subflavors__={})
    tokens = torch.tensor([5, 6, 1])
    sample = encoder._make_sample(
        source,
        input_ids=tokens,
        labels=tokens.clone(),
        vision_token_types=torch.full_like(tokens, -1, dtype=torch.int32),
        image_grid_hws=None,
        pixel_values=None,
    )
    batch = encoder.batch([sample])
    torch.testing.assert_close(batch.tokens, tokens.unsqueeze(0))
    assert batch.__key__ == ["sample"]
    assert batch.__restore_key__ == [()]


def test_bf16_hf_metadata_does_not_claim_quantized_or_dspark_weights(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from mcore_checkpoint_convert.huggingface.huggingface_checkpoint import save_dsv41_hf_config

    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {"num_nextn_predict_layers": 3, "dspark_target_layer_ids": [37, 38, 39]},
    }
    config_path = source / "config.json"
    config_path.write_text(json.dumps(config))
    (source / "tokenizer_config.json").write_text("{}")
    output = tmp_path / "output"
    save_dsv41_hf_config(config_path, output, torch.bfloat16)
    actual = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in actual
    assert actual["dtype"] == "bfloat16"
    assert actual["text_config"]["num_nextn_predict_layers"] == 0
    assert actual["text_config"]["dspark_target_layer_ids"] == []
    assert (output / "tokenizer_config.json").read_text() == "{}"
    assert json.loads(config_path.read_text()) == config
