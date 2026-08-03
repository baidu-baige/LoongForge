# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import PretrainedConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import loongforge.train.training_utils as training_utils
from loongforge.data.mm_plugin import MiniCPMV46Plugin
from loongforge.data.minicpm_v_4_6_image_processor import MiniCPMV46ImageProcessor
from loongforge.models.common.peft.canonical_lora import CanonicalLoRA, DenseLinearAdapter
from loongforge.models.common.peft.lora_layers import LinearAdapter
from loongforge.models.common.base_model_config import _initialize_pretrained_config
from loongforge.models.encoder.minicpm_v_4_6_vision_models import (
    MiniCPMV46MergerConfig,
)
from loongforge.models.encoder.minicpm_v_4_6_vision_models.vision_model import (
    MiniCPMV46VisionEmbeddings,
)
from loongforge.models.foundation.minicpm_v_4_6.minicpm_v_4_6_gated_deltanet import (
    MiniCPMV46GatedDeltaNet,
    Qwen3NextRMSNormGated,
    _torch_chunk_gated_delta_rule,
    _torch_l2norm,
    _torch_module_causal_conv1d,
)
from loongforge.models.omni_models.omni_encoder_model import OmniEncoderModel
from loongforge.data.multimodal import resolve_task_encoder
from convert_checkpoint.huggingface.huggingface_checkpoint import (
    _drop_duplicate_tied_lm_head,
)
from convert_checkpoint.common.common_checkpoint import (
    MTP_LAYER_PREFIX,
    MTP_WORD_EMBEDDING,
)
from convert_checkpoint.utils.config_utils import get_yaml_config, remap_state_dict_prefixes


def test_minicpm_examples_only_contain_supported_user_workflows():
    scripts = sorted(
        path.relative_to(REPO_ROOT / "examples" / "minicpm_v_4_6").as_posix()
        for path in (REPO_ROOT / "examples" / "minicpm_v_4_6").rglob("*.sh")
    )

    assert scripts == [
        "checkpoint_convert/convert_minicpm_v_4_6_hf_to_mcore.sh",
        "checkpoint_convert/convert_minicpm_v_4_6_mcore_to_hf.sh",
        "finetuning/sft_minicpm_v_4_6.sh",
        "pretrain/pretrain_minicpm_v_4_6.sh",
    ]


def test_minicpm_vlm_checkpoint_mapping_scopes_mtp_under_foundation_model(monkeypatch):
    monkeypatch.setenv("LOONGFORGE_PATH", str(REPO_ROOT))
    config = get_yaml_config(
        str(REPO_ROOT / "configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml"),
        str(
            REPO_ROOT
            / "configs/models/minicpm_v_4_6/ckpt_convert/minicpm_v_4_6_llm_convert.yaml"
        ),
        for_vlm=True,
    )
    mcore_names = config.get("name_map")["mcore"]

    assert mcore_names[MTP_WORD_EMBEDDING] == (
        "foundation_model.embedding.word_embeddings"
    )
    assert mcore_names[MTP_LAYER_PREFIX] == "foundation_model.mtp.layers"


def test_minicpm_examples_select_model_task_encoder():
    for script in (
        REPO_ROOT / "examples/minicpm_v_4_6/pretrain/pretrain_minicpm_v_4_6.sh",
        REPO_ROOT / "examples/minicpm_v_4_6/finetuning/sft_minicpm_v_4_6.sh",
    ):
        assert "--task-encoder MiniCPMV46TaskEncoder" in script.read_text()

    encoder_cls = resolve_task_encoder("MiniCPMV46TaskEncoder")
    assert encoder_cls.__name__ == "MiniCPMV46TaskEncoder"

    finetune_script = (
        REPO_ROOT / "examples/minicpm_v_4_6/finetuning/sft_minicpm_v_4_6.sh"
    ).read_text()
    assert "--sft-dataset openai" in finetune_script


def test_minicpm_task_encoder_preserves_image_and_rejects_video():
    encoder_cls = resolve_task_encoder("MiniCPMV46TaskEncoder")
    encoder = encoder_cls.__new__(encoder_cls)
    image = object()

    assert encoder._resize_image(image) is image
    with pytest.raises(ValueError, match="video preprocessing requires"):
        encoder._resize_video(object())


def test_minicpm_task_encoder_builds_pretrain_image_contract(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"scale_resolution": 112, "slice_mode": False}),
        encoding="utf-8",
    )

    class CharacterTokenizer:
        @staticmethod
        def tokenize(text, **kwargs):
            del kwargs
            return [ord(character) for character in text]

    encoder_cls = resolve_task_encoder("MiniCPMV46TaskEncoder")
    encoder = encoder_cls.__new__(encoder_cls)
    encoder.minicpm_plugin = MiniCPMV46Plugin()
    encoder.processor = SimpleNamespace(name_or_path=str(tmp_path))
    encoder.tokenizer = CharacterTokenizer()
    image = Image.fromarray(np.zeros((80, 160, 3), dtype=np.uint8))

    input_ids, labels, images, grid, attention_mask = encoder._process(
        image,
        "<|vision_start|><|image_pad|><|vision_end|> caption",
    )

    assert input_ids.shape == labels.shape == attention_mask.shape
    assert torch.count_nonzero(labels == -100) > 0
    assert images[0].shape == (1, 3, 14, 672)
    assert grid.tolist() == [[1, 4, 12]]


def test_minicpm_plugin_builds_slice_placeholders_from_processor_metadata():
    plugin = MiniCPMV46Plugin()
    processor = SimpleNamespace(
        default_use_image_id=True,
        image_processor=SimpleNamespace(slice_mode=True, downsample_mode="4x"),
    )
    mm_inputs = {
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]]),
        "grids": [[1, 1]],
        "num_patches_per_image": [2],
    }

    placeholders = plugin._build_image_placeholders(
        mm_inputs, processor, use_image_id=True
    )

    assert placeholders == [
        "<image_id>0</image_id><image>"
        "<|image_pad|><|image_pad|><|image_pad|><|image_pad|></image>"
        "<slice><|image_pad|></slice>"
    ]


def test_minicpm_local_image_processor_packs_real_image():
    image = Image.fromarray(np.full((80, 160, 3), [255, 0, 0], dtype=np.uint8))
    processor = MiniCPMV46ImageProcessor(
        scale_resolution=112,
        slice_mode=False,
    )

    result = processor(image, return_tensors="pt")

    assert result["target_sizes"].tolist() == [[4, 12]]
    assert result["pixel_values"].shape == (1, 3, 14, 672)
    assert result["grids"] == [[0, 0]]
    assert result["num_patches_per_image"] == [1]
    assert torch.equal(
        result["pixel_values"][:, :, 0, 0],
        torch.tensor([[1.0, -1.0, -1.0]]),
    )


def test_minicpm_packed_pixels_use_reference_conv2d_path():
    config = SimpleNamespace(
        hidden_size=4,
        image_size=28,
        patch_size=14,
        in_channels=3,
    )
    embeddings = MiniCPMV46VisionEmbeddings(config)
    pixels = torch.randn(1, 3, 14, 28)
    target_sizes = torch.tensor([[1, 2]])
    calls = []
    original_forward = embeddings.patch_embedding.forward

    def capturing_forward(value):
        calls.append(value)
        return original_forward(value)

    embeddings.patch_embedding.forward = capturing_forward
    output = embeddings(pixels, target_sizes)

    assert calls == [pixels]
    assert output.shape == (1, 2, 4)


def test_minicpm_plugin_loads_local_processor_for_real_image(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "max_slice_nums": 9,
                "scale_resolution": 112,
                "patch_size": 14,
                "slice_mode": False,
                "use_image_id": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    processor = SimpleNamespace(name_or_path=str(tmp_path), downsample_mode="16x")
    plugin = MiniCPMV46Plugin()
    image = Image.fromarray(np.zeros((80, 160, 3), dtype=np.uint8))

    messages, mm_inputs = plugin.process_messages(
        [{"role": "user", "content": "<image>describe"}],
        [image],
        [],
        processor,
    )

    assert isinstance(processor.image_processor, MiniCPMV46ImageProcessor)
    assert mm_inputs["pixel_values"].shape == (1, 3, 14, 672)
    assert mm_inputs["image_grid_thw"].tolist() == [[1, 4, 12]]
    assert messages[0]["content"].startswith("<image_id>0</image_id><image>")


def test_minicpm_local_processor_rejects_video_without_supported_backend(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    processor = SimpleNamespace(name_or_path=str(tmp_path))
    plugin = MiniCPMV46Plugin()

    with pytest.raises(ValueError, match="video preprocessing requires"):
        plugin.get_mm_inputs([], [object()], [], [1], [1], processor)


def test_model_input_dump_handles_context_parallel_label_length(monkeypatch, capsys):
    import loongforge.utils.global_vars as global_vars

    tokenizer = SimpleNamespace(detokenize=lambda ids, **kwargs: " ".join(map(str, ids)))
    monkeypatch.setattr(global_vars, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(training_utils, "_PRINTED_MODEL_INPUT_EXAMPLE", False)
    monkeypatch.setattr(training_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(training_utils.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(training_utils.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(training_utils.mpu, "get_pipeline_model_parallel_rank", lambda: 0)

    training_utils.dump_model_input_example_once(
        tokens=torch.arange(8).unsqueeze(0),
        labels=torch.tensor([[1, 2, -100, -100]]),
        attn_mask=torch.zeros(1, 4, dtype=torch.bool),
    )

    output = capsys.readouterr().out
    assert "tokens.shape=(1, 8) labels.shape=(1, 4)" in output
    assert "===== end model-input example =====" in output


def test_projector_output_flattening_is_shared_by_image_and_video_paths():
    embeddings = [torch.ones(1, 2, 3), torch.zeros(2, 3)]

    flattened = OmniEncoderModel._flatten_projected_embeddings(embeddings)

    assert flattened.shape == (4, 3)
    assert torch.equal(flattened[0], torch.ones(3))
    assert torch.equal(flattened[-1], torch.zeros(3))


def test_minicpm_merger_config_accepts_decoder_tensor_parallelism():
    config = MiniCPMV46MergerConfig(
        tensor_model_parallel_size=2,
        merge_kernel_size=[2, 2],
    )

    assert config.tensor_model_parallel_size == 2
    assert config.num_attention_heads == 1
    assert config.return_dict is True


def test_minicpm_gated_delta_rule_normalizes_before_fp32_recurrence():
    class ConstantProjection(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.register_buffer("output", output)

        def forward(self, hidden_states):
            return self.output.expand(*hidden_states.shape[:2], -1)

    class GatedNorm(torch.nn.Module):
        def forward(self, hidden_states, gate):
            del gate
            return hidden_states

    module = MiniCPMV46GatedDeltaNet.__new__(MiniCPMV46GatedDeltaNet)
    torch.nn.Module.__init__(module)
    module.sequence_parallel = False
    module.tp_size = 1
    module.cp_size = 1
    module.projection_split_mode = "merged"
    module.num_key_heads = 1
    module.num_value_heads = 1
    module.key_head_dim = 2
    module.value_head_dim = 2
    module.qk_dim = 2
    module.v_dim = 2
    module.activation = "silu"
    module.in_proj_qkvz = ConstantProjection(
        torch.arange(1, 9, dtype=torch.bfloat16)
    )
    module.in_proj_ba = ConstantProjection(
        torch.tensor([0.25, 0.5], dtype=torch.bfloat16)
    )
    module.conv1d = SimpleNamespace(
        weight=torch.ones(6, 1, 1, dtype=torch.bfloat16),
        bias=None,
    )
    module.causal_conv1d = lambda **kwargs: (kwargs["x"],)
    module.A_log = torch.nn.Parameter(torch.zeros(1))
    module.dt_bias = torch.nn.Parameter(torch.zeros(1))
    module.use_qk_l2norm = True
    captured_dtypes = {}

    def capture_recurrence(query, key, value, *, g, beta, **kwargs):
        del kwargs
        captured_dtypes.update(
            query=query.dtype,
            key=key.dtype,
            value=value.dtype,
            beta=beta.dtype,
            g=g.dtype,
        )
        return value, None

    module.chunk_gated_delta_rule = capture_recurrence
    module.out_norm = GatedNorm()
    module.use_torch_linear = False
    module.out_proj = torch.nn.Identity()

    output, bias = module(
        torch.ones(1, 1, 2, dtype=torch.bfloat16),
        attention_mask=None,
    )

    assert captured_dtypes == {
        "query": torch.bfloat16,
        "key": torch.bfloat16,
        "value": torch.bfloat16,
        "beta": torch.bfloat16,
        "g": torch.float32,
    }
    assert output.dtype == torch.bfloat16
    assert bias is None


def test_minicpm_gated_delta_l2norm_uses_fp32_reduction():
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.bfloat16)

    normalized = _torch_l2norm(values)
    expected = values * torch.rsqrt(
        (values * values).sum(dim=-1, keepdim=True, dtype=torch.float32)
        + 1e-6
    )

    assert normalized.dtype == torch.float32
    assert torch.equal(normalized, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_minicpm_torch_chunk_matches_bf16_autocast_execution():
    torch.manual_seed(7)
    shape = (1, 5, 2, 4)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(shape[:-1], device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(
        torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    )

    actual = _torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
    )[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule as reference_delta_rule,
        )

        expected = reference_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
        )[0]

    assert torch.equal(actual, expected)


def test_minicpm_torch_module_conv_preserves_reference_transpose_stride():
    inputs = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
    weight = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10

    output = _torch_module_causal_conv1d(
        inputs,
        weight,
        activation="silu",
    )[0]
    expected = torch.nn.functional.conv1d(
        inputs.transpose(1, 2).contiguous(),
        weight.unsqueeze(1),
        padding=2,
        groups=6,
    )[:, :, : inputs.shape[1]]
    expected = torch.nn.functional.silu(expected).transpose(1, 2)

    assert torch.equal(output, expected)
    assert output.stride() == expected.stride()
    assert not output.is_contiguous()


def test_minicpm_torch_gated_norm_preserves_reference_cast_boundary():
    norm = Qwen3NextRMSNormGated(4, eps=1e-6).to(dtype=torch.bfloat16)
    hidden_states = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4]], dtype=torch.bfloat16
    )
    gate = torch.tensor([[0.5, -0.5, 1.0, -1.0]], dtype=torch.bfloat16)

    normalized = hidden_states.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + 1e-6
    )
    expected = norm.weight * normalized.to(torch.bfloat16)
    expected = expected * torch.nn.functional.silu(gate.float())

    assert torch.equal(norm(hidden_states, gate), expected.to(torch.bfloat16))


def test_pretrained_config_initialization_prefers_post_init(monkeypatch):
    calls = []
    config = SimpleNamespace()
    monkeypatch.setattr(
        PretrainedConfig,
        "__post_init__",
        lambda instance: calls.append(("post_init", instance)),
        raising=False,
    )
    monkeypatch.setattr(
        PretrainedConfig,
        "__init__",
        lambda instance: calls.append(("init", instance)),
    )

    _initialize_pretrained_config(config)

    assert calls == [("post_init", config)]


def test_pretrained_config_initialization_falls_back_to_init(monkeypatch):
    calls = []
    config = SimpleNamespace()
    monkeypatch.delattr(PretrainedConfig, "__post_init__", raising=False)
    monkeypatch.setattr(
        PretrainedConfig,
        "__init__",
        lambda instance: calls.append(instance),
    )

    _initialize_pretrained_config(config)

    assert calls == [config]


def test_canonical_lora_preserves_model_output_dtype():
    base = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    lora = CanonicalLoRA(
        target_modules=["proj"],
        dim=2,
        alpha=4,
        lora_A_init_method="kaiming",
        lora_dtype=torch.float32,
    )

    adapted = lora.transform(base, name="proj")
    output = adapted(torch.ones(2, 3, dtype=torch.bfloat16))

    assert isinstance(adapted, LinearAdapter)
    assert adapted.linear_in.weight.dtype == torch.float32
    assert adapted.linear_out.weight.dtype == torch.float32
    assert output.dtype == torch.bfloat16


def test_dense_canonical_lora_exports_distributed_checkpoint_state(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: 0,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_world_size",
        lambda *, with_context_parallel: 1,
    )
    adapter = DenseLinearAdapter(
        3,
        4,
        dim=2,
        alpha=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        a_init="kaiming",
        b_init="zero",
    )

    state = adapter.sharded_state_dict("adapter.")

    assert set(state) == {
        "adapter.linear_in.weight",
        "adapter.linear_out.weight",
    }
    assert state["adapter.linear_in.weight"].data.shape == (2, 3)
    assert state["adapter.linear_out.weight"].data.shape == (4, 2)


def test_temporary_optimizer_backend_restores_global_state():
    import megatron.core.optimizer as mcore_optimizer

    config = SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False)
    original_adam = torch.optim.Adam
    original_adamw = torch.optim.AdamW
    original_backend_flag = mcore_optimizer.USING_PYTORCH_OPTIMIZER

    with training_utils._temporary_optimizer_backend(config, "torch-fused"):
        assert torch.optim.Adam is not original_adam
        assert torch.optim.AdamW is not original_adamw
        assert mcore_optimizer.USING_PYTORCH_OPTIMIZER is True

    assert torch.optim.Adam is original_adam
    assert torch.optim.AdamW is original_adamw
    assert mcore_optimizer.USING_PYTORCH_OPTIMIZER is original_backend_flag


def test_temporary_optimizer_backend_rejects_unsupported_optimizer():
    config = SimpleNamespace(optimizer="sgd", use_precision_aware_optimizer=False)

    with pytest.raises(ValueError, match="requires --optimizer adam"):
        with training_utils._temporary_optimizer_backend(config, "torch-fused"):
            pass


def test_peft_periodic_checkpoint_uses_loongforge_save(monkeypatch):
    args = SimpleNamespace(
        exit_signal_handler=False,
        save="checkpoint-dir",
        save_interval=2,
        non_persistent_save_interval=None,
        exit_duration_in_mins=None,
        exit_interval=None,
    )
    calls = []
    peft = object()
    monkeypatch.setattr(training_utils, "get_args", lambda: args)
    monkeypatch.setattr(
        training_utils,
        "save_checkpoint_and_time",
        lambda *call_args, **call_kwargs: calls.append((call_args, call_kwargs)),
    )

    should_exit = training_utils.checkpoint_and_decide_exit(
        model="model",
        ema="ema",
        optimizer="optimizer",
        opt_param_scheduler="scheduler",
        iteration=2,
        num_floating_point_operations_so_far=3,
        checkpointing_context={},
        train_data_iterator="iterator",
        peft_class=peft,
    )

    assert should_exit is False
    assert len(calls) == 1
    assert calls[0][1]["peft_class"] is peft


def test_prefix_remap_and_tied_weight_filter_are_directional():
    weight = torch.ones(2, 3)
    state_dict = {
        "model.embed_tokens.weight": weight,
        "lm_head.weight": weight,
    }
    _drop_duplicate_tied_lm_head(state_dict)
    assert set(state_dict) == {"model.embed_tokens.weight"}

    untied_head = weight.clone()
    untied_state_dict = {
        "model.embed_tokens.weight": weight,
        "lm_head.weight": untied_head,
    }
    _drop_duplicate_tied_lm_head(untied_state_dict)
    assert set(untied_state_dict) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }

    shared_storage = torch.arange(12).reshape(4, 3)
    disjoint_view_state_dict = {
        "model.embed_tokens.weight": shared_storage[:2],
        "lm_head.weight": shared_storage[2:],
    }
    _drop_duplicate_tied_lm_head(disjoint_view_state_dict)
    assert set(disjoint_view_state_dict) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }

    source = {"model.vision_tower.weight": weight}
    mcore = remap_state_dict_prefixes(
        source,
        {"vision_model": "model.vision_tower"},
        mcore_to_hf=False,
    )
    assert set(mcore) == {"vision_model.weight"}
    assert mcore["vision_model.weight"] is weight
