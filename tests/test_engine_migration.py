# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the engine migration boundaries."""

import argparse
import ast
import functools
import importlib
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import make_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_functions(relative_path, names, **namespace):
    """Run selected functions without importing the GPU stack."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def model_imports():
    """Keep model registration real and replace only heavy model dependencies."""
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name.split(".")[0] in {"loongforge", "torch", "transformers", "megatron"}:
                del sys.modules[name]

        def stub(name, **exports):
            module = sys.modules.setdefault(name, ModuleType(name))
            module.__path__ = []
            module.__dict__.update(exports)

        stub("torch")
        stub("transformers", AutoModel=NS(register=Mock()))
        stub("megatron")
        stub("megatron.core")
        stub("megatron.core.transformer")
        stub("megatron.core.transformer.spec_utils", import_module=Mock())
        stub("loongforge.engines.megatron.utils.runtime", get_transformers_version=Mock())
        stub(
            "loongforge.engines.megatron",
            build_transformer_config=Mock(), get_args=Mock(), print_rank_0=Mock(),
        )
        stub("loongforge.models.common", BaseModelStditConfig=type("BaseModelStditConfig", (), {}))
        stub("loongforge.models.diffusion.wan.wan_config", WanConfig=type("WanConfig", (), {}))
        stub(
            "loongforge.models.diffusion.qwen_image.qwen_image_model",
            QwenImageModel=type("QwenImageModel", (), {}),
        )
        stub(
            "loongforge.models.diffusion.qwen_image.qwen_image_layer_spec",
            get_qwen_image_layer_with_te_spec=Mock(),
        )
        for package in ("foundation", "encoder"):
            path = ROOT / f"loongforge/models/{package}/__init__.py"
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    name = f"loongforge.models.{package}.{node.module}"
                    stub(name, **{alias.name: type(alias.name, (), {}) for alias in node.names})

        yield importlib.import_module


@pytest.mark.parametrize("first", ["factory", "vlm.args", "foundation", "encoder", "diffusion"])
def test_model_imports_register_builtins_before_custom_providers(first, model_imports, monkeypatch):
    model_imports(
        "loongforge.training.vlm.args" if first == "vlm.args" else f"loongforge.models.{first}"
    )
    factory = model_imports("loongforge.models.factory")
    builtin = factory.get_model_provider("qwen_image")
    assert builtin is not None
    provider = model_imports("loongforge.models.diffusion.qwen_image.qwen_image_provider")
    assert builtin is provider.qwen_image_model_provider
    assert factory.get_model_config("qwen-image-edit-2511").model_type == "qwen_image"
    register = model_imports("transformers").AutoModel.register
    registrations = register.call_args_list[:]
    assert registrations
    for module in ("factory", "foundation", "encoder", "diffusion"):
        model_imports(f"loongforge.models.{module}")
    assert register.call_args_list == registrations

    with pytest.raises(ValueError, match="Cannot register duplicate model provider"):
        factory.register_model_provider("qwen_image")(lambda: None)
    assert factory.get_model_provider("qwen_image") is builtin

    vlm_args = model_imports("loongforge.training.vlm.args")
    query = factory.get_support_model_archs
    assert vlm_args.get_support_model_archs is query
    expected = query("qwen_image")
    assert expected
    monkeypatch.setattr(factory, "get_support_model_archs", Mock(side_effect=AssertionError("Late factory lookup")))
    assert vlm_args.get_support_model_archs("qwen_image") == expected


def test_model_catalog_import_keeps_torch_entry_lightweight():
    subprocess.run(
        [sys.executable, "-B", "-c", """
import sys
import loongforge.models.catalog
assert not any(name.split(".")[0] in {"torch", "transformers", "megatron"} for name in sys.modules)
assert not any(name in sys.modules for name in (
    "loongforge.models.factory",
    "loongforge.models.foundation", "loongforge.models.encoder", "loongforge.models.diffusion",
))
"""],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_recipe_registry_keeps_family_case_and_override_rules(monkeypatch):
    from loongforge.training import trainer_builder

    registry = {}
    monkeypatch.setitem(trainer_builder.TRAINERS_BY_ENGINE, "megatron", registry)
    original = Mock(__name__="original")
    replacement = Mock(__name__="replacement")
    sft = Mock(__name__="sft")
    register = trainer_builder.register_model_trainer
    assert register(["LLAMA", "qwen3"], "pretrain", engine="megatron")(original) is original
    assert registry == {
        "llama": {"pretrain": original},
        "qwen3": {"pretrain": original},
    }
    register("llama", "pretrain", engine="megatron")(replacement)
    assert registry["llama"]["pretrain"] is original
    register("llama", "pretrain", override=True, engine="megatron")(replacement)
    assert registry["llama"]["pretrain"] is replacement
    assert register("LLAMA", "sft", training_func=sft, engine="megatron") is None
    assert registry["llama"]["sft"] is sft


def test_builder_uses_the_registered_phase_of_the_selected_engine(monkeypatch):
    from loongforge.training import trainer_builder

    recipe = Mock()
    args = NS(model_name="qwen3-8b", model_family="qwen3", training_phase="pretrain")
    monkeypatch.setattr(
        trainer_builder,
        "TRAINERS_BY_ENGINE",
        {"megatron": {"qwen3": {"pretrain": recipe}}, "torch": {}},
    )
    monkeypatch.setitem(
        trainer_builder.ENGINES,
        "megatron",
        lambda: ("qwen3", "pretrain", None, (args,)),
    )
    trainer_builder.build_model_trainer("megatron")
    recipe.assert_called_once_with(args)


def test_engine_registers_loss_and_keeps_original_training_provider(monkeypatch):
    resolvers = {}
    register = Mock(side_effect=lambda name, fn, **kwargs: resolvers.update({name: fn}))
    training_provider = Mock()
    default_loss, internvl_loss, config = object(), object(), object()
    result = NS(model_family="llm", training_phase="pretrain")
    resume = Mock(side_effect=AssertionError("Resumption flags must remain opt-in"))
    resolver_namespace = _load_functions(
        "loongforge/engines/megatron/parser.py", {"register_custom_resolvers"},
        functools=functools, F=NS(relu=Mock(), gelu=Mock(), silu=Mock()),
        get_position_ids=Mock(), get_mrope_index=Mock(), get_rope_index_internvl=Mock(),
        get_rope_index_qwen3vl=Mock(), moe_freq_type=Mock(),
        OmegaConf=NS(register_new_resolver=register),
        default_loss_func=default_loss, loss_func_internvl=internvl_loss,
    )

    def parse_arguments(**kwargs):
        resolver_namespace["register_custom_resolvers"]()
        assert resolvers["loss_func"]("DEFAULT") is default_loss
        assert resolvers["loss_func"]("LOSS_FUNC_INTERNVL") is internvl_loss
        assert kwargs["extra_args_provider"] is training_provider
        assert "args_providers" not in kwargs
        assert "extra_validator" not in kwargs
        return result, config

    save_config = Mock()
    engine = _load_functions(
        "loongforge/engines/megatron/parser.py", {"parse_train_args"},
        parse_arguments=Mock(side_effect=parse_arguments), set_hydra_config=save_config,
        loongforge_extra_train_args_provider=training_provider,
        validate_loongforge_extra_args=Mock(), apply_resumption_flags=resume,
    )
    assert engine["parse_train_args"]() is result
    assert {"act", "position_func", "loss_func", "moe_freq"} <= resolvers.keys()
    assert register.call_args.kwargs == {"replace": True}
    save_config.assert_called_once_with(config)
    resume.assert_not_called()


def test_runner_imports_original_hf_tool_functions(monkeypatch):
    functions = {name: Mock() for name in (
        "load_hf_checkpoint_online", "save_hf_checkpoint_online",
    )}
    for name, export in (
        ("mcore_bridge.checkpoint.hf_checkpoint_loader", "load_hf_checkpoint_online"),
        ("mcore_bridge.checkpoint.hf_checkpoint_saver", "save_hf_checkpoint_online"),
    ):
        module = ModuleType(name)
        setattr(module, export, functions[export])
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "loongforge/engines/megatron/runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "project_root" for target in node.targets))
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.ImportFrom)
               and node.module == "mcore_bridge.checkpoint.hf_checkpoint_saver")
    paths = [entry for entry in sys.path if entry not in {str(ROOT), str(ROOT / "tools")}]
    monkeypatch.setattr(sys, "path", paths)
    namespace = {"__file__": str(path), "os": os, "sys": sys}
    exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), str(path), "exec"), namespace)
    for name, function in functions.items():
        assert namespace[name] is function
        function.assert_not_called()
    assert namespace["project_root"] == str(ROOT)
    assert namespace["tools_path"] == str(ROOT / "tools")
    assert sys.path[:2] == [str(ROOT / "tools"), str(ROOT)]


def test_megatron_provider_keeps_one_parser_and_hydra_overrides(monkeypatch):
    seen = []

    def provider(name):
        def register(parser):
            seen.append((name, parser))
            parser.add_argument(f"--{name}")
            return parser
        return register

    namespace = _load_functions(
        "loongforge/engines/megatron/parser.py", {"parse_megatron_arguments"},
        argparse=argparse, os=os, add_megatron_arguments=provider("upstream"),
    )
    for name, value in {
        "OMPI_COMM_WORLD_RANK": "-1", "OMPI_COMM_WORLD_SIZE": "-1",
        "RANK": "2", "WORLD_SIZE": "8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", [
        "train", "--upstream", "u", "--training", "t", "model.width=64",
    ])
    args, overrides = namespace["parse_megatron_arguments"](
        provider("training"), True,
    )
    assert [name for name, parser in seen] == ["upstream", "training"]
    assert all(parser is seen[0][1] for name, parser in seen)
    assert vars(args) == {
        "upstream": "u", "training": "t", "rank": 2, "world_size": 8,
    }
    assert overrides == ["model.width=64"]


def test_training_provider_registers_original_bridge_parameters_once(monkeypatch):
    calls = []
    add_argument = argparse._ActionsContainer.add_argument

    def register(container, *options, **kwargs):
        calls.extend(options)
        return add_argument(container, *options, **kwargs)

    monkeypatch.setattr(argparse._ActionsContainer, "add_argument", register)
    providers = (
        "_add_extra_tokenizer_args", "_add_extra_training_args", "_add_extra_parallel_args",
        "_add_run_args", "_add_freeze_and_init_args", "_add_ema_args", "_add_recipe_args",
        "_add_sft_args", "_add_loop_logging_args", "_add_log_tensor_args",
        "add_llm_args", "add_vlm_args", "add_diffusion_args",
    )
    namespace = _load_functions(
        "loongforge/training/megatron_args.py", {"_add_extra_bridge_args", "loongforge_extra_train_args_provider"},
        **{name: Mock(side_effect=lambda parser: parser) for name in providers},
    )
    parser = argparse.ArgumentParser()
    assert namespace["loongforge_extra_train_args_provider"](parser) is parser
    group = next(group for group in parser._action_groups if group.title == "extra-bridge")
    names = (
        "fp8-force-no-requant", "force-pow-2-scales", "amax-epsilon", "hf-dequantize-int4",
        "hf-dequantize-mxfp4", "hf-dequantize-dtype", "hf-quant-config-file", "quant-method", "pretrain-as-fp8",
    )
    assert len(group._group_actions) == len(names)
    for name in names:
        for option in (f"--{name}", f"--{name.replace('-', '_')}"):
            assert calls.count(option) == 1
    defaults = {
        "fp8_force_no_requant": False, "force_pow_2_scales": False, "amax_epsilon": 0.0,
        "hf_dequantize_int4": False, "hf_dequantize_mxfp4": False, "hf_dequantize_dtype": "bfloat16",
        "hf_quant_config_file": None, "quant_method": "te", "pretrain_as_fp8": False,
    }
    assert vars(parser.parse_args([])) == defaults
    expected = {
        "fp8_force_no_requant": True, "force_pow_2_scales": True, "amax_epsilon": 0.25,
        "hf_dequantize_int4": True, "hf_dequantize_mxfp4": True, "hf_dequantize_dtype": "float32",
        "hf_quant_config_file": "config.json", "quant_method": "pt", "pretrain_as_fp8": True,
    }
    for separator in ("-", "_"):
        argv = []
        for name in names:
            value = expected[name.replace("-", "_")]
            argv.append(f"--{name.replace('-', separator)}")
            if value is not True:
                argv.append(str(value))
        assert vars(parser.parse_args(argv)) == expected


def test_megatron_validators_keep_original_call_order():
    names = [
        "_validate_extra_model_args", "_validate_extra_tokenizer_args", "_validate_extra_sft_args",
        "_validate_extra_training_args", "_validate_extra_multimodal_args", "_validata_extra_custom_args",
        "_validata_extra_parallel_args", "_align_wan_packing_seq_length", "_validate_legacy_pipeline_args",
        "validate_args",
    ]
    calls = []
    callbacks = {name: Mock(side_effect=lambda *values, name=name: calls.append(name)) for name in names}
    namespace = _load_functions(
        "loongforge/training/megatron_args.py",
        {"validate_loongforge_extra_args", "validate_megatron_args"}, **callbacks,
    )
    args, config = NS(enable_one_logger=True), object()
    namespace["validate_loongforge_extra_args"](args, config)
    namespace["validate_megatron_args"](args)
    assert calls == names
    callbacks[names[0]].assert_called_once_with(args, config)
    for name in names[1:]:
        callbacks[name].assert_called_once_with(args)
    assert args.enable_one_logger is False


@pytest.mark.parametrize(("filenames", "expected"), [
    ([], None),
    (["sft_dataset_config.yaml"], "sft_dataset_config.yaml"),
    (["sft_dataset_config.yml"], "sft_dataset_config.yml"),
    (["sft_dataset_config.yml", "sft_dataset_config.yaml"], "sft_dataset_config.yaml"),
])
def test_relocated_runtime_keeps_default_config_root_and_search_order(tmp_path, filenames, expected):
    config_dir = tmp_path / "configs/data"
    config_dir.mkdir(parents=True)
    for filename in filenames:
        (config_dir / filename).touch()
    namespace = _load_functions(
        "loongforge/engines/megatron/utils/runtime.py", {"get_default_sft_dataset_config"},
        Path=Path, DEFAULT_DATASET_CONFIG="sft_dataset_config.yaml",
        __file__=str(tmp_path / "loongforge/engines/megatron/utils/runtime.py"),
    )
    assert namespace["get_default_sft_dataset_config"]() == (
        str(config_dir / expected) if expected else None
    )


@pytest.mark.parametrize(("initialized", "actual_rank", "explicit_rank", "expected"), [
    (False, 0, None, "message\n"), (False, 0, 1, ""),
    (True, 0, 1, "message\n"), (True, 1, 0, ""),
])
def test_runtime_print_rank_zero_keeps_actual_rank_priority(
    initialized, actual_rank, explicit_rank, expected, capsys,
):
    namespace = _load_functions(
        "loongforge/engines/megatron/utils/runtime.py", {"print_rank_0"},
        torch=NS(distributed=NS(is_initialized=lambda: initialized, get_rank=lambda: actual_rank)),
    )
    namespace["print_rank_0"]("message", explicit_rank)
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(("family", "length", "cp", "expected"), [
    ("wan", 5, 4, 8), ("wan", 8, 3, 9), ("wan", 8, 4, 8),
    ("wan", 5, 1, 5), ("wan", None, 4, None), ("llm", 5, 4, 5),
])
def test_diffusion_text_length_keeps_cp_alignment(family, length, cp, expected):
    namespace = _load_functions(
        "loongforge/training/diffusion/args.py", {"_validata_extra_custom_args"},
        catalog=NS(DiffusionModelFamilies=NS(names=lambda: {"wan"})),
    )
    args = NS(model_family=family, max_text_length=length, context_parallel_size=cp)
    namespace["_validata_extra_custom_args"](args)
    assert args.max_text_length == expected


def test_diffusion_latent_patch_size_keeps_tuple_cli_behavior():
    namespace = _load_functions(
        "loongforge/training/diffusion/args.py", {"add_diffusion_args"}, argparse=argparse,
    )
    parser = argparse.ArgumentParser()
    assert namespace["add_diffusion_args"](parser) is parser
    assert parser.parse_args([]).latent_patch_size == (1, 1, 1)
    assert parser.parse_args(["--latent-patch-size", "(1,2,2)"]).latent_patch_size == tuple("(1,2,2)")


@pytest.mark.parametrize("family", ["llm", "vlm", "diffusion"])
@pytest.mark.parametrize("yaml_dsa", [False, True], ids=["cli-dsa", "yaml-dsa"])
def test_master_schema_filters_cli_fields_and_global_dsa_keeps_yaml_priority(family, yaml_dsa, monkeypatch):
    base = make_dataclass("BaseConfig", [("hidden_size", int, 16), ("qk_pos_emb_head_dim", int, 64)])
    model = make_dataclass("ModelConfig", [("target_only_field", bool, False)], bases=(base,))
    target_module = ModuleType("engine_migration_test")
    target_module.ModelConfig = model
    monkeypatch.setitem(sys.modules, target_module.__name__, target_module)
    llm_args = _load_functions(
        "loongforge/training/llm/args.py", {"add_llm_args"}, argparse=argparse,
    )
    cli_parser = llm_args["add_llm_args"](argparse.ArgumentParser())
    cli_values = vars(cli_parser.parse_args([
        "--use-dsa-fused", "--use-dsa-sp-first",
        "--absorb-backend", "torch", "--dsa-kernel-backend", "torch_ref",
    ]))
    args = NS(
        **cli_values,
        hidden_size=32, qk_pos_emb_head_dim=96, unrelated_run_flag=True, target_only_field=True,
        rank=0, model_family=family,
        multi_latent_attention=True, no_persist_layer_norm=False, apply_layernorm_1p=False,
        norm_epsilon=1e-5, params_dtype="bf16", overlap_p2p_comm=False, num_experts=None,
        rotary_interleaved=False, decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None, fp8_param_gather=False, swiglu=False,
        bias_gelu_fusion=False, squared_relu=False, init_method_xavier_uniform=False,
        group_query_attention=False, cp_comm_type=["p2p"], config_logger_dir="", rope_type=None,
    )
    original = deepcopy(vars(args))
    class ConfigNode(dict):
        @property
        def model_type(self):
            return self["model_type"]

    node = ConfigNode(_target_="engine_migration_test.ModelConfig", hidden_size=48)
    dsa_fields = ("use_dsa_fused", "use_dsa_sp_first", "absorb_backend", "dsa_kernel_backend")
    expected_dsa = {name: cli_values[name] for name in dsa_fields}
    if yaml_dsa:
        expected_dsa = dict(
            use_dsa_fused=False, use_dsa_sp_first=False,
            absorb_backend="te", dsa_kernel_backend="sm80",
        )
        node.update(expected_dsa)
    if family == "llm":
        node["model_type"] = family
    config = NS(model=node if family == "llm" else ConfigNode(
        model_type=family, foundation=node, image_encoder=node, tag="kept",
    ))
    original_node = deepcopy(node)
    constants = NS(
        LanguageModelFamilies=NS(names=lambda: {"llm"}),
        VisionLanguageModelFamilies=NS(names=lambda: {"vlm"}, QWEN3_VL="qwen3_vl"),
        DiffusionModelFamilies=NS(names=lambda: {"diffusion"}),
        EmbodiedModelFamilies=NS(names=lambda: set()),
    )
    validator = _load_functions(
        "loongforge/training/megatron_args.py", {"_validate_extra_model_args"},
        _YAML_KEY_ALIASES={}, catalog=constants,
        _validate_vlm_model_args=Mock(), _validate_llm_model_args=Mock(),
        print_rank_0=Mock(),
    )
    namespace = _load_functions(
        "loongforge/models/utils.py",
        {"convert_megatron_transformer_config_args", "build_model_config"},
        Iterable=Iterable, deepcopy=deepcopy, TransformerConfig=base, MLATransformerConfig=base,
        OmegaConf=NS(to_container=lambda value, resolve: deepcopy(value)),
        instantiate=lambda node, **values: NS(**{key: val for key, val in values.items() if key != "_target_"}),
        constants=constants,
        get_args_dict=lambda: {"foundation": args, "image_encoder": vars(args)}, VLMModelConfig=NS,
    )
    filtered = namespace["convert_megatron_transformer_config_args"](vars(args))
    assert set(dsa_fields).isdisjoint(filtered)
    assert "target_only_field" not in filtered
    mla_config = next(node for node in ast.parse(
        (ROOT / "loongforge/models/common/base_model_config.py").read_text(encoding="utf-8")
    ).body if isinstance(node, ast.ClassDef) and node.name == "BaseModelMLAConfig")
    assert set(dsa_fields).isdisjoint(
        node.target.id for node in mla_config.body if isinstance(node, ast.AnnAssign)
    )
    built = namespace["build_model_config"](args, config)
    components = [built] if family == "llm" else [built.foundation, built.image_encoder]
    for component in components:
        assert component.hidden_size == 48
        assert component.qk_pos_emb_head_dim == 96
        if yaml_dsa:
            assert {name: getattr(component, name) for name in dsa_fields} == expected_dsa
        else:
            assert all(not hasattr(component, name) for name in dsa_fields)
        assert not hasattr(component, "unrelated_run_flag")
        assert not hasattr(component, "target_only_field")
    assert vars(args) == original
    assert node == original_node
    if family != "llm":
        assert built.tag == "kept"

    # Check hydration separately so it cannot hide config merge precedence.
    validator["_validate_extra_model_args"](args, node)
    assert {name: getattr(args, name) for name in dsa_fields} == expected_dsa
    assert vars(args) == {**original, "hidden_size": 48, **expected_dsa}
    assert node == original_node
    training = ModuleType("megatron.training")
    training.get_args = lambda: args
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setenv("LOONGFORGE_DSA_KERNEL_BACKEND", "tilelang")
    backend = _load_functions(
        "loongforge/models/common/experimental_attention_variant/dsa_kernel_backend.py", {"_requested_backend"},
        BACKEND_AUTO="auto", VALID_BACKENDS=("auto", "sm100", "sm80", "torch_ref", "tilelang"),
    )
    assert backend["_requested_backend"]() == expected_dsa["dsa_kernel_backend"]

    # Run the original consumer statements without loading CUDA modules.
    for model_family in ("deepseek", "glm"):
        spec_tree = ast.parse((ROOT / (
            f"loongforge/models/foundation/{model_family}/{model_family}_layer_spec.py"
        )).read_text(encoding="utf-8"))
        read_fused = next(node for node in ast.walk(spec_tree) if isinstance(node, ast.Assign)
                          and ast.unparse(node.targets[0]) == "use_dsa_fused")
        consumer = {"get_args": training.get_args}
        exec(compile(ast.Module(body=[read_fused], type_ignores=[]), "<DSA layer spec>", "exec"), consumer)
        assert consumer["use_dsa_fused"] == expected_dsa["use_dsa_fused"]

    consumer = {"self": NS(config=components[0]), "get_args": training.get_args}
    for filename, class_name, first_target, result_field in [
        ("dsa_fused.py", "DSAttentionFused", "args", "use_dsa_sp_first"),
        ("multi_latent_attention.py", "MLASelfAttentionFused", "self.absorb_backend", "absorb_backend"),
        ("multi_latent_attention.py", "MLASelfAttentionFused", "args", "use_dsa_sp_first"),
    ]:
        tree = ast.parse((ROOT / (
            f"loongforge/models/common/experimental_attention_variant/{filename}"
        )).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        start = next(i for i, node in enumerate(init.body) if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == first_target)
        exec(compile(ast.Module(body=init.body[start:start + 2], type_ignores=[]), "<DSA consumer>", "exec"), consumer)
        assert getattr(consumer["self"], result_field) == expected_dsa[result_field]


@pytest.mark.parametrize(
    ("kind", "finetune", "data", "rng", "optim_flag", "rng_flag", "expected"),
    [
        ("none", True, False, False, False, False, (True, False, False)),
        ("missing", True, False, False, False, False, (True, False, False)),
        ("release", True, True, True, False, False, (True, False, False)),
        ("empty", False, False, False, False, False, (False, False, False)),
        ("checkpoint", True, True, True, False, False, (False, True, False)),
        ("checkpoint", False, False, False, False, False, (False, True, True)),
        ("checkpoint", True, False, True, False, False, (True, True, False)),
        ("checkpoint", False, True, True, True, True, (False, True, True)),
    ],
)
def test_resumption_flags_remain_explicit_and_keep_existing_rules(
    tmp_path, kind, finetune, data, rng, optim_flag, rng_flag, expected,
):
    load = None if kind == "none" else str(tmp_path / "missing" if kind == "missing" else tmp_path)
    if kind == "release":
        (tmp_path / "release").mkdir()
    probes = None if kind == "empty" else NS(dataloader_present=data, rng_present=rng, iteration=7)
    probe = Mock(return_value=probes)
    namespace = _load_functions(
        "loongforge/engines/megatron/checkpointing.py", {"apply_resumption_flags"},
        os=os, probe_checkpoint_components=probe, print_rank_0=Mock(),
    )
    args = NS(load=load, finetune=finetune, no_load_optim=optim_flag, no_load_rng=rng_flag)
    probe.assert_not_called()
    assert (args.finetune, args.no_load_optim, args.no_load_rng) == (finetune, optim_flag, rng_flag)
    namespace["apply_resumption_flags"](args)
    assert (args.finetune, args.no_load_optim, args.no_load_rng) == expected
    if kind in {"none", "missing", "release"}:
        probe.assert_not_called()
    else:
        probe.assert_called_once_with(load, check_rng=True, dp_rank=0)


@pytest.mark.parametrize("family", ["vlm", "ernie4_5_vl", "intern_vl"])
def test_dp_balance_uses_the_model_family_without_registration(family):
    vlm_reorder = Mock(return_value="vlm-batch")
    internvl_reorder = Mock(return_value="internvl-batch")
    namespace = _load_functions(
        "loongforge/data/dp_balance/rebalance/balance.py",
        {"reorder_data_across_dp"},
        get_args=lambda: NS(model_family=family),
        constants=NS(VisionLanguageModelFamilies=NS(names=lambda: ["vlm", "ernie4_5_vl"])),
        reorder_data_for_vlm=vlm_reorder,
        reorder_data_for_internvl=internvl_reorder,
    )
    expected = "internvl-batch" if family == "intern_vl" else "vlm-batch"
    assert namespace["reorder_data_across_dp"](["metadata", "batch"]) == ["metadata", expected]
    selected, unused = (internvl_reorder, vlm_reorder) if family == "intern_vl" else (vlm_reorder, internvl_reorder)
    selected.assert_called_once_with("batch")
    unused.assert_not_called()


def test_finetune_keeps_the_original_base_contract_and_iterator_lifecycle():
    classes = []
    for path in (
        "loongforge/engines/torch/base_trainer.py",
        "loongforge/training/embodied/sft.py",
    ):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        classes.extend(node for node in tree.body if isinstance(node, ast.ClassDef))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[]))
    namespace = dict(
        ABC=ABC,
        abstractmethod=abstractmethod,
        contextmanager=contextmanager,
        register_model_trainer=lambda *args, **kwargs: (lambda cls: cls),
        EmbodiedModelFamilies=NS(names=lambda: []),
        TrainingPhase=NS(SFT="sft"),
    )
    exec(compile(module, "trainer_contract", "exec"), namespace)
    base, finetune = namespace["BaseTrainer"], namespace["FinetuneTrainer"]
    assert base.__abstractmethods__ == {
        "_build_model", "_build_dataloaders", "_train_forward", "_forward_backward",
        "_backward_loss", "_build_optimizer", "_build_scheduler", "_clip_gradients",
        "_clean_nan_gradients", "_load_pretrained", "_handle_resume", "_freeze_modules",
        "_save_checkpoint", "_init_data_iterator", "_advance_epoch", "_fetch_batch",
    }
    assert not finetune.__abstractmethods__

    class Loader(list):
        pass

    loader = Loader(["first", "second"])
    loader.sampler = Mock()
    trainer = finetune(NS(train_iters=1), NS(), NS())
    trainer.ctx = NS(is_main=False)
    trainer.model = NS(parameters=lambda: iter([NS(device="cpu")]))
    trainer.dataloaders = {"vla": loader}
    trainer._init_data_iterator("vla")
    assert [trainer._fetch_batch("vla") for _ in range(3)] == ["first", "second", "first"]
    assert trainer.current_epoch == 1
    assert [call.args for call in loader.sampler.set_epoch.call_args_list] == [(0,), (1,)]


@pytest.mark.parametrize("offload_enabled", [False, True])
def test_full_hetero_keeps_global_state_and_lazy_offload(offload_enabled, monkeypatch):
    offload = NS(enabled=offload_enabled, clear=Mock())
    manager = Mock(return_value=offload)
    module = ModuleType("loongforge.engines.megatron.full_hetero_cpu_offload")
    module.CpuOffloadManager = manager
    monkeypatch.setitem(sys.modules, module.__name__, module)
    lists = {name: [] for name in (
        "embedding_list", "grad_list", "visual_pos_masks_list",
        "deepstack_visual_embeds_list", "deepstack_grad_list",
    )}
    state = _load_functions(
        "loongforge/training/vlm/pretrain_vlm.py",
        {
            "get_cpu_offload_manager", "get_encoder_data_iterator", "set_encoder_data_iterator",
            "get_embedding_list", "clear_embedding_list", "get_grad_list", "clear_grad_list",
            "get_visual_pos_masks_list", "clear_visual_pos_masks_list", "get_deepstack_visual_embeds_list",
            "clear_deepstack_visual_embeds_list", "get_deepstack_grad_list", "clear_deepstack_grad_list",
            "clear_vpp0_batch_cache", "clear_vpp_counters", "set_count", "clear_full_hetero_info",
        },
        **lists, _cpu_offload_manager=None, _encoder_data_iterator=None,
        forward_step_calling_count=5, _vpp0_batch_cache=[object()], _vpp_counters={"higher": 2},
        get_args=lambda: NS(full_hetero_dp_cpu_offload=offload_enabled),
    )
    manager.assert_not_called()
    assert state["get_cpu_offload_manager"]() is offload
    assert state["get_cpu_offload_manager"]() is offload
    manager.assert_called_once_with(enabled=offload_enabled)
    assert state["get_encoder_data_iterator"]() is None
    for name, items in lists.items():
        assert state[f"get_{name}"]() is items
        items.append(object())
        state[f"clear_{name}"]()
        assert not items
        assert state[f"get_{name}"]() is items
    iterator = object()
    state["set_encoder_data_iterator"](iterator)
    for items in lists.values():
        items.append(object())
    state["clear_full_hetero_info"]()
    assert all(not items for items in lists.values())
    for name, items in lists.items():
        assert state[f"get_{name}"]() is items
    assert state["forward_step_calling_count"] == 0
    assert not state["_vpp0_batch_cache"]
    assert not state["_vpp_counters"]
    assert state["get_encoder_data_iterator"]() is iterator
    assert state["get_cpu_offload_manager"]() is offload
    manager.assert_called_once_with(enabled=offload_enabled)
    assert offload.clear.call_count == int(offload_enabled)


def test_vlm_recipe_uses_original_model_provider(monkeypatch):
    path = ROOT / "loongforge/training/vlm/pretrain_vlm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    source = "loongforge.models.omni_models.omni_model_provider"
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == source]
    provider = Mock()
    module = ModuleType(source)
    module.omni_model_provider = provider
    monkeypatch.setitem(sys.modules, source, module)
    namespace = {}
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["omni_model_provider"] is provider
    provider.assert_not_called()


@pytest.mark.parametrize("kind", ["disabled", "missing", "saved", "invalid"])
def test_vlm_recipe_preserves_dataloader_checkpoint_loading(tmp_path, kind):
    path = tmp_path / "train_dataloader_dprank003.pt"
    if kind in {"saved", "invalid"}:
        path.touch()
    saved_state = object()
    load = Mock(return_value={"dataloader_state_dict": saved_state})
    if kind == "invalid":
        load.side_effect = RuntimeError("invalid state")
    checkpoint_name = Mock(return_value=str(path))
    loader = NS(restore_state_rank=Mock())
    args = NS(load=None if kind == "disabled" else "model", dataloader_save="data", iteration=17)
    dataset = object()
    namespace = _load_functions(
        "loongforge/data/vlm/dataloader_provider.py", {"get_train_loader"},
        os=os, torch=NS(load=load), parallel_state=NS(get_data_parallel_rank=lambda: 3),
        energon=NS(get_savable_loader=Mock(return_value=loader)), get_args=lambda: args,
        EnergonDataloader=lambda dataloader, collator: dataloader,
        get_checkpoint_name=checkpoint_name,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("importlib.metadata.version", lambda package: "7.0.0")
        assert namespace["get_train_loader"](dataset) is loader
    if kind == "saved":
        loader.restore_state_rank.assert_called_once_with(saved_state)
    else:
        loader.restore_state_rank.assert_not_called()
    if kind == "disabled":
        checkpoint_name.assert_not_called()
    else:
        checkpoint_name.assert_called_once_with(
            "data", 17, pipeline_rank=0, basename="train_dataloader_dprank003.pt",
        )
    if kind in {"saved", "invalid"}:
        load.assert_called_once_with(str(path), map_location="cpu")
    else:
        load.assert_not_called()
