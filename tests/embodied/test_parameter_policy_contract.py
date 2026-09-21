# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Real CPU torch contracts. Run this file with Python >= 3.10 and torch.

Only the eager trainer package initializers are bypassed: they import the whole
training stack (transformers, megatron, etc.). All tested implementation modules
and all tensors/optimizers/serialization are real, not torch mocks.
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load leaf modules without executing unrelated Trainer dependency imports.
for package in ("loongforge.embodied.model",
                "loongforge.embodied.train",
                "loongforge.embodied.train.trainers",
                "loongforge.embodied.train.trainers.optimizer_state_shard"):
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(ROOT.joinpath(*package.split(".")))]
    sys.modules.setdefault(package, namespace)

from loongforge.embodied.model.precision_policy import (
    MarkerParameterPolicy, ParameterMetadata, build_parameter_policy,
    resolve_parameter_capabilities,
)
from loongforge.embodied.model.lingbot_vla_v2.parameter_policy import LingbotVlaV2ParameterPolicy
from loongforge.embodied.train.trainers.optimizer_state_shard.registry import ParameterRegistry
from loongforge.embodied.train.trainers.optimizer_state_shard.parameter_manager import OptimizerStateShardManager
from loongforge.embodied.train.trainers.optimizer_state_shard.checkpoint_io import OptimizerStateShardCheckpointIO


def model():
    module = nn.Module()
    module.register_parameter("weight", nn.Parameter(torch.full((3, 3), 1.001)))
    module.register_parameter("bias", nn.Parameter(torch.full((3,), 0.003)))
    module.register_parameter("frozen", nn.Parameter(torch.ones(2), requires_grad=False))
    return module


def registry(world=1, rank=0, **kwargs):
    return ParameterRegistry(model(), rank, world, MarkerParameterPolicy(), **kwargs)


def manager(world=1, rank=0):
    return OptimizerStateShardManager(
        model(), rank=rank, world_size=world,
        parameter_policy=MarkerParameterPolicy(muon_ndims=()),
        grad_overlap=False, param_overlap=False,
    )


def context(rank=0, world=1):
    return types.SimpleNamespace(rank=rank, world_size=world, is_main=rank == 0,
                                 barrier=lambda: None)


class PolicyContract(unittest.TestCase):
    def test_immutable_and_master_precision(self):
        module = model()
        original = module.weight.detach().clone()
        reg = ParameterRegistry(module, 0, 1, MarkerParameterPolicy())
        cap = reg.capabilities[0]
        with self.assertRaises(FrozenInstanceError):
            cap.compute_dtype = "fp32"
        torch.testing.assert_close(reg.master["weight"], original, rtol=0, atol=0)
        self.assertFalse(torch.equal(reg.master["weight"], module.weight.float()))
        self.assertEqual(module.weight.dtype, torch.bfloat16)
        self.assertEqual(reg.optimizer_kind("weight", reg.master["weight"]), "muon")

    def test_lingbot_sensitive_parameters_and_legacy_labels(self):
        legacy = LingbotVlaV2ParameterPolicy()
        public = legacy.as_parameter_policy({"grad_reduce_dtype": "mixed",
                                             "param_sync_precision": "bf16"})
        prefix = "qwenvl_with_expert.qwen_expert.model."
        cases = [
            (prefix + "layers.0.mlp.weight", (4, 4), "action_expert", "fp32", "bf16"),
            (prefix + "layers.0.mlp.gate.weight", (4, 4), "expert_gate_norm", "fp32", "fp32"),
            (prefix + "layers.0.shared_expert_gate.weight", (4, 4), "expert_gate_norm", "fp32", "fp32"),
            (prefix + "norm.weight", (4,), "expert_gate_norm", "fp32", "fp32"),
            ("vlm.layer.bias", (4,), "vlm_norm_bias", "bf16", "fp32"),
            ("vlm.attn.weight", (4, 4), "vlm_backbone", "bf16", "bf16"),
        ]
        for name, shape, label, compute, wire in cases:
            with self.subTest(name=name):
                tensor = nn.Parameter(torch.ones(shape))
                self.assertEqual(legacy.classify(name, tensor), label)
                cap = public.classify(ParameterMetadata(name, tensor))
                self.assertEqual((cap.compute_dtype, cap.grad_reduce_dtype,
                                  cap.parameter_sync_dtype), (compute, wire, wire))

    def test_historical_fp32_publish_is_not_always_fp32(self):
        reg = registry(param_sync_precision="fp32")
        self.assertEqual(reg.record("weight").param_wire_dtype, torch.bfloat16)
        self.assertEqual(reg.record("bias").param_wire_dtype, torch.float32)
        self.assertEqual(reg.record("weight").grad_wire_dtype, torch.float32)

    def test_fp8_opt_in_and_critical_exclusion(self):
        for shadow, expected in ((False, torch.bfloat16), (True, torch.uint8)):
            reg = registry(param_sync_precision="fp8_e4m3_delta",
                           param_sync_fp8_include=("weight", "bias"),
                           param_sync_bf16_with_fp8=shadow)
            self.assertEqual(reg.record("weight").param_wire_dtype, expected)
            self.assertEqual(reg.record("bias").param_wire_dtype, torch.float32)

    def test_marker_misses_duplicates_and_case_insensitive_adamw(self):
        policy = build_parameter_policy({
            "compute_fp32_markers": ("weight", "weight", "wei"),
            "adamw_markers": ("WEIGHT",),
            "comm_critical_markers": ("optional_gate",),
        })
        with self.assertLogs("loongforge.embodied.model.precision_policy", level="WARNING") as logs:
            caps = resolve_parameter_capabilities(model(), policy)
        self.assertEqual(caps[0].compute_dtype, "fp32")
        self.assertEqual(caps[0].optimizer_kind, "adamw")
        self.assertEqual(len(logs.output), 1)
        with self.assertLogs("loongforge.embodied.model.precision_policy", level="WARNING"):
            resolve_parameter_capabilities(model(), LingbotVlaV2ParameterPolicy().as_parameter_policy())

    def test_conflicts_fail_before_cast(self):
        module = model()
        policy = build_parameter_policy({"explicit_rules": [
            {"marker": "weight", "compute_dtype": "fp32"},
            {"marker": "wei", "compute_dtype": "bf16"},
        ]})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ParameterRegistry(module, 0, 1, policy)
        self.assertEqual(module.weight.dtype, torch.float32)

    def test_invalid_configuration(self):
        for runtime in ({"grad_reduce_dtype": "bad"}, {"param_sync_precision": "fp8"},
                        {"param_sync_fp8_include": "weight"}, {"grad_reduce_dtype": ""}):
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                build_parameter_policy(runtime_config=runtime)
        for kwargs in ({"param_sync_fp8_block": 3}, {"param_sync_fp8_reprime_interval": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                registry(**kwargs)
        with self.assertRaises(ValueError):
            registry(world=0)
        with self.assertRaises(ValueError):
            MarkerParameterPolicy(compute_fp32_markers=("",))

    def test_custom_public_policy_is_consumed(self):
        base = build_parameter_policy()

        class CustomPolicy:
            def classify(self, metadata):
                return replace(base.classify(metadata), optimizer_kind="adamw",
                               compute_dtype="fp32", parameter_sync_dtype="fp32")

            def validate(self, capabilities):
                pass

        reg = ParameterRegistry(model(), 0, 1, CustomPolicy())
        self.assertEqual(reg.record("weight").compute.dtype, torch.float32)
        self.assertEqual(reg.record("weight").optimizer_kind, "adamw")

    def test_builder_runtime_survives_registry_and_conflicts_fail(self):
        policy = build_parameter_policy(runtime_config={"grad_reduce_dtype": "bf16"})
        reg = ParameterRegistry(model(), 0, 1, policy)
        self.assertEqual(reg.record("weight").grad_wire_dtype, torch.bfloat16)
        with self.assertRaises(ValueError):
            ParameterRegistry(model(), 0, 1, policy, grad_reduce_dtype="compute")

    def test_explicit_fp32_conflicts_but_default_allows_registry_override(self):
        explicit = build_parameter_policy(
            runtime_config={"grad_reduce_dtype": "fp32", "param_sync_precision": "fp32"}
        )
        with self.assertRaises(ValueError):
            ParameterRegistry(model(), 0, 1, explicit, grad_reduce_dtype="bf16")
        with self.assertRaises(ValueError):
            ParameterRegistry(model(), 0, 1, explicit, param_sync_precision="bf16")
        default = build_parameter_policy()
        reg = ParameterRegistry(model(), 0, 1, default, grad_reduce_dtype="bf16",
                                param_sync_precision="bf16")
        self.assertEqual(reg.grad_reduce_mode, "bf16")
        self.assertEqual(reg.param_sync_precision, "bf16")

    def test_fp8_shadow_bool_conflict_and_manager_inheritance(self):
        explicit_true = build_parameter_policy(
            runtime_config={"param_sync_bf16_with_fp8": True}
        )
        with self.assertRaises(ValueError):
            ParameterRegistry(model(), 0, 1, explicit_true,
                              param_sync_bf16_with_fp8=False)
        explicit_false = build_parameter_policy(
            runtime_config={"param_sync_bf16_with_fp8": False}
        )
        with self.assertRaises(ValueError):
            ParameterRegistry(model(), 0, 1, explicit_false,
                              param_sync_bf16_with_fp8=True)
        inherited = manager()
        self.assertFalse(inherited.registry.param_sync_bf16_with_fp8)
        inherited_true = OptimizerStateShardManager(
            model(), rank=0, world_size=1, parameter_policy=explicit_true,
            grad_overlap=False, param_overlap=False,
        )
        self.assertTrue(inherited_true.registry.param_sync_bf16_with_fp8)
        default = build_parameter_policy(runtime_config={
            "grad_reduce_dtype": None, "param_sync_precision": None,
            "param_sync_bf16_with_fp8": None, "param_sync_fp8_include": None,
        })
        self.assertEqual(default.wire.explicit_fields, frozenset())
        self.assertIs(default.wire.param_sync_bf16_with_fp8, False)
        reg = ParameterRegistry(model(), 0, 1, default,
                                param_sync_bf16_with_fp8=True,
                                grad_reduce_dtype="bf16", param_sync_precision="bf16")
        self.assertTrue(reg.param_sync_bf16_with_fp8)
        same = ParameterRegistry(model(), 0, 1, explicit_false,
                                 param_sync_bf16_with_fp8=False)
        self.assertFalse(same.param_sync_bf16_with_fp8)

    def test_explicit_empty_whitelist_conflicts(self):
        policy = build_parameter_policy(runtime_config=types.SimpleNamespace(
            param_sync_fp8_include=(), param_sync_bf16_with_fp8=False,
        ))
        self.assertEqual(policy.wire.explicit_fields, frozenset({
            "param_sync_fp8_include", "param_sync_bf16_with_fp8",
        }))
        with self.assertRaisesRegex(ValueError, "param_sync_fp8_include"):
            ParameterRegistry(model(), 0, 1, policy, param_sync_fp8_include=("weight",))

    def test_custom_fp8_capability_requires_fp8_registry_mode(self):
        base = build_parameter_policy()

        class Fp8Policy:
            def classify(self, metadata):
                cap = base.classify(metadata)
                if metadata.name == "weight":
                    return replace(cap, compute_dtype="fp32",
                                   parameter_sync_dtype="fp8_e4m3_delta")
                return cap

            def validate(self, capabilities):
                pass

        module = model()
        # Guard only the setup method; tensors and dtype resolution remain real.
        with patch.object(ParameterRegistry, "_create_master") as create_master:
            with self.assertRaisesRegex(ValueError, "requires FP8"):
                ParameterRegistry(module, 0, 1, Fp8Policy())
            create_master.assert_not_called()
        self.assertEqual(module.weight.dtype, torch.float32)
        reg = ParameterRegistry(
            model(), 0, 1, Fp8Policy(), param_sync_precision="fp8_e4m3_delta",
            param_sync_fp8_include=("weight",),
        )
        self.assertEqual(reg.record("weight").param_wire_dtype, torch.uint8)

    def test_fp8_registry_allows_non_fp8_public_capabilities(self):
        base = build_parameter_policy()

        class PlainPolicy:
            def classify(self, metadata):
                return base.classify(metadata)

            def validate(self, capabilities):
                pass

        # FP8 mode may contain only ordinary records (e.g. no opted-in tensors).
        reg = ParameterRegistry(model(), 0, 1, PlainPolicy(),
                                param_sync_precision="fp8_e4m3_delta")
        self.assertEqual(reg.record("weight").param_wire_dtype, torch.bfloat16)
        self.assertEqual(reg.record("bias").param_wire_dtype, torch.float32)


class ManifestContract(unittest.TestCase):
    def test_manifest_roundtrip_and_frozen_parameters(self):
        reg = registry()
        manifest = json.loads(json.dumps(reg.manifest()))
        reg.validate_manifest(manifest)
        self.assertEqual(len(manifest["parameters"]), 3)
        self.assertIsNone(manifest["parameters"][2]["owner_rank"])
        self.assertEqual(manifest["parameters"][2]["compute_dtype"], "fp32")

    def test_same_world_rejects_every_changed_contract_field(self):
        reg = registry()
        changes = {"name": "renamed", "shape": [9], "numel": 12, "owner_rank": 1,
                   "compute_dtype": "fp32", "grad_reduce_dtype": "bf16",
                   "parameter_sync_dtype": "fp32", "optimizer_kind": "adamw",
                   "owner_policy": "other", "requires_grad": False}
        for key, value in changes.items():
            saved = copy.deepcopy(reg.manifest())
            saved["parameters"][0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                reg.validate_manifest(saved)
        saved = reg.manifest()
        saved["policy_version"] += 1
        with self.assertRaises(RuntimeError):
            reg.validate_manifest(saved)

    def test_reshard_allows_only_owner_changes(self):
        reg = registry()
        saved = registry(world=2).manifest()
        reg.validate_manifest(saved, allow_reshard=True, saved_world_size=2)
        with self.assertRaises(RuntimeError):
            reg.validate_manifest(saved)
        saved["parameters"][1]["owner_rank"] = 2
        with self.assertRaises(RuntimeError):
            reg.validate_manifest(saved, allow_reshard=True)
        saved = registry(world=2).manifest()
        saved["parameters"][0]["shape"] = [9]
        with self.assertRaises(RuntimeError):
            reg.validate_manifest(saved, allow_reshard=True)

    def test_save_detects_mutation(self):
        reg = registry()
        reg.compute["weight"].data = reg.compute["weight"].float()
        with self.assertRaises(RuntimeError):
            reg.manifest()

    def test_legacy_warning(self):
        with self.assertLogs(level="WARNING") as logs:
            registry().validate_manifest(None)
        self.assertIn("Legacy checkpoint", "\n".join(logs.output))

    def test_manager_roundtrip_and_legacy(self):
        src, dst = manager(), manager()
        state = src.state_dict()
        dst.load_state_dict(state)
        torch.testing.assert_close(src.master["weight"], dst.master["weight"], rtol=0, atol=0)
        del state["manifest"]
        with self.assertLogs(level="WARNING"):
            dst.load_state_dict(state)

    def test_checkpoint_same_world_and_reshard(self):
        for saved_world in (1, 2):
            with self.subTest(saved_world=saved_world), tempfile.TemporaryDirectory() as directory:
                metadata_path = str(Path(directory) / "metadata.json")
                for rank in range(saved_world):
                    src = manager(saved_world, rank)
                    optimizer = torch.optim.AdamW(list(src.master.values()), lr=0.01)
                    for parameter in src.master.values():
                        parameter.grad = torch.ones_like(parameter)
                    optimizer.step()
                    OptimizerStateShardCheckpointIO(src, optimizer).save_local_state(
                        directory, metadata_path, context(rank, saved_world),
                    )
                dst = manager()
                optimizer = torch.optim.AdamW(list(dst.master.values()), lr=0.01)
                io = OptimizerStateShardCheckpointIO(dst, optimizer)
                metadata = json.loads(Path(metadata_path).read_text())
                io.load_local_state(directory, metadata, context())
                self.assertEqual(len(optimizer.state), 2)
                for rank in range(saved_world):
                    saved = torch.load(str(Path(directory) / f"rank_{rank}.pt"),
                                       weights_only=False)
                    for name, value in saved["manager"]["master"].items():
                        torch.testing.assert_close(dst.master[name], value, rtol=0, atol=0)
                # A tampered manifest must fail before copying any masters.
                before = dst.master["weight"].detach().clone()
                metadata["manifest"]["parameters"][0]["shape"] = [9]
                with self.assertRaises(RuntimeError):
                    io.load_local_state(directory, metadata, context())
                torch.testing.assert_close(dst.master["weight"], before, rtol=0, atol=0)

    def test_legacy_checkpoint_io_and_invalid_rank_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            src = manager()
            optimizer = torch.optim.AdamW(list(src.master.values()))
            io = OptimizerStateShardCheckpointIO(src, optimizer)
            metadata_path = str(Path(directory) / "metadata.json")
            io.save_local_state(directory, metadata_path, context())
            metadata = json.loads(Path(metadata_path).read_text())
            rank_path = str(Path(directory) / "rank_0.pt")
            saved = torch.load(rank_path, weights_only=False)
            # Removing a saved master violates the manifest before state copying.
            invalid = copy.deepcopy(saved)
            invalid["manager"]["master"].pop("bias")
            torch.save(invalid, rank_path)
            with self.assertRaisesRegex(RuntimeError, "master keys"):
                io.load_local_state(directory, metadata, context())
            # Both missing manifests are accepted, with an explicit warning.
            del saved["manager"]["manifest"]
            del metadata["manifest"]
            torch.save(saved, rank_path)
            with self.assertLogs(level="WARNING"):
                io.load_local_state(directory, metadata, context())


if __name__ == "__main__":
    unittest.main(verbosity=2)
