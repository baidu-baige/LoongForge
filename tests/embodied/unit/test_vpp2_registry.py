# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""P2-1 registration and meta-construction smoke tests."""

import inspect
from unittest.mock import patch

from loongforge.embodied.model import build_model
from loongforge.embodied.model.registry import MODEL_REGISTRY
from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig
from loongforge.embodied.model.vpp2.modeling_vpp2 import VPP2Model


def test_vpp2_registry_and_meta_shell():
    model = build_model(VPP2ModelConfig())
    assert "vpp2" in MODEL_REGISTRY
    assert model.__class__.__name__ == "VPP2Model"
    assert all(parameter.is_meta for parameter in model.parameters())
    assert inspect.signature(model.forward).parameters["iteration"].default is None
    assert model.default_fp8_targets()["module_patterns"]
    assert model.default_lora_targets()["target_modules"] == []


def test_vpp2_action_stage_freeze_contract_survives_train_call():
    model = build_model(VPP2ModelConfig())
    model.train()

    assert sum(parameter.requires_grad for parameter in model.video_expert.parameters()) == 0
    assert sum(parameter.numel() for parameter in model.video_expert.parameters() if parameter.requires_grad) == 0
    assert model.video_expert.training is False
    assert model.vae.training is False
    assert model.clip_encoder.training is False
    assert model.mot.training is False
    assert model.action_expert.training is True
    assert model.proprio_encoder.training is True
    assert sum(parameter.numel() for parameter in model.action_expert.parameters() if parameter.requires_grad) == 929399303
    assert sum(parameter.numel() for parameter in model.proprio_encoder.parameters() if parameter.requires_grad) == 36864


def test_vpp2_eager_config_materializes_and_strict_loads_before_wrap():
    """The DDP launcher must not hand a meta tensor to the generic wrapper."""
    cfg = VPP2ModelConfig(init_on_meta=False)
    with patch.object(VPP2Model, "materialize") as materialize, patch.object(
        VPP2Model, "load_pretrained"
    ) as load_pretrained:
        build_model(cfg)

    materialize.assert_called_once()
    load_pretrained.assert_called_once_with(cfg.video_checkpoint_path, device=materialize.call_args.args[0])
