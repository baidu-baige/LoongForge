# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM arch routing preserves the legacy model and native data contract."""

import pytest
import torch
from omegaconf import OmegaConf

from loongforge.embodied.model.arch.model_configuration_arch import FastwamArchConfig
from loongforge.embodied.model.arch.modeling_arch import ArchWorldModel
from loongforge.embodied.model.registry import MODEL_REGISTRY


def test_fastwam_arch_config_builds_native_provider():
    config = FastwamArchConfig()
    spec = config.build_spec()
    assert config.model_type == "fastwam_arch"
    assert spec.model_type == "fastwam"
    assert spec.geometry.latent_channels == 48
    assert spec.geometry.vae_spatial_stride == 16


def test_fastwam_arch_yaml_and_data_schema():
    from loongforge.embodied.train.config_map import get_model_schema, get_config_path

    schema = get_model_schema("fastwam_arch")
    yaml = OmegaConf.load(get_config_path("fastwam_arch"))
    config = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(schema.model_config_cls), yaml.model))
    data = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(schema.data_config_cls), yaml.data))
    assert config.build_spec().geometry.num_frames == data.num_video_frames
    assert config.disable_train_autocast


def test_fastwam_arch_registry_keeps_legacy_model():
    from loongforge.embodied.model.fastwam.modeling_fastwam import FastWAMPolicy
    from loongforge.embodied.data.datasets.fastwam.transforms.fastwam_collator import (
        FastWAMPreprocessor,
    )
    from loongforge.embodied.data.datasets.transforms.collator import _PREPROCESSOR_REGISTRY
    from loongforge.embodied.data.datasets.transforms.registry import _TRANSFORM_BUILDER_REGISTRY
    from loongforge.embodied.data.datasets.sampler_builder import get_sampler_builder

    assert MODEL_REGISTRY["fastwam"] is FastWAMPolicy
    assert MODEL_REGISTRY["fastwam_arch"] is ArchWorldModel
    assert _PREPROCESSOR_REGISTRY["fastwam_arch"] is FastWAMPreprocessor
    assert _TRANSFORM_BUILDER_REGISTRY["fastwam_arch"] is _TRANSFORM_BUILDER_REGISTRY["fastwam"]
    assert get_sampler_builder("fastwam_arch") is get_sampler_builder("fastwam")


def test_fastwam_arch_rejects_implicit_scratch_initialization():
    with pytest.raises(ValueError, match="initialization checkpoints"):
        ArchWorldModel.from_pretrained(FastwamArchConfig())


def test_fastwam_config_rejects_half_initializer():
    with pytest.raises(ValueError, match="both checkpoints"):
        FastwamArchConfig(video_checkpoint_path="video.pt")


def test_fastwam_config_can_disable_proprioception():
    assert "proprio" not in {
        condition.name for condition in FastwamArchConfig(proprio_dim=None).build_spec().conditions
    }


def test_fastwam_wrapper_encodes_native_vae_and_preserves_masks():
    from types import SimpleNamespace

    config = FastwamArchConfig()
    spec = config.build_spec()

    class RecordingVAE(torch.nn.Module):
        def encode(self, video, *, device, tiled):
            self.call = (video, device, tiled)
            return video.new_zeros(
                (video.shape[0], spec.geometry.latent_channels,
                 *spec.geometry.latent_grid))

    vae = RecordingVAE()
    wrapper = ArchWorldModel(
        config, SimpleNamespace(spec=spec), vae=vae, image_encoder=None)
    sample = {
        "video": torch.zeros(1, 3, 9, 224, 448),
        "action": torch.zeros(1, 32, 7),
        "context": torch.zeros(1, 3, 4096),
        "context_mask": torch.tensor([[True, False, True]]),
        "proprio": torch.arange(16).reshape(1, 2, 8).float(),
        "action_is_pad": torch.zeros(1, 32, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, 9, dtype=torch.bool),
    }
    encoded = wrapper.encode_batch(sample)
    assert vae.call[1:] == (sample["video"].device, False)
    assert vae.call[0].dtype == torch.bfloat16
    assert encoded["video"].dtype == torch.bfloat16
    for name in ("context_mask", "action_is_pad", "image_is_pad"):
        assert torch.equal(encoded[name], sample[name])
    assert torch.equal(encoded["proprio"], sample["proprio"][:, 0].bfloat16())
