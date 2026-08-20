# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the GLM-5.2 MoonViT placeholder format."""

import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import loongforge.train  # noqa: F401 - initialize package imports in training order
import torch
from transformers.processing_utils import ProcessorMixin

from loongforge.data.chat_template import MAPPING_NAME_TO_TEMPLATE
from loongforge.data.kimi_k25_plugin import KimiK25Plugin
from loongforge.data.multimodal import dataloader_provider
from loongforge.data.multimodal.base.task_encoder import BaseTaskEncoder
from loongforge.data.multimodal.kimi_task_encoder import KimiVLMTaskEncoder
from loongforge.data.multimodal.vlm_task_encoder import VLMTaskEncoder
from loongforge.models.omni_models.omni_encoder_model import OmniEncoderModel
from loongforge.utils import constants


class Glm52KimiVitPluginTest(unittest.TestCase):
    """Verify MoonViT features map directly to GLM image tokens."""

    def test_kimi_template_keeps_default_image_markers(self):
        """Existing Kimi templates must retain their original image format."""
        plugin = MAPPING_NAME_TO_TEMPLATE["kimi-k2.6"].mm_plugin

        self.assertEqual(
            plugin._build_image_placeholder(2),
            "<|media_begin|>image<|media_content|><|media_content|><|media_end|>",
        )

    def test_glm_template_expands_glm_image_span(self):
        """The GLM template must wrap MoonViT features in GLM image markers."""
        plugin = MAPPING_NAME_TO_TEMPLATE["glm5.2-hf"].mm_plugin
        self.assertIsInstance(plugin, KimiK25Plugin)

        mm_inputs = {
            "image_grid_thw": [[1, 4, 6]],
            "pixel_values": object(),
        }
        with patch.object(plugin, "_get_mm_inputs", return_value=mm_inputs):
            messages, _ = plugin.process_messages(
                [{"role": "user", "content": "look <image>"}],
                images=[object()],
                videos=[],
                processor=object(),
            )

        self.assertEqual(
            messages[0]["content"],
            "look <|begin_of_image|>" + "<|image|>" * 6 + "<|end_of_image|>",
        )

    def test_vqa_uses_glm_template_with_kimi_media_processing(self):
        """Kimi pixels and GLM text tokens must share the normal SFT path."""
        encoder = object.__new__(KimiVLMTaskEncoder)
        encoder.chat_template = MAPPING_NAME_TO_TEMPLATE["glm5.2-hf"]
        processed = tuple(object() for _ in range(7))
        encoder.process_sft_qa = Mock(return_value=processed)
        image = object()

        result = encoder.process_sft_vqa("look <image>", "answer", image)

        self.assertEqual(result, processed[:5])
        encoder.process_sft_qa.assert_called_once_with(
            [
                {
                    "role": constants.DataRoles.USER,
                    "content": "look <image>",
                },
                {
                    "role": constants.DataRoles.ASSISTANT,
                    "content": "answer",
                },
            ],
            "",
            None,
            [image],
        )

    def test_vqa_adds_missing_image_placeholder(self):
        """The legacy VQA input contract adds one placeholder for its image."""
        encoder = object.__new__(KimiVLMTaskEncoder)
        encoder.chat_template = MAPPING_NAME_TO_TEMPLATE["glm5.2-hf"]
        encoder.process_sft_qa = Mock(return_value=tuple(object() for _ in range(7)))
        image = object()

        encoder.process_sft_vqa("look", "answer", image)

        messages = encoder.process_sft_qa.call_args.args[0]
        self.assertEqual(messages[0]["content"], "<image>look")


class VLMTaskEncoderCompatibilityTest(unittest.TestCase):
    """Check shared VLM changes keep legacy and registered HF paths isolated."""

    def test_registered_hf_templates_use_hf_message_encoding(self):
        encoder = object.__new__(VLMTaskEncoder)
        encoder.args = SimpleNamespace(
            train_on_prompt=False,
            history_mask_loss=False,
        )
        encoder.tokenizer = object()

        for template_name in ("kimi-k2.6-hf", "qwen2.5-hf", "glm5.2-hf"):
            with self.subTest(template=template_name):
                encoder.chat_template = MAPPING_NAME_TO_TEMPLATE[template_name]
                with patch.object(
                    encoder.chat_template,
                    "encode_openai",
                    return_value=([1], [2], None, None),
                ) as encode_openai:
                    input_ids, target = encoder._encode_sft_messages(
                        [{"role": constants.DataRoles.USER, "content": "question"}],
                        system="system",
                    )

                self.assertEqual(input_ids.tolist(), [1])
                self.assertEqual(target.tolist(), [2])
                self.assertEqual(
                    encode_openai.call_args.kwargs["messages"][0],
                    {"role": constants.DataRoles.SYSTEM, "content": "system"},
                )

    def test_legacy_template_encoding_is_unchanged(self):
        encoder = object.__new__(VLMTaskEncoder)
        encoder.chat_template = Mock()
        encoder.tokenizer = object()
        encoder.chat_template.encode_multiturn.return_value = [
            ([1], [2]),
            ([3], [4]),
        ]

        input_ids, target = encoder._encode_sft_messages(
            [{"role": constants.DataRoles.USER, "content": "question"}],
            system="system",
        )

        self.assertEqual(input_ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(target.tolist(), [-100, 2, -100, 4])

    def test_processor_repr_patch_is_only_used_for_separate_path(self):
        common_args = dict(
            training_phase="pretrain",
            hf_tokenizer_path="tokenizer",
            image_resolution=None,
            frame_min_pixels=None,
            frame_max_pixels=None,
            video_max_pixels=None,
            fps=None,
            fps_min_frames=None,
            fps_max_frames=None,
            min_pixels=None,
            max_pixels=None,
        )
        original_repr = ProcessorMixin.__repr__
        for processor_path, expected_path, expected_patched in (
            (None, "tokenizer", False),
            ("processor", "processor", True),
        ):
            with self.subTest(processor_path=processor_path):
                args = SimpleNamespace(
                    **common_args,
                    hf_processor_path=processor_path,
                )
                repr_was_patched = []

                def load_processor(path, **kwargs):
                    self.assertEqual(path, expected_path)
                    self.assertTrue(kwargs["trust_remote_code"])
                    repr_was_patched.append(
                        ProcessorMixin.__repr__ is object.__repr__
                    )
                    return Mock()

                with patch.object(
                    BaseTaskEncoder,
                    "__init__",
                    lambda encoder: setattr(encoder, "args", args),
                ), patch(
                    "loongforge.data.multimodal.vlm_task_encoder.AutoProcessor.from_pretrained",
                    side_effect=load_processor,
                ):
                    VLMTaskEncoder(args)
                self.assertEqual(repr_was_patched, [expected_patched])
                self.assertIs(ProcessorMixin.__repr__, original_repr)

class VLMValidationDatasetTest(unittest.TestCase):
    """Check validation data path selection for Energon datasets."""

    @staticmethod
    def _args(data_path, valid_data_path=None):
        return SimpleNamespace(
            data_path=data_path,
            valid_data_path=valid_data_path,
            micro_batch_size=1,
            num_workers=0,
            packing_buffer_size=0,
        )

    def _get_val_dataset(self, args):
        dataset = object()
        with patch.object(
            dataloader_provider, "get_args", return_value=args
        ), patch.object(
            dataloader_provider.parallel_state,
            "get_data_parallel_rank",
            return_value=0,
        ), patch.object(
            dataloader_provider.parallel_state,
            "get_data_parallel_world_size",
            return_value=1,
        ), patch.object(
            dataloader_provider.parallel_state,
            "get_data_parallel_group",
            return_value=None,
        ), patch.object(
            dataloader_provider.energon, "WorkerConfig", return_value=object()
        ), patch.object(
            dataloader_provider.energon, "get_val_dataset", return_value=dataset
        ) as get_val_dataset:
            result = dataloader_provider.get_val_dataset(Mock())
        return result, dataset, get_val_dataset

    def test_explicit_validation_path_is_used(self):
        result, dataset, get_val_dataset = self._get_val_dataset(
            self._args(["train"], ["valid"])
        )

        self.assertIs(result, dataset)
        self.assertEqual(get_val_dataset.call_args.args[0], "valid")
        self.assertEqual(get_val_dataset.call_args.kwargs["split_part"], "val")

    def test_single_training_path_without_validation_path_is_skipped(self):
        result, _, get_val_dataset = self._get_val_dataset(self._args(["train"]))

        self.assertIsNone(result)
        get_val_dataset.assert_not_called()

    def test_multiple_validation_paths_use_metadataset(self):
        args = self._args(["train"], ["0.6", "valid-a", "0.4", "valid-b"])
        with patch.object(
            dataloader_provider,
            "create_metadataset_yaml",
            return_value="validation-metadataset.yaml",
        ) as create_metadataset_yaml:
            result, dataset, get_val_dataset = self._get_val_dataset(args)

        self.assertIs(result, dataset)
        create_metadataset_yaml.assert_called_once_with(
            ["valid-a", "valid-b"], [0.6, 0.4], split="val"
        )
        self.assertEqual(
            get_val_dataset.call_args.args[0], "validation-metadataset.yaml"
        )


class MetaDatasetYamlTest(unittest.TestCase):
    def test_train_and_validation_files_are_split_specific(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            dataloader_provider.tempfile, "gettempdir", return_value=temp_dir
        ):
            train_path = dataloader_provider.create_metadataset_yaml(
                ["train-a"], [1.0], split="train"
            )
            val_path = dataloader_provider.create_metadataset_yaml(
                ["val-a"], [1.0], split="val"
            )
            self.assertNotEqual(train_path, val_path)
            with open(train_path) as train_file, open(val_path) as val_file:
                self.assertIn("train:", train_file.read())
                self.assertIn("val:", val_file.read())


class OmniImageFeatureValidationTest(unittest.TestCase):
    """Check the image token invariant shared by all Omni VLMs."""

    @staticmethod
    def _encoder_with_features(feature_count):
        encoder = object.__new__(OmniEncoderModel)
        encoder.image_encoder = Mock()
        encoder.image_encoder.config = SimpleNamespace(image_token_id=99)
        encoder.image_encoder.return_value = (
            torch.ones(feature_count, 4),
            None,
            None,
        )
        encoder.image_projector = None
        return encoder

    @patch(
        "loongforge.models.omni_models.omni_encoder_model.get_args",
        return_value=SimpleNamespace(use_vit_dp_balance=False),
    )
    def test_matching_image_tokens_and_features_are_accepted(self, _):
        encoder = self._encoder_with_features(2)
        input_ids = torch.tensor([[99, 1, 99]])
        input_embeds = torch.zeros(3, 1, 4)

        combined, mask, _ = encoder.image_forward(input_ids, input_embeds)

        self.assertEqual(combined.shape, input_embeds.shape)
        self.assertEqual(mask.sum().item(), 8)

    @patch(
        "loongforge.models.omni_models.omni_encoder_model.get_args",
        return_value=SimpleNamespace(use_vit_dp_balance=False),
    )
    def test_mismatched_image_tokens_and_features_are_rejected(self, _):
        encoder = self._encoder_with_features(1)
        input_ids = torch.tensor([[99, 1, 99]])
        input_embeds = torch.zeros(3, 1, 4)

        with self.assertRaisesRegex(ValueError, "Image features 1 != image tokens 2"):
            encoder.image_forward(input_ids, input_embeds)


if __name__ == "__main__":
    unittest.main()
