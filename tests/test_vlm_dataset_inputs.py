# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import ast
import base64
import binascii
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_functions(relative_path, names, namespace=None):
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {function.name for function in functions} == set(names)

    namespace = {} if namespace is None else namespace
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class VlmDatasetInputTest(unittest.TestCase):
    def test_raw_json_is_rejected_before_energon(self):
        namespace = _load_functions(
            "loongforge/data/multimodal/dataloader_provider.py",
            {"_validate_energon_data_paths"},
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Energon WebDataset.*convert_to_webdataset\.py",
        ):
            namespace["_validate_energon_data_paths"](["train.JSONL"])

        namespace["_validate_energon_data_paths"](["train_wds", "blend.yaml"])

    def test_multi_mix_qa_decodes_image_data_uri(self):
        namespace = self._load_converter_functions()
        payload = b"embedded-image"
        uri = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        args = SimpleNamespace(
            image_dir=None,
            video_dir=None,
            columns_messages="messages",
            sample_type="multi_mix_qa",
        )

        sample = namespace["construct_sample"](
            args,
            "image",
            [uri],
            3,
            {"messages": [{"role": "user", "content": "look"}]},
        )

        self.assertEqual(sample["0_image.png"], payload)
        self.assertEqual(
            json.loads(sample["json"]),
            {
                "texts": [{"role": "user", "content": "look"}],
                "media": "image",
                "name": ["0_image.png"],
            },
        )

    def test_chat_mix_writes_messages_and_tools(self):
        namespace = self._load_converter_functions()
        payload = base64.b64encode(b"embedded-image").decode("ascii")
        args = SimpleNamespace(
            image_dir=None,
            video_dir=None,
            columns_messages="messages",
            sample_type="chat_mix",
        )
        messages = [{"role": "user", "content": "look"}]
        tools = [{"type": "function", "function": {"name": "inspect"}}]

        sample = namespace["construct_sample"](
            args,
            "image",
            [f"data:image/jpeg;base64,{payload}"],
            4,
            {"messages": messages, "tools": tools},
        )
        content = json.loads(sample["json"])

        self.assertEqual(content["messages"], messages)
        self.assertEqual(content["tools"], tools)
        self.assertNotIn("texts", content)

    @staticmethod
    def _load_converter_functions():
        return _load_functions(
            "tools/data_preprocess/vlm/convert_to_webdataset.py",
            {"_read_media", "_build_content", "construct_sample"},
            {
                "base64": base64,
                "binascii": binascii,
                "json": json,
                "os": os,
            },
        )


if __name__ == "__main__":
    unittest.main()
