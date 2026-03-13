
"""latent dataset"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from megatron.training import get_args
from transformers import AutoProcessor

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, steps_per_epoch=0):
        args = get_args()
        self.metadata = []
        self.load_metadata(metadata_path)
        base_path = Path(args.data_path[0])
        if args.model_name == "wan2_2_i2v":
            self.path = [base_path / data["video"] for data in self.metadata]

        self.path = [
            p.with_suffix(p.suffix + ".tensors.pth")
            for p in self.path
            if (p.with_suffix(p.suffix + ".tensors.pth")).exists()
        ]
        self.steps_per_epoch = steps_per_epoch
        print(
            f"self.steps_per_epoch: {self.steps_per_epoch}, total_samples: {len(self.metadata)}"
        )
        assert len(self.path) > 0
        self.manual_seed = args.seed

    def load_metadata(self, metadata_path):
        """load metadata from different types of files"""
        if metadata_path is None:
            print("No metadata_path. Please provide metadata_path.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.metadata = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.metadata = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.metadata = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, index):
        seed = (self.manual_seed + index) % 2**32
        numpy_random_state = np.random.RandomState(seed=seed)
        data_id = numpy_random_state.randint(0, self.steps_per_epoch)
        data_id = data_id % len(self.path)
        # data_id = 0
        path = self.path[data_id]
        data = torch.load(path, weights_only=False, map_location="cpu")
        # used for generate timestep
        data["seed"] = seed
        return data

    def __len__(self):
        return self.steps_per_epoch


class ErnieImageDataset(torch.utils.data.Dataset):
    """Dataset for ernie-vl"""
    def __init__(self, args, metadata_path, steps_per_epoch=0):
        self.manual_seed = args.seed
        self.steps_per_epoch = steps_per_epoch
        self.processor = AutoProcessor.from_pretrained(args.hf_tokenizer_path,  trust_remote_code=True)
        self.file_names = []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.file_names.append(line.strip())

    def __getitem__(self, index):
        seed = (self.manual_seed + index) % 2**32
        numpy_random_state = np.random.RandomState(seed=seed)
        data_id = numpy_random_state.randint(0, self.steps_per_epoch)
        data_id = data_id % len(self.file_names)
        data_name = self.file_names[data_id]
        data = np.load(data_name)
        data_item = {
            "images": torch.from_numpy(data["images"]),
            "input_ids": torch.from_numpy(data["input_ids"]),
            "token_type_ids": torch.from_numpy(data["token_type_ids"])[:, :-1],
            "position_ids": torch.from_numpy(data["position_ids"]),
            "grid_thw": torch.from_numpy(data["grid_thw"]),
            "image_type_ids": torch.from_numpy(data["image_type_ids"]),
            "labels": torch.from_numpy(data["labels"])
        }
        return data_item

    def __len__(self):
        return self.steps_per_epoch