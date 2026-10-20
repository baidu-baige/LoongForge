import json
import os
from collections import OrderedDict
from typing import List

import torch.nn as nn
from peft import get_peft_model_state_dict, set_peft_model_state_dict


class LoRAPeftWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert hasattr(model, 'peft_config')
        self.model = model

    def to_fp8(self, **kwargs):
        self.model.to_fp8(**kwargs)

    def save_config(self, save_directory):
        if hasattr(self.model, 'fp8') and self.model.fp8:
            save_path = os.path.join(save_directory, 'config_fp8.json')
            json.dump(self.model.fp8_config, open(save_path, 'w'), indent=2)

    def state_dict(self, *args, destination=None, **kwargs):
        assert len(args) == 0
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        state_dict = self.model.state_dict(**kwargs)
        if hasattr(self.model, 'fp8') and self.model.fp8:
            for key, val in state_dict.items():
                if 'lora_' not in key and key.endswith('_extra_state'):
                    destination[key] = val
        state_dict = get_peft_model_state_dict(self.model, state_dict)
        destination.update(state_dict)
        return destination

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        incompatible_keys = set_peft_model_state_dict(self.model, state_dict)
        incompatible_keys.missing_keys.clear()
        error_msgs: List[str] = []
        if strict:
            if len(incompatible_keys.unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    'Unexpected key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in incompatible_keys.unexpected_keys)),
                )
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, '\n\t'.join(error_msgs)))
        return incompatible_keys

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
