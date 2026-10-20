import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .. import utils
from .base_dataset import BaseDataset
from .dataset import register_dataset


@register_dataset
class PklDataset(BaseDataset):
    """Dataset backed by a pickle list file ``data.pkl``."""

    def __init__(self, data_size: int | None, **kwargs: Any) -> None:
        super(PklDataset, self).__init__(**kwargs)
        self.data_size = data_size
        self.data_list: list | None = None

    @classmethod
    def load(cls, data_or_config: str | dict) -> 'PklDataset':
        from .dataset import load_config

        config = load_config(data_or_config)
        config_path = config.get('config_path', None)
        data_path = config.get('data_path', None)
        data_size = config['data_size']
        return cls(config_path=config_path, data_path=data_path, data_size=data_size)

    def save(self, save_path: str, copy_data: bool = False, store_rel_path: bool = True) -> None:
        from .dataset import get_rel_path

        if save_path.endswith('.json'):
            save_config_path = save_path
            save_path = os.path.dirname(save_config_path)
        else:
            save_config_path = os.path.join(save_path, 'config.json')
        config = utils.load_file(self.config_path)
        if copy_data:
            if self.data_list is None:
                os.makedirs(save_path, exist_ok=True)
                for data_file in Path(self.data_path).glob('*.pkl'):
                    shutil.copy2(data_file, save_path)
            else:
                assert self.data_size == len(self.data_list)
                config['data_size'] = self.data_size
                data_path = os.path.join(save_path, 'data.pkl')
                utils.save_file(data_path, self.data_list)
        else:
            if self.data_size is not None:
                assert config['data_size'] == self.data_size
            if store_rel_path:
                config['data_path'] = get_rel_path(self.data_path)
            else:
                config['data_path'] = self.data_path
        utils.save_file(save_config_path, config)

    def open(self) -> None:
        if self.data_list is None:
            # Load the list of per-sample dicts from a single pickle file
            data_path = os.path.join(self.data_path, 'data.pkl')
            self.data_list = utils.load_file(data_path)
            if self.data_size is not None:
                assert self.data_size == len(self.data_list)
            else:
                self.data_size = len(self.data_list)

    def close(self) -> None:
        if self.data_list is not None:
            self.data_list.clear()
            self.data_list = None
        super(PklDataset, self).close()

    def filter(self, mode: str, dataset_index: int = 0, **kwargs: Any) -> None:
        self.open()
        if mode == 'index':
            # Slice by index range/stride
            start = kwargs.get('start', 0)
            end = kwargs.get('end', None)
            step = kwargs.get('step', 1)
            self.data_list = self.data_list[start:end:step]
        elif mode == 'func':
            func = kwargs.pop('func')
            if isinstance(func, str):
                # Dynamically import custom filter function
                func = utils.import_function(func)
            self.data_list = func(self.data_list, dataset_index=dataset_index, **kwargs)
        else:
            assert False
        logging.info(f'filter dataset {dataset_index} from {self.data_size} to {len(self.data_list)}')
        self.data_size = len(self.data_list)

    def __len__(self) -> int:
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index: int) -> dict:
        return copy.deepcopy(self.data_list[index])


class PklWriter:
    """Writer helper for ``PklDataset``."""

    def __init__(self, data_path: str) -> None:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        self.data_path = data_path
        self.key_names: list[str] = []
        self.data_list: list[dict] = []

    def close(self) -> None:
        if len(self.data_list) > 0:
            # Persist accumulated list to data.pkl
            data_path = os.path.join(self.data_path, 'data.pkl')
            utils.save_file(data_path, self.data_list)
            self.data_list = []

    def write_dict(self, data: dict) -> None:
        assert isinstance(data, dict)
        # Track union of keys across all samples for schema recording
        self.key_names = list(set(self.key_names + list(data.keys())))
        self.data_list.append(data)

    def write_config(self, **kwargs: Any) -> None:
        config_path = os.path.join(self.data_path, 'config.json')
        self.key_names.sort()
        config = {
            '_class_name': 'PklDataset',
            '_key_names': self.key_names,
            'data_size': len(self.data_list),
        }
        config.update(kwargs)
        utils.save_file(config_path, config)
