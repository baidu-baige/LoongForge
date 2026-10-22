import numpy as np
import torch

from .. import utils


class DefaultCollator:
    """Default collator that stacks common array/tensor fields.

    Args:
        is_equal: If True, enforce equal-sized tensors/arrays when stacking.
    """

    def __init__(self, is_equal: bool = False) -> None:
        """Initialize the default collator.

        Args:
            is_equal (bool): If True, enforce equal-sized tensors/arrays when stacking.
        """
        self.is_equal = is_equal

    def __call__(self, batch: dict | list[dict]) -> dict:
        """Collate a list/dict batch into a dict of stacked tensors/arrays.

        Args:
            batch (dict | list[dict]): A batch either as a list of sample dicts or already as a dict of lists.

        Returns:
            dict: A dict whose values are collated tensors/arrays following field-wise stacking.
        """
        batch_dict = dict()
        if isinstance(batch, list):
            for key in batch[0]:
                # Column-wise collate across list of dicts
                batch_dict[key] = self._collate([d[key] for d in batch])
        elif isinstance(batch, dict):
            for key in batch:
                # Already column-oriented: collate each value container
                batch_dict[key] = self._collate(batch[key])
        else:
            assert False
        return batch_dict

    def _collate(self, batch: list | tuple | np.ndarray | torch.Tensor | dict):
        """Recursively collate nested containers into tensors where possible.

        Args:
            batch: A nested container of tensors/arrays/lists/dicts to be collated.

        Returns:
            Collated structure with tensors/arrays stacked where applicable.
        """
        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], torch.Tensor):
                # Stack tensors directly, with optional equal-shape enforcement
                batch = utils.stack_data(batch, is_equal=self.is_equal)
            elif isinstance(batch[0], np.ndarray):
                batch = utils.stack_data(batch, is_equal=self.is_equal)
                batch = torch.from_numpy(batch)
            elif isinstance(batch[0], (np.bool_, np.number, np.object_)):
                # Scalars to tensor
                batch = torch.as_tensor(batch)
            elif isinstance(batch[0], dict):
                # Dict of lists -> dict of tensors (recursive)
                batch = {key: self._collate([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], (list, tuple)):
                # Transpose list-of-lists/tuples, then collate each field
                batch = type(batch[0])([self._collate(d) for d in zip(*batch)])
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return batch
