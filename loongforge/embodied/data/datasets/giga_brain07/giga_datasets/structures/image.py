import torch

from .base_structure import BaseStructure


class Image(BaseStructure):
    """Image tensor wrapper with width/height helpers."""

    def __init__(self, tensor: torch.Tensor | list | tuple):
        """Wrap an image tensor with shape (H, W, C) or compatible input.

        Args:
            tensor: Image-like data convertible to a torch.Tensor of shape (H, W, C).
        """
        super(Image, self).__init__(tensor)
        assert self.tensor.ndim == 3
        self.tensor = self.tensor.to(torch.float32)

    @property
    def width(self) -> int:
        return self.tensor.shape[1]

    @property
    def height(self) -> int:
        return self.tensor.shape[0]
