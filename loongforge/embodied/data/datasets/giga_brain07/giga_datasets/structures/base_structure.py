import torch


class BaseStructure:
    """Lightweight tensor wrapper with convenience methods and immutability
    helpers.

    The structure wraps a tensor and preserves default parameters used to
    rebuild new instances after operations like ``to``/``clone``.
    """

    def __init__(self, tensor: torch.Tensor | list | tuple, **kwargs):
        """Construct a structure from a tensor-like input.

        Args:
            tensor: Backing tensor or array-like convertible via ``torch.as_tensor``.
            **kwargs: Default parameters carried to newly created structures
                (e.g., coordinate mode or offset), preserved across ops like ``to``/``clone``.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
        tensor = torch.as_tensor(tensor, device=device)
        self.tensor = tensor
        self.default_params = kwargs

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, *args, **kwargs):
        """Move/cast underlying tensor and return a new wrapped structure.

        Returns:
            Same structure type with updated tensor device/dtype, preserving defaults.
        """
        return type(self)(self.tensor.to(*args, **kwargs), **self.default_params)

    def clone(self):
        """Deep-copy tensor and return a new wrapped structure."""
        return type(self)(self.tensor.clone(), **self.default_params)

    def contiguous(self):
        """Return a contiguous copy wrapped in the same structure type."""
        return type(self)(self.tensor.contiguous(), **self.default_params)

    def cuda(self, device: int | str | torch.device):
        """Move tensor to CUDA on specified device and wrap it back."""
        return type(self)(self.tensor.cuda(device), **self.default_params)

    def cpu(self):
        """Move tensor to CPU and wrap it back."""
        return type(self)(self.tensor.contiguous().cpu(), **self.default_params)

    def numpy(self):
        """Return a contiguous CPU numpy array view of the tensor."""
        return self.tensor.contiguous().cpu().numpy()

    def new_structure(self, data: torch.Tensor | list | tuple):
        """Create a new structure of the same type from raw data.

        Args:
            data: Tensor or array-like to convert. If non-tensor, uses ``tensor.new_tensor``.

        Returns:
            Same structure type wrapping the new tensor.
        """
        if not isinstance(data, torch.Tensor):
            new_tensor = self.tensor.new_tensor(data)
        else:
            new_tensor = data.to(self.device)
        return type(self)(new_tensor, **self.default_params)

    def __getitem__(self, item):
        """Index/slice the underlying tensor and wrap the result.

        Notes:
            If result is 1D, it is reshaped to (1, -1) for consistency.
        """
        new_tensor = self.tensor[item]
        if new_tensor.ndim == 1:
            new_tensor = new_tensor.view(1, -1)
        return type(self)(new_tensor, **self.default_params)

    def __iter__(self):
        """Yield elements of the underlying tensor along the first
        dimension."""
        yield from self.tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'
