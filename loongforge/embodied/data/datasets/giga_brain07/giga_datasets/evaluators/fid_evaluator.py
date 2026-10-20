import logging
from typing import Any, Iterable

import numpy as np
import torch
import torchvision.transforms as TF
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


class FIDEvaluator:
    """Compute FID between reference dataset images and generated results."""

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 50, device: str = 'cuda', dims: int = 2048, num_workers: int = 0) -> None:
        """Initialize the FID evaluator.

        Args:
            dataset (torch.utils.data.Dataset): Reference dataset providing ground-truth images by index.
            batch_size (int): Mini-batch size for Inception activations.
            device (str): Torch device where the Inception model will run.
            dims (int): Feature dimensionality to use from InceptionV3.
            num_workers (int): DataLoader workers for feature extraction.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.dims = dims
        self.num_workers = num_workers

    def __call__(self, results: Iterable[dict[str, Any]]) -> dict[str, float]:
        """Compute FID between ground-truth images and provided results.

        The function aligns image sizes by resizing the dataset image to match
        the generated image resolution when needed.

        Args:
            results (Iterable[dict[str, Any]]): Generated results where each item contains
                an 'image' (PIL) and optional 'data_index' to align with the reference dataset.

        Returns:
            dict[str, float]: A dictionary containing the key 'fid'.
        """
        annos = []
        for i, result in enumerate(results):
            data_index = result.get('data_index', i)
            image_true = self.dataset[data_index]['image']
            image_test = result['image']
            if image_test.size != image_true.size:
                image_true = image_true.resize(image_test.size)
            annos.append({'image': image_true})
        fid_value = calculate_fid(
            [annos, results],
            batch_size=self.batch_size,
            device=self.device,
            dims=self.dims,
            num_workers=self.num_workers,
        )
        logging.info('FID: {}'.format(fid_value))
        return dict(fid=fid_value)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, data_list: list[dict[str, Any]], transforms=None) -> None:
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = self.data_list[i]['image']
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(data_list: list[dict[str, Any]], model: torch.nn.Module, batch_size: int, device: str, dims: int, num_workers: int) -> np.ndarray:
    """Calculate Inception activations (pool_3) for all images.

    Args:
        data_list: List of dicts with key 'image' as PIL.Image.
        model: InceptionV3 network wrapped to return requested feature maps.
        batch_size: Mini-batch size.
        device: Torch device.
        dims: Feature dimensionality (used to preallocate array).
        num_workers: DataLoader workers.

    Returns:
        np.ndarray: Array of shape (N, dims) with activations.
    """
    model.eval()

    if batch_size > len(data_list):
        print(('Warning: batch size is bigger than the data size. ' 'Setting batch size to data size'))
        batch_size = len(data_list)

    dataset = ImagePathDataset(data_list, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    pred_arr = np.empty((len(data_list), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(
    data_list: list[dict[str, Any]], model: torch.nn.Module, batch_size: int, device: str, dims: int, num_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance of activations used by the FID.

    Returns:
        tuple[np.ndarray, np.ndarray]: (mu, sigma)
    """
    act = get_activations(data_list, model, batch_size, device, dims, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid(data_list: list[list[dict[str, Any]]], batch_size: int, device: str, dims: int, num_workers: int) -> float:
    """Calculate the Frechet Inception Distance (FID).

    Args:
        data_list: A pair [reals, fakes], each a list of dicts with key 'image'.
        batch_size: Mini-batch size.
        device: Torch device.
        dims: Inception feature dimension (e.g., 2048).
        num_workers: DataLoader workers.

    Returns:
        float: The computed FID value.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics(data_list[0], model, batch_size, device, dims, num_workers)
    m2, s2 = calculate_activation_statistics(data_list[1], model, batch_size, device, dims, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
