import logging
from typing import Any, Iterable

import numpy as np
from skimage.metrics import peak_signal_noise_ratio


class PSNREvaluator:
    """Compute average PSNR between reference images and results."""

    def __init__(self, dataset) -> None:
        """Initialize PSNR evaluator.

        Args:
            dataset: Reference dataset mapping index -> dict with key 'image'.
        """
        self.dataset = dataset

    def __call__(self, results: Iterable[dict[str, Any]]) -> dict[str, float]:
        psnr_list = []
        for i, result in enumerate(results):
            data_index = result.get('data_index', i)
            image_true = self.dataset[data_index]['image']
            image_test = result['image']
            if image_test.size != image_true.size:
                image_true = image_true.resize(image_test.size)
            psnr = calculate_psnr(image_true, image_test)
            psnr_list.append(psnr)
        psnr = sum(psnr_list) / len(psnr_list)
        logging.info('PSNR: {}'.format(psnr))
        return dict(psnr=psnr)


def calculate_psnr(image_true: Any, image_test: Any) -> float:
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    psnr = peak_signal_noise_ratio(image_true, image_test, data_range=255)
    return psnr
