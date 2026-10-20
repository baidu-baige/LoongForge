import logging
from typing import Any, Iterable

import lpips as _lpips
import numpy as np
import torch


class LPIPSEvaluator:
    """Compute LPIPS distance between reference images and results."""

    def __init__(self, dataset, net: str = 'vgg', device: str = 'cuda') -> None:
        """Initialize LPIPS evaluator.

        Args:
            dataset: Reference dataset mapping index -> dict with key 'image'.
            net (str): Backbone name for LPIPS ('vgg','alex','squeeze').
            device (str): Torch device string.
        """
        self.dataset = dataset
        self.net = net
        self.device = device
        self.lpips_loss = None

    def __call__(self, results: Iterable[dict[str, Any]]) -> dict[str, float]:
        if self.lpips_loss is None:
            self.lpips_loss = _lpips.LPIPS(net=self.net)
            self.lpips_loss.to(self.device)
        lpips_list = []
        for i, result in enumerate(results):
            data_index = result.get('data_index', i)
            image_true = self.dataset[data_index]['image']
            image_test = result['image']
            if image_test.size != image_true.size:
                image_true = image_true.resize(image_test.size)
            lpips = calculate_lpips(image_true, image_test, self.lpips_loss, self.device)
            lpips_list.append(lpips)
        lpips = sum(lpips_list) / len(lpips_list)
        logging.info('LPIPS: {}'.format(lpips))
        return dict(lpips=lpips)


def calculate_lpips(image_true: Any, image_test: Any, lpips_net: _lpips.LPIPS, device: str) -> float:
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    image_true = _lpips.im2tensor(image_true).to(device)
    image_test = _lpips.im2tensor(image_test).to(device)
    with torch.no_grad():
        lpips = lpips_net(image_true, image_test)
        lpips = lpips.reshape(-1).item()
    return lpips
