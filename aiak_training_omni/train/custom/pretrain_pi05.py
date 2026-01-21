"""Standalone PI0.5 data pipeline demo aligned with `lerobot_train.py`.

This script focuses only on data pieces:
- dataset creation
- pre/post processors
- dataloader wiring (same shape/flags as the upstream trainer)

It is meant to sanity-check AIAK-Training-Omni's LeRobot data path without
spinning up the full training stack.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from aiak_training_omni.data.lerobot.lerobot_data_processor import (
	make_pi05_pre_post_processors,
)
from aiak_training_omni.data.lerobot.lerobot_dataset_builder import (
	build_lerobot_dataset,
	get_lerobot_dataset_stats,
)
from aiak_training_omni.data.lerobot.lerobot_dataset_config import LeRobotDatasetConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

DEFAULT_REPO_ID = "/Users/chen/Desktop/datasets/aloha_mobile_cabinet/"


def build_pi05_data_pipeline(
    repo_id: str,
    root: Optional[str] = None,
    episodes: Optional[list[int]] = None,
    revision: Optional[str] = None,
    use_imagenet_stats: bool = True,
    streaming: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
	"""Build dataset, processors, and dataloader mirroring `lerobot_train.py`.

	Returns (dataset, preprocessor, postprocessor, dataloader, policy_cfg).
	"""

	device = torch.device(device)

	ds_cfg = LeRobotDatasetConfig(
		repo_id=repo_id,
		root=root,
		episodes=episodes,
		revision=revision,
		use_imagenet_stats=use_imagenet_stats,
		streaming=streaming,
	)

	policy_cfg = PI05Config(device=device.type)

	dataset = build_lerobot_dataset(ds_cfg, policy=policy_cfg)
	dataset_stats = get_lerobot_dataset_stats(dataset)

	preprocessor, postprocessor = make_pi05_pre_post_processors(
		config=policy_cfg,
		dataset_stats=dataset_stats,
	)

	shuffle = True and not ds_cfg.streaming
	sampler = None

	dataloader = torch.utils.data.DataLoader(
		dataset,
		num_workers=num_workers,
		batch_size=batch_size,
		shuffle=shuffle,
		sampler=sampler,
		pin_memory=device.type == "cuda",
		drop_last=False,
		prefetch_factor=2 if num_workers > 0 else None,
	)

	return dataset, preprocessor, postprocessor, dataloader, policy_cfg





def demo():
    """
    该函数构建并测试PI0.5数据管道，包括数据集加载、预处理、后处理和数据迭代器的创建。

    Args:
        无参数，通过命令行参数接收配置：
        --repo_id: 数据集仓库ID或本地路径，默认指向本地Aloha样本
        --root: 可选的HF缓存/根目录覆盖
        --revision: 数据集版本
        --episodes: 逗号分隔的剧集索引列表
        --batch_size: 批处理大小，默认为2
        --num_workers: 数据加载工作进程数，默认为2
        --device: 目标设备，自动检测CUDA或使用CPU
        --streaming: 使用流式模式的标志

    Returns:
        无返回值，主要功能是打印数据集信息并验证管道流程
    """
    parser = argparse.ArgumentParser(description="PI0.5 data-only sanity check")
    parser.add_argument(
        "--repo_id",
        required=False,
        default=DEFAULT_REPO_ID,
        help="Dataset repo_id or local path (default points to local Aloha sample)",
    )
    parser.add_argument("--root", default=None, help="Optional HF cache/root override")
    parser.add_argument("--revision", default=None, help="Dataset revision")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for processors",
    )
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    episodes = None
    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",") if x.strip()]

    dataset, preprocessor, postprocessor, dataloader, policy_cfg = build_pi05_data_pipeline(
        repo_id=args.repo_id,
        root=args.root,
        episodes=episodes,
        revision=args.revision,
        use_imagenet_stats=True,
        streaming=args.streaming,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    print(f"Loaded dataset with {dataset.num_frames} frames across {dataset.num_episodes} episodes")
    print(f"Policy device: {policy_cfg.device}; batch_size={args.batch_size}; num_workers={args.num_workers}")

    dl_iter = iter(dataloader)
    batch = next(dl_iter)
    batch = preprocessor(batch)

    # # Print a compact summary of tensor keys/shapes to verify pipeline works
    # def fmt(obj):
    #     if isinstance(obj, torch.Tensor):
    #         return f"Tensor{tuple(obj.shape)} {obj.dtype} {obj.device}"
    #     return type(obj).__name__

    # summary = {k: fmt(v) for k, v in batch.items()}
    # print("Preprocessed batch keys → shapes/dtypes/devices:")
    # for k, v in summary.items():
    #     print(f"  {k}: {v}")

    # # Dummy postprocess of a zero action to ensure the postprocessor path is valid
    # dummy_action = {k: torch.zeros_like(v) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    # post_out = postprocessor(dummy_action)
    # print(f"Postprocessor ran; keys: {list(post_out.keys())}")


if __name__ == "__main__":
    demo()
