# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L0 几何：token 网格算术的唯一来源（Lane A）。

签名与不变量以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.1 为准，
字段清单以 ``mot_spec_config.zh-CN.md`` §3.3 为准（两者已对账一致）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["GeometrySpec"]


def _exact_div(numerator: int, denominator: int, *, what: str) -> int:
    """整除；不整除即构造期抛错，而不是静默向下取整。"""
    if denominator <= 0:
        raise ValueError(f"GeometrySpec.{what}: 除数必须为正，得到 {denominator}")
    if numerator % denominator != 0:
        raise ValueError(
            f"GeometrySpec.{what}: {numerator} 不能被 {denominator} 整除。"
            f"整条「像素 → VAE → latent → patch」链上的除法必须整除"
        )
    return numerator // denominator


@dataclass(frozen=True)
class GeometrySpec:
    """token 网格算术的唯一来源。

    职责：把「像素帧 → VAE 压缩 → latent 网格 → patch → 每帧 token 数 →
    序列长度 → 对齐 padding」这条链收成一个对象。除本类之外，
    任何地方不得再出现这条链上的算术表达式。

    不变量：
      - 所有派生量都是 property，不存字段，避免两份真相；
      - 整条链上的除法必须整除，不整除即构造期抛错；
      - 本类不含任何张量，可被安全地反复构造、可哈希、可进配置指纹。
    """

    frame_height: int                   # 像素帧高
    frame_width: int                    # 像素帧宽（与高分开：存在非正方形输入）
    num_frames: int                     # 视频帧数（时间维，压缩前）
    num_views: int = 1                  # 相机路数。VPP2 的 concat_multi_camera=horizontal ⇒ 2
    vae_spatial_stride: int = 16        # VAE 空间压缩比。与时间压缩比必须分开
    vae_temporal_stride: int = 4        # VAE 时间压缩比
    patch_size: tuple[int, int, int] = (1, 2, 2)   # (t, h, w)
    action_horizon: int = 0             # 动作预测步数。0 ⇒ 本模型无动作流
    action_video_freq_ratio: int = 1    # 每个视频 latent 帧对应几个动作步
    align_to: int = 1                   # 序列长度对齐粒度（归属待裁定，总纲 §10-1）

    def __post_init__(self) -> None:
        positive = (
            "frame_height", "frame_width", "num_frames", "num_views",
            "vae_spatial_stride", "vae_temporal_stride",
            "action_video_freq_ratio", "align_to",
        )
        for field_name in positive:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(
                    f"GeometrySpec.{field_name} 必须为正，得到 {value}"
                )
        if self.action_horizon < 0:
            raise ValueError(
                f"GeometrySpec.action_horizon 不得为负，得到 {self.action_horizon}"
            )
        if len(self.patch_size) != 3 or any(p <= 0 for p in self.patch_size):
            raise ValueError(
                f"GeometrySpec.patch_size 必须是三个正整数 (t, h, w)，"
                f"得到 {self.patch_size!r}"
            )
        # 触发整条链的整除校验：错误要在构造期暴露，不要等到第一次取 property
        _ = self.patch_grid
        _ = self.num_chunks

    # ---- 派生量：全部为 property，禁止在别处重算 ----

    @property
    def latent_grid(self) -> tuple[int, int, int]:
        """VAE 压缩后的 latent 网格 (t, h, w)。

        时间轴按**因果 VAE** 压缩：``1 + (num_frames - 1) / stride``。
        首帧自成一个 latent 帧，其后每 ``stride`` 帧压成一帧。现状五处
        各自写了同一个公式（``model/fastwam/mot/fastwam.py:873``、
        ``mot/idm.py:353``、``mot/joint.py:155``、``wan/core.py:358``、
        ``wan/vae.py:1141``），本属性是它们收敛后的唯一出处。
        直除会把 21 帧 / stride 4 判成非法，而那是 VPP2 与 EWAM 的实际取值
        （21 ⇒ 6 个 latent 帧）。``stride=1`` 时两式同值。

        ``num_views`` 作用在宽度上：多相机是沿宽度拼接的
        （VPP2 的 ``concat_multi_camera: horizontal``）。
        """
        return (
            1 + _exact_div(self.num_frames - 1, self.vae_temporal_stride,
                           what="(num_frames - 1) / vae_temporal_stride"),
            _exact_div(self.frame_height, self.vae_spatial_stride,
                       what="frame_height / vae_spatial_stride"),
            _exact_div(self.frame_width * self.num_views, self.vae_spatial_stride,
                       what="frame_width × num_views / vae_spatial_stride"),
        )

    @property
    def patch_grid(self) -> tuple[int, int, int]:
        """patch 化之后的网格 (t, h, w)，即每个轴上的 token 数。"""
        latent = self.latent_grid
        axis_names = ("t", "h", "w")
        return tuple(  # type: ignore[return-value]
            _exact_div(latent[i], self.patch_size[i],
                       what=f"latent_grid[{axis_names[i]}] / "
                            f"patch_size[{axis_names[i]}]")
            for i in range(3)
        )

    @property
    def tokens_per_frame(self) -> int:
        """单个 latent 帧的 token 数 = patch_grid[1] × patch_grid[2]。"""
        _, h, w = self.patch_grid
        return h * w

    @property
    def video_seq_len(self) -> int:
        """视频流的 token 总数（未对齐）= prod(patch_grid)。"""
        t, h, w = self.patch_grid
        return t * h * w

    @property
    def padded_seq_len(self) -> int:
        """按 align_to 向上对齐后的长度。pad 出来的位置在标签里标 pad=True。

        这是**静态形状的唯一来源**（总纲 INV-2）：CUDA graph 捕获要求
        序列长度在 step 之间不变，所以长度只能从本属性来，不能从数据算。
        """
        return math.ceil(self.video_seq_len / self.align_to) * self.align_to

    @property
    def num_chunks(self) -> int:
        """动作分块数 = action_horizon / action_video_freq_ratio。

        chunk 标签的唯一来源，交界面 I2 的内容之一。不整除即构造期抛错。
        """
        if self.action_horizon == 0:
            return 0
        return _exact_div(self.action_horizon, self.action_video_freq_ratio,
                          what="action_horizon / action_video_freq_ratio")

    def build_labels(self) -> dict[str, Tensor]:
        """产出本几何对应的 token 标签，交给 ``TokenLayout``。

        返回的键：
          frame  [L] int   每个 token 属于哪个 latent 帧
          row    [L] int   patch 行号（供空间局部性规则使用）
          col    [L] int   patch 列号
          pad    [L] bool  是否为对齐填充出来的位置

        注意：不产 stream / chunk / noise_level 这三个键 ——
        它们由各流自己贴（stream）、由动作流按 ``num_chunks`` 贴（chunk）、
        由 L3 的加噪方案贴（noise_level）。几何层只提供 chunk 的**个数**，
        不负责把它贴到 token 上。
        """
        _, h, w = self.patch_grid
        real = self.video_seq_len
        total = self.padded_seq_len

        index = torch.arange(real, dtype=torch.long)
        frame = torch.full((total,), -1, dtype=torch.long)
        row = torch.full((total,), -1, dtype=torch.long)
        col = torch.full((total,), -1, dtype=torch.long)
        pad = torch.ones(total, dtype=torch.bool)

        # 展平顺序 = (t, h, w) 的 C order，与 patchify 的 flatten 顺序一致
        frame[:real] = index // (h * w)
        row[:real] = (index // w) % h
        col[:real] = index % w
        pad[:real] = False
        return {"frame": frame, "row": row, "col": col, "pad": pad}
