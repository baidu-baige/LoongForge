import os
import random
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from moviepy import VideoFileClip

from ....utils import download_from_huggingface
from ...pipeline import BasePipeline


class TransNetV2Pipeline(BasePipeline):
    """Shot boundary detection pipeline using TransNetV2."""

    def __init__(self, model_path: str) -> None:
        """Initialize TransNetV2 model from a checkpoint.

        Args:
            model_path: Local path where the TransNetV2 weights are stored.
        """
        self.download(model_path)
        state_dict = torch.load(model_path)
        self.model = TransNetV2()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = 'cpu'

    def download(self, file_path: str) -> None:
        """Download model weights if absent at file_path.

        Args:
            file_path: Path where model weights should be stored.
        """
        if os.path.exists(file_path):
            return
        file_name = os.path.basename(file_path)
        download_from_huggingface(
            repo_id='Sn4kehead/TransNetV2',
            filenames=[file_name],
            local_dir=os.path.dirname(file_path),
        )

    def to(self, device: str):
        """Move model to device and return self.

        Args:
            device: Device string (e.g., "cpu", "cuda", "cuda:0").

        Returns:
            TransNetV2Pipeline: Self for method chaining.
        """
        self.device = device
        self.model.to(device)
        return self

    def __call__(
        self,
        video_path: str,
        video_length: Optional[int] = None,
        video_mode: str = 'cv2',
        dst_size: Tuple[int, int] = (48, 27),
    ) -> list:
        """Predict scene boundaries for a given video.

        Args:
            video_path: Path to the input video file.
            video_length: Optional number of frames to read; None for all.
            video_mode: Frame reader backend, one of {"cv2", "decord"}.
            dst_size: Frame size (width, height) to which frames are resized.

        Returns:
            list: A list of [start, end] frame indices for scenes. In rare
                cases where all predictions are 1, returns a flat list
                [0, num_frames - 1].
        """
        if video_mode == 'cv2':
            from giga_datasets import VideoReaderCV2 as VideoReader
        elif video_mode == 'decord':
            from giga_datasets import VideoReaderDecord as VideoReader
        else:
            assert False
        video = VideoReader(video_path, video_length, dst_size, queue_size=100)
        predictions = self.predict_video(video)
        scenes = self.predictions_to_scenes(predictions)
        video.close()
        return scenes

    def get_frames(self, video: Any, buffers: list) -> Tuple[Optional[list], list, bool]:
        """Read and buffer frames for sliding window processing.

        Args:
            video: Video reader object with read() method.
            buffers: List of previously read frames for temporal context.

        Returns:
            Tuple[Optional[list], list, bool]: (frames, updated_buffers, is_end).
                frames is None if video ends before reading 25 initial frames.
        """
        is_end = False
        if len(buffers) == 0:
            for _ in range(25):
                frame = video.read()
                if frame is None:
                    is_end = True
                    break
                buffers.append(frame)
            if is_end:
                return None, buffers, is_end
            for _ in range(25):
                buffers.insert(0, buffers[0])
        frames = []
        for _ in range(50):
            frame = video.read()
            if frame is None:
                is_end = True
                break
            frames.append(frame)
        frames, buffers = buffers + frames, frames
        return frames, buffers, is_end

    def predict_video(self, video: Any) -> np.ndarray:
        """Predict shot boundaries for entire video using sliding window.

        Args:
            video: Video reader object.

        Returns:
            np.ndarray: 1D array of boundary probabilities per frame, or empty if video too short.
        """
        predictions = []
        buffers = []
        is_end = False
        while not is_end:
            frames, buffers, is_end = self.get_frames(video, buffers)
            if frames is None:
                return []
            pred = self.forward_model(frames)
            predictions.append(pred)
        if len(buffers) > 25:
            pred = self.forward_model(buffers)
            predictions.append(pred)
        predictions = np.concatenate(predictions)
        assert len(predictions) == video.cur_frame_idx + 1
        return predictions

    def forward_model(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Run model inference on a batch of frames.

        Args:
            frames: List of frames (numpy arrays) with length in (25, 100].

        Returns:
            np.ndarray: Boundary probabilities for valid frames (excluding padding).
        """
        frame_size = len(frames)
        assert 25 < frame_size <= 100
        for i in range(frame_size, 100):
            frames.append(frames[-1])
        frames = torch.from_numpy(np.stack(frames)).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(frames[None])
        pred = pred[0, 25 : min(frame_size, 75), 0]
        pred = torch.sigmoid(pred).cpu().numpy()
        return pred

    def predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.5) -> list:
        """Convert per-frame probabilities to scene [start, end] ranges.

        Args:
            predictions: 1D float array of boundary probabilities.
            threshold: Decision threshold to binarize probabilities.

        Returns:
            list: List of [start, end] indices; or [0, N-1] if all ones.
        """
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])
        # just fix if all predictions are 1
        if len(scenes) == 0:
            return [0, len(predictions) - 1]
        return scenes

    def scenes_to_videos(self, video_path: str, scenes: list, save_dir: str) -> None:
        """Split video into separate files based on detected scenes.

        Args:
            video_path: Path to input video file.
            scenes: List of [start_frame, end_frame] pairs.
            save_dir: Directory where scene clips will be saved.
        """
        video = VideoFileClip(video_path)
        ext = video_path.split('.')[-1]
        os.makedirs(save_dir, exist_ok=True)
        for start, end in scenes:
            cropped_video = video.subclipped(start / video.fps, end / video.fps)
            output_path = os.path.join(save_dir, f'{start}_{end}.{ext}')
            cropped_video.write_videofile(output_path)
            cropped_video.close()


class TransNetV2(nn.Module):
    """TransNetV2 model for shot boundary detection."""

    def __init__(
        self,
        F: int = 16,
        L: int = 3,
        S: int = 2,
        D: int = 1024,
        use_many_hot_targets: bool = True,
        use_frame_similarity: bool = True,
        use_color_histograms: bool = True,
        use_mean_pooling: bool = False,
        dropout_rate: float = 0.5,
        use_convex_comb_reg: bool = False,
        use_resnet_features: bool = False,
        use_resnet_like_top: bool = False,
        frame_similarity_on_last_layer: bool = False,
    ):
        """Initialize TransNetV2 model.

        Args:
            F: Base number of filters.
            L: Number of DDCNN layers.
            S: Number of blocks per DDCNN layer.
            D: Dimension of fully connected layer.
            use_many_hot_targets: Whether to use many-hot targets for training.
            use_frame_similarity: Whether to use frame similarity features.
            use_color_histograms: Whether to use color histogram features.
            use_mean_pooling: Whether to use mean pooling instead of flatten.
            dropout_rate: Dropout probability.
            use_convex_comb_reg: Not supported in PyTorch version.
            use_resnet_features: Not supported in PyTorch version.
            use_resnet_like_top: Not supported in PyTorch version.
            frame_similarity_on_last_layer: Not supported in PyTorch version.
        """
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplementedError('Some options not implemented in Pytorch version of Transnet!')

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)]
            + [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i) for i in range(1, L)]
        )

        self.frame_sim_layer = (
            FrameSimilarity(
                sum([(F * 2**i) * 4 for i in range(L)]),
                lookup_window=101,
                output_dim=128,
                similarity_dim=128,
                use_bias=True,
            )
            if use_frame_similarity
            else None
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for TransNetV2.

        Args:
            inputs: Batch of video frames [B, T, H, W, C] as uint8 tensors.

        Returns:
            torch.Tensor: Shot boundary predictions [B, T, 1].
        """
        assert (
            isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8
        ), 'incorrect input type and/or shape'
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {'many_hot': self.cls_layer2(x)}

        return one_hot


class StackedDDCNNV2(nn.Module):
    """Stacked Dilated Dense CNN V2 block."""

    def __init__(
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        shortcut: bool = True,
        use_octave_conv: bool = False,
        pool_type: str = 'avg',
        stochastic_depth_drop_prob: float = 0.0,
    ):
        """Initialize Stacked DDCNN V2.

        Args:
            in_filters: Number of input channels.
            n_blocks: Number of DDCNN blocks.
            filters: Base number of filters.
            shortcut: Whether to use residual connections.
            use_octave_conv: Not supported in PyTorch version.
            pool_type: Pooling type, either "avg" or "max".
            stochastic_depth_drop_prob: Probability of dropping paths in stochastic depth.
        """
        super(StackedDDCNNV2, self).__init__()

        if use_octave_conv:
            raise NotImplementedError('Octave convolution not implemented in Pytorch version of Transnet!')

        assert pool_type == 'max' or pool_type == 'avg'
        if use_octave_conv and pool_type == 'max':
            print('WARN: Octave convolution was designed with average pooling, not max pooling.')

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    octave_conv=use_octave_conv,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == 'max' else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked DDCNN blocks.

        Args:
            inputs: Input tensor [B, C, T, H, W].

        Returns:
            torch.Tensor: Output after DDCNN blocks and pooling.
        """
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    """Dilated Dense CNN V2 block with multiple dilation rates."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        batch_norm: bool = True,
        activation: Optional[Any] = None,
        octave_conv: bool = False,
    ):
        """Initialize Dilated DCNN V2.

        Args:
            in_filters: Number of input channels.
            filters: Number of filters per dilation rate.
            batch_norm: Whether to use batch normalization.
            activation: Activation function to apply.
            octave_conv: Not supported in PyTorch version.
        """
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplementedError('Octave convolution not implemented in Pytorch version of Transnet!')

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with parallel dilated convolutions.

        Args:
            inputs: Input tensor [B, C, T, H, W].

        Returns:
            torch.Tensor: Concatenated outputs from all dilation rates.
        """
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    """Configurable 3D convolution with separable and dilation options."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        dilation_rate: int,
        separable: bool = True,
        octave: bool = False,
        use_bias: bool = True,
        kernel_initializer: Optional[Any] = None,
    ):
        """Initialize configurable 3D convolution.

        Args:
            in_filters: Number of input channels.
            filters: Number of output channels.
            dilation_rate: Dilation rate for temporal dimension.
            separable: Whether to use (2+1)D separable convolution.
            octave: Not supported in PyTorch version.
            use_bias: Whether to use bias in convolution.
            kernel_initializer: Not supported in PyTorch version.
        """
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplementedError('Octave convolution not implemented in Pytorch version of Transnet!')
        if kernel_initializer is not None:
            raise NotImplementedError('Kernel initializers are not implemented in Pytorch version of Transnet!')

        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D convolution layers.

        Args:
            inputs: Input tensor [B, C, T, H, W].

        Returns:
            torch.Tensor: Output tensor after convolution.
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    """Frame similarity module computing temporal similarity features."""

    def __init__(
        self,
        in_filters: int,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
        stop_gradient: bool = False,
        use_bias: bool = False,
    ):
        """Initialize Frame Similarity module.

        Args:
            in_filters: Number of input channels.
            similarity_dim: Dimension for similarity projection.
            lookup_window: Size of temporal lookup window (must be odd).
            output_dim: Output feature dimension.
            stop_gradient: Not supported in PyTorch version.
            use_bias: Whether to use bias in projection layer.
        """
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplementedError('Stop gradient not implemented in Pytorch version of Transnet!')

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, '`lookup_window` must be odd integer'

    def forward(self, inputs: list) -> torch.Tensor:
        """Compute frame similarity features.

        Args:
            inputs: List of feature tensors from different blocks.

        Returns:
            torch.Tensor: Frame similarity features [B, T, output_dim].
        """
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat([1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat([batch_size, 1, self.lookup_window])
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat([batch_size, time_window, 1]) + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    """Color histogram module computing color-based similarity features."""

    def __init__(self, lookup_window: int = 101, output_dim: Optional[int] = None):
        """Initialize Color Histograms module.

        Args:
            lookup_window: Size of temporal lookup window (must be odd).
            output_dim: Output feature dimension; None to return raw similarities.
        """
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, '`lookup_window` must be odd integer'

    @staticmethod
    def compute_color_histograms(frames: torch.Tensor) -> torch.Tensor:
        """Compute normalized color histograms for video frames.

        Args:
            frames: Video frames [B, T, H, W, C] as uint8.

        Returns:
            torch.Tensor: Normalized histograms [B, T, 512].
        """
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute color histogram similarity features.

        Args:
            inputs: Input video frames [B, T, H, W, C].

        Returns:
            torch.Tensor: Color histogram features [B, T, output_dim] or [B, T, lookup_window].
        """
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat([1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat([batch_size, 1, self.lookup_window])
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat([batch_size, time_window, 1]) + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities
