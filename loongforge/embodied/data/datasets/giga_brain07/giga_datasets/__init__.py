__version__ = '1.1.0'


def _preload_torchcodec_decoder_before_decord() -> None:
    try:
        # Decord can load FFmpeg symbols before torchcodec and break AV1 stream
        # detection in torchcodec. Preloading keeps torchcodec usable when this
        # package later imports decord-backed helpers.
        from torchcodec.decoders import VideoDecoder as _VideoDecoder  # noqa: F401
    except Exception:
        return


_preload_torchcodec_decoder_before_decord()

from .collators import DefaultCollator
from .datasets import (
    BaseDataset,
    BaseProcessor,
    ConcatDataset,
    Dataset,
    WeightedConcatDataset,
    load_config,
    load_dataset,
    register_dataset,
)
from .utils import is_lerobot_available

if is_lerobot_available():
    from .datasets import LeRobotDataset
from .samplers import (
    AspectRatioSampler,
    BucketBatchSampler,
    BucketSampler,
    DefaultSampler,
    SpecialDatasetSampler,
)
