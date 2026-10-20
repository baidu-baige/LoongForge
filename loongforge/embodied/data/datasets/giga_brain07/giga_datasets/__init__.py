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
    FileDataset,
    FileWriter,
    LmdbDataset,
    LmdbWriter,
    PklDataset,
    PklWriter,
    VQADataset,
    WeightedConcatDataset,
    WorkerRangeDataset,
    load_config,
    load_dataset,
    register_dataset,
)
from .utils import is_lerobot_available

if is_lerobot_available():
    from .datasets import LeRobotDataset, LeRobotVQADataset
from .evaluators import (
    AestheticScoreEvaluator,
    CLIPScoreEvaluator,
    FIDEvaluator,
    LPIPSEvaluator,
    MAPEvaluator,
    PSNREvaluator,
    SSIMEvaluator,
)
from .samplers import (
    AspectRatioSampler,
    BucketBatchSampler,
    BucketSampler,
    DefaultSampler,
    ListWeightedSampler,
    ShardedListWeightedSampler,
    SpecialDatasetSampler,
    WeightedSampler,
)
from .structures import (
    BaseStructure,
    Boxes,
    Boxes3D,
    CameraBoxes3D,
    DepthBoxes3D,
    Image,
    LidarBoxes3D,
    Mode3D,
    Points,
    Points3D,
    VideoReaderCV2,
    VideoReaderDecord,
    boxes3d_utils,
    boxes_utils,
    image_utils,
    points3d_utils,
    points_utils,
    video_utils,
)
from .transforms import PromptEncoderTransform, PromptTokenizerTransform, PromptTransform
from .visualization import ImageVisualizer
