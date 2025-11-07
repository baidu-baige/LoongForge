"""flavors"""

from aiak_training_omni.data.multimodal.flavors.packed_captioning import (
    PackedCaptioningSample,
)
from aiak_training_omni.data.multimodal.flavors.packed_vqa import PackedVQASample
from aiak_training_omni.data.multimodal.flavors.multi_vid_qa import MultiVidQASample
from aiak_training_omni.data.multimodal.flavors.multi_mix_qa import MultiMixQASample
from aiak_training_omni.data.multimodal.flavors.packed_multi_mix_qa import (
    PackedMultiMixQASample,
)

__all__ = [
    "PackedCaptioningSample",
    "PackedVQASample",
    "PackedMultiMixQASample",
    "MultiVidQASample",
    "MultiMixQASample",
]
