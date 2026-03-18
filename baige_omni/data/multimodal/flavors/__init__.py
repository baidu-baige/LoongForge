# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""flavors"""

from baige_omni.data.multimodal.flavors.packed_captioning import (
    PackedCaptioningSample,
)
from baige_omni.data.multimodal.flavors.packed_vqa import PackedVQASample
from baige_omni.data.multimodal.flavors.multi_vid_qa import MultiVidQASample
from baige_omni.data.multimodal.flavors.multi_mix_qa import MultiMixQASample
from baige_omni.data.multimodal.flavors.packed_multi_mix_qa import (
    PackedMultiMixQASample,
)

__all__ = [
    "PackedCaptioningSample",
    "PackedVQASample",
    "PackedMultiMixQASample",
    "MultiVidQASample",
    "MultiMixQASample",
]
