from importlib import import_module

from .pipeline import BasePipeline, get_pipelines, list_pipelines, load_pipeline
from .vla import GigaBrain07Pipeline

_LAZY_EXPORTS = {
    'Cosmos25Pipeline': '.diffusion',
    'GigaWorld0Pipeline': '.diffusion',
    'DepthAnythingPipeline': '.vision',
    'DPTForDepthEstimationPipeline': '.vision',
    'CannyPipeline': '.vision',
    'HEDPipeline': '.vision',
    'LineartPipeline': '.vision',
    'MLSDPipeline': '.vision',
    'PidiNetPipeline': '.vision',
    'FilmPipeline': '.vision',
    'PromptIRPipeline': '.vision',
    'OpenPosePipeline': '.vision',
    'RTMPosePipeline': '.vision',
    'SegmentAnythingPipeline': '.vision',
    'UperNetForSemanticSegmentationPipeline': '.vision',
    'TransNetV2Pipeline': '.vision',
    'Pi0Pipeline': '.vla',
}

__all__ = [
    'BasePipeline',
    'GigaBrain07Pipeline',
    'get_pipelines',
    'list_pipelines',
    'load_pipeline',
    *_LAZY_EXPORTS,
]


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
