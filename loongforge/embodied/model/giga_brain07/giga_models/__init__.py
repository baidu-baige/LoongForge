from importlib import import_module

__version__ = '1.1.0'

from .models import GigaBrain07Policy
from .pipelines import BasePipeline, GigaBrain07Pipeline, get_pipelines, list_pipelines, load_pipeline

_LAZY_EXPORTS = {
    'Cosmos25ControlNet3DModel': '.models',
    'Cosmos25MultiControlNet3DModel': '.models',
    'Cosmos25Pipeline': '.pipelines',
    'Cosmos25Transformer3DModel': '.models',
    'DepthAnythingPipeline': '.pipelines',
    'DPTForDepthEstimationPipeline': '.pipelines',
    'EDMRESMultistepScheduler': '.schedulers',
    'FilmPipeline': '.pipelines',
    'FlowUniPCMultistepScheduler': '.schedulers',
    'GigaWorld0Pipeline': '.pipelines',
    'GigaWorld0Transformer3DModel': '.models',
    'HEDPipeline': '.pipelines',
    'LineartPipeline': '.pipelines',
    'LoRAPeftWrapper': '.models',
    'MLSDPipeline': '.pipelines',
    'OpenPosePipeline': '.pipelines',
    'PI0Policy': '.models',
    'Pi0Pipeline': '.pipelines',
    'PidiNetPipeline': '.pipelines',
    'PromptIRPipeline': '.pipelines',
    'RTMPosePipeline': '.pipelines',
    'SegmentAnythingPipeline': '.pipelines',
    'TransNetV2Pipeline': '.pipelines',
    'UperNetForSemanticSegmentationPipeline': '.pipelines',
    'CannyPipeline': '.pipelines',
}

__all__ = [
    'BasePipeline',
    'GigaBrain07Policy',
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
