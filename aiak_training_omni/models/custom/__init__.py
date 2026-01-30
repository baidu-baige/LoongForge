"""aiak training omni"""
from .wan.wan_config import WanConfig

"""aiak training omni"""
# PI05 uses optional `lerobot`; import lazily.
try:
    from .pi05.configuration_pi05 import PI05Config
except ImportError:
    PI05Config = None
