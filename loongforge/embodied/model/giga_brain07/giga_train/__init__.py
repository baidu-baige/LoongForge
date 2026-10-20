__version__ = '1.1.0'

from .configs import Config, load_config
from .distributed import Launcher, launch_from_config, ssh_copy_id
from .modules import ModuleDict
from .optimizers import OPTIMIZERS, build_optimizer
from .registry import Registry, build_module, merge_params
from .samplers import SAMPLERS, ParallelBatchSampler, build_sampler
from .schedulers import SCHEDULERS, build_scheduler
from .strategies import EMAModel, module_auto_wrap_policy
from .testers import Tester
from .trainers import Trainer
from .transforms import TRANSFORMS, build_transform
from .utils import setup_environment
