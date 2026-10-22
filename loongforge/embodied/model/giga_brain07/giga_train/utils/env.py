import os
import sys


def setup_environment() -> None:
    """Add the current working directory to sys.path and PYTHONPATH.
    """
    work_dir = os.getcwd()
    sys.path.insert(0, work_dir)
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] += ':{}'.format(work_dir)
    else:
        os.environ['PYTHONPATH'] = work_dir
