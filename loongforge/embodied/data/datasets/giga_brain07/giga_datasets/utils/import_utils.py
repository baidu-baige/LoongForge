from diffusers.utils.import_utils import _is_package_available
from packaging.version import InvalidVersion, Version

_lerobot_available, _lerobot_version = _is_package_available('lerobot')


def is_lerobot_available() -> bool:
    """Check if the optional dependency 'lerobot' is available in a supported
    version.

    Returns:
        bool: True if the package 'lerobot' is installed and its version is
            greater than or equal to 0.4.0; otherwise False.
    """
    if not _lerobot_available:
        return False

    try:
        return Version(_lerobot_version) >= Version('0.4.0')
    except (InvalidVersion, TypeError):
        return False
