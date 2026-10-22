import os


def get_model_dir() -> str:
    default_model_dir = os.path.join(os.path.expanduser('~'), '.cache', 'giga_models', 'models')
    return os.environ.get('GIGA_MODELS_CACHE', default_model_dir)


def get_repo_dir() -> str:
    default_repo_dir = os.path.join(os.path.expanduser('~'), '.cache', 'giga_models', '3rdparty')
    return os.environ.get('GIGA_MODELS_REPO_CACHE', default_repo_dir)
