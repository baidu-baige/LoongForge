import logging
import os


def create_logger(log_path: str | None = None, log_format: str | None = None) -> logging.Logger:
    """Create a root logger that logs to file and stdout.

    Args:
        log_path: File path to write logs; when None, only stdout is used.
        log_format: Logging format string; defaults to a concise timestamp+message.
    """
    if log_format is None:
        log_format = '%(asctime)-15s %(message)s'
        # log_format = '%(asctime)-15s %(filename)s[line:%(lineno)d] %(message)s'
    if log_path is not None:
        if os.path.exists(log_path):
            os.remove(log_path)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler_s = logging.StreamHandler()
        handler_s.setFormatter(formatter)
        logger.addHandler(handler_s)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    return logger
