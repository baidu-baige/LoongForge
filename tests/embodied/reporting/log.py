# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Console (and optional file) logger with colored output.

Kept local to the embodied suite so that `tests/embodied/` stays self-contained and
importable without the LLM/VLM E2E stack (which pulls in torch).
"""

import logging
import os
import sys

_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[41m",
}
_RESET = "\033[0m"

_FMT = "[%(asctime)s] [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if sys.stdout.isatty():
            return f"{_COLORS.get(record.levelno, '')}{msg}{_RESET}"
        return msg


def create_logger(log_dir=None, name="embodied_regression", logfile_name="regression.log",
                  level=logging.INFO):
    """Return a console logger; when log_dir is given, also persist to log_dir/logfile_name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ColorFormatter(_FMT, _DATEFMT))
    logger.addHandler(console)

    if log_dir:
        file_handler = logging.FileHandler(os.path.join(log_dir, logfile_name), mode="a")
        file_handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(file_handler)

    return logger
