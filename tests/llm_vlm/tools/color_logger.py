# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""color logger"""

import os
import sys
import logging

fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d) %(levelname)s : %(message)s'
GLOBAL_LOG_LEVEL=logging._nameToLevel[os.getenv("FLYAGAIN_LOG_LEVEL", "INFO").upper()]
class CustomFormatter(logging.Formatter):
    """CustomFormatter."""
    grey = "\x1b[38;20m"
    green = "\x1b[40;32m"
    blue = "\x1b[40;34m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: blue + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
        
def create_color_logger(output_dir = "./", dist_rank=0, name=__name__, logfile_name = None, level=GLOBAL_LOG_LEVEL):
    """create_color_logger."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            CustomFormatter())
        logger.addHandler(console_handler)

    # create file handlers
    logfile_name = f'log_rank{dist_rank}.log' if logfile_name is None else logfile_name
    file_handler = logging.FileHandler(os.path.join(output_dir, logfile_name), mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    logger = create_color_logger("./", level = logging.DEBUG)
    logger.info("info")
    logger.debug("debug")
    logger.error("error")
    logger.critical("critical")
    logger.warning("warning")