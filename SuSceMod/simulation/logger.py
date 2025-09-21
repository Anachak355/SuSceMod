# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:07:39 2024

@author: Administrator
"""

import logging
import os
import sys

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        # If message is not empty, log it
        if message:
            # Strip only trailing newlines to avoid extra newlines in log
            if message != '\n':
                self.logger.log(self.level, message)

    def flush(self):
        pass

def setup_logger(log_file_path=None, log_time=True, log_level=logging.DEBUG):
    # Create or get a logger
    logger = logging.getLogger(__name__)

    # Clear existing handlers
    logger.handlers.clear()

    # Define formatter based on user preferences
    if log_time:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')  # No additional information

    # File handler setup
    if log_file_path:
        try:
            log_file_path = os.path.abspath(log_file_path)
            # print(f"Log file will be created at: {log_file_path}")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create log file: {e}")

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    # Redirect stdout and stderr to the logger
    if not hasattr(sys.stdout, '_original'):
        sys.stdout._original = sys.stdout
        sys.stderr._original = sys.stderr

    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

def disable_logging():
    """Disable logging and restore original stdout and stderr."""
    if hasattr(sys.stdout, '_original'):
        sys.stdout = sys.stdout._original
        sys.stderr = sys.stderr._original

    logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
