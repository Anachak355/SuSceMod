import logging
import os
import sys

# The stdout and stderr that were in place before setup_logger, restored by disable_logging
_original_streams = None

class LoggerWriter:
    def __init__(self, logger, level, stream):
        self.logger = logger
        self.level = level
        self.stream = stream

    def __getattr__(self, name):
        # Anything not defined here goes to the original stream. Jupyter, for example,
        # calls set_parent on sys.stdout to show output under the cell that is running.
        return getattr(self.stream, name)

    def write(self, message):
        # If message is not empty, log it
        if message:
            # Strip only trailing newlines to avoid extra newlines in log
            if message != '\n':
                self.logger.log(self.level, message)

    def flush(self):
        pass

def setup_logger(log_file_path=None, log_time=True, log_level=logging.DEBUG):
    """
    Send everything printed to stdout and stderr to a logger, which writes to the console and,
    if log_file_path is given, to that file. Undo it with disable_logging.
    """
    global _original_streams
    if _original_streams is None:
        _original_streams = (sys.stdout, sys.stderr)

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

    # Console handlers: what was printed goes back to the original stdout, errors to the original stderr
    stdout_handler = logging.StreamHandler(_original_streams[0])
    stdout_handler.setLevel(log_level)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    stderr_handler = logging.StreamHandler(_original_streams[1])
    stderr_handler.setLevel(max(log_level, logging.ERROR))
    for console_handler in (stdout_handler, stderr_handler):
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.setLevel(log_level)

    # Redirect stdout and stderr to the logger
    sys.stdout = LoggerWriter(logger, logging.INFO, _original_streams[0])
    sys.stderr = LoggerWriter(logger, logging.ERROR, _original_streams[1])

def disable_logging():
    """Disable logging and restore original stdout and stderr."""
    global _original_streams
    if _original_streams is not None:
        sys.stdout, sys.stderr = _original_streams
        _original_streams = None

    logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
