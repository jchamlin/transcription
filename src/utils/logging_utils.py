import os
import io
import logging
import sys
import warnings

# Fixes warning: Requested Pretrainer collection using symlinks on Windows. This might not work; see 'LocalStrategy' documentation. Consider unsetting 'collect_in' in Pretrainer to avoid symlinking altogether.
os.environ["SB_FORCE_LOCAL_STRATEGY"] = "true"

class FilteredStdout(io.TextIOBase):
    """
    A wrapper for stdout that filters out suppressed phrases.
    """
    suppress_phrases = set()

    def __init__(self, original):
        """
        Initializes the filtered stdout.

        Args:
            original (io.TextIOBase): The original stdout stream.
        """
        self.original = original
        self._suppress_next_newline = False

    def write(self, message):
        """
        Writes a message to stdout unless it matches a suppressed phrase.

        Args:
            message (str): The message to write.
        """
        if self._suppress_next_newline and message == "\n":
            self._suppress_next_newline = False
            return

        if any(phrase in message for phrase in self.suppress_phrases):
            self._suppress_next_newline = True
            return

        return self.original.write(message)

    def flush(self):
        """
        Flushes the underlying stdout stream.
        """
        return self.original.flush()

sys.stdout = FilteredStdout(sys.stdout)

class FilteredStderr(io.TextIOBase):
    """
    A wrapper for stderr that filters out suppressed phrases.
    """
    suppress_phrases = set()

    def __init__(self, original):
        """
        Initializes the filtered stderr.

        Args:
            original (io.TextIOBase): The original stderr stream.
        """
        self.original = original
        self._suppress_next_newline = False

    def write(self, message):
        """
        Writes a message to stderr unless it matches a suppressed phrase.

        Args:
            message (str): The message to write.
        """
        if self._suppress_next_newline and message == "\n":
            self._suppress_next_newline = False
            return

        if any(phrase in message for phrase in self.suppress_phrases):
            self._suppress_next_newline = True
            return

        return self.original.write(message)

    def flush(self):
        """
        Flushes the underlying stderr stream.
        """
        return self.original.flush()

sys.stderr = FilteredStderr(sys.stderr)

_OUTPUT_SUPPRESSIONS_ENABLED = False

def enable_output_suppressions(enabled=True):
    """
    Enables or disables output suppression globally.

    Args:
        enabled (bool): Whether suppression should be active.
    """
    global _OUTPUT_SUPPRESSIONS_ENABLED
    _OUTPUT_SUPPRESSIONS_ENABLED = enabled

def suppress_stdout(phrase):
    """
    Suppresses specific messages in stdout.

    Args:
        phrase (str): A substring to suppress.
    """
    if _OUTPUT_SUPPRESSIONS_ENABLED:
        FilteredStdout.suppress_phrases.add(phrase)

def suppress_stderr(phrase):
    """
    Suppresses specific messages in stderr.

    Args:
        phrase (str): A substring to suppress.
    """
    if _OUTPUT_SUPPRESSIONS_ENABLED:
        FilteredStderr.suppress_phrases.add(phrase)

def suppress_warning(message=None, category=UserWarning):
    """
    Suppresses warnings optionally filtered by message and category.

    Args:
        message (str, optional): Regex pattern for the warning message.
        category (Warning): The warning category to suppress.
    """
    if _OUTPUT_SUPPRESSIONS_ENABLED:
        if message in (None, "", "*"):
            warnings.filterwarnings("ignore", category=category)
        else:
            warnings.filterwarnings("ignore", message=message, category=category)

def suppress_logging(logger_name):
    """
    Suppresses a named logger by setting its level to WARNING.

    Args:
        logger_name (str): The name of the logger to suppress.
    """
    if _OUTPUT_SUPPRESSIONS_ENABLED:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

class ShortNameFormatter(logging.Formatter):
    """
    Custom formatter that adds a relative short path to each log record.
    """
    def format(self, record):
        """
        Adds 'shortname' to the record with a path relative to sys.path.

        Args:
            record (LogRecord): The log record.

        Returns:
            str: The formatted log message.
        """
        full_path = os.path.abspath(record.pathname)
        for path in map(os.path.abspath, sys.path):
            if full_path.startswith(path):
                rel_path = os.path.relpath(full_path, path)
                record.shortname = rel_path.replace(os.sep, "/")
                break
        else:
            record.shortname = os.path.basename(record.pathname)
        return super().format(record)

def setup_logging(logger_name):
    """
    Configures the root logger with stream handlers, formatting, and optional suppression.
    
    Args:
        logger_name (str): The name of the logger to return. If None, the root logger is configured.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = ShortNameFormatter(
        fmt="%(asctime)s [%(levelname)-5s] %(threadName)s - %(name)s - %(message)s (%(shortname)s:%(lineno)d)",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    
    if logger_name:
        logger = logging.getLogger(logger_name)

    return logger
