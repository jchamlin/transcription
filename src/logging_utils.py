import os
import io
import logging
import sys
import warnings

# Fixes warning: Requested Pretrainer collection using symlinks on Windows. This might not work; see 'LocalStrategy' documentation. Consider unsetting 'collect_in' in Pretrainer to avoid symlinking altogether.
os.environ["SB_FORCE_LOCAL_STRATEGY"] = "true"

# Filters out specific Pyannote version mismatch messages printed to stdout
class FilteredStdout(io.TextIOBase):
    def __init__(self, original):
        self.original = original
        self._suppress_next_newline = False

    def write(self, message):
        suppress_phrases = [
            "Model was trained with pyannote.audio",
            "Model was trained with torch",
            ">>Performing voice activity detection",
        ]

        if self._suppress_next_newline and message == "\n":
            self._suppress_next_newline = False
            return  # Skip the newline after a suppressed line

        if any(phrase in message for phrase in suppress_phrases):
            self._suppress_next_newline = True
            return  # Suppress line

        return self.original.write(message)

    def flush(self):
        return self.original.flush()

sys.stdout = FilteredStdout(sys.stdout)

def debug(msg, *args, **kwargs):
    logging.getLogger().debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logging.getLogger().info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logging.getLogger().warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logging.getLogger().error(msg, *args, **kwargs)

def setup_logging():
    # Suppress noisy or known irrelevant warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Using SYMLINK strategy on Windows*", category=UserWarning)
    warnings.filterwarnings("ignore", message="Requested Pretrainer collection using symlinks*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*speechbrain\\.pretrained.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Needed so your app logs work at all levels

    # Suppress DEBUG from third-party libraries (set them to WARNING or higher)
    for noisy_logger in sorted([
        "faster_whisper",
        "fsspec.local",
        "matplotlib",
        "numba",
        "pydub.converter",
        "pyannote",
        "pytorch_lightning",
        "pytorch_lightning.utilities.migration.utils",
        "speechbrain",
        "torch",
        "torio",
        "urllib3"
    ]):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Replicate Log4j logging pattern: %d{ISO8601} [%-5p] %t #%x - %c{1} - %m%n
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(threadName)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"  # ISO8601-style timestamp
    )

    # Handler for stdout (DEBUG and INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)

    # Handler for stderr (WARNING and above)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

