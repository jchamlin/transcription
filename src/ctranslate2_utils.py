import os
from logging_utils import info
from ctranslate2 import get_cuda_device_count, get_supported_compute_types

def is_mps_available():
    """
    Checks if Apple MPS (Metal Performance Shaders) is available.

    Returns:
        bool: Always False, as MPS is not currently supported in CTranslate2 as of version 4.6.0
    """
    return False

def is_cuda_available():
    """
    Checks if CUDA is available via CTranslate2.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return get_cuda_device_count() > 0

def is_gpu_available():
    """
    Checks if any GPU backend is available.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return is_mps_available() or is_cuda_available()

def get_available_devices():
    """Lists available compute devices in order of preference.

    Returns:
        list[str]: List of device names, ordered by priority (e.g., ["cuda", "cpu"])
    """
    devices = []
    if is_mps_available():
        devices.append("mps")
    if is_cuda_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices

def get_device():
    """
    Selects the best available device in priority order: mps > cuda > cpu.

    Returns:
        str: The selected device name.
    """
    return get_available_devices()[0]

def get_available_compute_types(device):
    """
    Returns the supported compute types for a device.

    Args:
        device (str): The device name (e.g., "cpu", "cuda", "mps").

    Returns:
        list[str]: Supported compute types such as ["float16", "int8_float32"]

    Raises:
        KeyError: If the model has not been loaded yet.
    """
    return get_supported_compute_types(device)

def get_compute_type(device):
    """
    Returns the recommended compute type for a given device.

    Args:
        device (str): The device name (e.g., "cpu", "cuda", "mps").

    Returns:
        str: The compute type (e.g., "float16" or "int8_float32").
    """
    if device == "mps":
        compute_type = "float16"
    elif device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "float32"
    return compute_type

def get_num_threads(device):
    """
    Determines the optimal number of processing threads for a given device.

    Args:
        device (str): The device name (e.g., "cpu", "cuda", "mps").

    Returns:
        int: Recommended number of processing threads.
    """
    if device == "mps":
        num_threads = 1
    elif device == "cuda":
        num_threads = 1
    else:
        num_threads = int(os.cpu_count() / 2)
    return num_threads

_ctranslate2_num_threads = None

def set_processing_threads(device=None, num_threads=None):
    """
    Sets the number of threads for CTranslate2 operations.

    This function avoids redundant reconfiguration by tracking the last-used thread count.

    Args:
        device (str, optional): Device to configure (default is auto-selected).
        num_threads (int, optional): Number of threads to set (default is calculated).

    Returns:
        None
    """
    global _ctranslate2_num_threads
    device = device or get_device()
    num_threads = num_threads or get_num_threads(device)

    if _ctranslate2_num_threads is None or _ctranslate2_num_threads != num_threads:
        info(f"🔹 Configuring CTranslate2 device {device} for {num_threads} threads")
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        _ctranslate2_num_threads = num_threads

    return
