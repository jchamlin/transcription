import os
from utils.logging_utils import info
from lightning.fabric.accelerators.cuda import is_cuda_available
import torch  # import this last so logging gets to set things up before it starts printing and warning

def is_mps_available():
    """
    Checks if Apple MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available, False otherwise.
    """
    return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

def is_cuda_available():
    """
    Checks if CUDA is available.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()

def is_gpu_available():
    """
    Checks if any GPU backend (MPI or CUDA) is available.

    Returns:
        bool: True if MPI or CUDA is available, False otherwise.
    """
    return is_mps_available() or is_cuda_available()

def get_available_devices():
    """
    Lists available compute devices in order of preference.

    Returns:
        list[str]: List of device names, ordered by priority (e.g., ["mps", "cuda", "cpu"])
    """
    devices = []
    if is_mps_available():
        devices.append("mps")
    if is_cuda_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices

_torch_compute_type_cache = {}

def get_available_compute_types(device):
    """
    Returns supported compute types for a given device.

    Args:
        device (str): The device name (e.g., "cpu", "cuda", "mps").

    Returns:
        set[str]: Supported compute types such as "float16", "float32", "int8_float32".
    """
    if device not in _torch_compute_type_cache:
        def is_dtype_supported(dtype):
            try:
                torch.tensor([1.0], dtype=dtype, device=device)
                return True
            except Exception:
                return False
        
        dtypes = [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.int32,
            torch.int64,
        ]
        supported = {
            str(dtype).split(".")[-1] for dtype in dtypes if is_dtype_supported(dtype)
        }
        _torch_compute_type_cache[device] = supported

    return _torch_compute_type_cache[device]

def get_device():
    """
    Selects the best available device in priority order: mps > cuda > cpu.

    Returns:
        str: The selected device name.
    """
    return get_available_devices()[0]

def get_compute_type(device):
    """
    Returns the default compute type for a given device.

    Args:
        device (str): The device name (e.g., "cpu", "cuda", "mps").

    Returns:
        str: The compute type (e.g., "float16" or "float32").
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

_torch_num_threads = None

def set_processing_threads(device=None, num_threads=None):
    """
    Sets the number of threads used by Torch for the given device.

    This function avoids redundant reconfiguration by tracking the last-used thread count.

    Args:
        device (str, optional): Device to configure (default is auto-selected).
        num_threads (int, optional): Number of threads to set (default is calculated).

    Returns:
        None
    """
    device = device or get_device()
    num_threads = num_threads or get_num_threads(device)
    global _torch_num_threads

    if _torch_num_threads is None or _torch_num_threads != num_threads:
        info(f"🔹 Configuring Torch device {device} for {num_threads} threads")
        torch.set_num_threads(num_threads)
        _torch_num_threads = num_threads

    return
