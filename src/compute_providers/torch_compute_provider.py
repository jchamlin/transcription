import os
import torch
from utils.logging_utils import info
from compute_providers.compute_provider import ComputeProvider


class TorchComputeProvider(ComputeProvider):
    """
    Torch-based implementation of the ComputeProvider interface.
    Provides access to device detection, compute type selection, and threading control using PyTorch.
    """
    _compute_type_cache = {}
    _num_threads = None

    def is_mps_available(self):
        return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    def is_cuda_available(self):
        return torch.cuda.is_available()

    def is_gpu_available(self):
        return self.is_mps_available() or self.is_cuda_available()

    def get_available_devices(self):
        devices = []
        if self.is_mps_available():
            devices.append("mps")
        if self.is_cuda_available():
            devices.append("cuda")
        devices.append("cpu")
        return devices

    def get_available_compute_types(self, device):
        if device not in self._compute_type_cache:

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
            self._compute_type_cache[device] = supported

        return self._compute_type_cache[device]

    def get_device(self):
        return self.get_available_devices()[0]

    def get_compute_type(self, device):
        if device == "mps":
            compute_type = "float16"
        elif device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "float32"
        return compute_type

    def get_num_threads(self, device):
        if device == "mps":
            num_threads = 1
        elif device == "cuda":
            num_threads = 1
        else:
            num_threads = int(os.cpu_count() / 2)
        return num_threads

    def set_processing_threads(self, device=None, num_threads=None):
        device = device or self.get_device()
        num_threads = num_threads or self.get_num_threads(device)

        if self._num_threads is None or self._num_threads != num_threads:
            info(f"🔹 Configuring Torch device {device} for {num_threads} threads")
            torch.set_num_threads(num_threads)
            self._num_threads = num_threads
