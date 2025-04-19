import os
from utils.logging_utils import info
from ctranslate2 import get_cuda_device_count, get_supported_compute_types
from compute_providers.compute_provider import ComputeProvider


class CTranslate2ComputeProvider(ComputeProvider):
    """
    CTranslate2-based implementation of the ComputeProvider interface.
    Provides access to device detection, compute type selection, and threading control using CTranslate2.
    """

    _num_threads = None

    def is_mps_available(self):
        return False

    def is_cuda_available(self):
        return get_cuda_device_count() > 0

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

    def get_device(self):
        return self.get_available_devices()[0]

    def get_available_compute_types(self, device):
        return get_supported_compute_types(device)

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
            info(f"🔹 Configuring CTranslate2 device {device} for {num_threads} threads")
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            self._num_threads = num_threads
