from abc import ABC, abstractmethod


class ComputeProvider(ABC):
    """
    Abstract interface for compute runtimes that support model execution across backends
    like Torch and CTranslate2.
    """

    @abstractmethod
    def is_mps_available(self):
        """
        Checks if Apple's Metal Performance Shaders (MPS) is available.

        Returns:
            bool: True if MPS is available, False otherwise.
        """
        pass

    @abstractmethod
    def is_cuda_available(self):
        """
        Checks if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        pass

    @abstractmethod
    def is_gpu_available(self):
        """
        Checks if any GPU backend (MPS or CUDA) is available.

        Returns:
            bool: True if a GPU is available, False otherwise.
        """
        pass

    @abstractmethod
    def get_available_devices(self):
        """
        Returns a list of available compute devices (e.g., ["cuda", "cpu", "mps"]).

        Returns:
            list[str]: List of device names.
        """
        pass

    @abstractmethod
    def get_available_compute_types(self, device):
        """
        Returns a set of supported compute types for the given device (e.g., float16, int8).

        Args:
            device (str): The device to query.

        Returns:
            set[str]: Supported compute types.
        """
        pass

    @abstractmethod
    def get_device(self):
        """
        Returns the preferred device based on availability.

        Returns:
            str: Preferred device.
        """
        pass

    @abstractmethod
    def get_compute_type(self, device):
        """
        Determines the best compute type for a given device.

        Args:
            device (str): Target device.

        Returns:
            str: Compute type (e.g., "float32", "float16").
        """
        pass

    @abstractmethod
    def get_num_threads(self, device):
        """
        Returns the optimal number of threads to use for the given device.

        Args:
            device (str): Device identifier.

        Returns:
            int: Number of threads.
        """
        pass

    @abstractmethod
    def set_processing_threads(self, device=None, num_threads=None):
        """
        Configures thread usage for model execution.

        Args:
            device (str, optional): Device to configure. Defaults to None.
            num_threads (int, optional): Number of threads to use. Defaults to None.
        """
        pass
