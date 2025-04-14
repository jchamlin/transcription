import os

def get_device():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_compute_type():
    device = get_device()
    compute_type = "float16" if device == "cuda" else "int8_float32"
    return compute_type

_torch_threads = None

def setup_torch(threads=int(os.cpu_count() / 2)):
    """
    Setup the number of threads torch should use 
    """
    from logging_utils import info
    global _torch_threads

    import torch
    if _torch_threads is None or _torch_threads != threads:
        info(f"🔹 Configuring Torch for {threads} threads")
        torch.set_num_threads(threads)
        _torch_threads = threads

    return

_audio_cache = {}

def load_audio_file(path):
    from pydub import AudioSegment
    if path in _audio_cache:
        return _audio_cache[path]
    audio = AudioSegment.from_file(path)
    _audio_cache[path] = audio
    return audio
