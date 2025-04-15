import os

def get_device():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_compute_type(device):
    compute_type = "float16" if device == "cuda" else "int8_float32"
    return compute_type

def get_num_threads(device):
    num_threads = 1 if device == "cuda" else int(os.cpu_count() / 2)
    return num_threads

_audio_cache = {}

def load_audio_file(path):
    from pydub import AudioSegment
    if path in _audio_cache:
        return _audio_cache[path]
    audio = AudioSegment.from_file(path)
    _audio_cache[path] = audio
    return audio
