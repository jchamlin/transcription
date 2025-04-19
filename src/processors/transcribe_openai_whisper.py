import time
from logging import info
from compute_providers.torch_utils import get_available_devices as get_available_torch_devices, get_device, get_compute_type, get_num_threads, set_processing_threads

def get_available_models():
    """
    A list of transcription models available, like tiny, small, medium, medium.en, large, etc
    
    Returns:
        A list of available transcription models.
    """
    return [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v1", "large-v2", "large-v3"
    ]

def get_available_devices():
    """
    Returns available compute devices using PyTorch backend.

    Returns:
        list[str]: List of devices in preferred order (e.g., ["cuda", "cpu"])
    """
    return get_available_torch_devices()    

def get_available_compute_types(device):
    """
    Returns available compute types for a given device.

    Args:
        device (str): The device name (e.g., "cpu", "cuda", "mps").

    Returns:
        set[str]: Supported compute types are "float16" and "float32".
    """
    return {"float16", "float32"}

_transcription_models = {}

def get_transcription_model(model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Loads and caches an OpenAI Whisper model configured for the given device and compute type.

    Args:
        model_size_or_path (str, optional): The model name or path (e.g., "medium"). Defaults to "medium".
        device (str, optional): The compute device (e.g., "cpu", "cuda", "mps"). Defaults to best available.
        compute_type (str, optional): The compute type (e.g., "float16"). Defaults to device-preferred.
        num_threads (int, optional): Number of threads to use. Defaults to optimal value based on device.

    Returns:
        whisper.model.Whisper: The loaded and configured model instance.
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)
    if model_size_or_path not in _transcription_models:
        import whisper
        info(f"🔄 Creating OpenAI Whisper model '{model_size_or_path}' device '{device}' compute_type '{compute_type}' num_threads '{num_threads}'")
        model = whisper.load_model(model_size_or_path)
        if compute_type == "float16":
            model = model.half()
        model = model.to(device)
        _transcription_models[model_size_or_path] = {
            "model": model,
            "model_size_or_path": model_size_or_path,
            "device": device,
            "compute_type": compute_type,
            "num_threads": num_threads
        }

    result = _transcription_models[model_size_or_path]
    return result["model"]

def transcribe_openai_whisper(audio_file, use_cache=True, model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_file (str): Path to the audio file to transcribe.
        use_cache (bool, optional): Ignored in OpenAI Whisper, included for compatibility. Defaults to True.
        model_size_or_path (str, optional): Whisper model name or path. Defaults to "medium".
        device (str, optional): Compute device (e.g., "cpu", "cuda", "mps"). Defaults to best available.
        compute_type (str, optional): Compute type (e.g., "float16", "float32"). Defaults to device-preferred.
        num_threads (int, optional): Number of threads to use. Defaults to optimal value.

    Returns:
        list[dict]: A list of transcription segments from the Whisper model.
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)

    start = time.time()
    device = "cpu" # Force CPU for OpenAI Whisper since it's faster than GPU
    info(f"🔹 Running OpenAI Whisper transcription on '{audio_file}' model '{model_size_or_path}' device '{device}' compute_type '{compute_type}' num_threads '{num_threads}'")
    set_processing_threads(1) # force threads to 1 even if in CPU mode, adding more threads makes this slower, not faster. 16 threads is 4x slower than 1 thread.
    whisper_model = get_transcription_model(model_size_or_path=model_size_or_path, device=device, compute_type=compute_type, num_threads=num_threads)
    whisper_result = whisper_model.transcribe(audio_file, fp16=(device == "cuda"))
    transcript = whisper_result["segments"]
    end = time.time()
    info(f"✅ OpenAI Whisper transcription complete in {end - start:.2f} seconds.")
    return transcript
