import os
import time
from logging_utils import info, error
from file_utils import write_file
from transcript_utils import load_transcript, save_transcript, format_segment
from ctranslate2_utils import get_available_devices as get_available_ct2_devices, get_available_compute_types as get_available_ct2_compute_types, get_device, get_compute_type, get_num_threads

def get_available_models():
    """
    A list of transcription models available, like tiny, small, medium, medium.en, large, etc

    Returns:
        A list of available transcription models.
    """
    from faster_whisper import available_models
    return available_models()

def get_available_devices():
    """
    Returns available compute devices using CTranslate2 backend.

    Returns:
        list[str]: List of devices in preferred order (e.g., ["cuda", "cpu"])
    """
    return get_available_ct2_devices()

def get_available_compute_types(device):
    """
    Returns available compute types using CTranslate2 backend.

    Args:
        device (str): The compute device (e.g., "cpu", "cuda", "mps").

    Returns:
        set[str]: Supported compute types for the given device.
    """
    return get_available_ct2_compute_types()

_transcription_models = {}

def get_transcription_model(model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Cache the transcription model globally to make runs on multiple files faster
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)
    if model_size_or_path not in _transcription_models:
        from faster_whisper import WhisperModel
        info(f"🔄 Creating faster-whisper model '{model_size_or_path}' device '{device}' compute_type '{compute_type}' num_threads '{num_threads}'")
        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_threads
        )
        _transcription_models[model_size_or_path] = {
            "model": model,
            "model_size_or_path": model_size_or_path,
            "device": device,
            "compute_type": compute_type,
            "num_threads": num_threads
        }

    result = _transcription_models[model_size_or_path]
    return result["model"]

def transcribe(audio_file, use_cache=True, model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Transcribe an audio file using Faster-Whisper, with optional caching.

    Args:
        audio_file (str): Path to the audio file to transcribe.
        use_cache (bool): Whether to use a cached .whisper file if it exists.
        model_size_or_path (str, optional): The model name or path to use. Defaults to "medium".
        device (str, optional): The device to run inference on (e.g., "cpu", "cuda", "mps").
        compute_type (str, optional): The compute type to use (e.g., "float16", "int8_float32").
        num_threads (int, optional): Number of threads to use (relevant for CPU).

    Returns:
        list[dict]: A list of transcript segments with start, end, and content.
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)

    base = os.path.splitext(audio_file)[0]
    suffix = f" ({compute_type})" if compute_type else ""
    transcript_file = base + suffix + ".whisper"
    cached_transcription_result = load_transcript(transcript_file)
    if use_cache and cached_transcription_result is not None:
        info(f"🔹 Skipping transcription on {audio_file} and using cached results from {transcript_file}")
        transcript = cached_transcription_result
    else:
        transcript = transcribe_fast_whisper(
            audio_file, 
            model_size_or_path=model_size_or_path, 
            device=device, 
            compute_type=compute_type, 
            num_threads=num_threads
        )
        if cached_transcription_result is None:
            save_transcript(transcript_file, transcript)
        else:
            # Compare to cached file if it existed before
            cached_lines = [format_segment(seg) for seg in cached_transcription_result]
            new_lines = [format_segment(seg) for seg in transcript]
            if cached_lines != new_lines:
                from difflib import unified_diff
                transcript_file2 = base + suffix + ".whisper2"
                save_transcript(transcript_file2, transcript)
                diff = "\n".join(unified_diff(cached_lines, new_lines, fromfile='cached', tofile='new', lineterm=''))
                diff_file = base + suffix + ".whisper2-diff"
                write_file(diff_file, diff)
                error(f"❌ Transcription output mismatch detected on {audio_file} new transcript saved to {transcript_file2} and diff to {diff_file}!")
                raise ValueError(f"Transcription output changed between runs. Reproducibility issue detected.")

    return transcript

def transcribe_fast_whisper(audio_file, model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Transcribe an audio file using Faster-Whisper for high-performance inference.

    Args:
        audio_file (str): Path to the audio file to transcribe.
        model_size_or_path (str, optional): The model name or path to use. Defaults to "medium".
        device (str, optional): The compute device (e.g., "cpu", "cuda", "mps").
        compute_type (str, optional): The compute type (e.g., "float16").
        num_threads (int, optional): Number of threads to use for inference.

    Returns:
        list[dict]: A list of transcript segments with start, end, and content.
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)

    start = time.time()
    info(f"🔹 Running faster-whisper transcription on '{audio_file}' using model '{model_size_or_path}' on device '{device}' compute_type '{compute_type}' and num_threads='{num_threads}'")
    whisper_model = get_transcription_model(
        model_size_or_path=model_size_or_path,
        device=device,
        compute_type=compute_type,
        num_threads=num_threads
    )
    segments, _ = whisper_model.transcribe(audio_file, beam_size=5, temperature=0.0, word_timestamps=True)
    transcript = []
    for segment in segments:
        transcript.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "content": segment.text.strip()
        })
    end = time.time()
    info(f"✅ Faster-whisper transcription complete in {end - start:.2f} seconds.")
    return transcript
