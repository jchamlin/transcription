import os
import time
from logging_utils import info, error
from transcript_utils import load_transcript, save_transcript, format_segment
from audio_utils import get_device, get_compute_type, get_num_threads, write_file

_transcription_models = {}

def available_models():
    """
    A list of transcription models available, like tiny, small, medium, medium.en, large, etc
    
    Returns:
        A list of available transcription models.
    """
    from faster_whisper import available_models as faster_whisper_available_models
    return faster_whisper_available_models()

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
        info(f"🔄 Creating Whisper model '{model_size_or_path}' device '{device}' compute_type '{compute_type}' num_threads `{num_threads}`")
    
        _transcription_models[model_size_or_path] = {
            "model": WhisperModel(
                model_size_or_path,
                device=device,
                compute_type=compute_type,
                cpu_threads=num_threads
            ),
            "model_size_or_path": model_size_or_path,
            "device": device,
            "compute_type": compute_type,
            "num_threads": num_threads
        }

    result = _transcription_models[model_size_or_path]
    return result["model"]

def transcribe(audio_file, use_cache=True, model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Transcribe an audio file.

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - model_size_or_path (str): The Whisper model_size_or_path to use (default: "medium").
    - use_cache (bool): Skip Whisper transcription step and use existing cached .whisper file
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)

    from difflib import unified_diff

    transcript_file = os.path.splitext(audio_file)[0] + ".whisper"
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
                transcript_file2 = os.path.splitext(audio_file)[0] + ".whisper2"
                save_transcript(transcript_file2, transcript)
                diff = "\n".join(unified_diff(cached_lines, new_lines, fromfile='cached', tofile='new', lineterm=''))
                diff_file = os.path.splitext(audio_file)[0] + ".whisper2-diff"
                write_file(diff_file, diff)
                error(f"❌ Transcription output mismatch detected on {audio_file} new transcript saved to {transcript_file2} and diff to {diff_file}!")
                raise ValueError(f"Transcription output changed between runs. Reproducibility issue detected.")

    return transcript

def transcribe_fast_whisper(audio_file, model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Transcribe the audio file using faster-whisper for improved performance.

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - model (str): The Whisper model to use (default: "medium").
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)

    start = time.time()
    info(f"🔹 Running faster-whisper transcription on '{audio_file}' using model '{model_size_or_path}' on device '{device}` compute_type '{compute_type}' and num_threads='{num_threads}`")
    whisper_model = get_transcription_model(
        model_size_or_path=model_size_or_path,
        device=device,
        compute_type=compute_type,
        num_threads=num_threads
    )
    segments, _ = whisper_model.transcribe(audio_file, beam_size=5, temperature=0.0, word_timestamps=True)
    transcript = []
    for segment in segments:
        #debug(f"DEBUG SEG: start={segment.start}, end={segment.end}, content={segment.content[:40]}")
        transcript.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "content": segment.text.strip()
        })
    end = time.time()
    info(f"✅ Faster-whisper transcription complete in {end - start:.2f} seconds.")
    return transcript

