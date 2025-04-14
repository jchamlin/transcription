import os
import time
from logging_utils import info, error
from transcript_utils import load_transcript, save_transcript, format_segment
from audio_utils import get_device, get_compute_type, setup_torch

_transcription_models = {}

def available_models():
    """
    A list of transcription models available, like tiny, small, medium, medium.en, large, etc
    
    Returns:
        A list of available transcription models.
    """
    from faster_whisper import available_models as faster_whisper_available_models
    return faster_whisper_available_models()

def get_transcription_model(model_size="medium"):
    """
    Cache the transcription model globally to make runs on multiple files faster
    """
    if model_size not in _transcription_models:
        from faster_whisper import WhisperModel
        device = get_device()
        compute_type = get_compute_type()
        info(f"🔄 Creating Whisper model '{model_size}' device '{device}' compute_type '{compute_type}'")
    
        _transcription_models[model_size] = {
            "model": WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            ),
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type
        }

    result = _transcription_models[model_size]
    model = result["model"]
    model_size = result["model_size"]
    device = result["device"]
    compute_type = result["compute_type"]
    
    if device == "cpu":
        setup_torch()
    else:
        setup_torch(1)

    return model


def transcribe(audio_file, model="medium", use_cache=True):
    """
    Transcribe an audio file.

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - model (str): The Whisper model to use (default: "medium").
    - use_cache (bool): Skip Whisper transcription step and use existing cached .whisper file
    """
    from difflib import unified_diff

    transcript_file = os.path.splitext(audio_file)[0] + ".whisper"
    cached_transcription_result = load_transcript(transcript_file)
    if use_cache and cached_transcription_result is not None:
        info(f"🔹 Skipping transcription on {audio_file} and using cached results from {transcript_file}")
        transcript = cached_transcription_result
    else:
        transcript = transcribe_fast_whisper(audio_file, model)
        if cached_transcription_result is None:
            save_transcript(transcript_file, transcript)
        else:
            # Compare to cached file if it existed before
            cached_lines = [format_segment(seg) for seg in cached_transcription_result]
            new_lines = [format_segment(seg) for seg in transcript]
            if cached_lines != new_lines:
                diff = "\n".join(unified_diff(cached_lines, new_lines, fromfile='cached', tofile='new', lineterm=''))
                error("❌ Transcription output mismatch detected!")
                error("🔍 Transcription diff:\n" + ("-" * 40) + f"\n{diff}\n" + ("-" * 40))
                raise ValueError("Transcription output changed between runs. Reproducibility issue detected.")

    return transcript

def transcribe_openai_whisper(audio_file, model):
    """
    Transcribe the audio file using OpenAI Whisper

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - model (str): The Whisper model to use (default: "medium").
    """
    import whisper

    start = time.time()
    device = "cpu" # Force CPU for OpenAI Whisper since it's faster than GPU
    info(f"🔹 Running OpenAI Whisper transcription on {audio_file} model '{model}' using device '{device}'")
    setup_torch(1) # force threads to 1 even if in CPU mode, adding more threads makes this slower, not faster. 16 threads is 4x slower than 1 thread.
    whisper_model = whisper.load_model(model).to(device)
    whisper_result = whisper_model.transcribe(audio_file, fp16=(device == "cuda"))
    transcript = whisper_result["segments"]
    end = time.time()
    info(f"✅ OpenAI Whisper transcription complete in {end - start:.2f} seconds.")
    return transcript

def transcribe_fast_whisper(audio_file, model):
    """
    Transcribe the audio file using faster-whisper for improved performance.

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - model (str): The Whisper model to use (default: "medium").
    """
    start = time.time()
    info(f"🔹 Running faster-whisper transcription on {audio_file} using model '{model}'")
    whisper_model = get_transcription_model(model)
    segments, _ = whisper_model.transcribe(audio_file, beam_size=5, temperature=0.0, word_timestamps=True)
    transcript = []
    for seg in segments:
        #debug(f"DEBUG SEG: start={seg.start}, end={seg.end}, content={seg.content[:40]}")
        transcript.append({
            "start": seg.start,
            "end": seg.end,
            "content": seg.text.strip()
        })
    end = time.time()
    info(f"✅ Faster-whisper transcription complete in {end - start:.2f} seconds.")
    return transcript

