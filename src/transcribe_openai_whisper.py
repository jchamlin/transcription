import time
from logging import info
from diarization_utils import setup_torch

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
