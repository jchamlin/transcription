import os
import time
import logging
from utils import write_file, load_transcript, save_transcript, format_segment
from compute_providers.torch_utils import get_device, get_num_threads, set_processing_threads

logger = logging.getLogger(__name__)

_torch_num_threads = None

_diarization_models = {}

def get_diarization_model(model_path="pyannote/speaker-diarization", device=None, num_threads=None):
    """
    Cache the diarization model globally to make runs on multiple files faster
    """
    device = device or get_device()
    num_threads = num_threads or get_num_threads(device)
    
    if model_path not in _diarization_models:
        import torch
        from pyannote.audio.pipelines import SpeakerDiarization
    
        device = get_device()
        #device="cpu" # Force CPU
        logger.info(f"🔄 Creating Pyannote model '{model_path}' device '{device}' num_threads '{num_threads}'")
        _diarization_models[model_path] = SpeakerDiarization.from_pretrained(model_path)
        _diarization_models[model_path].to(torch.device(device))

    result = _diarization_models[model_path]
    set_processing_threads()
    return result

def diarize(audio_file, use_cache=True, model_path="pyannote/speaker-diarization", device=None, num_threads=None):
    """
    Diarize an audio file.

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - skip_transcription (bool): Skip Whisper transcription step and use existing .whisper file for diarization testing.
    """
    device = device or get_device()
    num_threads = num_threads or get_num_threads(device)

    diarization_file = os.path.splitext(audio_file)[0] + ".diarization"
    cached_diarization_result = load_transcript(diarization_file)
    if use_cache and cached_diarization_result is not None:
        logger.info(f"🔹 Skipping diarization on {audio_file} and using cached results")
        diarization = cached_diarization_result
    else:
        diarization = diarize_pyannote(audio_file, model_path, device, num_threads)
        if cached_diarization_result is None:
            save_transcript(diarization_file, diarization)
        else:
            # Compare to cached file if it existed before
            cached_lines = [format_segment(seg) for seg in cached_diarization_result]
            new_lines = [format_segment(seg) for seg in diarization]
            if cached_lines != new_lines:
                from difflib import unified_diff
                diarization_file2 = os.path.splitext(audio_file)[0] + ".diarization2"
                save_transcript(diarization_file2, diarization)
                diff = "\n".join(unified_diff(cached_lines, new_lines, fromfile='cached', tofile='new', lineterm=''))
                diff_file = os.path.splitext(audio_file)[0] + ".diarization2-diff"
                write_file(diff_file, diff)
                logger.error(f"❌ Transcription output mismatch detected on {audio_file} new transcript saved to {diarization_file2} and diff to {diff_file}!")

                diff = "\n".join(unified_diff(cached_lines, new_lines, fromfile='cached', tofile='new', lineterm=''))
                logger.error(f"❌ Diarization output mismatch detected on file {audio_file}!")
                logger.error(f"🔍 Diarization diff:\n" + ("-" * 40) + f"\n{diff}\n" + ("-" * 40))
                raise ValueError("Diarization output changed between runs. Reproducibility issue detected.")

    return diarization;

def diarize_pyannote(audio_file, model_path="pyannote/speaker-diarization", device=None, num_threads=None):
    """
    Diarize (identify different speakers) the audio file using Pyannote
    See https://huggingface.co/pyannote/speaker-diarization-3.1

    Parameters:
    - audio_file (str): Path to the audio file to process.
    """
    device = device or get_device()
    num_threads = num_threads or get_num_threads(device)

    logger.info(f"🔹 Running Pyannote / SpeechBrain speaker diarization on {audio_file} on model '{model_path}' using device '{device}' and num_threads '{num_threads}'")

    start = time.time()
    diarization_model = get_diarization_model(model_path=model_path, device=device, num_threads=num_threads)
    diarization_result = diarization_model(audio_file)
    # Store speaker timestamps
    speaker_labels = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_labels.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": speaker
        })
    end = time.time()
    logger.info(f"✅ Pyannote / SpeechBrain speaker diarization complete in {end - start:.2f} seconds.")
    return speaker_labels

