"""
transcribe.py

Runs a complete speaker-attributed transcription pipeline on one or more audio files.

Steps:
1. Transcribe audio to text
2. Diarize speaker turns
3. (Future) Attribute speakers to known identities
"""

import os
import time
import argparse
import traceback
import logging
from logging_utils import setup_logging, info, error
from audio_utils import get_device, get_compute_type, get_num_threads
from transcript_utils import save_transcript
from transcription_utils import available_models, transcribe as run_transcription
from diarization_utils import diarize, merge_transcription_and_diarization

_USE_WHISPERX = False

def transcribe(audio_file, output_format="transcript", use_cache=True, model_size_or_path=None, device=None, compute_type=None, num_threads=None):
    """
    Executes the full transcription pipeline on a given audio file.

    Steps:
    1. Transcribes audio to timestamped text
    2. Performs speaker diarization to segment by speaker
    3. (Future) Performs speaker attribution using known identities

    Args:
        audio_file (str): Path to the audio file to process.
        output_format (str): Extension for the output file (e.g., .transcript, .html, .json, .xml, .md).
        use_cache (bool): Whether to reuse cached results if available.
        model_size_or_path (str): Model used for transcription (default: "medium").
        device (str): Device used for transcription (default: "cuda" if available, otherwise "cpu")
        compute_type (str): Compute type used for transcription (default: "float16" if device is "cuda", otherwise "int8_float32")
        num_threads (int): Number of threads to use for transcription (default: 1 if device is "cuda", otherwise int(os.cpu_count() / 2))

    Returns:
        None (writes output to a file)
    """
    model_size_or_path = model_size_or_path or "medium"
    device = device or get_device()
    compute_type = compute_type or get_compute_type(device)
    num_threads = num_threads or get_num_threads(device)

    start = time.time()
    info(f"🔹 Transcribing '{audio_file}' using output_format '{output_format}' model `{model_size_or_path}` device `{device}` compute_type=`{compute_type}` num_threads=`{num_threads}`")
    if _USE_WHISPERX:
        from transcribe_and_diarize_whisperx import transcribe_and_diarize_whisperx
        merged_transcript = transcribe_and_diarize_whisperx(audio_file, model_size_or_path)
    else:
        transcript = run_transcription(
            audio_file, 
            use_cache=use_cache, 
            model_size_or_path=model_size_or_path, 
            device=device, 
            compute_type=compute_type, 
            num_threads=num_threads
        )
        diarization = diarize(audio_file, use_cache)
        merged_transcript = merge_transcription_and_diarization(transcript, diarization)
        
    # TODO add SpeechBrain speaker identification to convert merged_transcript to final_transcript
    final_transcript = merged_transcript

    transcript_file = os.path.splitext(audio_file)[0] + output_format
    save_transcript(transcript_file, final_transcript)
    end = time.time()
    info(f"⏱️Total processing time for {audio_file}: {end - start:.2f} seconds.")

def build_arg_parser():
    """
    Create and configure the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured parser with supported arguments.
    """

    device = get_device()
    compute_type = get_compute_type(device)
    threads = get_num_threads(device)
    parser = argparse.ArgumentParser(description="Transcribe Audio Files.")
    parser.add_argument("files", nargs='+', help="List of audio files.")
    parser.add_argument("-f", "--format", dest="output_format", default=".transcript", help="Output file extension (e.g., .transcript, .html, .json, .xml, .md). Default is '.transcript'.")
    parser.add_argument("-m", "--model", dest="model_size_or_path", default="medium", help=f"Whisper model to use ({', '.join(available_models())}). Default is 'medium'.")
    parser.add_argument("-d", "--device", default=device, help=f"Device to use: cuda or cpu. Default is `{device}`.")
    parser.add_argument("-c", "--compute-type", default=compute_type, help=f"Whisper compute_type to use, Default is `{compute_type}`.")
    parser.add_argument("-t", "--threads", dest="num_threads", type=int, default=threads, help=f"Number of threads to use in CPU mode. Default is `{threads}`.")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", help="Use cached transcription and diarization results if available (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable cache use and always regenerate results.")
    parser.set_defaults(use_cache=True)
    return parser
  
def main():
    """
    Main entry point for CLI execution. Parses arguments and runs the pipeline
    for each specified audio file.
    """
    setup_logging()
    parser = build_arg_parser();
    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        return

    # Validate model name after parsing
    if args.model_size_or_path not in available_models():
        parser.error(f"Invalid model '{args.model_size_or_path}'. Supported: {', '.join(available_models())}")
        return

    if not args.output_format.startswith("."):
        args.output_format = "." + args.output_format

    transcribe_args = {k: v for k, v in vars(args).items() if k != "files"}
  
    for file in args.files:
        try:
            transcribe(file, **transcribe_args)
        except Exception as e:
            error(f"❌ Failed to transcribe {file}: {e}")
            error(traceback.format_exc())  # Captures full traceback as a string and logs it

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        info("⛔ Transcription interrupted by user.")
    except Exception as e:
        # Emergency fallback logger
        logging.basicConfig(level=logging.DEBUG)
        logging.error(f"❌ Unhandled exception during startup: {e}")
        logging.error(traceback.format_exc())

