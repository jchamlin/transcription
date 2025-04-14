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
from transcript_utils import save_transcript
from transcription_utils import available_models, transcribe as run_transcription
from diarization_utils import diarize, merge_transcription_and_diarization

def transcribe(audio_file, model="medium", output_format="transcript", use_cache=True):
    """
    Executes the full transcription pipeline on a given audio file.

    Steps:
    1. Transcribes audio to timestamped text
    2. Performs speaker diarization to segment by speaker
    3. (Future) Performs speaker attribution using known identities

    Args:
        audio_file (str): Path to the audio file to process.
        model (str): Model used for transcription (default: "medium").
        output_format (str): Extension for the output file (e.g., .transcript, .html, .json, .xml, .md).
        use_cache (bool): Whether to reuse cached results if available.

    Returns:
        None
    """
    _USE_WHISPERX = False
    start = time.time()
    info(f"🔹 Transcribing {audio_file} using model '{model}' output_format '{output_format}'")
    if _USE_WHISPERX:
        from transcribe_and_diarize_whisperx import transcribe_and_diarize_whisperx
        merged_transcript = transcribe_and_diarize_whisperx(audio_file, model)
    else:
        transcript = run_transcription(audio_file, model, use_cache)
        diarization = diarize(audio_file, use_cache)
        merged_transcript = merge_transcription_and_diarization(transcript, diarization)
        
    # TODO add SpeechBrain speaker identification to convert merged_transcript to final_transcript
    final_transcript = merged_transcript

    transcript_file = os.path.splitext(audio_file)[0] + output_format
    save_transcript(transcript_file, final_transcript)
    end = time.time()
    info(f"⏱️ Total processing time for {audio_file}: {end - start:.2f} seconds.")

def build_arg_parser():
    """
    Create and configure the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured parser with supported arguments.
    """

    parser = argparse.ArgumentParser(description="Transcribe Audio Files.")
    parser.add_argument("files", nargs='+', help="List of audio files.")
    parser.add_argument("--model", default="medium", help=f"Whisper model to use ({', '.join(available_models())}). Default is 'medium'.")
    parser.add_argument("--format", default=".transcript", help="Output file extension (e.g., .transcript, .html, .json, .xml, .md). Default is '.transcript'.")
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
    if args.model not in available_models():
        parser.error(f"Invalid model '{args.model}'. Supported: {', '.join(available_models())}")
        return

    output_format = args.format
    if not output_format.startswith("."):
        output_format = "." + output_format

    for file in args.files:
        try:
            transcribe(file, model=args.model, output_format=output_format, use_cache=args.use_cache)
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

