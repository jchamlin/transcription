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
from argparse import ArgumentError
from utils import setup_logging, suppress_noisy_audio_processing_output, save_transcript
from compute_providers.torch_utils import get_device, get_compute_type, get_num_threads
from compute_providers.torch_utils import get_available_devices as torch_available_devices, get_available_compute_types as torch_available_compute_types
from processors import get_available_models as torch_available_models, transcribe as run_transcription
from processors import diarize
from processors import merge_transcription_and_diarization

script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = setup_logging(script_name)
suppress_noisy_audio_processing_output()

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
    logger.info(f"🔹 Transcribing '{audio_file}' using output_format '{output_format}' model '{model_size_or_path}' device '{device}' compute_type='{compute_type}' num_threads='{num_threads}'")
    if _USE_WHISPERX:
        from processors import transcribe_and_diarize_whisperx
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
    logger.info(f"⏱️Total processing time for {audio_file}: {end - start:.2f} seconds.")


def get_available_models():
    """
    Returns the list of models available for the active transcription backend.

    Returns:
        list[str]: List of supported model identifiers (e.g., ['tiny', 'base', 'small', ...]).
    """
    # TODO: later dispatch based on environment/backend
    return torch_available_models()


def get_available_devices():
    """
    Returns the list of devices available for the active transcription backend.

    Returns:
        list[str]: List of supported device strings (e.g., ['cuda', 'cpu']).
    """
    # TODO: later dispatch based on environment/backend
    return torch_available_devices()


def get_available_compute_types(device):
    """
    Returns the list of compute types available for the given device and backend.

    Args:
        device (str): The target compute device (e.g., 'cuda', 'cpu')

    Returns:
        list[str]: Supported compute types for this device.
    """
    # TODO: later dispatch based on environment/backend
    return torch_available_compute_types(device)


def validate_args(args):
    """
    Validates parsed arguments to ensure they are correct and complete.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ArgumentError: If validation fails (e.g., missing files or invalid model).
    """
    if not args.files:
        raise ArgumentError(None, "No files were provided. Use -h for help.")

    if args.model_size_or_path and args.model_size_or_path not in get_available_models():
        raise ArgumentError(None, f"Invalid model '{args.model_size_or_path}'. Supported: {', '.join(get_available_models())}")

    if args.output_format and args.output_format != ".transcript":
        raise ArgumentError(None, "Only '.transcript' output format is currently supported.")

    if args.device and args.device not in get_available_devices():
        raise ArgumentError(None, f"Invalid device '{args.device}'. Supported devices: {', '.join(get_available_devices())}")

    if args.compute_type and args.compute_type not in get_available_compute_types(args.device or get_device()):
        raise ArgumentError(None, f"Invalid compute_type '{args.compute_type}' for device '{args.device or get_device()}'. Supported: {', '.join(get_available_compute_types(args.device or get_device()))}")

    if args.num_threads is not None:
        max_threads = os.cpu_count()
        if args.num_threads < 1 or args.num_threads > max_threads:
            raise ArgumentError(None, f"Invalid thread count '{args.num_threads}'. Must be between 1 and {max_threads}.")


def execute(args):
    validate_args(args)

    if not args.output_format.startswith("."):
        args.output_format = "." + args.output_format

    transcribe_args = {k: v for k, v in vars(args).items() if k != "files"}

    for file in args.files:
        try:
            transcribe(file, **transcribe_args)
        except Exception as e:
            logger.error(f"❌ Failed to transcribe {file}: {e}")
            logger.error(traceback.format_exc())


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
    parser.add_argument("-m", "--model", dest="model_size_or_path", default=None, help=f"Whisper model to use ({', '.join(get_available_models())}). Default is 'medium'.")
    parser.add_argument("-d", "--device", default=None, help=f"Device to use: {', '.join(get_available_devices())}. Default is '{device}'.")
    parser.add_argument("-c", "--compute-type", default=None, help=f"Whisper compute_type to use, based on device. Options: {', '.join(get_available_compute_types(device))}. Default is '{compute_type}'.")
    parser.add_argument("-t", "--threads", dest="num_threads", type=int, default=None, help=f"Number of threads to use in CPU mode. Default is '{threads}'.")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", help="Use cached transcription and diarization results if available (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable cache use and always regenerate results.")
    parser.set_defaults(use_cache=True)
    return parser


def main():
    """
    Main entry point for CLI execution. Parses arguments and runs the pipeline
    for each specified audio file.
    """
    args = None
    try:
        parser = build_arg_parser()
        args = parser.parse_args()
        execute(args)
    except ArgumentError as ae:
        parser.error(str(ae))
    except KeyboardInterrupt:
        logger.info("⛔ Interrupted by user")
    except Exception:
        logging.basicConfig(level=logging.DEBUG)
        logging.exception(f"❌ Unhandled exception in main({args})")


if __name__ == "__main__":
    main()
