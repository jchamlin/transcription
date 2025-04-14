import os
import re
from logging_utils import info, error

def format_timestamp(seconds):
    """
    Convert time in seconds to a mm:ss.sss format string.

    Parameters:
    - seconds (float): Time in seconds.

    Returns:
    - str: Formatted timestamp.
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"

def format_segment(seg):
    """
    Format a transcript segment dictionary into a standardized string.

    Supports both Whisper-only segments and those with speaker/score labels.

    Returns:
    - str: Formatted transcript line
    """
    start_fmt = format_timestamp(seg["start"])
    end_fmt = format_timestamp(seg["end"])

    speaker = seg.get("speaker")
    confidence = seg.get("confidence")
    content = seg.get("content")

    confidence_str = f" ({confidence}% match)" if confidence is not None else ""
    speaker_str = f" {speaker}{confidence_str}" if speaker else ""
    content_str = content if content is not None else ""

    return f"[{start_fmt} - {end_fmt}]{speaker_str}: {content_str}".rstrip()

SEGMENT_LINE_RE = re.compile(
    r"\[(\d+):([\d.]+) - (\d+):([\d.]+)](?: ([^:(]+?)(?: \(([\d.]+)% match\))?:)? ?(.*)"
)

def parse_segment(line):
    """
    Parse a transcript line (with or without speaker/confidence/content) into a segment dict.
    Supports Whisper-only, diarization-only, and speaker-labeled formats.

    Returns:
    - dict with keys: start, end, optionally speaker, confidence, content
    """
    match = SEGMENT_LINE_RE.match(line.strip())
    if not match:
        raise ValueError(f"Line format not recognized: {line.strip()}")

    start_min, start_sec, end_min, end_sec, speaker, confidence, content = match.groups()
    segment = {
        "start": int(start_min) * 60 + float(start_sec),
        "end": int(end_min) * 60 + float(end_sec)
    }
    if speaker:
        segment["speaker"] = speaker.strip()
    if confidence:
        segment["confidence"] = round(float(confidence), 2)
    if content:
        segment["content"] = content.strip()

    return segment

def load_transcript(transcript_file):
    if not os.path.exists(transcript_file):
        return None

    info(f"🔹 Loading transcript from {transcript_file}")
    with open(transcript_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    transcript = []
    for line in lines:
        try:
            segment = parse_segment(line)
            transcript.append(segment)
        except Exception as e:
            error(f"⚠️ Failed to parse line in cached transcript: {line.strip()} - {e}")

    return transcript

def save_transcript(transcript_file, transcript):
    """
    Save the transcript to the disk
    """
    info(f"🔹 Saving transcript to {transcript_file}")
    with open(transcript_file, "w", encoding="utf-8") as f:
        for seg in transcript:
            f.write(format_segment(seg) + "\n")

_transcription_models = {}
