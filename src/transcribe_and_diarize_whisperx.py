def transcribe_and_diarize_whisperx(audio_file, model="medium", language_code="en"):
    """
    Transcribe the audio file using WhisperX for word-level timestamps and speaker diarization.

    - Uses OpenAI Whisper transcription_model via WhisperX for transcription.
    - Performs word-level alignment for precise timestamps.
    - Performs speaker diarization using Pyannote pipeline via WhisperX.
    - Matches each word to the diarization segment with the highest overlap.
    - Groups consecutive words from the same speaker into transcript segments.

    Parameters:
    - audio_file (str): Path to the audio file to process.
    - model (str): Whisper model to use (e.g., tiny, tiny.en, small, small.en, medium, medium.en, large, large.en).
    - language_code (str): Language code to use for Whisper and alignment (default is "en").

    Returns:
    - transcript (list): List of dicts containing start, end, speaker, content, and confidence.
    """
    import time
    import torch
    import whisperx
    import pandas as pd
    from diarize_pyannote import set_processing_threads
    from logging_utils import info, warning

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"🔹 Running WhisperX transcription on {audio_file} using transcription_model '{model}' and device '{device}'")

    set_processing_threads(1)
    transcription_model = whisperx.load_model(model, device, language=language_code)
    raw_result = transcription_model.transcribe(audio_file)

    alignment_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    aligned_result = whisperx.align(raw_result["segments"], alignment_model, metadata, audio_file, device)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=True, device=device)
    diarized_output = diarize_model(audio_file)

    if isinstance(diarized_output, dict) and "segments" in diarized_output:
        speaker_segments = diarized_output["segments"]
    elif isinstance(diarized_output, pd.DataFrame):
        speaker_segments = diarized_output.to_dict(orient="records")
    else:
        raise ValueError("Unexpected diarized result format: expected dict with 'segments' or DataFrame.")

    info(f"🔍 Found {len(speaker_segments)} speaker entries.")
    if speaker_segments:
        info(f"🔹 Example speaker entry: {speaker_segments[0]}")

    # Assign speaker label to each word using maximum overlap
    for seg in aligned_result["segments"]:
        for word in seg.get("words", []):
            word_start = word.get("start")
            word_end = word.get("end")
            max_overlap = 0
            matched_speaker = None

            for speaker_seg in speaker_segments:
                seg_start = speaker_seg["start"]
                seg_end = speaker_seg["end"]
                overlap = min(word_end, seg_end) - max(word_start, seg_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    matched_speaker = speaker_seg["speaker"]

            if matched_speaker:
                word["speaker"] = matched_speaker

    transcript = []
    current_words = []
    current_speaker = None
    for seg in aligned_result["segments"]:
        for word in seg.get("words", []):
            speaker = word.get("speaker")
            if speaker is None:
                continue

            if current_speaker is not None and speaker != current_speaker and current_words:
                transcript.append({
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "speaker": current_speaker,
                    "content": " ".join(w["word"] for w in current_words),
                    "confidence": None
                })
                current_words = []

            current_words.append(word)
            current_speaker = speaker

    if current_words:
        transcript.append({
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "speaker": current_speaker,
            "content": " ".join(w["word"] for w in current_words),
            "confidence": None
        })

    if not any(seg.get("speaker") for seg in transcript):
        warning("⚠️ No speaker labels detected — falling back to raw aligned segments.")
        transcript = []
        for seg in aligned_result["segments"]:
            transcript.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": None,
                "content": seg.get("text", seg.get("content", "")).strip(),
                "confidence": None
            })

    # Merge short orphan segments into neighboring speaker blocks
    smoothed = []
    i = 0
    while i < len(transcript):
        current = transcript[i]
        duration = current["end"] - current["start"]
        is_short = duration < 0.75

        prev_seg = smoothed[-1] if smoothed else None
        next_seg = transcript[i + 1] if i + 1 < len(transcript) else None

        if is_short:
            merged = False
            # Case: orphan between same-speaker neighbors
            if prev_seg and next_seg and prev_seg["speaker"] == next_seg["speaker"]:
                prev_seg["end"] = next_seg["end"]
                prev_seg["content"] += " " + current["content"] + " " + next_seg["content"]
                smoothed.pop()
                smoothed.append(prev_seg)
                i += 2
                merged = True
            # Case: match next
            elif next_seg and next_seg["speaker"] == current["speaker"]:
                next_seg["start"] = current["start"]
                next_seg["content"] = current["content"] + " " + next_seg["content"]
                i += 1
                merged = True
            # Case: match prev
            elif prev_seg and prev_seg["speaker"] == current["speaker"]:
                prev_seg["end"] = current["end"]
                prev_seg["content"] += " " + current["content"]
                smoothed[-1] = prev_seg
                i += 1
                merged = True
            if not merged:
                smoothed.append(current)
                i += 1
        else:
            smoothed.append(current)
            i += 1

    end = time.time()
    info(f"✅ WhisperX transcription + diarization complete in {end - start:.2f} seconds.")
    return smoothed
