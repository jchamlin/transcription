import os
import subprocess
import re
import argparse
import time
import atexit
from logging_utils import setup_logging, debug, info, warning
from diarization_utils import diarize
from transcript_utils import load_transcript, format_timestamp, format_segment

temporary_files = []

def mark_for_deletion(path):
    """
    Marks a file for deletion upon program exit.

    Args:
        path (str): Path to the file to delete later.
    """
    temporary_files.append(path)

def cleanup():
    """
    Deletes all files that were marked for deletion.
    """
    for path in temporary_files:
        if os.path.exists(path):
            os.remove(path)
            info(f"🗑️ Deleted temporary file: {path}")

atexit.register(cleanup)

def convert_to_pcm_s16le_if_needed(audio_path):
    """
    Converts ADPCM IMA WAV files to PCM S16LE format if needed.

    Args:
        audio_path (str): Path to the audio file to inspect and possibly convert.

    Returns:
        str: Path to the original or converted audio file.
    """
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    try:
        codec = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        raise RuntimeError(f"❌ Failed to probe audio codec for {audio_path}")

    if codec != "adpcm_ima_wav":
        return audio_path

    base, ext = os.path.splitext(audio_path)
    converted_path = f"{base} (pcm_s16le){ext}"

    if not os.path.exists(converted_path):
        info(f"🔄 Converting {audio_path} to standard PCM format...")
        convert_cmd = ["ffmpeg", "-y", "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", converted_path]
        subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        info(f"✅ Converted and saved to {converted_path}")
        mark_for_deletion(converted_path)

    return converted_path

padding = 50

def extract_training_audio(audio_file, transcript_file, output_dir, exclude_file=None):
    """
    Extracts training audio segments based on the provided transcript. Skips extraction if output is already up to date.

    Args:
        audio_file (str): Path to the source audio file.
        transcript_file (str): Path to the cleaned transcript file.
        output_dir (str): Path to directory where output slices should be written.
        exclude_file (str, optional): Path to a text file of line numbers to exclude.

    Returns:
        list: A list of dictionaries with details about each extracted segment.
    """
    from pydub import AudioSegment

    start = time.time()
    audio_mtime = os.path.getmtime(audio_file)
    transcript_mtime = os.path.getmtime(transcript_file)
    segments = load_transcript(transcript_file)
    
    all_slice_files = []
    if os.path.isdir(output_dir):
        for dp, _, filenames in os.walk(output_dir):
            for f in filenames:
                if f.endswith('.wav'):
                    all_slice_files.append(os.path.join(dp, f))
    
    if all_slice_files:
        slices_newer = all(os.path.getmtime(f) > max(audio_mtime, transcript_mtime) for f in all_slice_files)
        if slices_newer:
            info("⚡ Skipping audio extraction: all slices already up to date.")
            exported = []
            for f in all_slice_files:
                match = re.search(r"\((\d{3}) ", os.path.basename(f))
                if not match:
                    continue
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(segments):
                    exported.append({
                        "index": idx + 1,
                        "segment": segments[idx],
                        "filename": os.path.basename(f),
                        "path": f
                    })
            exported.sort(key=lambda x: x["index"])
            info(f"↩️ Reusing {len(exported)} previously extracted slices.")
            return exported
  
    info(f"🔹 Extracting training audio from audio file '{audio_file}' using transcript file '{transcript_file}' and writing training slices to '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(audio_file))[0]
    audio_file = convert_to_pcm_s16le_if_needed(audio_file)
    audio = AudioSegment.from_file(audio_file)

    exclude_indices = set()
    if exclude_file and os.path.exists(exclude_file):
        with open(exclude_file, 'r') as ef:
            for line in ef:
                if line.strip().isdigit():
                    exclude_indices.add(int(line.strip()))

    exported = []

    for i, segment in enumerate(segments):
        if (i + 1) in exclude_indices:
            continue

        start_ms = segment["start"] * 1000
        end_ms = segment["end"] * 1000
        if start_ms >= end_ms:
            warning(f"⚠️ Skipping transcript line {i+1}: start >= end → {format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}")
            continue

        segment_speaker = segment["speaker"]
        speaker_folder_name = re.sub(r'[<>:"/\\|?*]', '_', segment["speaker"].strip())
        speaker_folder = os.path.join(output_dir, speaker_folder_name + "_training_audio")
        debug(f"segment_speaker = {segment_speaker} speaker_folder_name = {speaker_folder_name} speaker_folder = {speaker_folder}")
        os.makedirs(speaker_folder, exist_ok=True)

        segment_audio = audio[max(0, int(start_ms - padding)):min(len(audio), int(end_ms + padding))]
        start_fmt = format_timestamp(segment["start"]).replace(':', '-')
        end_fmt = format_timestamp(segment["end"]).replace(':', '-')
        filename = f"{input_basename} ({i+1:03d} {start_fmt} - {end_fmt}).wav"
        output_path = os.path.join(speaker_folder, filename)
        segment_audio.export(output_path, format="wav", codec="pcm_s16le")

        exported.append({
            "index": i + 1,
            "segment": segment,
            "filename": filename,
            "path": output_path
        })

    end = time.time()
    info(f"✅ Extract training audio complete in {end - start:.2f} seconds.")
    return exported

def diarize_training_audio(exported):
    """
    Runs diarization on training audio segments and logs any multi-speaker segments.

    Args:
        exported (list): List of dictionaries containing segment metadata and audio paths.
    """
    from pydub import AudioSegment

    multi_speaker_flags = []

    for entry in exported:
        segment = entry["segment"]
        output_path = entry["path"]
        filename = entry["filename"]

        audio = AudioSegment.from_file(output_path)
        audio_duration = audio.duration_seconds
        expected_duration = (segment["end"] - segment["start"]) + (2 * padding / 1000)

        if abs(audio_duration - expected_duration) > 0.01:
            warning(f"⚠️ Audio duration mismatch for {filename}: expected {expected_duration:.3f}s, got {audio_duration:.3f}s")
        else:
            info(f"🧪 Exported audio duration for {filename}: {audio_duration:.3f}s (✔ matches expected {expected_duration:.3f}s)")

        diarization_result = diarize(output_path)
        adjusted_turns = []
        seg_dur = segment["end"] - segment["start"]

        for turn in diarization_result:
            rel_start = turn["start"] - (padding / 1000)
            rel_end = min(turn["end"] - (padding / 1000), audio_duration - (padding / 1000))
            absolute_start = max(segment["start"], rel_start + segment["start"])
            absolute_end = min(segment["end"], rel_end + segment["start"])
            if absolute_start < segment["start"]:
                warning(f"🚨 Even after clipping, adjusted start {absolute_start:.3f}s < segment start {segment['start']:.3f}s")
            if absolute_end > segment["end"]:
                warning(f"🚨 Even after clipping, adjusted end {absolute_end:.3f}s > segment end {segment['end']:.3f}s")

            overlap_start = max(rel_start, 0)
            overlap_end = min(rel_end, seg_dur)
            overlap_dur = max(0.0, overlap_end - overlap_start)

            adjusted_turns.append({
                "start": absolute_start,
                "end": absolute_end,
                "speaker": turn["speaker"],
                "text": "",
                "content": f"= {100 * overlap_dur / seg_dur:.2f}% ({overlap_dur:.3f}s) offset_start={rel_start:.3f}s offset_end={seg_dur - rel_end:.3f}s"
            })

        if len(set(t["speaker"] for t in adjusted_turns)) > 1:
            lines = []
            lines.append(format_segment(segment))
            for t in adjusted_turns:
                lines.append(format_segment(t))
            multi_speaker_flags.append(f"\n📄 {filename}\n" + "\n".join(lines))

    if multi_speaker_flags:
        warning("\n⚠️ Summary of multi-speaker detections:\n" + "\n\n".join(multi_speaker_flags))
    else:
        info("✅ No multi-speaker segments detected.")

def create_speaker_embeddings(exported, output_dir):
    """
    Create speaker embeddings using SpeechBrain and save them to disk.

    Args:
        exported (list): List of dictionaries containing audio segment info.
        output_dir (str): Path to output directory to save embeddings.
    """
    import torchaudio
    import numpy as np
    from speechbrain.pretrained import EncoderClassifier


    slice_paths = [e["path"] for e in exported]
    embedding_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_embedding.npy")]
    if slice_paths and embedding_paths:
        latest_slice = max(os.path.getmtime(p) for p in slice_paths)
        all_embeddings_newer = all(os.path.getmtime(ep) > latest_slice for ep in embedding_paths)
        if all_embeddings_newer:
            info("⚡ Skipping speaker embedding: embeddings are already up to date.")
            return

    info("🎤 Generating speaker embeddings using SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    speaker_segments = {}

    for entry in exported:
        speaker = entry["segment"]["speaker"]
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(entry["path"])

    for speaker, paths in speaker_segments.items():
        embeddings = []
        for path in paths:
            signal, _ = torchaudio.load(path)
            embedding = classifier.encode_batch(signal).squeeze().detach().numpy()
            embeddings.append(embedding)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            embed_path = os.path.join(output_dir, f"{speaker}_embedding.npy")
            np.save(embed_path, avg_embedding)
            info(f"✅ Saved speaker embedding for {speaker} → {embed_path}")

def verify_speaker_embeddings(output_dir):
    """
    Verifies that each audio file in subfolders matches its speaker embedding.

    Args:
        output_dir (str): Directory containing speaker folders and saved embeddings.
    """
    import torchaudio
    import numpy as np
    from speechbrain.pretrained import EncoderClassifier
    from scipy.spatial.distance import cosine

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    info("🔎 Verifying speaker identities for extracted training slices...")

    # Load all speaker embeddings from the output_dir
    embeddings = {}
    for file in os.listdir(output_dir):
        if file.endswith("_embedding.npy"):
            speaker = file.replace("_embedding.npy", "")
            embeddings[speaker] = np.load(os.path.join(output_dir, file))

    mismatches = []
    for speaker_folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, speaker_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(folder_path, file)
            signal, _ = torchaudio.load(path)
            test_embedding = classifier.encode_batch(signal).squeeze().detach().numpy()

            scores = {
                s: 1 - cosine(test_embedding, emb) for s, emb in embeddings.items()
            }
            best_match = max(scores, key=scores.get)
            best_score = scores[best_match]

            expected = speaker_folder.replace("_training_audio", "")
            detected = best_match.strip()

            if expected != detected:
                mismatches.append((file, expected, detected, best_score))

    for file, expected, detected, score in mismatches:
        warning(f"❌ {file}: expected {expected}, detected {detected} (score: {score:.4f})")

    if not mismatches:
        info("✅ All speaker slices matched their expected embeddings.")
    else:
        warning(f"⚠️ {len(mismatches)} mismatches found during speaker verification.")

def main():
    """
    Main entry point for the training audio extraction pipeline.

    Parses command-line arguments, extracts speaker-specific audio clips,
    optionally performs diarization verification, and builds/verifies speaker embeddings.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Extract training audio by speaker from audio and labeled transcript.")
    parser.add_argument("audio_file", help="Path to the input audio file")
    parser.add_argument("transcript_file", help="Path to the manual .tmp transcript file")
    parser.add_argument("output_dir", help="Directory where extracted segments will be saved")
    parser.add_argument("--exclude-file", help="Optional path to exclusion list (one-based line numbers)", default=None)
    parser.add_argument("--diarize-check", action="store_true", help="Run diarization check on extracted segments")
    args = parser.parse_args()

    extracted_slices = extract_training_audio(args.audio_file, args.transcript_file, args.output_dir, args.exclude_file)
    if args.diarize_check:
        diarize_training_audio(extracted_slices)

    create_speaker_embeddings(extracted_slices, args.output_dir)
    verify_speaker_embeddings(args.output_dir)

if __name__ == "__main__":
    main()
