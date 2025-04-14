import whisper
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
import torchaudio
import json

# ---------------------------
# 🔹 CONFIGURATION
# ---------------------------
AUDIO_FILE = "V20250111-024819 - Sameria's Threats and Blowup.WAV"  # Change to your file name
OUTPUT_FILE = AUDIO_FILE.replace(".WAV", ".txt")  # Output transcript filename

# Optional: Manually assign speaker names after first run
SPEAKER_NAMES = {
    "SPEAKER_00": "Sameria_0",
    "SPEAKER_01": "J.C._1",
    "SPEAKER_02": "J.C._2",
    "SPEAKER_03": "J.C._3",
    "SPEAKER_04": "Sameria_4"
}

# ---------------------------
# 🔹 STEP 1: Run Whisper for Transcription
# ---------------------------
print("🔹 Running Whisper transcription...")
model = whisper.load_model("medium")
whisper_result = model.transcribe(AUDIO_FILE)

# Extract text + timestamps from Whisper
segments = whisper_result["segments"]

# ---------------------------
# 🔹 STEP 2: Run Pyannote for Speaker Diarization
# ---------------------------
print("🔹 Running Pyannote speaker diarization...")
diarization_pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization")
diarization_result = diarization_pipeline(AUDIO_FILE)

# Store speaker timestamps
speaker_labels = []
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    speaker_labels.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
    })

# ---------------------------
# 🔹 STEP 3: Improved Speaker Matching Logic
# ---------------------------
print("🔹 Matching transcript segments to speakers...")

final_transcript = []
for seg in segments:
    start_time = seg["start"]
    end_time = seg["end"]
    text = seg["text"]

    # Find best matching speaker (overlapping segment)
    best_match = None
    max_overlap = 0

    for label in speaker_labels:
        # Calculate overlap
        overlap = min(end_time, label["end"]) - max(start_time, label["start"])
        if overlap > max_overlap and overlap > 0:
            max_overlap = overlap
            best_match = label["speaker"]

    # Assign speaker name
    speaker = SPEAKER_NAMES.get(best_match, best_match) if best_match else "Unknown"

    # Format transcript entry
    final_transcript.append(f"[{start_time:.2f} - {end_time:.2f}] {speaker}: {text}")

# ---------------------------
# 🔹 STEP 4: Save Final Transcript
# ---------------------------
print(f"🔹 Saving transcript to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(final_transcript))

print("✅ Done! Transcript is saved.")
