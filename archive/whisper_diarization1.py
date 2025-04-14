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
    "SPEAKER_00": "J.C.",
    "SPEAKER_01": "Sameria",
    "SPEAKER_02": "J.C._2",
    "SPEAKER_03": "J.C._3",
    "SPEAKER_04": "J.C._4"
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
# 🔹 STEP 3: Merge Transcription with Speaker Labels
# ---------------------------
print("🔹 Merging Whisper transcript with speakers...")

final_transcript = []
for seg in segments:
    start_time = seg["start"]
    end_time = seg["end"]
    text = seg["text"]

    # Find matching speaker based on timestamps
    speaker = "Unknown"
    for label in speaker_labels:
        if label["start"] <= start_time <= label["end"]:
            speaker = SPEAKER_NAMES.get(label["speaker"], label["speaker"])  # Replace with real name if available
            break

    # Format transcript entry
    final_transcript.append(f"[{start_time:.2f} - {end_time:.2f}] {speaker}: {text}")

# ---------------------------
# 🔹 STEP 4: Save Final Transcript
# ---------------------------
print(f"🔹 Saving transcript to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(final_transcript))

print("✅ Done! Transcript is saved.")

