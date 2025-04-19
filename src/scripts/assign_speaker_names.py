# ---------------------------
# 🔹 CONFIGURATION
# ---------------------------
INPUT_FILE = "V20250111-024819 - Sameria's Threats and Blowup.tmp"  # Change to your tmp file name
OUTPUT_FILE = INPUT_FILE.replace(".tmp", ".txt")  # Final output file

# Manually assign real names to speakers
SPEAKER_NAMES = {
    "SPEAKER_00": "Sameria_0",
    "SPEAKER_01": "J.C._1",
    "SPEAKER_02": "Sameria_2",
    "SPEAKER_03": "J.C._3",
    "SPEAKER_04": "Sameria_4"
}

# ---------------------------
# 🔹 STEP 1: Read Temporary Transcript
# ---------------------------
print("🔹 Reading temporary transcript...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    transcript_lines = f.readlines()

# ---------------------------
# 🔹 STEP 2: Replace Speaker Labels with Real Names
# ---------------------------
print("🔹 Replacing speaker labels...")
updated_transcript = []
for line in transcript_lines:
    for generic_speaker, real_name in SPEAKER_NAMES.items():
        line = line.replace(generic_speaker, real_name)
    updated_transcript.append(line)

# ---------------------------
# 🔹 STEP 3: Save Final Transcript
# ---------------------------
print(f"🔹 Saving final transcript as {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("".join(updated_transcript))

print("✅ Done! Final transcript saved as a .txt file.")
