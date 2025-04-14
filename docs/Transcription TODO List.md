# **📌 Transcription TODO List**

## 🔧 Core Refactors *(~6–9 hrs)*

- **Unify Audio Loading**: Refactor code to use audio_utils.py load_audio_file() everywhere. Work with ChatGPT if the existing loading is using torchaudio instead of using pydub AudioSegment to convert the code. *(~1.5 hrs)*

- **Signal-Based Refactor**: Refactor transcription, diarization, and speaker identification to be able to process both a file on the disk and a signal loaded by audio_utils.py load_audio_file(). Refactor code to use the signal version everywhere. *(~2 hrs)*

- **Speaker Utils Module**: Create speaker_identification_utils.py and add in SpeechBrain logic to mirror transcription_utils.py and diarization_utils.py and use everywhere. Move all SpeechBrain-specific preprocessing logic (e.g., converting AudioSegment to tensor) into `speaker_identification_utils.py` Standardize on using `pydub.AudioSegment` for all speaker matching and embedding creation to avoid inconsistencies with torchaudio *(~2 hrs)*

- **Move Checks to Audio Utils**: Move audio file loading and other checks (loudness, clipping, etc) to audio_utils.py *(~1 hr)*

## 🧠 Data Preparation & Exclusion Logic *(~3–5 hrs)*

- **Add Diarization-Based Exclusion**: Add diarization check and setting of exclusion flags (multiple_speakers) to clip_exclusion_gui.py *(~1.5 hrs)*

- **Rename Extracted Training Files**: Change the output format of the extract_training_audio.py to use the original audio file name and add the parenthesis at the end, but this time include the speaker name in there too, so like (Sameria 001 start - end).wav. *(~0.5 hrs)*

- **Restructure Training Folder Layout**: Restructure training audio folders to place everything in ../data/training instead of having a folder per speaker. If we need to reorganize that folder in the future because it gets too big, we can do that again at a later time. *(~1–2 hrs)*

## 🖥️ GUI Improvements *(~6–10 hrs)*

- **Add Sort and Filter UI**: Add the ability to sort and filter the data in the table. Most important filters would be: actual speaker, and the ability to show only rows with a particular exclusion reason or reasons, and the ability to show rows without a particular exclusion reason or reasons. *(~2 hrs)*

- **Keyboard Shortcuts for Manual Review**:
  → Spacebar opens audio preview and starts playback
  → Spacebar again closes it
  → Up/down arrow keys change rows and auto-play new clip
  → Enter opens the manual exclusions dialog
  → Assign unique shortcuts (e.g., 1, 2, 3...) to toggle manual exclusion criteria
  → Automatic exclusions remain read-only and only change via scan *(~2 hrs)*

- **Audio Preview Flow (Ocenaudio Style)**:
  → When a row is selected, spacebar opens an embedded audio preview panel (or lightweight popup) and starts playback.
  → Pressing space again stops playback and closes the preview.
  → Up/down arrow keys change rows, and the preview auto-updates to the new clip.
  → This mimics the fast review workflow of Ocenaudio while keeping the interface simple and keyboard-driven.
  → Use pyqtgraph or similar to quickly display waveform without playing audio. Avoids the lag of launching full audio playback. Cache waveforms once generated. *(~2–4 hrs)*

- **Refactor Speaker Column Behavior**: Refactor GUI to no longer require a speaker, and instead show the actual speaker and detected speaker (with the full speaker name including the _segment part) in each row. *(~1–2 hrs)*

## 🧪 Embedding & Transcription Enhancements *(~4–6 hrs)*

- **Embed Per Segment**: Create embeddings for each training segment, instead of trying to average them all. Use the original audio file name but use the .npy suffix instead. *(~1–2 hrs)*

- **Evaluate Self-Match Accuracy**: See if each embedding is the one that matches exactly to its own training segment during testing. *(~1 hr)*

- **Batch Mode for Transcribe.py**: Allow transcribe.py to accept multiple files and/or directories and pass the entire list to transcribe and have it process the list, recognizing directories and recursively processing all files in that directory and all subdirectories *(~1–2 hrs)*

- **Normalize WAV Files + Metadata**: Create code to cleanup the original WAV files: change name from .WAV to .wav, sanity check the date specified in the filename with the file's actual date created and date modified on the file system. Also, check metadata in the wav file, and adjust accordingly (adding location where the audio clip was recorded which is the folder name, the date the recording was created, and any other standard metadata fields which can be added). Preserve the file's date created and date modified attributes anytime the file is updated, so the original date created and date modified timestamps on the file are not lost when the file's metadata is updated. *(~1–2 hrs)*

## ⚙️ Optional & Stability Improvements *(~2–3 hrs)*

- **OPTIONAL: Add GUI Skip Flags**: (if scan is too slow after adding diarization): Add skip flags for diarization and embedding from GUI. → Adds flexibility for developers to re-scan clips without recomputing all intermediate steps. A checkable option in the GUI could control whether embedding/diarization runs. *(~1 hr)*

- **OPTIONAL: Robust Model Error Handling**: (tool doesn't fail now, but should log something if it does, show an error dialog, and recover) Improve error handling in transcription and diarization functions. → Wrap all model runs with try/except. Log traceback. Recover gracefully if segment fails. *(~1–2 hrs)*
