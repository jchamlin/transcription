# 🔁 Continuation Summary – Start of Transcription 9 Chat (April 18, 2025)

The project contents are uploaded at the project level in a zip file called transcription_v9.zip (the "9" is a sequence number for versioning the zips between chats and matches the chat sequence number, this is "Transcription 9"" chat so the zip version we start with is v9). Always use the bash shell command unzip to zip archives. Doing it with your own Python code leads to unpredicable results.

We are now working from the TODO list, recording start times and end times and actual duration with the task and will compare estimates vs. actual in a retrospective after each task is complete. Restrospective notes will be added to the task in the TODO.

## ✅ Progress

- **Extra Syntax Highlighting build automation**: The Extra Syntax Highlighing functionality we added in for transcript files was lost when Ecilpse updated the plugin and deleted our custom jar and replaced it with an updated jar. The project that builds our custom jar now has automation to build and deploy the patched jar through an ant bulid and custom ant tasks.

- **Apple MPS Support**: faster-whisper and CTranslate2 do not support mps, but torch does. So, added mps support for torch. Tested and on the first try it failed, tried to run faster-whisper with mps. So need to probably separate whiper/pyannote/speechbrain devices, or do something smart in faster-whisper if device is mps

- **ComputeDevices and Processors**: Refactored all torch related logic into a torch_utils.py file and in the process of converting it and all related code into classes This will help solve the problem when using device="mps" which does't work with faster-whisper.

## 🗂️ Canvases

| Name                      | Type              | Description                                                                      |
| ------------------------- | ----------------- | -------------------------------------------------------------------------------- |
| transcribe.py             | code/python       | Transcription + diarization pipeline                                             |
| extract_training_audio.py | code/python       | Audio slicing and embedding                                                      |
| clip_exclusion_gui.py     | code/python       | GUI for managing clip exclusions                                                 |
| Active Code               | code/python       | We can interactively pair program on whatever content is loaded in this canvas   |
| Console Log               | document/markdown | Track stdout/debug/errors from recent runs                                       |
| Transcription 9 Summary   | document/markdown | This continuation summary & task list                                            |
| Transcription TODO List   | document/markdown | An active TODO list of tasks                                                     |
| Rules for ChatGPT         | document/markdown | A collection of rules that are VERY important to have submitted with each prompt |
