"""
clip_exclusion_gui.py — GUI tool to manage and review clip exclusion_manager for audio training slices.

- Loads and displays exclusion_manager for each clip (auto and manual).
- Supports segment timing parsing and playback-friendly info.
- Allows manual exclusion_manager via checkbox GUI.
- Can populate automated exclusion_manager (volume, duration, clipping).
- Optionally integrates SpeechBrain and diarization for smarter filtering.

Dependencies: tkinter, json, argparse, pydub, speechbrain
"""

import os
import time
import json
import re
import glob
import argparse
import threading
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from utils import setup_logging, suppress_noisy_audio_processing_output, center_popup, load_geometry, save_geometry

script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = setup_logging(script_name)
suppress_noisy_audio_processing_output()

class ExclusionManager:

    def __init__(self, output_dir, speaker):
        self.output_dir = output_dir
        self.speaker = speaker
        self.speaker_folder = os.path.join(output_dir, f"{speaker}_training_audio")
        self.exclusion_file = os.path.join(output_dir, f"{speaker}_training_audio_exclusions.json")
        self.data = {}
        if os.path.exists(self.exclusion_file):
            with open(self.exclusion_file, "r") as f:
                self.data = json.load(f)
        self.sync_with_filesystem()

    def save(self):
        with open(self.exclusion_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_entry(self, filename):
        return self.data.setdefault(filename, {"automatic_exclusion_reasons": [], "manual_exclusion_reasons": [], "confidence": None, "matched_speaker": None})

    def get_exclusion_reasons(self, filename):
        entry = self.get_entry(filename)
        return {
            "automatic": entry.get("automatic_exclusion_reasons", []),
            "manual": entry.get("manual_exclusion_reasons", [])
        }

    def set_confidence(self, filename, score, match):
        entry = self.get_entry(filename)
        entry["confidence"] = score
        entry["matched_speaker"] = match

    def get_all_filenames(self):
        return sorted(self.data.keys())

    def list_audio_files(self, folder):
        return sorted([f for f in os.listdir(folder) if f.endswith(".wav")])

    def sync_with_filesystem(self):
        current_files = self.list_audio_files(self.speaker_folder)
        known_files = set(self.data.keys())
        current_set = set(current_files)

        # Remove files that no longer exist
        for fname in known_files - current_set:
            self.data.pop(fname, None)

        # Add new files with empty info
        for fname in current_set - known_files:
            self.data[fname] = {"automatic_exclusion_reasons": [], "manual_exclusion_reasons": [], "confidence": None, "matched_speaker": None}


class ClipExclusionApp:

    COLUMNS = [
        ("clip", 420, "w"),
        ("segment", 30, "e"),
        ("start", 30, "e"),
        ("end", 30, "e"),
        ("duration", 30, "e"),
        ("confidence", 30, "e"),
        ("matched", 30, "w"),
        ("automatic_exclusion_reasons", 300, "w"),
        ("manual_exclusion_reasons", 300, "w"),
    ]

    EXCLUSION_OPTIONS = [
        "too_short", "too_quiet", "clipped", "multiple_speakers", "low_confidence",
        "speaker_misidentification", "background_noise", "echo", "mumbling",
        "loud", "distant", "poor_quality"
    ]

    def __init__(self, root, exclusion_manager: ExclusionManager, embeddings):
        self.root = root
        self.exclusion_manager = exclusion_manager
        self.embeddings = embeddings
        self.geometry_file = f"../data/{script_name}_geometry.txt"
        self.tree = ttk.Treeview(root, columns=[c[0] for c in self.COLUMNS], show="headings")
        for name, width, anchor in self.COLUMNS:
            self.tree.heading(name, text=" ".join(w.capitalize() for w in name.split("_")))
            self.tree.column(name, width=width, anchor=anchor)
        self.tree.pack(expand=True, fill="both")
        self.tree.bind("<Double-1>", self.on_double_click)
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Rescan", command=self.rescan).pack(side="left", padx=10, pady=5)
        tk.Button(btn_frame, text="Save", command=self.save).pack(side="right", padx=10, pady=5)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        load_geometry(root, self.geometry_file);

    def parse_segment_info(self, filename):
        match = re.search(r"\((\d+)\s+(\d+)-(\d+\.\d+)\s+-\s+(\d+)-(\d+\.\d+)\)", filename)
        if not match:
            return ("", "", "", "")
        seg = int(match.group(1))
        start = f"{match.group(2)}:{match.group(3)}"
        end = f"{match.group(4)}:{match.group(5)}"
        try:
            fmt = "%M:%S.%f"
            delta = datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
            dur = f"{delta.total_seconds():.2f}"
        except Exception:
            dur = ""
        return (seg, start, end, dur)

    def load_data(self):
        self.tree.delete(*self.tree.get_children())
        for fname in self.exclusion_manager.get_all_filenames():
            exclusion_reasons = self.exclusion_manager.get_exclusion_reasons(fname)
            entry = self.exclusion_manager.get_entry(fname)
            seg, start, end, dur = self.parse_segment_info(fname)
            conf_val = entry.get("confidence")
            conf = f"{conf_val * 100:.1f}%" if conf_val is not None else ""
            matched = entry.get("matched_speaker") or ""
            self.tree.insert("", "end", iid=fname, values=(fname, seg, start, end, dur, conf, matched, ", ".join(exclusion_reasons["automatic"]), ", ".join(exclusion_reasons["manual"])))

    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        fname = item
        current = self.exclusion_manager.get_exclusion_reasons(fname)
        updated = self.exclusion_popup(self.root, fname, current["manual"])
        entry = self.exclusion_manager.data.setdefault(fname, {})
        entry["manual_exclusion_reasons"] = updated
        self.load_data()

    def exclusion_popup(self, filename, existing):
        popup = tk.Toplevel(self.root)
        popup.title(f"Edit Manual Exclusions — {filename}")
        popup.grab_set()
        tk.Label(popup, text="Select reasons to exclude this clip:").pack(padx=10, pady=5)
        frame = tk.Frame(popup)
        frame.pack(padx=10, pady=5)
        vars_dict = {}
        for reason in self.EXCLUSION_OPTIONS:
            var = tk.BooleanVar(value=(reason in existing))
            vars_dict[reason] = var
            cb = tk.Checkbutton(frame, text=reason, variable=var)
            cb.pack(anchor="w")
    
        result = []
    
        def on_ok():
            result.extend([k for k, v in vars_dict.items() if v.get()])
            popup.destroy()
    
        tk.Button(popup, text="OK", command=on_ok).pack(pady=5)
        center_popup(self.root, popup)
        popup.wait_window()
        return result

    def rescan(self):
        scan_training_clips(self.exclusion_manager, self.embeddings, self.root, self.load_data)

    def save(self):
        self.exclusion_manager.save()
        messagebox.showinfo("Saved", f"Exclusions saved to {self.exclusion_manager.path}")

    def on_close(self):
        save_geometry(self.root, self.geometry_file)
        self.root.destroy()


def load_embeddings(embeddings_dir):
    import torch
    import numpy as np
    logger.debug(f"Loading embeddings")
    embeddings = {}
    for file in glob.glob(os.path.join(embeddings_dir, "*_embedding.npy")):
        name = os.path.basename(file).replace("_embedding.npy", "")
        vec = np.load(file)
        vec = vec / np.linalg.norm(vec)  # Normalize to unit vector
        embeddings[name] = torch.tensor(vec)
    logger.debug(f"Loaded {len(embeddings)} embeddings")
    return embeddings


_speechbrain_classifiers = {}


def get_speechbrain_classifier(source="speechbrain/spkrec-ecapa-voxceleb"):
    """
    Cache the SpeechBrain classifier globally to make runs on multiple files faster
    """
    from speechbrain.pretrained import EncoderClassifier

    if source not in _speechbrain_classifiers:
        # device="cpu" # Force CPU
        logger.info(f"🔄 Creating SpeechBrain classifier '{source}''")
        _speechbrain_classifiers[source] = EncoderClassifier.from_hparams(source)

    result = _speechbrain_classifiers[source]
    return result


def analyze_with_speechbrain(signal, embeddings):
    import torch
    from scipy.spatial.distance import cosine

    classifier = get_speechbrain_classifier()  # Use cached model
    test_embedding = classifier.encode_batch(signal).squeeze().detach().cpu().numpy()

    matched_speaker = None
    confidence = -1

    for speaker, ref in embeddings.items():
        ref_np = ref.squeeze().detach().numpy() if torch.is_tensor(ref) else ref
        score = 1 - cosine(test_embedding, ref_np)
        if score > confidence:
            confidence = score
            matched_speaker = speaker

    return matched_speaker, confidence


def scan_training_clips(mgr: ExclusionManager, embeddings, root, done_callback):
    import torch
    import numpy as np
    from pydub import AudioSegment

    mgr.sync_with_filesystem()
    total = len(mgr.data)
    popup = tk.Toplevel(root)
    popup.title("Scanning Clips")
    popup.transient(root)
    popup.grab_set()
    popup.focus_force()
    center_popup(root, popup)

    status = tk.StringVar()
    tk.Label(popup, textvariable=status, padx=20, pady=10).pack()
    cancel = [False]

    def on_cancel():
        cancel[0] = True
        popup.destroy()

    tk.Button(popup, text="Cancel", command=on_cancel).pack(pady=10)
    popup.update()

    def run():
        try:
            start = time.time()
            logger.info(f"🔹 Scanning {len(mgr.data)} clips from {mgr.speaker_folder}")
            for i, fname in enumerate(sorted(mgr.data)):
                if cancel[0]:
                    logger.debug("Scan cancelled")
                    break
                entry = mgr.get_entry(fname)
                path = os.path.join(mgr.speaker_folder, fname)
                logger.debug(f"Processing {fname}")
                status.set(f"Processing {i+1}/{total} ({int((i+1)/total*100)}%)\n\n{fname}")
                popup.update()
    
                reasons = []
                logger.debug(f"Loading audio: {fname}")
                audio = AudioSegment.from_file(path)
                dur = len(audio) / 1000.0
                logger.debug(f"Loaded audio: duration = {len(audio) / 1000.0:.2f}s")
                if dur < 0.8: reasons.append("too_short")
                if audio.dBFS < -40: reasons.append("too_quiet")
                if audio.max_dBFS > -1.0: reasons.append("clipped")
 
                samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                signal = torch.tensor(samples).unsqueeze(0) / 32768.0

                # TODO diarization and setting the multiple_speakers automatic_exclusion flag 

                expected_speaker = mgr.speaker
                matched_speaker, confidence = analyze_with_speechbrain(signal, embeddings)
                if confidence < 0.65:
                    matched_speaker = "Unknown"
                    reasons.append("speaker_unknown")
                else:
                    if confidence < 0.80:
                        reasons.append("low_confidence")
                    if matched_speaker != expected_speaker:
                        reasons.append("speaker_misidentification")

                # Update entry
                entry["automatic_exclusion_reasons"] = reasons
                mgr.set_confidence(fname, confidence, matched_speaker)
        
            root.after(0, done_callback)
            popup.destroy()
            end = time.time()
            logger.info(f"⏱️ Scanning {len(mgr.data)} clips from {mgr.speaker_folder} completed in {end - start:.2f} seconds.")

        except Exception as e:
            logger.error(f"Exception in scan thread: {e}", exc_info=True)

    threading.Thread(target=run, daemon=True).start()


from argparse import ArgumentError


def validate_args(args):
    """
    Validates that output_dir exists and speaker folder exists inside it.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ArgumentError: If validation fails.
    """
    if not os.path.isdir(args.output_dir):
        raise ArgumentError(None, f"Output directory '{args.output_dir}' does not exist or is not a directory.")

    speaker_path = os.path.join(args.output_dir, f"{args.speaker}_training_audio")
    if not os.path.isdir(speaker_path):
        raise ArgumentError(None, f"Speaker folder '{speaker_path}' does not exist.")


def execute(args):
    validate_args(args)
    
    logger.debug(f"Initializing ExclusionManager")
    speaker = args.speaker
    exclusions_folder = args.output_dir
    mgr = ExclusionManager(exclusions_folder, speaker)

    logger.debug(f"Loading embeddings")
    embeddings = load_embeddings(args.output_dir)

    logger.debug(f"Initializing Tk GUI")
    root = tk.Tk()
    root.withdraw()
    root.update()
    logger.debug(f"Initializing App")
    app = ClipExclusionApp(root, mgr, embeddings)
    logger.debug(f"Displaying GUI")
    root.deiconify()
    root.title(f"Clip Exclusion Manager — {speaker}")
    
    # Uncomment to simulate a failure for testing error handling:
    # raise Exception("Simulated test failure")

    scan_training_clips(mgr, embeddings, root, app.load_data)

    logger.debug("Starting mainloop()")
    root.mainloop()
    logger.debug("mainloop() exited")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("speaker", help="Speaker name")
    parser.add_argument("--output_dir", default="../data/training")
    return parser


def main(argv=None):
    args = None
    try:
        parser = build_arg_parser()
        args = parser.parse_args(argv)
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
