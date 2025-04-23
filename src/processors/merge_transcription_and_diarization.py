import logging
import time
from utils import format_timestamp

logger = logging.getLogger(__name__)

def merge_transcription_and_diarization(transcript, diarization):
    """
    Merge transcription with speaker labels and show a confidence percentage.

    Parameters:
    - transcript (list): Whisper transcript segments with start/end/content.
    - diarization (list): Pyannote segments with start/end/speaker.

    Returns:
    - list of transcript segments with added speaker and confidence.
    """
    logger.info(f"🔹 Matching transcript segments to speakers...")
    start = time.time()
    merged_transcript = []

    def calculate_confidence(start, end, speaker):
        duration = end - start
        for label in diarization:
            if label["speaker"] != speaker:
                continue
            overlap = min(end, label["end"]) - max(start, label["start"])
            if overlap > 0:
                return round((overlap / duration) * 100, 2)
        return 0.0

    for i, seg in enumerate(transcript):
        start_time = seg["start"]
        end_time = seg["end"]
        content = seg["content"]

        if i > 0 and content.strip() == transcript[i - 1]["content"].strip():  # Detects Whisper hallucination after silence
            max_overlap = 0
            for label in diarization:
                overlap = min(end_time, label["end"]) - max(start_time, label["start"])
                if overlap > max_overlap:
                    max_overlap = overlap
            if max_overlap == 0:
                logger.warning(f"⚠️ Removing hallucinated duplicate: '{content}' at {format_timestamp(start_time)}")
                continue

        best_match = None
        best_confidence = 0.0
        speaker_overlap_stats = {}

        overlapping_segments = [label for label in diarization if min(end_time, label["end"]) - max(start_time, label["start"]) > 0]
        merged_by_speaker = {}
        for label in overlapping_segments:
            spk = label["speaker"]
            if spk in merged_by_speaker:
                merged_by_speaker[spk]["start"] = min(merged_by_speaker[spk]["start"], label["start"])
                merged_by_speaker[spk]["end"] = max(merged_by_speaker[spk]["end"], label["end"])
            else:
                merged_by_speaker[spk] = label.copy()

        for spk, label in merged_by_speaker.items():
            overlap_start = max(start_time, label["start"])
            overlap_end = min(end_time, label["end"])
            overlap = overlap_end - overlap_start
            if overlap > 0:
                duration = end_time - start_time
                ratio = overlap / duration if duration > 0 else 0
                start_offset = label["start"] - start_time
                end_offset = label["end"] - end_time
                speaker_overlap_stats[spk] = {
                    "duration": overlap,
                    "percent": ratio,
                    "start_offset": start_offset,
                    "end_offset": end_offset
                }
                if ratio > best_confidence:
                    best_match = spk
                    best_confidence = ratio

        filtered_stats = {
            spk: vals for spk, vals in speaker_overlap_stats.items()
            if vals["duration"] > 0.5 and vals["percent"] > 0.1
        }

        sorted_stats = sorted(filtered_stats.items(), key=lambda x: x[1]["percent"], reverse=True)
        if len(sorted_stats) > 1:
            dominant = sorted_stats[0][1]["percent"]
            secondary = [val["percent"] for _, val in sorted_stats[1:]]
            overlap_segments = [merged_by_speaker[spk] for spk in filtered_stats]
            overlap_segments.sort(key=lambda s: s["start"])

            overlaps_non_overlapping = all(
                overlap_segments[i]["end"] <= overlap_segments[i + 1]["start"]
                for i in range(len(overlap_segments) - 1)
            )

            if not (dominant >= 0.80 and all(p < 0.50 for p in secondary)) and overlaps_non_overlapping:
                left_speaker = sorted_stats[0][0]
                right_speaker = sorted_stats[1][0]
                left_seg = merged_by_speaker[left_speaker]
                right_seg = merged_by_speaker[right_speaker]

                left_end = left_seg["end"]
                right_start = right_seg["start"]

                if not (start_time < left_end < end_time and start_time < right_start < end_time):
                    split_time = (left_end + right_start) / 2
                else:
                    split_time = (left_end + right_start) / 2

                estimated_split_index = int(len(content) * ((split_time - start_time) / (end_time - start_time)))
                punctuations = [',', '.', ';', '?', '!']  # Added '!' for better punctuation-based splits
                max_distance = int(len(content) * 0.30)

                best_split_point = None
                for direction in (-1, 1):
                    for dist in range(1, max_distance + 1):
                        pos = estimated_split_index + dist * direction
                        if 0 <= pos < len(content) and content[pos] in punctuations:
                            best_split_point = pos + 1
                            break
                    if best_split_point is not None:
                        break

                if best_split_point is None:
                    best_split_point = content.rfind(' ', 0, estimated_split_index)
                    if best_split_point == -1:
                        best_split_point = content.find(' ', estimated_split_index)
                    if best_split_point == -1:
                        best_split_point = estimated_split_index

                left_text = content[:best_split_point].strip()
                right_text = content[best_split_point:].strip()

                left_conf = calculate_confidence(start_time, left_end, left_speaker)
                right_conf = calculate_confidence(right_start, end_time, right_speaker)

                message = [
                    "⚠️ Split multi-speaker segment:",
                    f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}] {content}",
                ]
                for spk, vals in filtered_stats.items():
                    message.append(
                        f"[{format_timestamp(merged_by_speaker[spk]['start'])} - {format_timestamp(merged_by_speaker[spk]['end'])}] {spk} = {vals['percent'] * 100:.2f}% ({vals['duration']:.3f}s) offset_start={vals['start_offset']:.3f}s offset_end={vals['end_offset']:.3f}s"
                    )
                message.append("-> Split:")
                message.append(
                    f"[{format_timestamp(start_time)} - {format_timestamp(left_end)}] {left_speaker} ({left_conf}%): {left_text}"
                )
                message.append(
                    f"[{format_timestamp(right_start)} - {format_timestamp(end_time)}] {right_speaker} ({right_conf}%): {right_text}"
                )
                logger.warning("\n".join(message))

                merged_transcript.append({
                    "start": start_time,
                    "end": left_end,
                    "speaker": left_speaker,
                    "confidence": left_conf,
                    "content": left_text
                })
                merged_transcript.append({
                    "start": right_start,
                    "end": end_time,
                    "speaker": right_speaker,
                    "confidence": right_conf,
                    "content": right_text
                })
                continue

        speaker = best_match or "SPEAKER_UNKNOWN"
        confidence_pct = round(best_confidence * 100, 2) if best_match else 0.0

        merged_transcript.append({
            "start": start_time,
            "end": end_time,
            "speaker": speaker,
            "confidence": confidence_pct,
            "content": content
        })

    end = time.time()
    logger.info(f"✅ Segment-to-speaker matching complete in {end - start:.2f} seconds.")
    return merged_transcript
