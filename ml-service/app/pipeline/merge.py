"""Merge word-level STT output with diarization speaker turns.

Critical-path algorithm (see ARCHITECTURE.md): assign each WORD (not whole
segment) to the diarization turn whose interval contains the word's
midpoint, then re-group consecutive same-speaker words into speaker turns.
Word-level assignment gives materially better speaker-boundary accuracy
than segment-level overlap, without a separate forced-alignment model.
"""

from typing import TypedDict

from app.pipeline.diarize import SpeakerTurn
from app.pipeline.stt import Word


class DiarizedTurn(TypedDict):
    start: float
    end: float
    speaker: str
    text: str


def _find_speaker(word: Word, turns: list[SpeakerTurn]) -> str:
    midpoint = (word["start"] + word["end"]) / 2
    for turn in turns:
        if turn["start"] <= midpoint <= turn["end"]:
            return turn["speaker"]
    return "unknown"


def merge_transcript(words: list[Word], turns: list[SpeakerTurn]) -> list[DiarizedTurn]:
    if not words:
        return []

    merged: list[DiarizedTurn] = []
    current: DiarizedTurn | None = None

    for word in words:
        speaker = _find_speaker(word, turns)
        if current is not None and current["speaker"] == speaker:
            current["end"] = word["end"]
            current["text"] += word["word"]
        else:
            if current is not None:
                merged.append(current)
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "text": word["word"],
            }
    if current is not None:
        merged.append(current)
    return merged


def format_for_llm(turns: list[DiarizedTurn]) -> str:
    lines = []
    for turn in turns:
        minutes, seconds = divmod(int(turn["start"]), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {turn['speaker']}: {turn['text'].strip()}")
    return "\n".join(lines)
