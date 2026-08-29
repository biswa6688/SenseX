"""Merge word-level STT output with diarization speaker turns.

Critical-path algorithm (see ARCHITECTURE.md): assign each WORD (not whole
segment) to the diarization turn whose interval contains the word's
midpoint, then re-group consecutive same-speaker words into speaker turns.
Word-level assignment gives materially better speaker-boundary accuracy
than segment-level overlap, without a separate forced-alignment model.
"""

from typing import TypedDict

from app.pipeline.confidence import count_sentences, score_turn_confidence
from app.pipeline.diarize import SpeakerTurn
from app.pipeline.stt import Word


class DiarizedTurn(TypedDict):
    start: float
    end: float
    speaker: str
    text: str
    confidence: float
    uncertain: bool


def _find_speaker(word: Word, turns: list[SpeakerTurn]) -> str:
    """Assign the word to whichever diarization turn contains its midpoint;
    if it falls in a gap between turns (pyannote turns rarely tile the
    audio with zero gaps — short pauses, breaths, and overlap-resolution
    all leave slivers), fall back to the nearest turn by distance instead
    of "unknown". A silence gap belongs to whichever speaker is talking
    on either side of it, not to a phantom extra speaker."""
    if not turns:
        return "unknown"
    midpoint = (word["start"] + word["end"]) / 2
    nearest_speaker = None
    nearest_distance = None
    for turn in turns:
        if turn["start"] <= midpoint <= turn["end"]:
            return turn["speaker"]
        distance = turn["start"] - midpoint if midpoint < turn["start"] else midpoint - turn["end"]
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = turn["speaker"]
    return nearest_speaker if nearest_speaker is not None else "unknown"


def _finalize_turn(turn: DiarizedTurn) -> DiarizedTurn:
    """Scores confidence now that the turn's full span/text is known —
    see confidence.py. This is what surfaces the "long region that's
    actually two speakers, merged because segmentation missed the
    boundary" failure mode instead of silently presenting it as fact."""
    duration = turn["end"] - turn["start"]
    result = score_turn_confidence(duration, count_sentences(turn["text"]))
    turn["confidence"] = result.confidence
    turn["uncertain"] = result.uncertain
    return turn


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
                merged.append(_finalize_turn(current))
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "text": word["word"],
                "confidence": 1.0,
                "uncertain": False,
            }
    if current is not None:
        merged.append(_finalize_turn(current))
    return merged


def format_for_llm(turns: list[DiarizedTurn]) -> str:
    lines = []
    for turn in turns:
        minutes, seconds = divmod(int(turn["start"]), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {turn['speaker']}: {turn['text'].strip()}")
    return "\n".join(lines)
