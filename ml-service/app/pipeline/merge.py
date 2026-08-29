"""Merge word-level STT output with diarization speaker turns.

Critical-path algorithm (see ARCHITECTURE.md): assign each WORD (not whole
segment) to the diarization turn whose interval contains the word's
midpoint, then re-group consecutive same-speaker words into speaker turns.
Word-level assignment gives materially better speaker-boundary accuracy
than segment-level overlap, without a separate forced-alignment model.

Gap words (not covered by any diarization turn — pyannote turns rarely
tile the audio with zero gaps) are resolved by embedding, not by time,
when an embedding model + audio are available (see DECISIONS.md #18):
verified on real audio (ground truth from an independent transcript)
that a short confirming word like "Perfect." spoken by one speaker was
getting glued onto the END of the OTHER speaker's preceding turn purely
because it was closer in time — assigning it by comparing its own voice
against both adjacent turns' voices instead fixes that class of error.
Falls back to time-nearest when no embedding model is available or the
gap is too short to embed meaningfully.
"""

from typing import TypedDict

import numpy as np

from app.pipeline import short_turn_resolver
from app.pipeline.confidence import count_sentences, score_turn_confidence
from app.pipeline.diarize import SpeakerTurn
from app.pipeline.embedding_utils import embed_span
from app.pipeline.stt import Word


class DiarizedTurn(TypedDict):
    start: float
    end: float
    speaker: str
    text: str
    confidence: float
    uncertain: bool
    # True once boundary_refinement.py has actually re-checked this turn
    # with an independent embedding signal and found no evidence of a
    # missed speaker change — distinct from "uncertain=False" turns that
    # were simply short/simple enough to never get flagged in the first
    # place. See apply_review_results() below and DECISIONS.md #16.
    reviewed: bool


def _nearest_turn_speaker(midpoint: float, turns: list[SpeakerTurn]) -> str | None:
    nearest_speaker = None
    nearest_distance = None
    for turn in turns:
        distance = turn["start"] - midpoint if midpoint < turn["start"] else midpoint - turn["end"]
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = turn["speaker"]
    return nearest_speaker


def _classify_gap(
    embedding_model,
    audio: np.ndarray,
    sample_rate: int,
    gap_start: float,
    gap_end: float,
    prev_turn: SpeakerTurn | None,
    next_turn: SpeakerTurn | None,
) -> str | None:
    """Resolves a gap by comparing its own voice against whichever of the
    two adjacent turns actually differ in speaker — cheaper and more
    locally relevant than building a full speaker centroid. Returns None
    if it can't decide (too short to embed, or an adjacent turn's own
    span is too short to embed), letting the caller fall back to
    time-nearest."""
    if prev_turn is None:
        return next_turn["speaker"] if next_turn else None
    if next_turn is None:
        return prev_turn["speaker"]
    if prev_turn["speaker"] == next_turn["speaker"]:
        return prev_turn["speaker"]  # no ambiguity — both sides agree

    gap_embedding = embed_span(embedding_model, audio, sample_rate, gap_start, gap_end)
    prev_embedding = embed_span(embedding_model, audio, sample_rate, prev_turn["start"], prev_turn["end"])
    next_embedding = embed_span(embedding_model, audio, sample_rate, next_turn["start"], next_turn["end"])
    if gap_embedding is None or prev_embedding is None or next_embedding is None:
        return None

    similarity_to_prev = float(np.dot(gap_embedding, prev_embedding))
    similarity_to_next = float(np.dot(gap_embedding, next_embedding))
    return prev_turn["speaker"] if similarity_to_prev >= similarity_to_next else next_turn["speaker"]


def _assign_speakers(
    words: list[Word],
    turns: list[SpeakerTurn],
    embedding_model,
    audio: np.ndarray | None,
    sample_rate: int | None,
) -> list[str]:
    """One speaker label per word in `words`, same order. Words covered by
    a turn get that turn's speaker directly; words falling in a gap are
    grouped into contiguous runs and resolved together (see
    _classify_gap), not word-by-word — a run of gap words is one
    utterance, not independent samples."""
    sorted_turns = sorted(turns, key=lambda t: t["start"])
    covering: list[SpeakerTurn | None] = []
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        candidates = [t for t in sorted_turns if t["start"] <= midpoint <= t["end"]]
        # pyannote turns can genuinely overlap (a short interjection from one
        # speaker nested inside a longer turn from another) — picking the
        # first by start order always favors whichever turn started earliest,
        # silently discarding the other speaker's real, shorter turn every
        # time. The narrowest covering turn is the most specific match for
        # this exact word, so prefer that instead. Verified on real audio: a
        # 0.63s SPEAKER_01 turn nested inside a 0.95s SPEAKER_00 turn was
        # being swallowed entirely before this fix (see DECISIONS.md #21).
        match = min(candidates, key=lambda t: t["end"] - t["start"], default=None)
        covering.append(match)

    speakers: list[str | None] = [t["speaker"] if t else None for t in covering]

    can_embed = embedding_model is not None and audio is not None and sample_rate is not None
    i = 0
    while i < len(words):
        if speakers[i] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and speakers[j] is None:
            j += 1
        # words[i:j] is one contiguous gap run
        gap_start, gap_end = words[i]["start"], words[j - 1]["end"]
        prev_turn = next((t for t in reversed(sorted_turns) if t["end"] <= gap_start), None)
        next_turn = next((t for t in sorted_turns if t["start"] >= gap_end), None)

        resolved = None
        if can_embed:
            resolved = _classify_gap(embedding_model, audio, sample_rate, gap_start, gap_end, prev_turn, next_turn)
        if resolved is None:
            midpoint = (gap_start + gap_end) / 2
            resolved = _nearest_turn_speaker(midpoint, sorted_turns) or "unknown"

        for k in range(i, j):
            speakers[k] = resolved
        i = j

    return [s if s is not None else "unknown" for s in speakers]


def _finalize_turn(turn: DiarizedTurn) -> DiarizedTurn:
    """Scores confidence now that the turn's full span/text is known —
    see confidence.py. This is what surfaces the "long region that's
    actually two speakers, merged because segmentation missed the
    boundary" failure mode instead of silently presenting it as fact."""
    duration = turn["end"] - turn["start"]
    result = score_turn_confidence(duration, count_sentences(turn["text"]))
    turn["confidence"] = result.confidence
    turn["uncertain"] = result.uncertain
    turn["reviewed"] = False
    return turn


def merge_transcript(
    words: list[Word],
    turns: list[SpeakerTurn],
    embedding_model=None,
    audio: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> list[DiarizedTurn]:
    """embedding_model/audio/sample_rate are optional — when provided
    (jobs.py passes them once the diarization embedding is available),
    gap words are resolved by voice instead of by time (see module
    docstring), and short sandwiched word runs are re-checked via
    short_turn_resolver.py (multi-signal — see its module docstring for
    why sub-1.5s spans need more than just an embedding check). Without
    them, falls back to pure time-nearest gap resolution and no
    short-run correction."""
    if not words:
        return []

    word_speakers = _assign_speakers(words, turns, embedding_model, audio, sample_rate)
    if embedding_model is not None and audio is not None and sample_rate is not None:
        word_speakers = short_turn_resolver.resolve_short_runs(words, word_speakers, embedding_model, audio, sample_rate)

    merged: list[DiarizedTurn] = []
    current: DiarizedTurn | None = None

    for word, speaker in zip(words, word_speakers):
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
                "reviewed": False,
            }
    if current is not None:
        merged.append(_finalize_turn(current))
    return merged


# Confidence assigned to a turn that boundary_refinement.py actually
# checked and found no contradicting evidence for — high enough to clear
# the `uncertain` threshold (see confidence.py), but deliberately not 1.0:
# it's corroborated by a second signal, not verified ground truth (see
# DECISIONS.md #16, "never fabricate certainty").
REVIEWED_NO_SPLIT_CONFIDENCE = 0.65


def apply_review_results(turns: list[DiarizedTurn], checked_no_split_spans: list[tuple[float, float]]) -> None:
    """Mutates `turns` in place: for any turn whose span was independently
    re-checked by boundary_refinement.py and found to have no evidence of
    a missed speaker change, clears the `uncertain` flag it got purely
    from the duration/sentence-count heuristic and marks it reviewed —
    otherwise a turn that WAS checked and passed still shows the same
    unexplained "uncertain" warning as one that was never checked at all
    (see DECISIONS.md #16)."""
    for turn in turns:
        for span_start, span_end in checked_no_split_spans:
            if turn["start"] >= span_start and turn["end"] <= span_end:
                turn["reviewed"] = True
                turn["uncertain"] = False
                turn["confidence"] = max(turn["confidence"], REVIEWED_NO_SPLIT_CONFIDENCE)
                break


def format_for_llm(turns: list[DiarizedTurn]) -> str:
    lines = []
    for turn in turns:
        minutes, seconds = divmod(int(turn["start"]), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {turn['speaker']}: {turn['text'].strip()}")
    return "\n".join(lines)
