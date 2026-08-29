"""Turn-boundary refinement orchestration.

Independently re-checks turns that are either flagged by confidence.py
(long, content-dense — the "segmentation missed a boundary in a long
region" failure mode, see DECISIONS.md #13) OR simply long enough to be
worth a check regardless of the flag (see MIN_CHECK_DURATION_SECONDS,
DECISIONS.md #18) — verified on real audio (ground truth from an
independent transcript) that a genuine two-speaker merge can be short
enough to never trip confidence.py's heuristic at all, so gating solely
on that flag misses real errors.

Embedding source is caller-supplied — jobs.py prefers three_d_speaker.py
(an independent model, see DECISIONS.md #17), falling back to
diarize.get_embedding_model() (reusing the diarization pipeline's own
embedding, see DECISIONS.md #15) if that's unavailable.

The actual boundary search is long_turn_refiner.py — genuine multi-
boundary change-point detection (every local similarity-drop, not just
the single strongest one), since a long turn can contain MORE THAN ONE
missed speaker change (see DECISIONS.md #20). refine_transcript() calls
refine_diarization_turns() repeatedly (bounded by MAX_REFINEMENT_ROUNDS)
so a still-long remainder gets its own chance to be checked instead of
being left unreviewed. This is a bounded process, not a guarantee: an
embedding source that can't distinguish two speakers in a given
recording won't find a boundary no matter how many rounds run.
"""

import numpy as np
import soundfile as sf

from app.pipeline import long_turn_refiner, merge
from app.pipeline.diarize import SpeakerTurn
from app.pipeline.embedding_utils import embed_span
from app.pipeline.merge import DiarizedTurn
from app.pipeline.stt import Word

# Turns at least this long get a boundary check even if confidence.py
# never flagged them — see module docstring: a real two-speaker merge can
# be short enough to stay under confidence.py's duration/sentence
# thresholds. Low enough to catch a ~10-15s merged exchange, high enough
# to skip trivial single-word acknowledgments ("Yes.", "No.") that are a
# poor match for 2-second window scanning anyway.
MIN_CHECK_DURATION_SECONDS = 4.0
# refine_transcript() re-checks newly-created uncertain/long-enough turns
# (a split leaves a remainder that can itself be worth checking) up to
# this many rounds — bounded so a genuinely long single-speaker monologue
# doesn't get pathologically subdivided. Matches the
# DIARIZATION_MAX_ATTEMPTS=3 precedent in diarize.py.
MAX_REFINEMENT_ROUNDS = 3


def needs_boundary_check(turn: DiarizedTurn) -> bool:
    return turn["uncertain"] or (turn["end"] - turn["start"]) >= MIN_CHECK_DURATION_SECONDS


def read_mono(audio_path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    return audio[:, 0], sample_rate


def _speaker_centroids(
    embedding_model, audio: np.ndarray, sample_rate: int, confident_turns: list[DiarizedTurn]
) -> dict[str, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for turn in confident_turns:
        emb = embed_span(embedding_model, audio, sample_rate, turn["start"], turn["end"])
        if emb is None:
            continue
        sums[turn["speaker"]] = sums.get(turn["speaker"], np.zeros_like(emb)) + emb
        counts[turn["speaker"]] = counts.get(turn["speaker"], 0) + 1
    return {speaker: sums[speaker] / counts[speaker] for speaker in sums}


def refine_diarization_turns(
    embedding_model,
    audio: np.ndarray,
    sample_rate: int,
    raw_turns: list[SpeakerTurn],
    diarized_turns: list[DiarizedTurn],
) -> tuple[list[SpeakerTurn], list[tuple[float, float]]]:
    """Returns (refined_raw_turns, checked_no_split_spans).

    refined_raw_turns: a possibly-modified copy of raw_turns (the
    pyannote-level turn timeline, before word-level merge) with an extra
    split inserted for each checked turn (see needs_boundary_check) where
    a clear boundary candidate was found. Caller should re-run
    merge.merge_transcript() with it to get correctly re-split
    text/confidence — this only touches the speaker-turn timeline, not
    transcript text.

    checked_no_split_spans: (start, end) of every checked turn that found
    no evidence of a missed speaker change (no candidate boundary, or a
    candidate whose two halves both matched the same known speaker).
    Caller should pass this to merge.apply_review_results() after
    re-merging, so a turn that was checked-and-passed doesn't keep
    showing the same unexplained "uncertain" warning as one that was
    never checked (see DECISIONS.md #16)."""
    candidates = [t for t in diarized_turns if needs_boundary_check(t)]
    if not candidates:
        return raw_turns, []

    # Deliberately NOT "not needs_boundary_check(t)" — that would exclude
    # every long-but-probably-fine turn from the centroid pool too (since
    # needs_boundary_check() now fires on length alone, not just
    # confidence.py's uncertain flag), starving it down to only very short
    # turns. Verified this was a real bug: on one real job it left only
    # one speaker with any centroid at all, so every comparison trivially
    # "agreed" and no split could ever be applied — see DECISIONS.md #18.
    # A turn is still a valid reference for building OTHER turns'
    # centroids as long as confidence.py itself didn't flag it, regardless
    # of whether it's independently long enough to also be a check
    # candidate.
    confident = [t for t in diarized_turns if not t["uncertain"]]
    if sample_rate != embedding_model.sample_rate:
        return raw_turns, []  # shouldn't happen (canonical audio is 16kHz) — bail safely rather than embed garbage

    centroids = _speaker_centroids(embedding_model, audio, sample_rate, confident)

    refined = list(raw_turns)
    checked_no_split_spans: list[tuple[float, float]] = []
    for turn in candidates:
        segments = long_turn_refiner.refine_long_turn(embedding_model, audio, sample_rate, turn, centroids)
        if segments is None:
            checked_no_split_spans.append((turn["start"], turn["end"]))
            continue

        # Replace whichever raw diarization turns this span overlaps with the multi-piece split.
        refined = [t for t in refined if not (t["start"] < turn["end"] and t["end"] > turn["start"])]
        refined.extend(segments)

    refined.sort(key=lambda t: t["start"])
    return refined, checked_no_split_spans


def refine_transcript(
    embedding_model,
    audio: np.ndarray,
    sample_rate: int,
    words: list[Word],
    turns: list[SpeakerTurn],
    diarized: list[DiarizedTurn],
) -> list[DiarizedTurn]:
    """Repeatedly calls refine_diarization_turns() + re-merges, so a split's
    remainder gets its own chance to be checked instead of being left as a
    fresh, never-reviewed turn — bounded by MAX_REFINEMENT_ROUNDS. Applies
    merge.apply_review_results() once at the end across every round's
    checked-no-split spans (a span that wasn't split stays intact through
    later rounds, so accumulating and applying at the end is safe).

    Takes the already-loaded audio (jobs.py reads it once for the first
    merge_transcript() pass and reuses it here) — merge.merge_transcript()
    also receives it, so gap-word resolution during re-merges stays
    embedding-aware too (see merge.py's module docstring)."""
    all_checked_no_split_spans: list[tuple[float, float]] = []

    for _ in range(MAX_REFINEMENT_ROUNDS):
        if not any(needs_boundary_check(t) for t in diarized):
            break
        refined_turns, checked_no_split_spans = refine_diarization_turns(
            embedding_model, audio, sample_rate, turns, diarized
        )
        all_checked_no_split_spans.extend(checked_no_split_spans)
        if refined_turns == turns:
            break  # no split happened this round; further rounds would repeat the same result
        turns = refined_turns
        diarized = merge.merge_transcript(words, turns, embedding_model, audio, sample_rate)

    merge.apply_review_results(diarized, all_checked_no_split_spans)
    return diarized
