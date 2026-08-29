"""Multi-boundary change-point detection for long diarization turns.

boundary_refinement.py's old single-boundary logic found at most the
SINGLE strongest similarity drop in a turn and split it in two — correct
when a long turn contains exactly one missed speaker change, but a
genuinely long turn can contain MULTIPLE real changes (e.g. a short
customer "Yes." buried inside a long agent explanation). A single-
boundary split, even applied iteratively (boundary_refinement.refine_transcript's
rounds), only ever re-checks the REMAINDER of a previous split — it
won't find a short middle segment whose surrounding context on both
sides still looks like the original speaker.

This instead computes similarity across the WHOLE window trajectory for
a candidate turn, finds every local-minimum drop below threshold (not
just the global minimum), and independently reclusters each resulting
segment against the known speaker centroids — genuine change-point
detection instead of one-boundary bisection.
"""

import numpy as np

from app.pipeline.diarize import SpeakerTurn
from app.pipeline.embedding_utils import embed_span
from app.pipeline.merge import DiarizedTurn

WINDOW_SECONDS = 2.0
HOP_SECONDS = 0.5
SIMILARITY_DROP_THRESHOLD = 0.5
# A segment shorter than this (turn edge to nearest boundary, or between
# two boundaries) is dropped rather than kept as its own tiny segment —
# avoids degenerate near-edge or near-duplicate boundaries.
MIN_SEGMENT_SECONDS = 1.0
# Candidate boundaries within this many seconds of each other are merged
# into one, at their local-minimum similarity — adjacent low-similarity
# windows from the SAME real change point shouldn't become two boundaries.
MIN_BOUNDARY_SEPARATION_SECONDS = 1.0


def _sliding_embeddings(embedding_model, audio: np.ndarray, sample_rate: int, start: float, end: float):
    window_starts = []
    t = start
    while t + WINDOW_SECONDS <= end:
        window_starts.append(t)
        t += HOP_SECONDS
    embeddings = [embed_span(embedding_model, audio, sample_rate, ws, ws + WINDOW_SECONDS) for ws in window_starts]
    return window_starts, embeddings


def _find_all_boundaries(embedding_model, audio: np.ndarray, sample_rate: int, start: float, end: float) -> list[float]:
    """Every candidate boundary timestamp where adjacent-window similarity
    drops below threshold — not just the single strongest one."""
    window_starts, embeddings = _sliding_embeddings(embedding_model, audio, sample_rate, start, end)
    if len(window_starts) < 2:
        return []

    drops: list[tuple[float, float]] = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        if a is None or b is None:
            continue
        similarity = float(np.dot(a, b))
        if similarity < SIMILARITY_DROP_THRESHOLD:
            boundary_time = window_starts[i] + WINDOW_SECONDS / 2 + HOP_SECONDS / 2
            drops.append((boundary_time, similarity))

    if not drops:
        return []

    merged: list[float] = []
    cluster = [drops[0]]
    for time, sim in drops[1:]:
        if time - cluster[-1][0] <= MIN_BOUNDARY_SEPARATION_SECONDS:
            cluster.append((time, sim))
        else:
            merged.append(min(cluster, key=lambda d: d[1])[0])
            cluster = [(time, sim)]
    merged.append(min(cluster, key=lambda d: d[1])[0])

    return [b for b in merged if b - start >= MIN_SEGMENT_SECONDS and end - b >= MIN_SEGMENT_SECONDS]


def refine_long_turn(
    embedding_model,
    audio: np.ndarray,
    sample_rate: int,
    turn: DiarizedTurn,
    centroids: dict[str, np.ndarray],
) -> list[SpeakerTurn] | None:
    """Returns replacement segments for `turn` (each independently
    classified against `centroids`), or None if no internal boundary was
    found / everything reclustered back to one speaker — caller should
    leave the turn unchanged in that case."""
    boundaries = _find_all_boundaries(embedding_model, audio, sample_rate, turn["start"], turn["end"])
    if not boundaries:
        return None

    edges = [turn["start"], *boundaries, turn["end"]]
    segments: list[SpeakerTurn] = []
    for seg_start, seg_end in zip(edges, edges[1:]):
        speaker = turn["speaker"]
        embedding = embed_span(embedding_model, audio, sample_rate, seg_start, seg_end)
        if embedding is not None and centroids:
            best_speaker, best_similarity = None, -1.0
            for candidate, centroid in centroids.items():
                similarity = float(np.dot(embedding, centroid))
                if similarity > best_similarity:
                    best_similarity, best_speaker = similarity, candidate
            if best_speaker is not None:
                speaker = best_speaker
        segments.append({"start": seg_start, "end": seg_end, "speaker": speaker})

    collapsed: list[SpeakerTurn] = []
    for seg in segments:
        if collapsed and collapsed[-1]["speaker"] == seg["speaker"]:
            collapsed[-1]["end"] = seg["end"]
        else:
            collapsed.append(dict(seg))

    if len(collapsed) < 2:
        return None  # everything reclustered to one speaker -- no real change

    return collapsed
