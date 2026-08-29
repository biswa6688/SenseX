"""Turn-boundary refinement via short-window re-embedding.

Reuses the diarization pipeline's OWN already-loaded embedding model (see
diarize.get_embedding_model()) — no new dependency, no new RAM — to
independently re-check turns confidence.py flagged as suspicious: long,
content-dense turns that are exactly the observed failure mode
(pyannote's segmentation missing a speaker change; see DECISIONS.md #13).

Only runs on flagged turns, not the whole recording. Finds at most one
additional boundary per flagged turn (the single strongest similarity
drop between adjacent short windows) — doesn't attempt to recover
multiple missed boundaries within one turn. This is a bounded first
pass, not a guarantee (see DECISIONS.md #15): it reuses the same
embedding space as the first diarization pass, so it can't catch cases
where that embedding space itself can't distinguish the two speakers.
"""

import numpy as np
import soundfile as sf
import torch

from app.pipeline.diarize import SpeakerTurn
from app.pipeline.merge import DiarizedTurn

WINDOW_SECONDS = 2.0
HOP_SECONDS = 0.5
# Below this cosine similarity between adjacent short windows is treated
# as a candidate speaker-change point.
SIMILARITY_DROP_THRESHOLD = 0.5
# A candidate boundary must leave at least this much audio on each side
# to be worth splitting on — avoids degenerate near-edge splits.
MIN_SEGMENT_SECONDS = 1.5


def _read_mono(audio_path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    return audio[:, 0], sample_rate


def _embed_window(embedding_model, audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray | None:
    s = max(0, int(start * sample_rate))
    e = min(len(audio), int(end * sample_rate))
    if e - s < int(0.3 * sample_rate):  # too short to embed meaningfully
        return None
    chunk = audio[s:e]
    waveform = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    embedding = embedding_model(waveform)[0]
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 1e-8 else None


def _find_best_boundary(
    embedding_model, audio: np.ndarray, sample_rate: int, start: float, end: float
) -> float | None:
    window_starts = []
    t = start
    while t + WINDOW_SECONDS <= end:
        window_starts.append(t)
        t += HOP_SECONDS
    if len(window_starts) < 2:
        return None

    embeddings = [_embed_window(embedding_model, audio, sample_rate, ws, ws + WINDOW_SECONDS) for ws in window_starts]

    best_index = None
    best_similarity = 1.0
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        if a is None or b is None:
            continue
        similarity = float(np.dot(a, b))
        if similarity < best_similarity:
            best_similarity = similarity
            best_index = i

    if best_index is None or best_similarity >= SIMILARITY_DROP_THRESHOLD:
        return None

    boundary = window_starts[best_index] + WINDOW_SECONDS / 2 + HOP_SECONDS / 2
    if boundary - start < MIN_SEGMENT_SECONDS or end - boundary < MIN_SEGMENT_SECONDS:
        return None
    return boundary


def _speaker_centroids(
    embedding_model, audio: np.ndarray, sample_rate: int, confident_turns: list[DiarizedTurn]
) -> dict[str, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for turn in confident_turns:
        emb = _embed_window(embedding_model, audio, sample_rate, turn["start"], turn["end"])
        if emb is None:
            continue
        sums[turn["speaker"]] = sums.get(turn["speaker"], np.zeros_like(emb)) + emb
        counts[turn["speaker"]] = counts.get(turn["speaker"], 0) + 1
    return {speaker: sums[speaker] / counts[speaker] for speaker in sums}


def _nearest_speaker(embedding: np.ndarray | None, centroids: dict[str, np.ndarray], fallback: str) -> str:
    if embedding is None or not centroids:
        return fallback
    best_speaker = fallback
    best_similarity = -1.0
    for speaker, centroid in centroids.items():
        similarity = float(np.dot(embedding, centroid))
        if similarity > best_similarity:
            best_similarity = similarity
            best_speaker = speaker
    return best_speaker


def refine_diarization_turns(
    embedding_model,
    audio_path: str,
    raw_turns: list[SpeakerTurn],
    diarized_turns: list[DiarizedTurn],
) -> list[SpeakerTurn]:
    """Returns a possibly-modified copy of raw_turns (the pyannote-level
    turn timeline, before word-level merge) with an extra split inserted
    for each uncertain diarized turn where a clear boundary candidate was
    found. Caller should re-run merge.merge_transcript() with the result
    to get correctly re-split text/confidence — this only touches the
    speaker-turn timeline, not transcript text."""
    uncertain = [t for t in diarized_turns if t["uncertain"]]
    if not uncertain:
        return raw_turns

    confident = [t for t in diarized_turns if not t["uncertain"]]
    audio, sample_rate = _read_mono(audio_path)
    if sample_rate != embedding_model.sample_rate:
        return raw_turns  # shouldn't happen (canonical audio is 16kHz) — bail safely rather than embed garbage

    centroids = _speaker_centroids(embedding_model, audio, sample_rate, confident)

    refined = list(raw_turns)
    for turn in uncertain:
        boundary = _find_best_boundary(embedding_model, audio, sample_rate, turn["start"], turn["end"])
        if boundary is None:
            continue

        left_embedding = _embed_window(embedding_model, audio, sample_rate, turn["start"], boundary)
        right_embedding = _embed_window(embedding_model, audio, sample_rate, boundary, turn["end"])
        left_speaker = _nearest_speaker(left_embedding, centroids, turn["speaker"])
        right_speaker = _nearest_speaker(right_embedding, centroids, turn["speaker"])
        if left_speaker == right_speaker:
            continue  # re-embedding agrees with the original single-speaker call; nothing to change

        # Replace whichever raw diarization turns this span overlaps with a clean two-piece split.
        refined = [t for t in refined if not (t["start"] < turn["end"] and t["end"] > turn["start"])]
        refined.append({"start": turn["start"], "end": boundary, "speaker": left_speaker})
        refined.append({"start": boundary, "end": turn["end"], "speaker": right_speaker})

    refined.sort(key=lambda t: t["start"])
    return refined
