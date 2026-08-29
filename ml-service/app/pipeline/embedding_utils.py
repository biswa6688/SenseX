"""Shared audio-span embedding helper — used by merge.py (gap resolution),
boundary_refinement.py (turn-boundary scanning), and short_turn_resolver.py
(short-span classification), so the "embed this time range, normalize,
skip if too short" logic exists in exactly one place."""

import numpy as np
import torch

# A span shorter than this can't be embedded meaningfully — too few
# samples for any embedding model to produce a reliable vector.
MIN_EMBED_SECONDS = 0.3


def embed_span(embedding_model, audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray | None:
    s = max(0, int(start * sample_rate))
    e = min(len(audio), int(end * sample_rate))
    if e - s < int(MIN_EMBED_SECONDS * sample_rate):
        return None
    chunk = audio[s:e]
    waveform = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    embedding = embedding_model(waveform)[0]
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 1e-8 else None
