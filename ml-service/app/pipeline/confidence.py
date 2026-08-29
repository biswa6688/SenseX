"""Confidence scoring for diarized turns.

No refinement backend (3D-Speaker/WeSpeaker) exists yet — see
DECISIONS.md #14 — so this can't escalate a low-confidence turn to a
second pass. What it does now: surface which turns are *likely* wrong,
using signals that are already available for free, so the data model and
UI can mark them uncertain instead of silently presenting a bad merge as
ground truth. This is exactly the failure mode reported on real audio —
a long region that's actually two speakers, folded into one turn because
pyannote's segmentation never emitted a boundary there.
"""

import re
from dataclasses import dataclass

# A diarization turn longer than this is exactly the observed failure
# mode: segmentation missing a speaker change through a fast back-and-
# forth exchange. Length alone is suspicious past this point.
LONG_TURN_THRESHOLD_SECONDS = 12.0
# Beyond LONG_TURN_THRESHOLD_SECONDS, confidence falls off linearly until
# a turn this long is treated as very likely wrong.
MAX_PENALIZED_OVERAGE_SECONDS = 60.0
# More than this many sentence-like chunks packed into one diarization
# turn is itself suspicious — a real single utterance rarely contains
# this many complete sentences back to back.
MANY_SENTENCES_THRESHOLD = 3

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+")


@dataclass
class TurnConfidence:
    confidence: float
    uncertain: bool

    def to_dict(self) -> dict:
        return {"confidence": self.confidence, "uncertain": self.uncertain}


def count_sentences(text: str) -> int:
    """Cheap sentence-like-chunk count via punctuation splitting — not
    real sentence segmentation, just a proxy for "how much distinct
    content is packed into this one diarization turn"."""
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]
    return max(1, len(parts))


def score_turn_confidence(turn_duration: float, sentence_count: int) -> TurnConfidence:
    """Returns confidence in [0, 1]; lower means this turn is more likely
    to actually contain a missed speaker change. `uncertain` is a simple
    thresholded view of the same score for UI/consumers that just want a
    boolean."""
    confidence = 1.0

    if turn_duration > LONG_TURN_THRESHOLD_SECONDS:
        overage = min(turn_duration - LONG_TURN_THRESHOLD_SECONDS, MAX_PENALIZED_OVERAGE_SECONDS)
        confidence -= (overage / MAX_PENALIZED_OVERAGE_SECONDS) * 0.7

    if sentence_count > MANY_SENTENCES_THRESHOLD:
        confidence -= min(sentence_count - MANY_SENTENCES_THRESHOLD, 5) * 0.05

    confidence = max(0.0, min(1.0, confidence))
    # turn_duration can be numpy-derived (pyannote/whisper timestamps are
    # np.float64) — `confidence < 0.6` on that lineage yields np.bool_,
    # which json.dumps rejects (unlike np.float64, which it happens to
    # accept). Cast explicitly rather than rely on that asymmetry.
    return TurnConfidence(confidence=round(float(confidence), 3), uncertain=bool(confidence < 0.6))
