"""Speaker diarization via pyannote.audio (CPU).

Model: settings.diarization_model (pyannote/speaker-diarization-3.1).
Gated HF model: requires HF_TOKEN env var set to a token that has accepted
the model's license on huggingface.co. This cannot be auto-bundled.

If no token / no accepted license, we fall back to a single "Speaker 1"
turn spanning the whole file rather than failing the whole job — the rest
of the pipeline (transcript, summary, sentiment, QA) still has value even
without real speaker separation.
"""

import logging
from functools import lru_cache
from typing import TypedDict

import soundfile as sf

from app.core.config import settings

logger = logging.getLogger(__name__)


class SpeakerTurn(TypedDict):
    start: float
    end: float
    speaker: str


@lru_cache(maxsize=1)
def get_diarization_pipeline():
    if not settings.hf_token:
        return None
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(settings.diarization_model, token=settings.hf_token)
    except Exception:
        logger.exception("Failed to load diarization pipeline; falling back to single-speaker")
        return None


def _audio_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def diarize(audio_path: str) -> list[SpeakerTurn]:
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return [{"start": 0.0, "end": _audio_duration_seconds(audio_path), "speaker": "Speaker 1"}]

    diarization = pipeline(audio_path)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
