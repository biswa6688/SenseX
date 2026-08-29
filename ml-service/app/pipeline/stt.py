"""Speech-to-text via faster-whisper (CTranslate2, CPU, int8).

Model: settings.whisper_model (default distil-large-v3, English-tuned).
Swap to large-v3-turbo for multilingual audio via WHISPER_MODEL env var.
"""

import threading
from functools import lru_cache
from typing import TypedDict

from faster_whisper import WhisperModel

from app.core.config import settings


class Word(TypedDict):
    start: float
    end: float
    word: str
    probability: float


class TranscriptResult(TypedDict):
    language: str
    words: list[Word]


_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(
        settings.whisper_model,
        device="cpu",
        compute_type=settings.whisper_compute_type,
        download_root=str(settings.models_cache_dir / "whisper"),
    )


def transcribe(audio_path: str) -> TranscriptResult:
    model = get_whisper_model()
    with _lock:  # model.transcribe is not thread-safe; queue is single-worker anyway, belt+suspenders
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        words: list[Word] = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
            for segment in segments
            for w in (segment.words or [])
        ]
    return {"language": info.language, "words": words}
