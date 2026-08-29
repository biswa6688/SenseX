"""Speaker diarization via pyannote.audio (CPU).

Model: settings.diarization_model (pyannote/speaker-diarization-community-1,
see DECISIONS.md #13 for why not speaker-diarization-3.1). Gated HF model:
requires HF_TOKEN env var set to a token that has accepted the model's
license on huggingface.co. This cannot be auto-bundled.

If no token / no accepted license, we fall back to a single "Speaker 1"
turn spanning the whole file rather than failing the whole job — the rest
of the pipeline (transcript, summary, sentiment, QA) still has value even
without real speaker separation.

KNOWN ISSUE, mitigated not fixed (see DECISIONS.md #14): the pipeline's VBx
clustering is an iterative Variational Bayes refinement (like k-means) —
verified on real audio that identical input + identical settings can
converge to different local optima across separate runs, occasionally
collapsing most of a call onto one speaker even though num_speakers is
already forced correctly. We retry up to DIARIZATION_MAX_ATTEMPTS times and
keep whichever attempt has the most balanced per-speaker talk time, since a
real conversation is rarely extremely lopsided but a bad local optimum is —
this doesn't guarantee a correct result, just reduces how often a bad one
reaches the user.
"""

import logging
from typing import TypedDict

import soundfile as sf

from app.core.config import settings

logger = logging.getLogger(__name__)

_pipeline_cache = None

# See module docstring: VBx clustering's local-optima sensitivity means a
# single run isn't reliable. Retry up to this many times, keeping the most
# balanced result; stop early once one looks clearly good (see
# _is_good_enough) rather than always paying the full cost.
DIARIZATION_MAX_ATTEMPTS = 3
# Ratio of (least-talkative / most-talkative) speaker's total time. A real
# two-party conversation is essentially never this lopsided; a collapsed/bad
# clustering run is (e.g. 9s vs 190s -> 0.05). Anything at or above this on
# the first attempt is accepted without spending time on further retries.
DIARIZATION_GOOD_ENOUGH_BALANCE = 0.3


class SpeakerTurn(TypedDict):
    start: float
    end: float
    speaker: str


def get_diarization_pipeline():
    """Loads and memoizes the diarization pipeline — but only on success.
    A gated-access failure (missing token / license not yet accepted) is
    NOT cached, so a later retry (e.g. the Models page Validate button,
    after the user accepts the license) can succeed without a process
    restart."""
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    if not settings.hf_token:
        return None
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(settings.diarization_model, token=settings.hf_token)
    except Exception:
        logger.exception("Failed to load diarization pipeline; falling back to single-speaker")
        return None

    # Mirrors the .complete marker convention in routers/models.py — lets the
    # Models page report real diarization status instead of the static
    # "gated, can't auto-download" note once the token/license actually work.
    marker = settings.models_cache_dir / "diarization" / ".complete"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    _pipeline_cache = pipeline
    return pipeline


def validate_pipeline() -> bool:
    """Used by the Models page's Validate action."""
    return get_diarization_pipeline() is not None


def _audio_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def _talk_time_balance(turns: list[SpeakerTurn]) -> float:
    """(least-talkative speaker's total time) / (most-talkative speaker's
    total time), in [0, 1]. Higher = more balanced = more likely correct.
    0 if fewer than 2 speakers have any speech at all."""
    talk: dict[str, float] = {}
    for turn in turns:
        talk[turn["speaker"]] = talk.get(turn["speaker"], 0.0) + (turn["end"] - turn["start"])
    values = sorted(v for v in talk.values() if v > 0)
    if len(values) < 2:
        return 0.0
    return values[0] / values[-1]


def _run_pipeline_once(pipeline, audio_path: str) -> list[SpeakerTurn]:
    kwargs = {}
    if settings.diarization_num_speakers is not None:
        kwargs["num_speakers"] = settings.diarization_num_speakers
    output = pipeline(audio_path, **kwargs)
    # pyannote.audio 4.x returns a DiarizeOutput wrapper; older versions
    # returned the Annotation directly. Handle both.
    annotation = output.speaker_diarization if hasattr(output, "speaker_diarization") else output
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def diarize(audio_path: str) -> list[SpeakerTurn]:
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return [{"start": 0.0, "end": _audio_duration_seconds(audio_path), "speaker": "Speaker 1"}]

    best_turns: list[SpeakerTurn] | None = None
    best_balance = -1.0
    for attempt in range(1, DIARIZATION_MAX_ATTEMPTS + 1):
        turns = _run_pipeline_once(pipeline, audio_path)
        balance = _talk_time_balance(turns)
        logger.info("diarization attempt %d/%d: talk-time balance=%.3f", attempt, DIARIZATION_MAX_ATTEMPTS, balance)
        if balance > best_balance:
            best_balance = balance
            best_turns = turns
        if balance >= DIARIZATION_GOOD_ENOUGH_BALANCE:
            break

    assert best_turns is not None  # loop always runs at least once
    return best_turns
