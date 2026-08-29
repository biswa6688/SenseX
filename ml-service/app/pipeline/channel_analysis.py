"""Stereo / dual-channel detection.

Call-center recordings sometimes put each party on a separate audio
channel (e.g. LEFT=customer, RIGHT=agent). When that's genuinely the
case, per-channel speaker attribution is far more reliable than
diarization — no clustering/segmentation guesswork needed at all. This
inspects the ORIGINAL file's channels (before audio_preprocess.py's mono
downmix) and decides whether that bypass is safe.

Not every stereo file is dual-speaker — some are just a duplicated mono
signal on both channels. Two signals distinguish them: each channel must
independently contain a meaningful amount of speech (a channel that's
mostly silent isn't carrying its own speaker), and the channels must NOT
be highly correlated (a duplicated-mono file has near-1.0 correlation;
genuinely separate mics/lines are much lower).
"""

from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf

# A channel needs at least this fraction of frames above the energy
# threshold to count as "has its own speech" (not near-silent/crosstalk
# bleed only).
MIN_SPEECH_RATIO = 0.05
# Above this, channels are treated as duplicates of each other rather
# than independent speakers, regardless of speech ratio.
MAX_CORRELATION_FOR_DUAL_SPEAKER = 0.5


@dataclass
class ChannelAnalysis:
    channel_count: int
    is_dual_speaker_candidate: bool
    channel_speech_ratio: list[float]
    correlation: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_channels(audio_path: str) -> ChannelAnalysis:
    data, sample_rate = sf.read(audio_path, always_2d=True)
    channel_count = data.shape[1]

    if channel_count < 2:
        return ChannelAnalysis(
            channel_count=channel_count,
            is_dual_speaker_candidate=False,
            channel_speech_ratio=[1.0] if channel_count == 1 else [],
            correlation=1.0,
            confidence=1.0,
        )

    # Only the first two channels matter for the dual-speaker case this
    # bypasses diarization for; anything beyond that still needs normal
    # diarization regardless.
    ch0 = data[:, 0].astype(np.float64)
    ch1 = data[:, 1].astype(np.float64)

    speech_ratio_0 = _speech_activity_ratio(ch0, sample_rate)
    speech_ratio_1 = _speech_activity_ratio(ch1, sample_rate)
    correlation = _correlation(ch0, ch1)

    is_candidate = (
        speech_ratio_0 >= MIN_SPEECH_RATIO
        and speech_ratio_1 >= MIN_SPEECH_RATIO
        and correlation <= MAX_CORRELATION_FOR_DUAL_SPEAKER
    )

    # Low correlation + both channels carrying meaningful independent
    # speech pushes confidence up; either signal being weak pulls it down.
    confidence = max(0.0, min(1.0, (1.0 - correlation) * min(speech_ratio_0, speech_ratio_1) * 4))

    return ChannelAnalysis(
        channel_count=channel_count,
        is_dual_speaker_candidate=is_candidate,
        channel_speech_ratio=[round(speech_ratio_0, 4), round(speech_ratio_1, 4)],
        correlation=round(correlation, 4),
        confidence=round(confidence, 4),
    )


def _speech_activity_ratio(
    channel: np.ndarray, sample_rate: int, frame_ms: int = 30, rms_threshold: float = 0.01
) -> float:
    """Fraction of frames whose RMS energy exceeds a simple silence
    threshold — a cheap energy-based proxy for "this channel has its own
    speech", not a real VAD model (that's what the segmentation stage is
    for downstream)."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(channel) // frame_len
    if n_frames == 0:
        return 0.0
    trimmed = channel[: n_frames * frame_len].reshape(n_frames, frame_len)
    peak = np.max(np.abs(channel)) or 1.0
    rms = np.sqrt(np.mean((trimmed / peak) ** 2, axis=1))
    return float(np.mean(rms > rms_threshold))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    a, b = a[:n], b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(abs(corr)) if not np.isnan(corr) else 1.0
