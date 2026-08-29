"""Canonical audio preprocessing.

Both faster-whisper (via PyAV) and pyannote (via torchcodec) previously
decoded the ORIGINAL uploaded file independently, through two different
codec libraries, on every job. This produces one deterministic
mono/16kHz/s16 WAV via ffmpeg per job, and every downstream model reads
from that instead — avoids redundant decode work and removes "different
libraries decoding the same file slightly differently" as a variable.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPreprocessError(Exception):
    pass


@dataclass
class AudioMetadata:
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    original_codec: str | None
    original_sample_rate: int | None
    original_channels: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def canonicalize_audio(input_path: str, output_path: Path) -> AudioMetadata:
    """Runs ffmpeg to produce a deterministic mono/16kHz/s16 WAV at
    output_path. Returns metadata about both the canonical output and
    (best-effort, via ffprobe) the original input."""
    original_info = _probe_original(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_path.exists():
        raise AudioPreprocessError(f"ffmpeg canonicalization failed: {result.stderr[-2000:]}")

    info = sf.info(str(output_path))
    return AudioMetadata(
        duration_seconds=info.frames / info.samplerate,
        sample_rate=info.samplerate,
        channels=info.channels,
        bit_depth=16,  # -sample_fmt s16 above is fixed, not derived from info
        original_codec=original_info.get("codec"),
        original_sample_rate=original_info.get("sample_rate"),
        original_channels=original_info.get("channels"),
    )


def _probe_original(input_path: str) -> dict:
    """Best-effort ffprobe of the original file, for metadata/provenance
    only (see AudioMetadata) — never fatal if it fails, since it doesn't
    affect the actual canonicalization."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", input_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return {
                    "codec": stream.get("codec_name"),
                    "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                    "channels": stream.get("channels"),
                }
    except Exception:
        logger.exception("ffprobe of original audio failed (non-fatal, metadata only)")
    return {}
