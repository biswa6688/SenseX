"""Text-to-speech via Piper (subprocess, CPU).

Used two ways (per product decision, see DECISIONS.md):
1. Pipeline stage: synthesize the generated Summary text -> summary.wav.
2. Standalone free-text tool: synthesize arbitrary user-supplied text,
   independent of any audio-analysis job.

Piper invoked as a subprocess against the venv's own piper.exe (installed
via the piper-tts wheel), not the piper-tts Python API, for closer parity
with how the model files are laid out (.onnx + .onnx.json pairs).
"""

import subprocess
import sys
from pathlib import Path

from app.core.config import settings

_PIPER_BIN = Path(sys.executable).with_name("piper.exe")

# rhasspy/piper-voices repo layout: <lang>/<lang_region>/<name>/<quality>/<file>.
# hf_hub_download preserves this nested path under local_dir, so lookups here
# must mirror app/routers/models.py's PIPER_VOICE_REPO_PATHS download target.
PIPER_VOICE_REPO_PATHS = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
}


def _model_path(voice: str) -> Path:
    rel = PIPER_VOICE_REPO_PATHS.get(voice, voice)
    return settings.models_cache_dir / "piper" / f"{rel}.onnx"


def synthesize(text: str, output_path: Path, voice: str | None = None) -> Path:
    voice = voice or settings.piper_voice
    model_path = _model_path(voice)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [str(_PIPER_BIN), "-m", str(model_path), "-f", str(output_path)],
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
    )
    return output_path
