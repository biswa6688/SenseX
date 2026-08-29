from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage_dir: Path = Path(__file__).resolve().parents[3] / "storage"
    models_cache_dir: Path = Path(__file__).resolve().parents[3] / "storage" / "models"

    whisper_model: str = "distil-large-v3"
    whisper_compute_type: str = "int8"

    diarization_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str | None = None
    # Defaults to 2 (agent + customer) — this app's QA rubric (greeting,
    # empathy, resolution, compliance...) is inherently a 2-party call-center
    # framework, and pyannote's unconstrained clustering is unreliable on
    # calls like these: it both split one speaker across two clusters and
    # merged rapid back-and-forth exchanges into one giant single-speaker
    # turn (see DECISIONS.md). Set to None to let pyannote auto-detect the
    # speaker count instead (e.g. for multi-party calls), or override per
    # deployment via DIARIZATION_NUM_SPEAKERS in ml-service/.env.
    diarization_num_speakers: int | None = 2

    llm_model_path: str = "llm/qwen2.5-3b-instruct-q4_k_m.gguf"  # relative to models_cache_dir
    llm_context_size: int = 8192

    piper_voice: str = "en_US-lessac-medium"

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)
settings.models_cache_dir.mkdir(parents=True, exist_ok=True)
(settings.storage_dir / "audio-jobs").mkdir(parents=True, exist_ok=True)
