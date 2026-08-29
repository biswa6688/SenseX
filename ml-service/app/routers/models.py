from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.pipeline.tts import PIPER_VOICE_REPO_PATHS

router = APIRouter(prefix="/models", tags=["models"])

REQUIRED_MODELS = [
    {
        "id": "whisper",
        "name": f"faster-whisper {settings.whisper_model}",
        "repo": f"Systran/faster-whisper-{settings.whisper_model}",
        "requiresAuth": False,
    },
    {
        "id": "diarization",
        "name": settings.diarization_model,
        "repo": settings.diarization_model,
        "requiresAuth": True,
        "authNote": "Gated model. Requires a free HuggingFace account, accepting "
        "the model license, and HF_TOKEN set in ml-service/.env. Cannot be "
        "downloaded automatically from this page.",
    },
    {
        "id": "llm",
        "name": "Qwen2.5-3B-Instruct (GGUF Q4_K_M)",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "requiresAuth": False,
    },
    {
        "id": "piper-voice",
        "name": settings.piper_voice,
        "repo": "rhasspy/piper-voices",
        "requiresAuth": False,
    },
]


def _marker(model_id: str):
    return settings.models_cache_dir / model_id / ".complete"


def _is_downloaded(model_id: str) -> bool:
    return _marker(model_id).exists()


def _do_download(model_id: str) -> None:
    marker = _marker(model_id)
    marker.parent.mkdir(parents=True, exist_ok=True)

    if model_id == "whisper":
        from app.pipeline.stt import get_whisper_model

        get_whisper_model()  # faster-whisper downloads+converts on first construction

    elif model_id == "llm":
        from huggingface_hub import hf_hub_download

        target_dir = settings.models_cache_dir / "llm"
        target_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            local_dir=str(target_dir),
        )

    elif model_id == "piper-voice":
        from huggingface_hub import hf_hub_download

        base = PIPER_VOICE_REPO_PATHS.get(settings.piper_voice)
        if base is None:
            raise HTTPException(400, f"unknown piper voice mapping: {settings.piper_voice}")
        target_dir = settings.models_cache_dir / "piper"
        target_dir.mkdir(parents=True, exist_ok=True)
        for ext in (".onnx", ".onnx.json"):
            hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=f"{base}{ext}",
                local_dir=str(target_dir),
            )

    elif model_id == "diarization":
        from app.pipeline.diarize import validate_pipeline

        # Not a real "download" — this is a validation probe: try to load the
        # gated pipeline (token + accepted license) in the same dedicated
        # subprocess diarize() actually uses (see diarize.py), which also
        # warms/confirms the local HF cache and touches the .complete marker
        # on success.
        if not validate_pipeline():
            raise HTTPException(
                409,
                "diarization model is gated: accept the license at "
                f"https://huggingface.co/{settings.diarization_model} and set "
                "HF_TOKEN in ml-service/.env, then click Validate again.",
            )
    else:
        raise HTTPException(404, "unknown model id")

    marker.touch()


@router.get("")
async def list_models():
    return [{**m, "downloaded": _is_downloaded(m["id"])} for m in REQUIRED_MODELS]


@router.post("/{model_id}/download")
async def download_model(model_id: str):
    if not any(m["id"] == model_id for m in REQUIRED_MODELS):
        raise HTTPException(404, "unknown model id")
    await run_in_threadpool(_do_download, model_id)
    return {"id": model_id, "downloaded": _is_downloaded(model_id)}


@router.get("/{model_id}/status")
async def model_status(model_id: str):
    if not any(m["id"] == model_id for m in REQUIRED_MODELS):
        raise HTTPException(404, "unknown model id")
    return {"id": model_id, "downloaded": _is_downloaded(model_id)}
