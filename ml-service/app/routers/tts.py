import uuid

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.config import settings
from app.pipeline import tts as tts_pipeline

router = APIRouter(prefix="/tts", tags=["tts"])


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None


@router.post("")
async def speak(req: SpeakRequest):
    """Standalone free-text TTS, independent of any audio-analysis job."""
    output_dir = settings.storage_dir / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4()}.wav"
    await run_in_threadpool(tts_pipeline.synthesize, req.text, output_path, req.voice)
    return {"audioPath": str(output_path)}
