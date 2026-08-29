from fastapi import FastAPI

from app.core.job_queue import job_queue
from app.routers import jobs, models, tts

app = FastAPI(title="SenseX ML Service")

app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(tts.router)


@app.on_event("startup")
async def on_startup() -> None:
    job_queue.reconcile_orphaned_sidecars()
    job_queue.start()
    # TODO (Phase 2): eager-load whisper/diarization/llm/piper singletons here
    # so first request isn't cold. See PLAN.md RAM budget (~10GB all loaded).


@app.get("/health")
async def health():
    return {"status": "ok"}
