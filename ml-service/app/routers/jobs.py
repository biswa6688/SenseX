import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.job_queue import PIPELINE_STAGES, Job, JobCancelled, JobStatus, job_queue
from app.pipeline import audio_preprocess, boundary_refinement, diarize, llm_analysis, merge, stt, three_d_speaker, tts

router = APIRouter(prefix="/jobs", tags=["jobs"])

# How often the SSE stream re-checks and re-emits a job's status. Short
# enough that the UI's elapsed-time/ETA display ticks smoothly, cheap enough
# (single-user, in-memory job lookup) that busy-waiting at this interval
# costs nothing measurable.
STREAM_POLL_INTERVAL_SECONDS = 1.0


def _status_payload(job: Job) -> dict:
    return {
        "jobId": job.id,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
        "stages": PIPELINE_STAGES,
        "elapsedSeconds": time.time() - job.created_at if job.status == JobStatus.PROCESSING else None,
        "etaSeconds": job_queue.estimate_remaining_seconds(job),
        "queuePosition": job_queue.queue_position(job.id) if job.status == JobStatus.QUEUED else None,
        "queueLength": job_queue.queue_length() if job.status == JobStatus.QUEUED else None,
    }


def run_audio_pipeline(job, set_stage) -> dict:
    """Synchronous by design — run via asyncio.to_thread in job_queue's
    worker, not awaited directly, so this CPU-blocking work never freezes
    the event loop (see job_queue.py)."""
    staging, transcribing, diarizing, merging, analyzing, synthesizing = PIPELINE_STAGES

    def checkpoint(stage: str) -> None:
        """Advance to `stage`, unless cancellation was requested — cancellation
        is only checked between stages (see JobCancelled), not mid-stage."""
        if job.cancel_requested:
            raise JobCancelled()
        set_stage(stage)

    checkpoint(staging)
    source_path = Path(job.payload["filePath"])
    original_path = job.dir() / f"original{source_path.suffix}"
    original_path.write_bytes(source_path.read_bytes())
    audio_path = str(original_path)

    # Canonicalize once here rather than letting STT (PyAV) and diarization
    # (torchcodec) each independently decode the original file through
    # different codec libraries — see audio_preprocess.py.
    canonical_path = job.dir() / "canonical.wav"
    audio_metadata = audio_preprocess.canonicalize_audio(audio_path, canonical_path)
    (job.dir() / "audio_metadata.json").write_text(json.dumps(audio_metadata.to_dict(), indent=2))
    canonical_audio_path = str(canonical_path)

    checkpoint(transcribing)
    transcript = stt.transcribe(canonical_audio_path)

    checkpoint(diarizing)
    turns = diarize.diarize(canonical_audio_path)

    checkpoint(merging)
    diarized = merge.merge_transcript(transcript["words"], turns)

    # Turn-boundary refinement (Phase 8, see boundary_refinement.py): only
    # runs if confidence.py flagged something. Prefers 3D-Speaker (an
    # independently-trained embedding — verified on real audio to catch
    # splits community-1's own embedding space couldn't, see
    # DECISIONS.md #17), falling back to reusing the diarization
    # pipeline's own embedding if 3D-Speaker fails to load (e.g. offline
    # on first download).
    embedding_model = three_d_speaker.get_embedding_model() or diarize.get_embedding_model()
    if embedding_model is not None and any(t["uncertain"] for t in diarized):
        diarized = boundary_refinement.refine_transcript(
            embedding_model, canonical_audio_path, transcript["words"], turns, diarized
        )

    diarized_text = merge.format_for_llm(diarized)

    checkpoint(analyzing)
    analysis = llm_analysis.analyze(diarized_text)

    checkpoint(synthesizing)
    summary_wav = job.dir() / "summary.wav"
    tts.synthesize(analysis["summary"], summary_wav)

    result = {
        "transcript": diarized,
        "summary": analysis["summary"],
        "sentiment": analysis["sentiment"],
        "qaRatings": analysis["qaRatings"],
        "originalAudioPath": audio_path,
        "summaryAudioPath": str(summary_wav),
    }
    (job.dir() / "result.json").write_text(json.dumps(result, indent=2))
    return result


@router.post("", status_code=202)
async def create_job(file_path: str):
    """file_path points into the shared storage dir; NestJS writes the
    upload there and passes the path — no re-uploading bytes over HTTP."""
    if not Path(file_path).exists():
        raise HTTPException(400, f"file not found: {file_path}")
    job = job_queue.submit("audio-pipeline", {"filePath": file_path}, run_audio_pipeline)
    return {"jobId": job.id, "status": job.status}


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _status_payload(job)


@router.get("/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Server-Sent Events version of GET /{job_id} — pushes a status
    snapshot roughly once a second over one long-lived connection instead
    of the client re-polling, and closes the stream itself once the job
    reaches a terminal state so the client doesn't need to."""
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    async def event_stream() -> AsyncIterator[str]:
        while True:
            current = job_queue.get(job_id)
            if current is None:
                yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
                return
            payload = _status_payload(current)
            yield f"data: {json.dumps(payload)}\n\n"
            if current.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return
            await asyncio.sleep(STREAM_POLL_INTERVAL_SECONDS)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    cancelled = job_queue.cancel(job_id)
    if not cancelled:
        raise HTTPException(409, f"job is {job.status}, cannot cancel")
    return _status_payload(job)


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"job is {job.status}, not completed")
    return job.result
