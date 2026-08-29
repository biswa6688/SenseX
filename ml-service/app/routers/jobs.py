import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.job_queue import JobStatus, job_queue
from app.pipeline import diarize, llm_analysis, merge, stt, tts

router = APIRouter(prefix="/jobs", tags=["jobs"])


def run_audio_pipeline(job, set_stage) -> dict:
    """Synchronous by design — run via asyncio.to_thread in job_queue's
    worker, not awaited directly, so this CPU-blocking work never freezes
    the event loop (see job_queue.py)."""
    set_stage("staging")
    source_path = Path(job.payload["filePath"])
    original_path = job.dir() / f"original{source_path.suffix}"
    original_path.write_bytes(source_path.read_bytes())
    audio_path = str(original_path)

    set_stage("transcribing")
    transcript = stt.transcribe(audio_path)

    set_stage("diarizing")
    turns = diarize.diarize(audio_path)

    set_stage("merging")
    diarized = merge.merge_transcript(transcript["words"], turns)
    diarized_text = merge.format_for_llm(diarized)

    set_stage("analyzing")
    analysis = llm_analysis.analyze(diarized_text)

    set_stage("synthesizing")
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
    return {"jobId": job.id, "status": job.status, "stage": job.stage, "error": job.error}


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"job is {job.status}, not completed")
    return job.result
