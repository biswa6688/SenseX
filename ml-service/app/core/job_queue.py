import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from statistics import mean
from typing import Any, Callable

from app.core.config import settings

# Order matters: used both to drive the pipeline (see routers/jobs.py) and to
# know which stages are still ahead of the current one for ETA estimation.
PIPELINE_STAGES = ["staging", "transcribing", "diarizing", "merging", "analyzing", "synthesizing"]

# How many past completed jobs to sample per stage when estimating ETA.
ETA_HISTORY_SAMPLE = 5


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCancelled(Exception):
    """Raised by a pipeline handler's stage checkpoint when the job has been
    cancelled — caught by the worker loop and turned into JobStatus.CANCELLED
    instead of FAILED. Only takes effect between stages (see run_audio_pipeline
    in routers/jobs.py) — CPU-blocking work already in flight for the current
    stage runs to completion; there's no way to preempt a whisper/pyannote/
    llama.cpp call mid-call without native cancellation support in those
    libraries."""


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stage_started_at: float = field(default_factory=time.time)
    stage_durations: dict[str, float] = field(default_factory=dict)
    cancel_requested: bool = False

    def dir(self):
        path = settings.storage_dir / "audio-jobs" / self.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def set_stage(self, stage: str) -> None:
        """Advance to `stage`, recording how long the previous stage took."""
        now = time.time()
        if self.stage is not None:
            elapsed = now - self.stage_started_at
            self.stage_durations[self.stage] = self.stage_durations.get(self.stage, 0.0) + elapsed
        self.stage = stage
        self.stage_started_at = now
        self.updated_at = now
        self.persist()

    def close_stage(self) -> None:
        """Record the final stage's duration once the job finishes (completed or failed)."""
        if self.stage is not None:
            elapsed = time.time() - self.stage_started_at
            self.stage_durations[self.stage] = self.stage_durations.get(self.stage, 0.0) + elapsed

    def persist(self) -> None:
        sidecar = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "stageDurations": self.stage_durations,
        }
        (self.dir() / "job.json").write_text(json.dumps(sidecar, indent=2))


JobHandler = Callable[[Job, Callable[[str], None]], dict[str, Any]]


class JobQueue:
    """Single-worker async queue. Concurrency is pinned to 1 by design —
    CPU inference activation memory (whisper decode buffers, pyannote
    clustering, llama.cpp KV cache) must never run two jobs at once on a
    16GB box. See PLAN.md RAM budget."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[tuple[Job, JobHandler]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._pending_ids: list[str] = []

    def reconcile_orphaned_sidecars(self) -> None:
        """Jobs are tracked in-memory (self._jobs); a process restart (crash,
        deploy, forced kill mid-job) wipes that dict but leaves the on-disk
        sidecar frozen at whatever status it last had. Without this, a job
        interrupted mid-run shows up in history as "processing" forever —
        permanently "active" in the UI with a Cancel button that 404s,
        since job_queue no longer knows about it. Called once at startup,
        before serving traffic, to mark those as failed instead."""
        audio_jobs_dir = settings.storage_dir / "audio-jobs"
        if not audio_jobs_dir.exists():
            return
        for job_dir in audio_jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            sidecar_path = job_dir / "job.json"
            if not sidecar_path.exists():
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except (OSError, ValueError):
                continue
            if sidecar.get("status") not in (JobStatus.QUEUED, JobStatus.PROCESSING):
                continue
            sidecar["status"] = JobStatus.FAILED
            sidecar["error"] = "Interrupted by ml-service restart"
            sidecar["updatedAt"] = time.time()
            sidecar_path.write_text(json.dumps(sidecar, indent=2))

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            job, handler = await self._queue.get()
            if job.id in self._pending_ids:
                self._pending_ids.remove(job.id)
            if job.cancel_requested:
                # Cancelled while it was still sitting in the queue — never ran.
                self._queue.task_done()
                continue
            job.status = JobStatus.PROCESSING
            job.updated_at = time.time()
            job.persist()
            try:
                # Pipeline handlers are synchronous, CPU-blocking calls (whisper/
                # pyannote/llama.cpp). Run in a thread so the event loop stays free
                # to serve status polls and other requests while a job processes.
                result = await asyncio.to_thread(handler, job, job.set_stage)
                job.result = result
                job.status = JobStatus.COMPLETED
            except JobCancelled:
                job.status = JobStatus.CANCELLED
            except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the job record
                job.status = JobStatus.FAILED
                job.error = str(exc)
            finally:
                job.close_stage()
                job.updated_at = time.time()
                job.persist()
                self._queue.task_done()

    def submit(self, kind: str, payload: dict[str, Any], handler: JobHandler) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind, payload=payload)
        self._jobs[job.id] = job
        self._pending_ids.append(job.id)
        job.persist()
        self._queue.put_nowait((job, handler))
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> int | None:
        """1-indexed position among jobs still waiting for the single worker
        (the currently processing job, if any, is not in this list)."""
        try:
            return self._pending_ids.index(job_id) + 1
        except ValueError:
            return None

    def queue_length(self) -> int:
        return len(self._pending_ids)

    def cancel(self, job_id: str) -> bool:
        """Returns False if the job doesn't exist or is already in a terminal
        state. A queued job is cancelled immediately (removed from the pending
        list, never runs). A processing job is only flagged — it stops at the
        next stage checkpoint (see JobCancelled), it does not interrupt
        in-flight CPU work."""
        job = self._jobs.get(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.PROCESSING):
            return False
        job.cancel_requested = True
        if job.status == JobStatus.QUEUED:
            if job.id in self._pending_ids:
                self._pending_ids.remove(job.id)
            job.status = JobStatus.CANCELLED
            job.updated_at = time.time()
            job.persist()
        return True

    def _historical_stage_durations(self, exclude_job_id: str) -> dict[str, list[float]]:
        """Per-stage durations sampled from recently completed jobs' sidecars,
        so ETA improves over time without a DB (see DECISIONS.md #10)."""
        audio_jobs_dir = settings.storage_dir / "audio-jobs"
        if not audio_jobs_dir.exists():
            return {}
        job_dirs = sorted(
            (d for d in audio_jobs_dir.iterdir() if d.is_dir() and d.name != exclude_job_id),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        history: dict[str, list[float]] = {}
        samples_needed = {s: ETA_HISTORY_SAMPLE for s in PIPELINE_STAGES}
        for d in job_dirs:
            if not samples_needed:
                break
            sidecar_path = d / "job.json"
            if not sidecar_path.exists():
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except (OSError, ValueError):
                continue
            if sidecar.get("status") != JobStatus.COMPLETED:
                continue
            durations = sidecar.get("stageDurations") or {}
            for stage, seconds in durations.items():
                if samples_needed.get(stage, 0) <= 0:
                    continue
                history.setdefault(stage, []).append(seconds)
                samples_needed[stage] -= 1
                if samples_needed[stage] <= 0:
                    del samples_needed[stage]
        return history

    def estimate_remaining_seconds(self, job: Job) -> float | None:
        """Best-effort ETA: average historical duration of the current stage
        (minus time already spent in it) plus average duration of stages still
        ahead. Returns None until there's at least one completed job to learn from."""
        if job.status != JobStatus.PROCESSING or job.stage is None or job.stage not in PIPELINE_STAGES:
            return None
        history = self._historical_stage_durations(exclude_job_id=job.id)
        if not history:
            return None
        idx = PIPELINE_STAGES.index(job.stage)
        remaining = 0.0
        have_data = False
        if history.get(job.stage):
            elapsed_current = time.time() - job.stage_started_at
            remaining += max(mean(history[job.stage]) - elapsed_current, 0.0)
            have_data = True
        for stage in PIPELINE_STAGES[idx + 1 :]:
            if history.get(stage):
                remaining += mean(history[stage])
                have_data = True
        return remaining if have_data else None


job_queue = JobQueue()
