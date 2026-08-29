import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from app.core.config import settings


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


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

    def dir(self):
        path = settings.storage_dir / "audio-jobs" / self.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist(self) -> None:
        sidecar = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
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

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            job, handler = await self._queue.get()
            job.status = JobStatus.PROCESSING
            job.updated_at = time.time()
            job.persist()
            try:
                def set_stage(stage: str) -> None:
                    job.stage = stage
                    job.updated_at = time.time()
                    job.persist()

                # Pipeline handlers are synchronous, CPU-blocking calls (whisper/
                # pyannote/llama.cpp). Run in a thread so the event loop stays free
                # to serve status polls and other requests while a job processes.
                result = await asyncio.to_thread(handler, job, set_stage)
                job.result = result
                job.status = JobStatus.COMPLETED
            except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the job record
                job.status = JobStatus.FAILED
                job.error = str(exc)
            finally:
                job.updated_at = time.time()
                job.persist()
                self._queue.task_done()

    def submit(self, kind: str, payload: dict[str, Any], handler: JobHandler) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind, payload=payload)
        self._jobs[job.id] = job
        job.persist()
        self._queue.put_nowait((job, handler))
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


job_queue = JobQueue()
