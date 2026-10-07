"""
Background summarization jobs with progress reporting.

Abstractive summarization of a long document can take minutes on a CPU. A
single blocking HTTP request cannot tell the user how far it has got, so the
API can also run a summarization as a *job*: the client starts it, receives a
job id immediately, and polls ``GET /api/jobs/{id}`` for status and progress
("summarizing chunks 3/7") until the result is ready.

Jobs are held in memory (no database is needed for this) and removed an hour
after finishing. They run on a small thread pool whose default size is 1:
on a CPU, two BART runs in parallel are each slower than running them one
after the other, and queued jobs simply wait their turn.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.errors import IntelliSumError

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 3600


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | completed | failed
    progress: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobNotFoundError(IntelliSumError):
    status_code = 404
    default_message = "No such job. It may have expired; please run the summarization again."


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="summarize")

    def submit(self, task: Callable[[Callable[[str, int, int], None]], Any]) -> Job:
        """
        Run ``task(on_progress)`` in the background. ``task`` receives a
        progress callback and returns the job result (already JSON-ready).
        """
        self._cleanup()
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job

        def on_progress(stage: str, done: int, total: int) -> None:
            job.progress = {"stage": stage, "done": done, "total": total}

        def run() -> None:
            job.status = "running"
            try:
                job.result = task(on_progress)
                job.progress = {"stage": "Completed", "done": 1, "total": 1}
                job.status = "completed"
            except IntelliSumError as exc:
                job.error, job.status = exc.message, "failed"
            except Exception:  # a bug: log it, but never expose the traceback
                logger.exception("Job %s failed", job.id)
                job.error, job.status = "An internal error occurred. Please try again.", "failed"
            finally:
                job.finished_at = time.time()

        self._executor.submit(run)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError()
        return job

    def _cleanup(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            for job_id in [j.id for j in self._jobs.values() if j.finished_at and j.finished_at < cutoff]:
                del self._jobs[job_id]


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager(max_workers=get_settings().max_concurrent_jobs)
        return _manager
