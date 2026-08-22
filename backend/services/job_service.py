"""In-memory indexing job tracker with cooperative cancel."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class IndexJob:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed | cancelled
    progress: float = 0.0
    total: int = 0
    processed: int = 0
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    cancel_requested: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "processed": self.processed,
            "logs": list(self.logs),
            "errors": list(self.errors),
            "document_ids": list(self.document_ids),
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobService:
    def __init__(self) -> None:
        self._jobs: Dict[str, IndexJob] = {}
        self._lock = threading.RLock()

    def create_job(self, total: int = 0) -> IndexJob:
        job_id = str(uuid.uuid4())
        job = IndexJob(job_id=job_id, total=total, status="queued")
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[IndexJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def request_cancel(self, job_id: str) -> Optional[IndexJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.cancel_requested = True
            if job.status in ("queued", "running"):
                job.status = "cancelled" if job.status == "queued" else job.status
            job.touch()
            return job


job_service = JobService()
