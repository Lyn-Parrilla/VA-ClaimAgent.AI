"""In-memory analysis job foundation.

This module owns the *lifecycle metadata* for analysis jobs and nothing else.
It is intentionally standalone: it imports no application code (and in
particular nothing from ``main``), so it carries no Google Cloud credential
requirements and can be exercised in isolation.

Privacy contract
----------------
``AnalysisJob`` stores metadata only. It must never hold veteran document
text, raw PII, PII token maps, prompts, model responses, or uploaded
documents. Two mechanisms enforce this structurally rather than by
convention:

* ``model_config = ConfigDict(extra="forbid")`` means a caller cannot attach
  an undeclared field to a job.
* ``error`` is typed as :class:`JobFailureCode`, a closed enum. There is no
  free-text field anywhere on the model, so a raw exception string such as
  ``str(exc)`` -- which can embed document text, prompt fragments, or model
  output -- is rejected by validation instead of being persisted.

Nothing in this module logs, so no sensitive value can leak through it.

Production note
---------------
``JobStore`` is a deliberate *development* implementation. State lives in a
single process's heap, which means:

* it is lost on restart and not shared across workers or replicas,
* expiration is lazy -- entries are only reaped when a sweep method is
  called, so nothing evicts a job on its own.

A production deployment will need a durable, shared job mechanism with an
explicit TTL/cleanup strategy (for example a scheduled sweep or a store with
native record expiry). Choosing that mechanism is out of scope here; this
module exists to pin down the model, the lifecycle, and the storage contract.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator

JOB_TTL_ENV_VAR = "CLAIM_AGENT_JOB_TTL_SECONDS"
DEFAULT_JOB_TTL_SECONDS = 3600


class JobError(Exception):
    """Base class for every error raised by this module."""


class JobNotFoundError(JobError):
    """Raised when a job ID is not present in the store."""


class InvalidJobTransitionError(JobError):
    """Raised when a caller requests a status change the lifecycle forbids."""


class JobConfigurationError(JobError):
    """Raised when the configured TTL is not a usable positive integer."""


class JobStatus(str, Enum):
    """The five lifecycle states an analysis job may occupy."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class JobFailureCode(str, Enum):
    """The closed set of reasons a job may record for reaching ``FAILED``.

    This is the *only* value ``AnalysisJob.error`` can hold. The set is
    deliberately coarse: each member names a stage of the pipeline, never a
    specific cause. Anything finer risks a caller reaching for ``str(exc)``
    and writing document text, prompt fragments, or model output into a
    record that is supposed to be metadata.

    Members map onto pipeline stages as follows:

    ``DOCUMENT_EXTRACTION_FAILED``
        Reading text out of the submitted document did not succeed.
    ``PII_PROTECTION_FAILED``
        PII detection or tokenization did not succeed, so nothing could
        safely be forwarded to an external service.
    ``MODEL_ANALYSIS_FAILED``
        The model call, or parsing/validating its response, did not succeed.
    ``REGULATORY_ANALYSIS_FAILED``
        Regulatory lookup did not succeed.
    ``INTERNAL_ERROR``
        Catch-all for anything unattributable. Prefer a specific member.

    Extending this enum is a deliberate act: add a member here rather than
    widening the field to accept text.
    """

    DOCUMENT_EXTRACTION_FAILED = "DOCUMENT_EXTRACTION_FAILED"
    PII_PROTECTION_FAILED = "PII_PROTECTION_FAILED"
    MODEL_ANALYSIS_FAILED = "MODEL_ANALYSIS_FAILED"
    REGULATORY_ANALYSIS_FAILED = "REGULATORY_ANALYSIS_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


#: States that end a job's life. Nothing may leave these.
TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.EXPIRED}
)

#: The complete transition table. Any pair absent from this mapping is invalid
#: and is rejected rather than silently applied.
ALLOWED_TRANSITIONS: Dict[JobStatus, frozenset] = {
    JobStatus.QUEUED: frozenset(
        {JobStatus.PROCESSING, JobStatus.FAILED, JobStatus.EXPIRED}
    ),
    JobStatus.PROCESSING: frozenset(
        {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.EXPIRED}
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.EXPIRED: frozenset(),
}


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def get_job_ttl_seconds() -> int:
    """Read the job TTL from the environment.

    Falls back to :data:`DEFAULT_JOB_TTL_SECONDS` when unset or blank. A
    malformed or non-positive value raises rather than silently defaulting,
    because the TTL is a data-retention control and quietly using the wrong
    window would be worse than failing loudly.
    """
    raw = os.environ.get(JOB_TTL_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_JOB_TTL_SECONDS
    try:
        ttl = int(raw.strip())
    except ValueError as exc:
        raise JobConfigurationError(
            f"{JOB_TTL_ENV_VAR} must be a positive integer number of seconds, "
            f"got {raw!r}"
        ) from exc
    if ttl <= 0:
        raise JobConfigurationError(
            f"{JOB_TTL_ENV_VAR} must be a positive integer number of seconds, "
            f"got {ttl}"
        )
    return ttl


class AnalysisJob(BaseModel):
    """Lifecycle metadata for a single analysis job.

    Metadata only -- see the module docstring's privacy contract. ``extra`` is
    forbidden so document text or PII maps cannot be attached to an instance.
    """

    # extra="forbid" blocks undeclared (potentially sensitive) fields at
    # construction; validate_assignment keeps that guard and the timezone check
    # active for later mutations too.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    completed_at: Optional[datetime] = None
    # Closed enum, never free text: see JobFailureCode and the module's
    # privacy contract. Validation rejects arbitrary strings outright.
    error: Optional[JobFailureCode] = None

    @field_validator("created_at", "updated_at", "expires_at", "completed_at")
    @classmethod
    def _require_timezone_aware_utc(
        cls, value: Optional[datetime]
    ) -> Optional[datetime]:
        """Reject naive datetimes and normalise everything to UTC."""
        if value is None:
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("datetime fields must be timezone-aware")
        return value.astimezone(timezone.utc)

    @property
    def is_terminal(self) -> bool:
        """True when the job has reached a state it can never leave."""
        return self.status in TERMINAL_STATUSES

    def can_transition_to(self, new_status: JobStatus) -> bool:
        """True when ``new_status`` is reachable from the current status."""
        return new_status in ALLOWED_TRANSITIONS[self.status]

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """True when this job's TTL window has elapsed."""
        return (now or utc_now()) >= self.expires_at


class JobStore:
    """Thread-safe in-memory registry of :class:`AnalysisJob` records.

    A :class:`threading.RLock` guards every read and write so concurrent
    FastAPI requests -- which FastAPI may run in its worker threadpool -- cannot
    interleave a read-modify-write and corrupt the registry. The lock is
    re-entrant because the convenience helpers delegate to
    :meth:`update_status` while already holding it.

    Accessors hand back deep copies. Callers therefore cannot mutate stored
    state directly and bypass the transition table; every change must go
    through this class.
    """

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        """Create a store.

        :param ttl_seconds: TTL override, mainly for tests. When omitted the
            value is read from :data:`JOB_TTL_ENV_VAR`.
        """
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise JobConfigurationError(
                f"ttl_seconds must be a positive integer, got {ttl_seconds}"
            )
        self._ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else get_job_ttl_seconds()
        )
        self._lock = threading.RLock()
        self._jobs: Dict[str, AnalysisJob] = {}

    @property
    def ttl_seconds(self) -> int:
        """The TTL, in seconds, applied to jobs created by this store."""
        return self._ttl_seconds

    # ---- creation ----------------------------------------------------------

    def create(self) -> AnalysisJob:
        """Register a new job in ``QUEUED`` and return it.

        ``expires_at`` is computed once, at creation, as
        ``created_at + ttl_seconds``.
        """
        now = utc_now()
        job = AnalysisJob(
            id=str(uuid.uuid4()),
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )
        with self._lock:
            self._jobs[job.id] = job
            return job.model_copy(deep=True)

    # ---- retrieval ---------------------------------------------------------

    def get(self, job_id: str) -> Optional[AnalysisJob]:
        """Return a copy of the job, or ``None`` when the ID is unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job is not None else None

    def require(self, job_id: str) -> AnalysisJob:
        """Return a copy of the job, raising when the ID is unknown."""
        job = self.get(job_id)
        if job is None:
            raise JobNotFoundError(f"No job with id {job_id!r}")
        return job

    def list_jobs(self) -> List[AnalysisJob]:
        """Return copies of every job, ordered by creation time."""
        with self._lock:
            return [
                job.model_copy(deep=True)
                for job in sorted(self._jobs.values(), key=lambda j: j.created_at)
            ]

    def __contains__(self, job_id: object) -> bool:
        with self._lock:
            return job_id in self._jobs

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    # ---- transitions -------------------------------------------------------

    def update_status(
        self,
        job_id: str,
        new_status: JobStatus,
        *,
        failure_code: Optional[JobFailureCode] = None,
    ) -> AnalysisJob:
        """Move a job to ``new_status``, enforcing the transition table.

        :param failure_code: a :class:`JobFailureCode` member. Required for
            ``FAILED`` and rejected for every other target state. Only true
            enum members are accepted -- a bare string, including one that
            happens to spell a valid code, raises :class:`TypeError`. That
            keeps ``str(exc)`` from ever reaching a stored job.
        :raises TypeError: when ``failure_code`` is not a
            :class:`JobFailureCode`.
        :raises JobNotFoundError: when the ID is unknown.
        :raises InvalidJobTransitionError: when the lifecycle forbids the move.
        """
        new_status = JobStatus(new_status)

        if failure_code is not None and not isinstance(failure_code, JobFailureCode):
            raise TypeError(
                "failure_code must be a JobFailureCode member, not "
                f"{type(failure_code).__name__}. Arbitrary failure text cannot "
                "be stored on a job; pick the closest JobFailureCode instead."
            )

        if new_status is JobStatus.FAILED:
            if failure_code is None:
                raise ValueError("A JobFailureCode is required for FAILED")
        elif failure_code is not None:
            raise ValueError(
                f"A failure_code is only valid for FAILED, not {new_status.value}"
            )

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"No job with id {job_id!r}")

            if not job.can_transition_to(new_status):
                if job.is_terminal:
                    raise InvalidJobTransitionError(
                        f"Job {job_id} is in terminal state "
                        f"{job.status.value} and cannot transition to "
                        f"{new_status.value}"
                    )
                raise InvalidJobTransitionError(
                    f"Job {job_id} cannot transition from {job.status.value} "
                    f"to {new_status.value}"
                )

            now = utc_now()
            job.status = new_status
            job.updated_at = now
            if new_status is JobStatus.COMPLETED:
                job.completed_at = now
            if new_status is JobStatus.FAILED:
                job.error = failure_code
            return job.model_copy(deep=True)

    def mark_processing(self, job_id: str) -> AnalysisJob:
        """``QUEUED -> PROCESSING``."""
        return self.update_status(job_id, JobStatus.PROCESSING)

    def mark_completed(self, job_id: str) -> AnalysisJob:
        """``PROCESSING -> COMPLETED``, stamping ``completed_at``."""
        return self.update_status(job_id, JobStatus.COMPLETED)

    def mark_failed(
        self, job_id: str, failure_code: JobFailureCode
    ) -> AnalysisJob:
        """``QUEUED|PROCESSING -> FAILED`` with a controlled failure code.

        Accepts only :class:`JobFailureCode` members, so a caller cannot
        record raw exception text against a job.
        """
        return self.update_status(
            job_id, JobStatus.FAILED, failure_code=failure_code
        )

    def mark_expired(self, job_id: str) -> AnalysisJob:
        """``QUEUED|PROCESSING -> EXPIRED``."""
        return self.update_status(job_id, JobStatus.EXPIRED)

    # ---- expiration --------------------------------------------------------

    def find_expired(self, now: Optional[datetime] = None) -> List[AnalysisJob]:
        """Return copies of every job whose ``expires_at`` has passed.

        Includes terminal jobs, which are eligible for purging but not for a
        further transition. ``now`` is injectable so callers and tests can
        evaluate a specific instant instead of sleeping.
        """
        moment = now or utc_now()
        with self._lock:
            return [
                job.model_copy(deep=True)
                for job in sorted(self._jobs.values(), key=lambda j: j.created_at)
                if job.is_expired(moment)
            ]

    def expire_due_jobs(self, now: Optional[datetime] = None) -> List[str]:
        """Transition every non-terminal expired job to ``EXPIRED``.

        Terminal jobs are left untouched. Returns the IDs transitioned.
        """
        moment = now or utc_now()
        with self._lock:
            due = [
                job.id
                for job in self._jobs.values()
                if job.is_expired(moment) and not job.is_terminal
            ]
            for job_id in due:
                self.update_status(job_id, JobStatus.EXPIRED)
            return due

    def purge_expired(self, now: Optional[datetime] = None) -> List[str]:
        """Delete every job whose ``expires_at`` has passed.

        This is the reclamation half of expiry; pair it with
        :meth:`expire_due_jobs` when observers need to see the ``EXPIRED``
        state before the record disappears. Returns the IDs removed.
        """
        moment = now or utc_now()
        with self._lock:
            removed = [
                job.id for job in self._jobs.values() if job.is_expired(moment)
            ]
            for job_id in removed:
                del self._jobs[job_id]
            return removed

    # ---- deletion ----------------------------------------------------------

    def delete(self, job_id: str) -> bool:
        """Remove a job. Returns ``True`` if it existed, ``False`` otherwise."""
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def clear(self) -> None:
        """Remove every job. Intended for test isolation."""
        with self._lock:
            self._jobs.clear()


_job_store: Optional[JobStore] = None
_job_store_lock = threading.Lock()


def get_job_store() -> JobStore:
    """Return the process-wide :class:`JobStore`, creating it on first use.

    A single accessor keeps job state behind one object instead of loose
    module-level dictionaries. Nothing is wired into the request path yet.
    """
    global _job_store
    with _job_store_lock:
        if _job_store is None:
            _job_store = JobStore()
        return _job_store


def reset_job_store() -> None:
    """Drop the process-wide store so the next access rebuilds it."""
    global _job_store
    with _job_store_lock:
        _job_store = None
