"""HTTP surface for the analysis job foundation.

A thin adapter over :mod:`claim_backend.jobs`. It owns no state of its own:
every job lives in the single application-level :class:`JobStore` resolved
through :func:`claim_backend.jobs.get_job_store`, which stays the one source
of truth.

Exposure contract
-----------------
Storage models and API models are kept separate on purpose. Handlers never
return :class:`AnalysisJob`; they project it field-by-field into
:class:`JobResponse` via :func:`_to_response`. A field added to the storage
model in a later step is therefore *not* published by accident -- it has to be
added here deliberately. ``extra="forbid"`` on the response model makes the
published shape exact.

The request body is an empty model, also ``extra="forbid"``, so an attempt to
submit document text, PII, prompts, or model parameters is rejected with a
422 rather than silently ignored. Nothing in this module logs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .jobs import AnalysisJob, JobFailureCode, JobStatus, JobStore, get_job_store

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    """An intentionally empty request body.

    Creating a job takes no input at this stage. ``extra="forbid"`` turns any
    attempt to attach document text, PII, prompts, or model parameters into a
    validation error instead of a silently discarded field.
    """

    model_config = ConfigDict(extra="forbid")


class JobResponse(BaseModel):
    """The safe, outward-facing view of a job.

    Metadata only. ``failure_code`` is the controlled
    :class:`JobFailureCode` enum, so no free-form failure text can reach a
    client through this contract.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    completed_at: Optional[datetime] = None
    failure_code: Optional[JobFailureCode] = None


class JobErrorResponse(BaseModel):
    """A concise, non-sensitive error payload."""

    model_config = ConfigDict(extra="forbid")

    detail: str


JOB_NOT_FOUND_DETAIL = "Job not found"


def _to_response(job: AnalysisJob) -> JobResponse:
    """Project a stored job onto the API contract, one field at a time.

    Deliberately explicit rather than ``model_validate(job)``: new storage
    fields must be published on purpose, never by inheritance.
    """
    return JobResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        expires_at=job.expires_at,
        completed_at=job.completed_at,
        failure_code=job.error,
    )


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an analysis job",
)
def create_job(
    body: Optional[CreateJobRequest] = None,
    store: JobStore = Depends(get_job_store),
) -> JobResponse:
    """Register a new job in ``QUEUED`` and return its metadata.

    The body is optional and must be empty when supplied. Nothing is queued
    for processing yet; this establishes the identifier a client will poll.
    """
    return _to_response(store.create())


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": JobErrorResponse, "description": "Job not found"}},
    summary="Get analysis job status",
)
def get_job(
    job_id: uuid.UUID,
    store: JobStore = Depends(get_job_store),
) -> JobResponse:
    """Return the current metadata for a job.

    ``job_id`` is typed as a UUID so FastAPI rejects a malformed identifier
    with its standard 422 before any handler code runs -- no manual parsing,
    no broad ``except``. Unknown-but-well-formed IDs get a fixed 404 message
    that reveals nothing about the store.
    """
    job = store.get(str(job_id))
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=JOB_NOT_FOUND_DETAIL
        )
    return _to_response(job)


#: Validation-error keys that are safe to publish. Notably excludes ``input``,
#: which Pydantic populates with the *rejected value itself*, and ``ctx``,
#: which can also carry it.
SAFE_VALIDATION_ERROR_KEYS = frozenset({"type", "loc", "msg"})


def _sanitize_validation_errors(errors: list) -> list:
    """Drop echoed input values from a validation error list."""
    return [
        {key: value for key, value in error.items() if key in SAFE_VALIDATION_ERROR_KEYS}
        for error in errors
    ]


async def jobs_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Strip echoed request values from validation errors on job routes.

    ``extra="forbid"`` makes Pydantic reject an unexpected field, but its
    default error payload includes ``input`` -- the rejected value. A client
    that mistakenly posts veteran text to this API would get that text
    reflected back, where proxies and error trackers can capture it. For job
    routes we publish only the error type, location, and fixed message.

    Every other path, including ``/api/extract``, is delegated to FastAPI's
    default handler so existing response contracts are unchanged.
    """
    if request.url.path.startswith(router.prefix):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {"detail": _sanitize_validation_errors(exc.errors())}
            ),
        )
    return await request_validation_exception_handler(request, exc)


def register_jobs_api(app: FastAPI) -> None:
    """Attach the job routes and their scoped validation handler to ``app``."""
    app.include_router(router)
    app.add_exception_handler(
        RequestValidationError, jobs_validation_exception_handler
    )
