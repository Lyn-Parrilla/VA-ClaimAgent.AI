"""Focused tests for the in-memory analysis job foundation.

These exercise claim_backend.jobs in isolation. The module imports no
application code, so nothing here needs Google Cloud credentials or the
FastAPI app.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from claim_backend.jobs import (
    ALLOWED_TRANSITIONS,
    DEFAULT_JOB_TTL_SECONDS,
    JOB_TTL_ENV_VAR,
    TERMINAL_STATUSES,
    AnalysisJob,
    InvalidJobTransitionError,
    JobConfigurationError,
    JobFailureCode,
    JobNotFoundError,
    JobStatus,
    JobStore,
    get_job_store,
    get_job_ttl_seconds,
    reset_job_store,
)

# Sensitive material that must never survive a round trip through the job layer.
VETERAN_TEXT = "John Doe, SSN 123-45-6789, is rated 10% for tinnitus."
PII_MAP = {"<PERSON_1>": "John Doe", "<SSN_1>": "123-45-6789"}

# The shape of an unsanitized exception string a careless caller might forward.
SENSITIVE_FAILURE_TEXT = "Veteran John Doe SSN 123-45-6789"


@pytest.fixture
def store() -> JobStore:
    """A store with a short, explicit TTL so expiry math is easy to assert."""
    return JobStore(ttl_seconds=60)


def _is_utc(value: datetime) -> bool:
    """True when value is timezone-aware and offset zero."""
    return value.tzinfo is not None and value.utcoffset() == timedelta(0)


def _code_for(target: JobStatus):
    """The failure_code a transition to `target` requires, if any."""
    return JobFailureCode.INTERNAL_ERROR if target is JobStatus.FAILED else None


# --- A. creation ------------------------------------------------------------


def test_new_job_gets_uuid_and_starts_queued(store):
    import uuid

    job = store.create()

    assert job.status is JobStatus.QUEUED
    # Round-tripping through UUID() proves it is a well-formed UUID string.
    assert str(uuid.UUID(job.id)) == job.id


def test_each_job_gets_a_distinct_id(store):
    ids = {store.create().id for _ in range(25)}
    assert len(ids) == 25
    assert len(store) == 25


def test_new_job_has_no_completion_or_error(store):
    job = store.create()
    assert job.completed_at is None
    assert job.error is None


# --- B. timezone-aware UTC timestamps ---------------------------------------


def test_timestamps_are_timezone_aware_utc(store):
    job = store.create()

    assert _is_utc(job.created_at)
    assert _is_utc(job.updated_at)
    assert _is_utc(job.expires_at)


def test_completed_at_is_timezone_aware_utc(store):
    job = store.create()
    store.mark_processing(job.id)
    completed = store.mark_completed(job.id)

    assert completed.completed_at is not None
    assert _is_utc(completed.completed_at)


def test_model_rejects_naive_datetimes():
    naive = datetime(2026, 1, 1, 12, 0, 0)

    with pytest.raises(ValidationError, match="timezone-aware"):
        AnalysisJob(
            id="naive-job",
            created_at=naive,
            updated_at=naive,
            expires_at=naive,
        )


def test_model_normalises_non_utc_offsets_to_utc():
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))

    job = AnalysisJob(
        id="offset-job", created_at=aware, updated_at=aware, expires_at=aware
    )

    assert _is_utc(job.created_at)
    assert job.created_at == aware


# --- C. TTL -----------------------------------------------------------------


def test_expires_at_is_created_at_plus_ttl(store):
    job = store.create()
    assert job.expires_at - job.created_at == timedelta(seconds=60)


def test_ttl_defaults_to_one_hour(monkeypatch):
    monkeypatch.delenv(JOB_TTL_ENV_VAR, raising=False)

    assert DEFAULT_JOB_TTL_SECONDS == 3600
    assert get_job_ttl_seconds() == 3600
    assert JobStore().ttl_seconds == 3600


def test_ttl_is_read_from_environment(monkeypatch):
    monkeypatch.setenv(JOB_TTL_ENV_VAR, "120")

    assert get_job_ttl_seconds() == 120
    job = JobStore().create()
    assert job.expires_at - job.created_at == timedelta(seconds=120)


def test_blank_ttl_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(JOB_TTL_ENV_VAR, "   ")
    assert get_job_ttl_seconds() == DEFAULT_JOB_TTL_SECONDS


@pytest.mark.parametrize("bad_value", ["not-a-number", "0", "-30", "1.5"])
def test_malformed_ttl_is_rejected(monkeypatch, bad_value):
    monkeypatch.setenv(JOB_TTL_ENV_VAR, bad_value)

    with pytest.raises(JobConfigurationError):
        get_job_ttl_seconds()


def test_non_positive_ttl_override_is_rejected():
    with pytest.raises(JobConfigurationError):
        JobStore(ttl_seconds=0)


# --- D. retrieval -----------------------------------------------------------


def test_job_can_be_retrieved_by_id(store):
    created = store.create()

    fetched = store.get(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.status is JobStatus.QUEUED
    assert store.require(created.id).id == created.id
    assert created.id in store


def test_retrieval_returns_a_copy_so_callers_cannot_bypass_transitions(store):
    """Mutating a returned job must not corrupt the registry."""
    created = store.create()

    leaked = store.get(created.id)
    leaked.status = JobStatus.COMPLETED

    assert store.require(created.id).status is JobStatus.QUEUED


def test_list_jobs_returns_all_jobs(store):
    ids = {store.create().id for _ in range(3)}
    assert {job.id for job in store.list_jobs()} == ids


# --- E. deletion ------------------------------------------------------------


def test_job_can_be_explicitly_deleted(store):
    job = store.create()

    assert store.delete(job.id) is True

    assert store.get(job.id) is None
    assert job.id not in store
    assert len(store) == 0


def test_deleting_unknown_job_returns_false(store):
    assert store.delete("does-not-exist") is False


def test_clear_removes_every_job(store):
    for _ in range(4):
        store.create()

    store.clear()

    assert len(store) == 0


# --- F. missing jobs --------------------------------------------------------


def test_get_returns_none_for_missing_job(store):
    assert store.get("missing") is None


def test_require_raises_for_missing_job(store):
    with pytest.raises(JobNotFoundError, match="missing"):
        store.require("missing")


def test_transitioning_missing_job_raises(store):
    with pytest.raises(JobNotFoundError):
        store.mark_processing("missing")

    with pytest.raises(JobNotFoundError):
        store.mark_failed("missing", JobFailureCode.INTERNAL_ERROR)


# --- G. valid transitions ---------------------------------------------------


def test_queued_to_processing(store):
    job = store.create()

    updated = store.mark_processing(job.id)

    assert updated.status is JobStatus.PROCESSING
    assert updated.updated_at >= job.updated_at
    assert updated.completed_at is None


def test_queued_to_failed(store):
    job = store.create()

    updated = store.mark_failed(job.id, JobFailureCode.PII_PROTECTION_FAILED)

    assert updated.status is JobStatus.FAILED
    assert updated.error is JobFailureCode.PII_PROTECTION_FAILED


def test_queued_to_expired(store):
    job = store.create()
    assert store.mark_expired(job.id).status is JobStatus.EXPIRED


def test_processing_to_completed(store):
    job = store.create()
    store.mark_processing(job.id)

    updated = store.mark_completed(job.id)

    assert updated.status is JobStatus.COMPLETED
    assert updated.completed_at is not None
    assert updated.error is None


def test_processing_to_failed(store):
    job = store.create()
    store.mark_processing(job.id)

    updated = store.mark_failed(job.id, JobFailureCode.MODEL_ANALYSIS_FAILED)

    assert updated.status is JobStatus.FAILED
    assert updated.error is JobFailureCode.MODEL_ANALYSIS_FAILED
    # FAILED is not a completion; completed_at stays unset.
    assert updated.completed_at is None


def test_processing_to_expired(store):
    job = store.create()
    store.mark_processing(job.id)
    assert store.mark_expired(job.id).status is JobStatus.EXPIRED


def test_every_documented_valid_transition_is_accepted(store):
    """Walk the published transition table and assert each edge works."""
    for source, targets in ALLOWED_TRANSITIONS.items():
        for target in targets:
            job = store.create()
            if source is JobStatus.PROCESSING:
                store.mark_processing(job.id)
            assert store.require(job.id).status is source

            code = _code_for(target)
            updated = store.update_status(job.id, target, failure_code=code)

            assert updated.status is target, f"{source.value} -> {target.value}"


# --- H. invalid transitions -------------------------------------------------


def test_queued_cannot_jump_straight_to_completed(store):
    job = store.create()

    with pytest.raises(InvalidJobTransitionError, match="QUEUED"):
        store.mark_completed(job.id)

    assert store.require(job.id).status is JobStatus.QUEUED


def test_processing_cannot_return_to_queued(store):
    job = store.create()
    store.mark_processing(job.id)

    with pytest.raises(InvalidJobTransitionError):
        store.update_status(job.id, JobStatus.QUEUED)

    assert store.require(job.id).status is JobStatus.PROCESSING


def test_queued_cannot_transition_to_itself(store):
    job = store.create()

    with pytest.raises(InvalidJobTransitionError):
        store.update_status(job.id, JobStatus.QUEUED)


def test_every_transition_absent_from_the_table_is_rejected(store):
    """Exhaustively confirm the complement of ALLOWED_TRANSITIONS is refused."""
    reachable = {
        JobStatus.QUEUED: lambda job_id: None,
        JobStatus.PROCESSING: lambda job_id: store.mark_processing(job_id),
        JobStatus.COMPLETED: lambda job_id: (
            store.mark_processing(job_id),
            store.mark_completed(job_id),
        ),
        JobStatus.FAILED: lambda job_id: store.mark_failed(
            job_id, JobFailureCode.INTERNAL_ERROR
        ),
        JobStatus.EXPIRED: lambda job_id: store.mark_expired(job_id),
    }

    for source, arrive in reachable.items():
        for target in JobStatus:
            if target in ALLOWED_TRANSITIONS[source]:
                continue
            job = store.create()
            arrive(job.id)
            assert store.require(job.id).status is source

            code = _code_for(target)
            with pytest.raises(InvalidJobTransitionError):
                store.update_status(job.id, target, failure_code=code)

            assert store.require(job.id).status is source


def test_failed_requires_a_failure_code(store):
    job = store.create()

    with pytest.raises(ValueError, match="JobFailureCode is required"):
        store.update_status(job.id, JobStatus.FAILED)

    assert store.require(job.id).status is JobStatus.QUEUED


def test_failure_code_rejected_for_non_failure_states(store):
    job = store.create()

    with pytest.raises(ValueError, match="only valid for FAILED"):
        store.update_status(
            job.id,
            JobStatus.PROCESSING,
            failure_code=JobFailureCode.INTERNAL_ERROR,
        )


# --- I. terminal states -----------------------------------------------------


def test_terminal_states_are_exactly_the_documented_three():
    assert TERMINAL_STATUSES == {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.EXPIRED,
    }
    for status in TERMINAL_STATUSES:
        assert ALLOWED_TRANSITIONS[status] == frozenset()


@pytest.mark.parametrize(
    "terminal", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.EXPIRED]
)
def test_terminal_states_cannot_transition_again(store, terminal):
    job = store.create()
    if terminal is JobStatus.COMPLETED:
        store.mark_processing(job.id)
        store.mark_completed(job.id)
    elif terminal is JobStatus.FAILED:
        store.mark_failed(job.id, JobFailureCode.INTERNAL_ERROR)
    else:
        store.mark_expired(job.id)

    assert store.require(job.id).is_terminal is True

    for target in JobStatus:
        code = _code_for(target)
        with pytest.raises(InvalidJobTransitionError, match="terminal"):
            store.update_status(job.id, target, failure_code=code)

    assert store.require(job.id).status is terminal


# --- J. expiration ----------------------------------------------------------


def test_job_is_not_expired_before_its_ttl_elapses(store):
    job = store.create()

    assert job.is_expired(job.created_at) is False
    assert job.is_expired(job.expires_at - timedelta(seconds=1)) is False
    assert store.find_expired(job.created_at) == []


def test_expired_jobs_can_be_identified(store):
    fresh = store.create()
    stale = store.create()
    after_expiry = stale.expires_at + timedelta(seconds=1)

    # Move only `stale` into the past by evaluating a later instant for both,
    # then confirm identification keys off expires_at.
    assert stale.is_expired(after_expiry) is True
    expired_ids = {job.id for job in store.find_expired(after_expiry)}
    assert expired_ids == {fresh.id, stale.id}

    assert store.find_expired(fresh.created_at) == []


def test_expire_due_jobs_transitions_non_terminal_jobs(store):
    queued = store.create()
    processing = store.create()
    store.mark_processing(processing.id)
    after_expiry = queued.expires_at + timedelta(seconds=1)

    transitioned = store.expire_due_jobs(after_expiry)

    assert set(transitioned) == {queued.id, processing.id}
    assert store.require(queued.id).status is JobStatus.EXPIRED
    assert store.require(processing.id).status is JobStatus.EXPIRED


def test_expire_due_jobs_leaves_terminal_jobs_untouched(store):
    completed = store.create()
    store.mark_processing(completed.id)
    store.mark_completed(completed.id)
    after_expiry = completed.expires_at + timedelta(seconds=1)

    transitioned = store.expire_due_jobs(after_expiry)

    assert transitioned == []
    assert store.require(completed.id).status is JobStatus.COMPLETED


def test_expire_due_jobs_is_idempotent(store):
    job = store.create()
    after_expiry = job.expires_at + timedelta(seconds=1)

    assert store.expire_due_jobs(after_expiry) == [job.id]
    assert store.expire_due_jobs(after_expiry) == []


def test_purge_expired_removes_records(store):
    job = store.create()
    after_expiry = job.expires_at + timedelta(seconds=1)

    removed = store.purge_expired(after_expiry)

    assert removed == [job.id]
    assert store.get(job.id) is None
    assert len(store) == 0


def test_purge_expired_keeps_live_jobs(store):
    job = store.create()

    assert store.purge_expired(job.created_at) == []
    assert len(store) == 1


def test_expire_then_purge_is_the_full_cleanup_path(store):
    job = store.create()
    after_expiry = job.expires_at + timedelta(seconds=1)

    store.expire_due_jobs(after_expiry)
    assert store.require(job.id).status is JobStatus.EXPIRED

    store.purge_expired(after_expiry)
    assert store.get(job.id) is None


# --- K. privacy contract ----------------------------------------------------


def test_job_model_declares_only_metadata_fields():
    """The schema must not grow a field that could hold a document or PII."""
    assert set(AnalysisJob.model_fields) == {
        "id",
        "status",
        "created_at",
        "updated_at",
        "expires_at",
        "completed_at",
        "error",
    }


def test_job_model_forbids_extra_fields():
    """Sensitive payloads cannot be smuggled in as undeclared fields."""
    now = datetime.now(timezone.utc)

    for sensitive in ("text", "document_text", "pii_map", "prompt", "model_response"):
        with pytest.raises(ValidationError):
            AnalysisJob(
                id="x",
                created_at=now,
                updated_at=now,
                expires_at=now,
                **{sensitive: VETERAN_TEXT},
            )


def test_job_model_forbids_extra_fields_on_assignment(store):
    job = store.create()

    with pytest.raises(ValidationError):
        job.pii_map = PII_MAP


def test_store_never_holds_veteran_text_or_pii_mappings(store):
    """Serialize the whole store and assert no sensitive substring appears."""
    completed = store.create()
    store.mark_processing(completed.id)
    store.mark_completed(completed.id)

    failed = store.create()
    store.mark_failed(failed.id, JobFailureCode.MODEL_ANALYSIS_FAILED)

    dumped = "".join(job.model_dump_json() for job in store.list_jobs())

    assert "John Doe" not in dumped
    assert "123-45-6789" not in dumped
    assert "tinnitus" not in dumped
    assert "<PERSON_1>" not in dumped
    assert "<SSN_1>" not in dumped
    for value in PII_MAP.values():
        assert value not in dumped


# --- controlled failure codes -----------------------------------------------


def test_failure_code_set_is_closed_and_documented():
    assert {code.value for code in JobFailureCode} == {
        "DOCUMENT_EXTRACTION_FAILED",
        "PII_PROTECTION_FAILED",
        "MODEL_ANALYSIS_FAILED",
        "REGULATORY_ANALYSIS_FAILED",
        "INTERNAL_ERROR",
    }


@pytest.mark.parametrize("code", list(JobFailureCode))
def test_every_valid_failure_code_can_be_stored(store, code):
    job = store.create()

    updated = store.mark_failed(job.id, code)

    assert updated.status is JobStatus.FAILED
    assert updated.error is code
    # Survives a store round trip and JSON serialization as the bare code.
    assert store.require(job.id).error is code
    assert code.value in updated.model_dump_json()


def test_error_field_is_typed_as_the_failure_code_enum():
    """The model must offer no free-text failure field at all."""
    annotation = AnalysisJob.model_fields["error"].annotation
    assert JobFailureCode in getattr(annotation, "__args__", (annotation,))


@pytest.mark.parametrize(
    "bad",
    [
        SENSITIVE_FAILURE_TEXT,
        "Extraction failed: Veteran John Doe SSN 123-45-6789",
        "ValueError: unparsable JSON from model",
        "INTERNAL_ERROR",  # right spelling, still a bare str
        "",
        123,
        {"code": "INTERNAL_ERROR"},
    ],
)
def test_mark_failed_rejects_anything_that_is_not_a_failure_code(store, bad):
    """Arbitrary exception text cannot be recorded against a job."""
    job = store.create()

    with pytest.raises((TypeError, ValueError)):
        store.mark_failed(job.id, bad)

    # Rejected before any mutation: the job is untouched.
    unchanged = store.require(job.id)
    assert unchanged.status is JobStatus.QUEUED
    assert unchanged.error is None


def test_sensitive_string_cannot_enter_error_via_update_status(store):
    job = store.create()
    store.mark_processing(job.id)

    with pytest.raises(TypeError, match="must be a JobFailureCode"):
        store.update_status(
            job.id, JobStatus.FAILED, failure_code=SENSITIVE_FAILURE_TEXT
        )

    assert store.require(job.id).status is JobStatus.PROCESSING
    assert store.require(job.id).error is None


def test_sensitive_string_cannot_enter_error_via_model_construction():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError):
        AnalysisJob(
            id="x",
            status=JobStatus.FAILED,
            created_at=now,
            updated_at=now,
            expires_at=now,
            error=SENSITIVE_FAILURE_TEXT,
        )


def test_sensitive_string_cannot_enter_error_via_assignment(store):
    """Direct attribute assignment is validated, so it cannot smuggle text in."""
    job = store.create()

    with pytest.raises(ValidationError):
        job.error = SENSITIVE_FAILURE_TEXT

    assert job.error is None


def test_failed_job_never_serializes_sensitive_text(store):
    job = store.create()
    store.mark_processing(job.id)
    store.mark_failed(job.id, JobFailureCode.PII_PROTECTION_FAILED)

    dumped = store.require(job.id).model_dump_json()

    assert "John Doe" not in dumped
    assert "123-45-6789" not in dumped
    assert SENSITIVE_FAILURE_TEXT not in dumped
    assert "PII_PROTECTION_FAILED" in dumped


# --- store lifecycle --------------------------------------------------------


def test_get_job_store_returns_a_singleton():
    reset_job_store()
    try:
        assert get_job_store() is get_job_store()
    finally:
        reset_job_store()


def test_reset_job_store_discards_state():
    reset_job_store()
    try:
        job = get_job_store().create()
        reset_job_store()
        assert get_job_store().get(job.id) is None
    finally:
        reset_job_store()


def test_concurrent_creates_and_transitions_do_not_corrupt_the_registry():
    """Hammer the store from several threads and assert nothing is lost."""
    import threading

    store = JobStore(ttl_seconds=600)
    threads_count = 8
    per_thread = 40
    created_ids: list[str] = []
    created_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            local: list[str] = []
            for _ in range(per_thread):
                job = store.create()
                store.mark_processing(job.id)
                store.mark_completed(job.id)
                local.append(job.id)
            with created_lock:
                created_ids.extend(local)
        except BaseException as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(created_ids) == threads_count * per_thread
    assert len(set(created_ids)) == len(created_ids)
    assert len(store) == len(created_ids)
    assert all(
        store.require(job_id).status is JobStatus.COMPLETED for job_id in created_ids
    )
