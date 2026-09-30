"""Focused tests for asynchronous analysis-job execution (Step 5).

Every seam is injected, so nothing here builds a DLP client, makes a network
call, or needs credentials. All values are synthetic.
"""

import io
import threading
import time
from contextlib import redirect_stderr, redirect_stdout

import pytest

from claim_backend.analysis import (
    AnalysisResult,
    PotentialIssueType,
    RegulatoryAnalysisStatus,
    build_analysis_from_extraction,
)
from claim_backend.document_processing import (
    DocumentProtectionError,
    PiiTokenMap,
    ProtectedDocument,
    ProtectionMetadata,
)
from claim_backend.job_execution import (
    ExecutionOutcome,
    InlineDispatcher,
    ThreadDispatcher,
    default_analyze,
    execute_analysis_job,
    get_dispatcher,
    reset_dispatcher,
    set_dispatcher,
)
from claim_backend.jobs import (
    JobFailureCode,
    JobNotFoundError,
    JobStatus,
    JobStore,
)

# Synthetic only. Nothing here is real and nothing leaves the process.
VETERAN_NAME = "John Doe"
SSN = "123-45-6789"
PHONE = "555-0199"
RAW_DOCUMENT = (
    f"{VETERAN_NAME}, SSN {SSN}, phone {PHONE}, granted service connection "
    "for tinnitus, diagnostic code 6260, effective 2023-08-14, 10 percent."
)
SANITIZED = (
    "<PERSON_1>, SSN <SSN_1>, phone <PHONE_NUMBER_1>, granted service "
    "connection for tinnitus, diagnostic code 6260, effective 2023-08-14, "
    "10 percent."
)
TOKEN_MAP = {"<PERSON_1>": VETERAN_NAME, "<SSN_1>": SSN, "<PHONE_NUMBER_1>": PHONE}
SENSITIVE_VALUES = (VETERAN_NAME, SSN, PHONE, RAW_DOCUMENT)

MARKER = "SENSITIVE_EXCEPTION_MARKER"


@pytest.fixture
def store() -> JobStore:
    return JobStore(ttl_seconds=600)


def fake_protect(document_text: str) -> ProtectedDocument:
    """Stand-in for the Step 4 boundary; returns a realistic tokenized result."""
    return ProtectedDocument(
        sanitized_text=SANITIZED,
        token_map=PiiTokenMap(TOKEN_MAP),
        protection=ProtectionMetadata(
            finding_count=3,
            token_count=3,
            entity_types=["PERSON", "PHONE_NUMBER", "SSN"],
            sanitized_length=len(SANITIZED),
        ),
    )


class RecordingAnalyze:
    """Captures exactly what the analysis step was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, sanitized_text: str, job_id: str) -> AnalysisResult:
        self.calls.append((sanitized_text, job_id))
        return build_analysis_from_extraction({"conditions": []}, job_id=job_id)


def _run(store, job_id, text=RAW_DOCUMENT, **kwargs):
    kwargs.setdefault("protect", fake_protect)
    kwargs.setdefault("analyze", RecordingAnalyze())
    return execute_analysis_job(job_id, text, store=store, **kwargs)


# --- happy path: QUEUED -> PROCESSING -> COMPLETED ---------------------------


def test_job_runs_from_queued_to_completed(store):
    job = store.create()
    assert job.status is JobStatus.QUEUED

    outcome = _run(store, job.id)

    assert outcome.executed is True
    assert outcome.status is JobStatus.COMPLETED
    assert outcome.failure_code is None
    assert store.require(job.id).status is JobStatus.COMPLETED


def test_completion_stamps_the_job(store):
    job = store.create()

    _run(store, job.id)

    stored = store.require(job.id)
    assert stored.completed_at is not None
    assert stored.updated_at >= job.updated_at
    assert stored.error is None


def test_job_passes_through_processing(store):
    """The intermediate PROCESSING state is really entered, not skipped."""
    observed = []
    job = store.create()

    def observing_protect(text):
        observed.append(store.require(job.id).status)
        return fake_protect(text)

    _run(store, job.id, protect=observing_protect)

    assert observed == [JobStatus.PROCESSING]
    assert store.require(job.id).status is JobStatus.COMPLETED


def test_outcome_carries_the_analysis(store):
    job = store.create()

    outcome = _run(store, job.id)

    assert isinstance(outcome.analysis, AnalysisResult)
    assert outcome.analysis.job_id == job.id
    assert (
        outcome.analysis.regulatory_analysis_status
        is RegulatoryAnalysisStatus.NOT_PERFORMED
    )


# --- only sanitized text goes downstream -------------------------------------


def test_analysis_receives_sanitized_text_never_raw(store):
    job = store.create()
    analyze = RecordingAnalyze()

    _run(store, job.id, analyze=analyze)

    assert len(analyze.calls) == 1
    received_text, received_job_id = analyze.calls[0]
    assert received_text == SANITIZED
    assert received_job_id == job.id
    for value in SENSITIVE_VALUES:
        assert value not in received_text


def test_analysis_never_receives_the_token_map(store):
    """The analysis seam is handed a string, not the ProtectedDocument."""
    job = store.create()
    analyze = RecordingAnalyze()

    _run(store, job.id, analyze=analyze)

    received_text, _ = analyze.calls[0]
    assert isinstance(received_text, str)
    for original in TOKEN_MAP.values():
        assert original not in received_text


def test_protection_receives_the_raw_text(store):
    """Protection is the one place raw text legitimately arrives."""
    seen = []
    job = store.create()

    def capturing_protect(text):
        seen.append(text)
        return fake_protect(text)

    _run(store, job.id, protect=capturing_protect)

    assert seen == [RAW_DOCUMENT]


# --- protection failure -> FAILED / PII_PROTECTION_FAILED --------------------


def test_protection_failure_marks_the_job_failed(store):
    job = store.create()

    def failing_protect(text):
        raise DocumentProtectionError(
            failure_code=JobFailureCode.PII_PROTECTION_FAILED,
            stage="detection",
            cause_type="RuntimeError",
        )

    outcome = _run(store, job.id, protect=failing_protect)

    assert outcome.executed is True
    assert outcome.status is JobStatus.FAILED
    assert outcome.failure_code is JobFailureCode.PII_PROTECTION_FAILED

    stored = store.require(job.id)
    assert stored.status is JobStatus.FAILED
    assert stored.error is JobFailureCode.PII_PROTECTION_FAILED
    assert stored.completed_at is None


@pytest.mark.parametrize("code", list(JobFailureCode))
def test_controlled_failure_code_is_carried_through_verbatim(store, code):
    job = store.create()

    def failing_protect(text):
        raise DocumentProtectionError(failure_code=code, stage="detection")

    outcome = _run(store, job.id, protect=failing_protect)

    assert outcome.failure_code is code
    assert store.require(job.id).error is code


def test_analysis_is_not_run_when_protection_fails(store):
    job = store.create()
    analyze = RecordingAnalyze()

    def failing_protect(text):
        raise DocumentProtectionError(stage="detection")

    _run(store, job.id, protect=failing_protect, analyze=analyze)

    assert analyze.calls == [], "fail closed: nothing may go downstream"


# --- unexpected failure -> FAILED / INTERNAL_ERROR ---------------------------


@pytest.mark.parametrize(
    "boom",
    [
        RuntimeError(MARKER),
        ValueError(MARKER),
        KeyError(MARKER),
        TypeError(MARKER),
    ],
    ids=lambda e: type(e).__name__,
)
def test_unexpected_protection_failure_maps_to_internal_error(store, boom):
    job = store.create()

    def exploding_protect(text):
        raise boom

    outcome = _run(store, job.id, protect=exploding_protect)

    assert outcome.status is JobStatus.FAILED
    assert outcome.failure_code is JobFailureCode.INTERNAL_ERROR
    assert store.require(job.id).error is JobFailureCode.INTERNAL_ERROR


def test_unexpected_analysis_failure_maps_to_internal_error(store):
    job = store.create()

    def exploding_analyze(sanitized_text, job_id):
        raise RuntimeError(MARKER)

    outcome = _run(store, job.id, analyze=exploding_analyze)

    assert outcome.status is JobStatus.FAILED
    assert outcome.failure_code is JobFailureCode.INTERNAL_ERROR


def test_exception_text_never_reaches_the_job(store):
    """str(exception) must not be stored, even indirectly."""
    job = store.create()

    def exploding_protect(text):
        raise RuntimeError(f"{MARKER} for {VETERAN_NAME} SSN {SSN}")

    _run(store, job.id, protect=exploding_protect)

    dumped = store.require(job.id).model_dump_json()
    assert MARKER not in dumped
    for value in SENSITIVE_VALUES:
        assert value not in dumped
    assert store.require(job.id).error is JobFailureCode.INTERNAL_ERROR


def test_outcome_never_carries_exception_text(store):
    job = store.create()

    def exploding_protect(text):
        raise RuntimeError(f"{MARKER} {SSN}")

    outcome = _run(store, job.id, protect=exploding_protect)

    rendered = outcome.model_dump_json() + repr(outcome)
    assert MARKER not in rendered
    assert SSN not in rendered


# --- raw text and PII never reach the stored job -----------------------------


@pytest.mark.parametrize(
    "protect_impl",
    [
        fake_protect,
        pytest.param(
            lambda text: (_ for _ in ()).throw(DocumentProtectionError()),
            id="protection_failure",
        ),
        pytest.param(
            lambda text: (_ for _ in ()).throw(RuntimeError(RAW_DOCUMENT)),
            id="unexpected_failure",
        ),
    ],
)
def test_stored_job_never_contains_raw_text_or_pii(store, protect_impl):
    job = store.create()

    _run(store, job.id, protect=protect_impl)

    dumped = store.require(job.id).model_dump_json()
    for value in SENSITIVE_VALUES:
        assert value not in dumped
    assert SANITIZED not in dumped


def test_job_field_set_is_unchanged_by_execution(store):
    job = store.create()

    _run(store, job.id)

    stored = store.require(job.id)
    assert set(stored.model_dump()) == {
        "id",
        "status",
        "created_at",
        "updated_at",
        "expires_at",
        "completed_at",
        "error",
    }


def test_token_map_is_not_persisted_to_the_job(store):
    job = store.create()

    _run(store, job.id)

    stored = store.require(job.id)
    dumped = stored.model_dump_json()
    assert "token_map" not in dumped
    assert "<PERSON_1>" not in dumped
    for original in TOKEN_MAP.values():
        assert original not in dumped
    assert not hasattr(stored, "token_map")


def test_token_map_is_not_carried_on_the_outcome(store):
    job = store.create()

    outcome = _run(store, job.id)

    assert "token_map" not in outcome.model_dump()
    assert set(outcome.model_dump()) == {
        "job_id",
        "status",
        "executed",
        "failure_code",
        "analysis",
    }
    rendered = outcome.model_dump_json()
    for original in TOKEN_MAP.values():
        assert original not in rendered


def test_outcome_model_rejects_sensitive_fields():
    for field in (
        "document_text",
        "raw_text",
        "token_map",
        "pii_map",
        "sanitized_text",
        "prompt",
        "model_response",
        "metadata",
        "api_key",
    ):
        with pytest.raises(Exception):
            ExecutionOutcome(
                job_id="j",
                status=JobStatus.COMPLETED,
                executed=True,
                **{field: "injected"},
            )


# --- missing / invalid job IDs ----------------------------------------------


def test_missing_job_id_raises_a_controlled_error(store):
    with pytest.raises(JobNotFoundError):
        _run(store, "00000000-0000-0000-0000-000000000000")


def test_missing_job_error_contains_no_document_text(store):
    with pytest.raises(JobNotFoundError) as exc_info:
        _run(store, "unknown-id", text=RAW_DOCUMENT)

    rendered = str(exc_info.value) + repr(exc_info.value)
    for value in SENSITIVE_VALUES:
        assert value not in rendered


def test_no_job_is_created_as_a_side_effect(store):
    with pytest.raises(JobNotFoundError):
        _run(store, "unknown-id")

    assert len(store) == 0


# --- no accidental reprocessing ----------------------------------------------


def test_completed_job_is_not_reprocessed(store):
    job = store.create()
    _run(store, job.id)
    analyze = RecordingAnalyze()

    outcome = _run(store, job.id, analyze=analyze)

    assert outcome.executed is False
    assert outcome.status is JobStatus.COMPLETED
    assert analyze.calls == []


def test_failed_job_is_not_reprocessed(store):
    job = store.create()
    store.mark_failed(job.id, JobFailureCode.PII_PROTECTION_FAILED)
    analyze = RecordingAnalyze()

    outcome = _run(store, job.id, analyze=analyze)

    assert outcome.executed is False
    assert outcome.status is JobStatus.FAILED
    assert outcome.failure_code is JobFailureCode.PII_PROTECTION_FAILED
    assert analyze.calls == []


def test_expired_job_is_not_reprocessed(store):
    job = store.create()
    store.mark_expired(job.id)
    analyze = RecordingAnalyze()

    outcome = _run(store, job.id, analyze=analyze)

    assert outcome.executed is False
    assert outcome.status is JobStatus.EXPIRED
    assert analyze.calls == []


def test_job_already_processing_is_not_claimed_twice(store):
    job = store.create()
    store.mark_processing(job.id)
    analyze = RecordingAnalyze()

    outcome = _run(store, job.id, analyze=analyze)

    assert outcome.executed is False
    assert outcome.status is JobStatus.PROCESSING
    assert analyze.calls == []


def test_concurrent_execution_claims_a_job_exactly_once(store):
    """The QUEUED -> PROCESSING transition is the concurrency guard."""
    job = store.create()
    started = threading.Barrier(8)
    executed_flags = []
    analyze = RecordingAnalyze()
    lock = threading.Lock()

    def slow_protect(text):
        time.sleep(0.01)
        return fake_protect(text)

    def worker():
        started.wait()
        outcome = execute_analysis_job(
            job.id, RAW_DOCUMENT, store=store, protect=slow_protect, analyze=analyze
        )
        with lock:
            executed_flags.append(outcome.executed)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sum(1 for flag in executed_flags if flag) == 1
    assert len(analyze.calls) == 1
    assert store.require(job.id).status is JobStatus.COMPLETED


# --- dispatchers -------------------------------------------------------------


def test_inline_dispatcher_runs_synchronously(store):
    job = store.create()
    dispatcher = InlineDispatcher(
        store=store, protect=fake_protect, analyze=RecordingAnalyze()
    )

    outcome = dispatcher.dispatch(job.id, RAW_DOCUMENT)

    assert outcome.status is JobStatus.COMPLETED
    assert store.require(job.id).status is JobStatus.COMPLETED


def test_thread_dispatcher_runs_in_the_background(store):
    job = store.create()
    dispatcher = ThreadDispatcher(
        store=store, protect=fake_protect, analyze=RecordingAnalyze()
    )

    dispatcher.dispatch(job.id, RAW_DOCUMENT)
    assert dispatcher.wait(timeout=10) is True

    assert store.require(job.id).status is JobStatus.COMPLETED
    assert dispatcher.pending == 0


def test_thread_dispatcher_handles_many_jobs(store):
    jobs = [store.create() for _ in range(10)]
    dispatcher = ThreadDispatcher(
        store=store, protect=fake_protect, analyze=RecordingAnalyze()
    )

    for job in jobs:
        dispatcher.dispatch(job.id, RAW_DOCUMENT)
    assert dispatcher.wait(timeout=20) is True

    assert all(
        store.require(job.id).status is JobStatus.COMPLETED for job in jobs
    )


def test_thread_dispatcher_records_failures_on_the_job(store):
    job = store.create()

    def failing_protect(text):
        raise DocumentProtectionError(
            failure_code=JobFailureCode.PII_PROTECTION_FAILED
        )

    dispatcher = ThreadDispatcher(store=store, protect=failing_protect)
    dispatcher.dispatch(job.id, RAW_DOCUMENT)
    assert dispatcher.wait(timeout=10) is True

    assert store.require(job.id).error is JobFailureCode.PII_PROTECTION_FAILED


def test_thread_dispatcher_swallows_unknown_job_ids_quietly(store):
    """A bad id must not crash a thread or print a traceback."""
    dispatcher = ThreadDispatcher(store=store, protect=fake_protect)
    err = io.StringIO()

    with redirect_stderr(err):
        dispatcher.dispatch("unknown-id", RAW_DOCUMENT)
        assert dispatcher.wait(timeout=10) is True

    assert RAW_DOCUMENT not in err.getvalue()
    assert SSN not in err.getvalue()


def test_dispatcher_does_not_retain_document_text(store):
    job = store.create()
    dispatcher = ThreadDispatcher(
        store=store, protect=fake_protect, analyze=RecordingAnalyze()
    )
    dispatcher.dispatch(job.id, RAW_DOCUMENT)
    dispatcher.wait(timeout=10)

    rendered = repr(dispatcher.__dict__)
    for value in SENSITIVE_VALUES:
        assert value not in rendered


def test_result_sink_receives_the_outcome(store):
    job = store.create()
    received = []
    dispatcher = InlineDispatcher(
        store=store,
        protect=fake_protect,
        analyze=RecordingAnalyze(),
        result_sink=received.append,
    )

    dispatcher.dispatch(job.id, RAW_DOCUMENT)

    assert len(received) == 1
    assert received[0].status is JobStatus.COMPLETED


def test_process_dispatcher_accessor_is_a_singleton():
    reset_dispatcher()
    try:
        assert get_dispatcher() is get_dispatcher()
        assert isinstance(get_dispatcher(), ThreadDispatcher)
    finally:
        reset_dispatcher()


def test_dispatcher_can_be_overridden_for_tests(store):
    inline = InlineDispatcher(store=store, protect=fake_protect)
    set_dispatcher(inline)
    try:
        assert get_dispatcher() is inline
    finally:
        reset_dispatcher()


# --- default analyze seam ----------------------------------------------------


def test_default_analyze_invents_nothing(store):
    """The placeholder reports incomplete extraction rather than fabricating."""
    result = default_analyze(SANITIZED, "job-1")

    assert result.conditions == []
    assert [i.issue_type for i in result.potential_issues] == [
        PotentialIssueType.INCOMPLETE_EXTRACTION
    ]
    assert result.overall_confidence.score is None
    assert result.regulatory_analysis_status is RegulatoryAnalysisStatus.NOT_PERFORMED


def test_default_analyze_does_not_echo_its_input(store):
    result = default_analyze(SANITIZED, "job-1")
    dumped = result.model_dump_json()

    assert SANITIZED not in dumped
    for value in SENSITIVE_VALUES:
        assert value not in dumped


# --- no logging or printing --------------------------------------------------


def test_module_contains_no_logging_or_printing():
    import claim_backend.job_execution as module

    source = open(module.__file__).read()
    assert "import logging" not in source
    assert "logging.getLogger" not in source
    assert "print(" not in source
    assert "@traceable" not in source
    assert "langsmith" not in source


@pytest.mark.parametrize(
    "protect_impl",
    [
        fake_protect,
        pytest.param(
            lambda text: (_ for _ in ()).throw(DocumentProtectionError()),
            id="protection_failure",
        ),
        pytest.param(
            lambda text: (_ for _ in ()).throw(RuntimeError(f"{MARKER} {SSN}")),
            id="unexpected_failure",
        ),
    ],
)
def test_execution_writes_nothing_to_stdout_or_stderr(store, protect_impl):
    job = store.create()
    out, err = io.StringIO(), io.StringIO()

    with redirect_stdout(out), redirect_stderr(err):
        _run(store, job.id, protect=protect_impl)

    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_execution_logs_nothing_sensitive(store, caplog):
    import logging

    job = store.create()
    with caplog.at_level(logging.DEBUG):
        _run(store, job.id)

    for value in SENSITIVE_VALUES:
        assert value not in caplog.text


# --- existing behavior preserved ---------------------------------------------


def test_job_api_still_creates_queued_jobs_without_processing():
    """Step 2's contract is untouched: no body accepted, nothing dispatched."""
    from fastapi.testclient import TestClient

    from claim_backend.jobs import get_job_store, reset_job_store
    from claim_backend.main import app

    reset_job_store()
    try:
        client = TestClient(app)
        created = client.post("/api/jobs")
        assert created.status_code == 201
        assert created.json()["status"] == "QUEUED"

        # Submitting document text is still refused.
        rejected = client.post("/api/jobs", json={"text": RAW_DOCUMENT})
        assert rejected.status_code == 422
        assert VETERAN_NAME not in rejected.text

        # And the job stays QUEUED: Step 5 adds no implicit dispatch.
        assert client.get(f"/api/jobs/{created.json()['id']}").json()["status"] == (
            "QUEUED"
        )
    finally:
        reset_job_store()
