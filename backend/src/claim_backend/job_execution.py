"""Asynchronous execution of analysis jobs.

Wires the four existing pieces into one pipeline::

    job (QUEUED)
        |  mark_processing
        v
    PROCESSING -- protect_document_text() --> ProtectedDocument
                                                    |
                                          sanitized_text only
                                                    v
                                          structured analysis
                                                    |
                                        COMPLETED / FAILED

This module owns no PII logic of its own. Detection, tokenization,
rehydration, and the token map all remain in
:mod:`claim_backend.document_processing`; the analysis contract remains in
:mod:`claim_backend.analysis`; lifecycle and storage remain in
:mod:`claim_backend.jobs`.

What crosses the boundary
-------------------------
Raw document text exists only as a parameter on the transient execution call
stack. It is never written to the job, never attached to an exception, never
placed in :class:`ExecutionOutcome`, and never handed to the analysis step --
only :attr:`ProtectedDocument.sanitized_text` goes downstream. The token map
stays inside its ``ProtectedDocument`` and is dropped when the call returns.

Failure handling
----------------
Fail closed. A :class:`~claim_backend.document_processing.DocumentProtectionError`
moves the job to ``FAILED`` with the controlled
:class:`~claim_backend.jobs.JobFailureCode` it already carries. Anything
unexpected moves the job to ``FAILED`` with ``INTERNAL_ERROR``. No exception
string is ever stored, because :meth:`JobStore.mark_failed` structurally
refuses anything that is not a ``JobFailureCode``.

Reprocessing
------------
Execution claims a job by transitioning ``QUEUED -> PROCESSING`` through the
Step 1 transition table, which is guarded by the store's lock. That single
atomic step is the concurrency control: a job already processing, completed,
failed, or expired cannot be claimed, so it is skipped rather than
reprocessed.

Nothing here logs, prints, or emits telemetry.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, ConfigDict

from .analysis import AnalysisResult, build_analysis_from_extraction
from .document_processing import (
    DocumentProtectionError,
    ProtectedDocument,
    protect_document_text,
)
from .jobs import (
    InvalidJobTransitionError,
    JobFailureCode,
    JobStatus,
    JobStore,
    get_job_store,
)

#: Protects raw text. Takes the raw document text, returns a ProtectedDocument.
ProtectFn = Callable[[str], ProtectedDocument]

#: Analyses *sanitized* text. Takes (sanitized_text, job_id).
#: Note the first parameter is sanitized text -- raw text never reaches here.
AnalyzeFn = Callable[[str, str], AnalysisResult]

#: Receives a finished outcome. A seam for a future result store; by default
#: nothing is registered and outcomes are simply returned to the caller.
ResultSink = Callable[["ExecutionOutcome"], None]


class ExecutionOutcome(BaseModel):
    """The non-sensitive result of one execution attempt.

    Carries no raw document text and no token map -- only the job's resulting
    status, the controlled failure code if any, and the structured analysis
    (which is itself derived from sanitized text).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    status: JobStatus
    executed: bool
    failure_code: Optional[JobFailureCode] = None
    analysis: Optional[AnalysisResult] = None


def default_protect(document_text: str) -> ProtectedDocument:
    """Protect text using the established Step 4 boundary.

    Delegates straight to :func:`protect_document_text`, which resolves a DLP
    client lazily. Tests inject a fake instead of calling this.
    """
    return protect_document_text(document_text)


def default_analyze(sanitized_text: str, job_id: str) -> AnalysisResult:
    """Produce a structured analysis from sanitized text.

    **Placeholder seam.** Turning sanitized text into extracted conditions
    requires the model-extraction step, which is not wired into jobs yet, so
    no model call is made here. This runs the existing Step 3 adapter over an
    empty extraction, which yields a well-formed :class:`AnalysisResult`
    carrying an ``INCOMPLETE_EXTRACTION`` review item. That is the honest
    representation of "nothing has been extracted yet" -- it invents no
    findings and asserts no confidence.

    A later step replaces this function via the ``analyze`` parameter of
    :func:`execute_analysis_job`; no other code needs to change.
    """
    # sanitized_text is accepted (and deliberately not inspected here) so the
    # signature is already correct for the real extractor.
    return build_analysis_from_extraction({"conditions": []}, job_id=job_id)


def execute_analysis_job(
    job_id: str,
    document_text: str,
    *,
    store: Optional[JobStore] = None,
    protect: ProtectFn = default_protect,
    analyze: AnalyzeFn = default_analyze,
) -> ExecutionOutcome:
    """Run one analysis job to a terminal state.

    Synchronous and directly callable, which is what makes it testable: the
    background dispatcher below is a thin wrapper around this function.

    Claiming is atomic. The ``QUEUED -> PROCESSING`` transition is the only
    way in, so a job that is already processing or already terminal is
    skipped and returned with ``executed=False`` rather than being run twice.

    :param document_text: raw, document-derived text. Lives only on this call
        stack; it is never stored, logged, or passed downstream unprotected.
    :param store: job store; defaults to the application-level store.
    :param protect: protection seam, for injecting fakes or failures.
    :param analyze: analysis seam, which receives *sanitized* text only.
    :raises JobNotFoundError: if ``job_id`` is unknown.
    """
    active_store = store if store is not None else get_job_store()

    # Claim the job. mark_processing raises JobNotFoundError for an unknown
    # id, and InvalidJobTransitionError for anything not in QUEUED.
    try:
        active_store.mark_processing(job_id)
    except InvalidJobTransitionError:
        current = active_store.require(job_id)
        return ExecutionOutcome(
            job_id=job_id,
            status=current.status,
            executed=False,
            failure_code=current.error,
        )

    protection_failure: Optional[JobFailureCode] = None
    unexpected_failure = False
    analysis: Optional[AnalysisResult] = None

    try:
        protected = protect(document_text)
        # Only the sanitized representation crosses into analysis. The token
        # map stays on `protected` and dies with this frame.
        analysis = analyze(protected.sanitized_text, job_id)
    except DocumentProtectionError as exc:
        # Take only the controlled code; the exception's message is never read.
        protection_failure = exc.failure_code
    except Exception:
        # Deliberately bare: the exception object is not inspected, so no
        # provider text, document text, or PII can be carried forward.
        unexpected_failure = True

    if protection_failure is not None:
        failed = active_store.mark_failed(job_id, protection_failure)
        return ExecutionOutcome(
            job_id=job_id,
            status=failed.status,
            executed=True,
            failure_code=failed.error,
        )

    if unexpected_failure:
        failed = active_store.mark_failed(job_id, JobFailureCode.INTERNAL_ERROR)
        return ExecutionOutcome(
            job_id=job_id,
            status=failed.status,
            executed=True,
            failure_code=failed.error,
        )

    completed = active_store.mark_completed(job_id)
    return ExecutionOutcome(
        job_id=job_id,
        status=completed.status,
        executed=True,
        analysis=analysis,
    )


class InlineDispatcher:
    """Runs jobs on the calling thread.

    The default for tests and for any caller that wants deterministic,
    synchronous behavior.
    """

    def __init__(
        self,
        *,
        store: Optional[JobStore] = None,
        protect: ProtectFn = default_protect,
        analyze: AnalyzeFn = default_analyze,
        result_sink: Optional[ResultSink] = None,
    ) -> None:
        self._store = store
        self._protect = protect
        self._analyze = analyze
        self._result_sink = result_sink

    def dispatch(self, job_id: str, document_text: str) -> ExecutionOutcome:
        outcome = execute_analysis_job(
            job_id,
            document_text,
            store=self._store,
            protect=self._protect,
            analyze=self._analyze,
        )
        if self._result_sink is not None:
            self._result_sink(outcome)
        return outcome

    def wait(self, timeout: Optional[float] = None) -> bool:
        """No-op: inline dispatch has already finished."""
        return True


class ThreadDispatcher:
    """Runs jobs on short-lived daemon threads.

    The minimal background mechanism that suits the current in-memory,
    single-process architecture: no broker, no worker service, no external
    dependency. The job store it writes to is already lock-guarded, so
    concurrent executions are safe, and the transition table prevents two
    threads from claiming the same job.

    Suitable for development and single-process deployment only. Work is lost
    on restart and is not shared across workers -- the same constraint the job
    store already carries, and it needs the same durable replacement when the
    store gets one.

    Document text is passed as a thread argument, so it lives on that
    thread's stack and nowhere else. It is never stored on this object.
    """

    def __init__(
        self,
        *,
        store: Optional[JobStore] = None,
        protect: ProtectFn = default_protect,
        analyze: AnalyzeFn = default_analyze,
        result_sink: Optional[ResultSink] = None,
    ) -> None:
        self._store = store
        self._protect = protect
        self._analyze = analyze
        self._result_sink = result_sink
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []

    def _run(self, job_id: str, document_text: str) -> None:
        try:
            outcome = execute_analysis_job(
                job_id,
                document_text,
                store=self._store,
                protect=self._protect,
                analyze=self._analyze,
            )
        except Exception:
            # The worker already converts failures into job state. Anything
            # escaping is swallowed rather than printed, because the default
            # threading excepthook would write a traceback -- which could
            # quote arguments -- to stderr.
            return
        if self._result_sink is not None:
            try:
                self._result_sink(outcome)
            except Exception:
                return

    def dispatch(self, job_id: str, document_text: str) -> threading.Thread:
        """Start execution on a background thread and return immediately."""
        thread = threading.Thread(
            target=self._run,
            args=(job_id, document_text),
            name=f"analysis-job-{job_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until dispatched work finishes. Returns True if all joined."""
        with self._lock:
            pending = list(self._threads)
        for thread in pending:
            thread.join(timeout)
            if thread.is_alive():
                return False
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
        return True

    @property
    def pending(self) -> int:
        """How many dispatched threads are still running."""
        with self._lock:
            return sum(1 for t in self._threads if t.is_alive())


_dispatcher: Optional[Any] = None
_dispatcher_lock = threading.Lock()


def get_dispatcher() -> Any:
    """Return the process-wide dispatcher, creating it on first use.

    Mirrors :func:`claim_backend.jobs.get_job_store`: one accessor, no loose
    module-level state, and overridable in tests via :func:`set_dispatcher`.
    """
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = ThreadDispatcher()
        return _dispatcher


def set_dispatcher(dispatcher: Optional[Any]) -> None:
    """Replace the process-wide dispatcher; pass ``None`` to reset it."""
    global _dispatcher
    with _dispatcher_lock:
        _dispatcher = dispatcher


def reset_dispatcher() -> None:
    """Drop the process-wide dispatcher so the next access rebuilds it."""
    set_dispatcher(None)
