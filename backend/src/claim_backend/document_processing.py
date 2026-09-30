"""The secure document-processing boundary.

Every path from document-derived text to downstream analysis runs through
here::

    raw text -> DLP detection -> tokenization -> ProtectedDocument
                                                        |
                                            sanitized_text -> analysis

This module holds the *only* Cloud DLP implementation in the project.
``main`` imports from here rather than defining its own, so there is exactly
one detector, one tokenizer, and one rehydrator.

Importability
-------------
Defining these contracts must never require credentials. The DLP client is
built lazily and is always injectable, so the module imports, and its models
validate, with no Google Cloud configuration present. A credential is needed
only when a caller actually inspects text *and* declines to supply a client.

The security boundary
---------------------
:func:`protect_document_text` is the boundary. It returns a
:class:`ProtectedDocument` whose ``sanitized_text`` is the one representation
intended for downstream model or regulatory analysis. The original text is
never stored on the result -- there is no field for it -- so downstream code
cannot reach back through the boundary even by accident.

The token map needed for controlled rehydration is application-owned,
in-memory, and deliberately hostile to escape: see :class:`PiiTokenMap`, which
redacts its own ``repr``, refuses to pickle, and is excluded from every
serialization of :class:`ProtectedDocument`.

Rehydration is a separate, explicit, final-stage operation
(:func:`rehydrate_text`, :func:`rehydrate_structure`). It never happens
automatically inside sanitization, and it always requires the caller to hand
over the map on purpose.

Fail closed
-----------
If detection or tokenization fails, no partially-protected text is returned.
:class:`DocumentProtectionError` is raised instead, carrying a controlled
:class:`~claim_backend.jobs.JobFailureCode` and a fixed public message. The
underlying exception's message is deliberately dropped so provider error text
-- which may quote the payload -- can never surface to a client or land in
``AnalysisJob.error``.

Observability
-------------
Nothing here logs, prints, or emits telemetry, and no tracing decorator is
applied to any function that can see raw text or the token map.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Tuple

from google.cloud import dlp_v2
from pydantic import BaseModel, ConfigDict, Field

from .jobs import JobFailureCode

# ---------------------------------------------------------------------------
# Cloud DLP configuration
#
# Extracted verbatim from main.py so a single implementation serves both the
# existing /api/extract pipeline and this boundary. Behavior is unchanged:
# the same built-in info types, the same custom SSN and local-phone
# detectors, and the same info-type -> token mapping.
# ---------------------------------------------------------------------------

# Built-in DLP info types. DATE and generic numeric types are deliberately
# omitted so VA effective dates and diagnostic codes survive untouched.
DLP_BUILTIN_INFO_TYPES = [
    "PERSON_NAME",
    "US_SOCIAL_SECURITY_NUMBER",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "STREET_ADDRESS",
    "LOCATION",
]

# Custom regex info types cover the edge cases the built-in detectors miss:
# dashless 9-digit SSNs and 7-digit local numbers without an area code.
DLP_CUSTOM_INFO_TYPES = [
    dlp_v2.CustomInfoType(
        info_type=dlp_v2.InfoType(name="CUSTOM_SSN"),
        regex=dlp_v2.CustomInfoType.Regex(pattern=r"\b\d{3}-?\d{2}-?\d{4}\b"),
        likelihood=dlp_v2.Likelihood.VERY_LIKELY,
    ),
    dlp_v2.CustomInfoType(
        info_type=dlp_v2.InfoType(name="CUSTOM_LOCAL_PHONE"),
        regex=dlp_v2.CustomInfoType.Regex(
            pattern=r"\b(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
        ),
        likelihood=dlp_v2.Likelihood.LIKELY,
    ),
]

# Map DLP info type names onto the stable token prefixes used in the map.
DLP_INFO_TYPE_TO_ENTITY = {
    "PERSON_NAME": "PERSON",
    "US_SOCIAL_SECURITY_NUMBER": "SSN",
    "CUSTOM_SSN": "SSN",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "CUSTOM_LOCAL_PHONE": "PHONE_NUMBER",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "STREET_ADDRESS": "LOCATION",
    "LOCATION": "LOCATION",
}

#: Identifies which protection implementation produced a result.
PROTECTION_ENGINE = "google-cloud-dlp"

#: Fixed, client-safe text for any protection failure. Never interpolated.
PROTECTION_FAILURE_MESSAGE = (
    "Document protection failed; no sanitized text was produced."
)

_dlp_client: Optional[dlp_v2.DlpServiceClient] = None


def _get_dlp_client() -> dlp_v2.DlpServiceClient:
    """Build the DLP client on first use.

    Lazy so that importing this module never touches credentials. Tests
    inject a fake client instead of calling this.
    """
    global _dlp_client
    if _dlp_client is None:
        _dlp_client = dlp_v2.DlpServiceClient()
    return _dlp_client


def dlp_parent(project_id: Optional[str] = None) -> str:
    """Build the DLP resource parent for a project.

    Reads ``GOOGLE_CLOUD_PROJECT`` at call time rather than import time, so
    this module stays importable without configuration.
    """
    project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise DocumentProtectionError(stage="configuration")
    return f"projects/{project}/locations/global"


class PiiFinding(NamedTuple):
    """A single PII span detected by Cloud DLP, in codepoint offsets."""

    entity_type: str
    start: int
    end: int


class DocumentProtectionError(Exception):
    """A controlled protection failure.

    Carries a :class:`~claim_backend.jobs.JobFailureCode` and a fixed public
    message. The triggering exception's message is intentionally *not*
    captured: provider errors can quote the payload that caused them, so
    propagating that text would defeat the entire boundary. Only the cause's
    class name is retained, via :attr:`cause_type`, which is safe to surface
    to operators and useful for diagnosis.
    """

    def __init__(
        self,
        *,
        failure_code: JobFailureCode = JobFailureCode.PII_PROTECTION_FAILED,
        stage: str = "detection",
        cause_type: Optional[str] = None,
    ) -> None:
        self.failure_code = failure_code
        self.stage = stage
        self.cause_type = cause_type
        # Defence in depth: if this error is ever raised from inside an
        # `except` block, suppress display of the implicit chain. The real
        # guarantee comes from _controlled_failure plus raising outside the
        # handler, which leaves __context__ itself empty.
        self.__suppress_context__ = True
        super().__init__(PROTECTION_FAILURE_MESSAGE)

    def __str__(self) -> str:  # noqa: D105 - fixed, content-free
        return PROTECTION_FAILURE_MESSAGE

    def __repr__(self) -> str:  # noqa: D105 - fixed, content-free
        return (
            f"DocumentProtectionError(failure_code={self.failure_code.value}, "
            f"stage={self.stage!r}, cause_type={self.cause_type!r})"
        )


def _controlled_failure(
    *,
    failure_code: JobFailureCode,
    stage: str,
    cause_type: Optional[str] = None,
) -> DocumentProtectionError:
    """Build a controlled failure with every exception link severed.

    Callers **must** return this from an ``except`` block and raise it only
    *after* the handler has exited. ``raise ... from None`` is not sufficient:
    it clears ``__cause__`` but Python still attaches the exception being
    handled to ``__context__``, leaving the provider's message -- which can
    quote the document payload -- reachable on the controlled error. Raising
    outside the handler means there is no exception being handled, so no
    context is attached at all. The explicit clearing below covers the
    remaining case where the object already carried links.
    """
    error = DocumentProtectionError(
        failure_code=failure_code, stage=stage, cause_type=cause_type
    )
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True
    return error


class PiiTokenMap:
    """Temporary, application-owned token -> original-value mapping.

    Deliberately *not* a Pydantic model and deliberately awkward to move
    around. It exists for one purpose: controlled rehydration during the
    current request, in memory, and then it is dropped.

    The defenses are structural rather than advisory:

    * ``__repr__``/``__str__`` report only a token count, so interpolating
      one into a message or traceback cannot leak originals.
    * pickling raises, so it cannot be written to disk, pushed into a cache,
      or sent over a wire by accident.
    * iteration yields tokens, never values.
    * reading originals requires the explicitly-named :meth:`resolve` or
      :meth:`as_dict`, which are easy to spot in review.
    """

    __slots__ = ("_mapping",)

    def __init__(self, mapping: Optional[Mapping[str, str]] = None) -> None:
        self._mapping: Dict[str, str] = dict(mapping or {})

    def __repr__(self) -> str:
        return f"<PiiTokenMap tokens={len(self._mapping)} values=redacted>"

    __str__ = __repr__

    def __len__(self) -> int:
        return len(self._mapping)

    def __bool__(self) -> bool:
        return bool(self._mapping)

    def __contains__(self, token: object) -> bool:
        return token in self._mapping

    def __iter__(self) -> Iterator[str]:
        """Iterate tokens only. Originals require an explicit accessor."""
        return iter(self._mapping)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PiiTokenMap):
            return NotImplemented
        return self._mapping == other._mapping

    def tokens(self) -> List[str]:
        """The token placeholders, safe to surface."""
        return list(self._mapping)

    def resolve(self, token: str) -> Optional[str]:
        """Return the original value behind one token."""
        return self._mapping.get(token)

    def as_dict(self) -> Dict[str, str]:
        """Return a copy of the raw mapping.

        Named to be conspicuous: every call site is a deliberate decision to
        handle original PII.
        """
        return dict(self._mapping)

    def __reduce__(self):
        raise TypeError(
            "PiiTokenMap is deliberately not picklable: the mapping must stay "
            "in memory for the current request and must never be persisted."
        )

    def __getstate__(self):
        raise TypeError(
            "PiiTokenMap is deliberately not serializable: the mapping must "
            "never be persisted or transmitted."
        )

    def __copy__(self) -> "PiiTokenMap":
        return PiiTokenMap(self._mapping)

    def __deepcopy__(self, memo) -> "PiiTokenMap":
        return PiiTokenMap(self._mapping)


class ProtectionMetadata(BaseModel):
    """Non-sensitive facts about a protection run.

    Counts and entity-type names only -- enough for downstream code and
    operators to reason about coverage, with no value ever recorded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: str = PROTECTION_ENGINE
    finding_count: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)
    entity_types: List[str] = Field(default_factory=list)
    sanitized_length: int = Field(default=0, ge=0)


class ProtectedDocument(BaseModel):
    """The sanitized result of the protection boundary.

    ``sanitized_text`` is the only representation intended for downstream
    model or regulatory analysis.

    There is no field for the original document text, and ``extra="forbid"``
    means one cannot be attached. ``frozen=True`` means nothing here can be
    swapped out after construction. The token map is carried for controlled
    rehydration but is marked ``exclude=True``, so it is absent from
    ``model_dump()`` and ``model_dump_json()`` -- serializing a
    ``ProtectedDocument``, including through an API response, cannot emit the
    mapping.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    sanitized_text: str
    token_map: PiiTokenMap = Field(default_factory=PiiTokenMap, exclude=True, repr=False)
    protection: ProtectionMetadata = Field(default_factory=ProtectionMetadata)

    @property
    def has_protected_content(self) -> bool:
        """True when at least one value was tokenized."""
        return len(self.token_map) > 0


# ---------------------------------------------------------------------------
# Detection and tokenization
#
# Moved from main.py unchanged except that the DLP client and resource parent
# are now explicit parameters. main.py keeps a thin wrapper that supplies
# both, which preserves its public behavior and its existing test seams.
# ---------------------------------------------------------------------------


def detect_pii(
    text: str,
    *,
    client: Any,
    parent: str,
) -> List[PiiFinding]:
    """Inspect text with Cloud DLP and return the PII spans to tokenize.

    Intentionally not traced: the input here still contains raw PII.

    :param client: a DLP client. Explicit so tests inject a fake and no live
        call can occur.
    """
    response = client.inspect_content(
        request={
            "parent": parent,
            "inspect_config": {
                "info_types": [{"name": name} for name in DLP_BUILTIN_INFO_TYPES],
                "custom_info_types": DLP_CUSTOM_INFO_TYPES,
                "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
                "include_quote": False,
                "limits": {"max_findings_per_request": 0},
            },
            "item": {"value": text},
        }
    )

    findings: List[PiiFinding] = []
    for finding in response.result.findings:
        entity_type = DLP_INFO_TYPE_TO_ENTITY.get(finding.info_type.name)
        if not entity_type:
            continue
        span = finding.location.codepoint_range
        if span.end <= span.start:
            continue
        findings.append(
            PiiFinding(entity_type=entity_type, start=span.start, end=span.end)
        )
    return findings


def tokenize_pii(
    text: str, findings: List[PiiFinding]
) -> Tuple[str, Dict[str, str]]:
    """Replace detected PII with safe tokens and return the tokenized text + map.

    The map is token -> original value, kept in request-local memory only.
    """
    pii_map: Dict[str, str] = {}
    type_counts: Dict[str, int] = {}
    # Walk backward through the text so earlier indices stay valid, preferring
    # the longest span whenever two findings start at the same offset.
    sorted_findings = sorted(
        findings, key=lambda f: (f.start, f.end - f.start), reverse=True
    )
    tokenized = text
    last_start = len(text)

    for finding in sorted_findings:
        if finding.end > last_start:
            # Skip findings that overlap a span we already tokenized.
            continue
        original = text[finding.start : finding.end]
        entity_type = finding.entity_type
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        token = f"<{entity_type}_{type_counts[entity_type]}>"
        pii_map[token] = original
        tokenized = tokenized[: finding.start] + token + tokenized[finding.end :]
        last_start = finding.start

    return tokenized, pii_map


def rehydrate_mapping_text(text: str, entity_map: Dict[str, str]) -> str:
    """Replace safe tokens with original PII values, longest tokens first."""
    result = text
    for token in sorted(entity_map.keys(), key=len, reverse=True):
        result = result.replace(token, entity_map[token])
    return result


def rehydrate_mapping_structure(obj: Any, entity_map: Dict[str, str]) -> Any:
    """Recursively replace safe tokens in parsed JSON with the original values."""
    if isinstance(obj, dict):
        return {k: rehydrate_mapping_structure(v, entity_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rehydrate_mapping_structure(item, entity_map) for item in obj]
    if isinstance(obj, str):
        result = obj
        for token in sorted(entity_map.keys(), key=len, reverse=True):
            result = result.replace(token, entity_map[token])
        return result
    return obj


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def protect_document_text(
    text: str,
    *,
    client: Optional[Any] = None,
    parent: Optional[str] = None,
) -> ProtectedDocument:
    """Protect document-derived text and return only its sanitized form.

    This is the security boundary. Downstream model and regulatory analysis
    must consume :attr:`ProtectedDocument.sanitized_text` and nothing else;
    the original text is not carried on the result and is unreachable from it.

    Fails closed. If detection or tokenization raises, no partially-protected
    text is returned -- :class:`DocumentProtectionError` is raised with a
    controlled failure code and a fixed message, and the cause's message is
    dropped.

    Rehydration is *not* performed here and never happens implicitly; call
    :func:`rehydrate_text` deliberately if a final-stage caller needs it.

    :param text: document-derived text. Must be non-empty.
    :param client: DLP client. Supply a fake in tests; when omitted a real
        client is built lazily, which is the only path that needs credentials.
    :param parent: DLP resource parent. Derived from ``GOOGLE_CLOUD_PROJECT``
        when omitted.
    :raises DocumentProtectionError: on invalid input or any protection failure.
    """
    if not isinstance(text, str) or not text.strip():
        # Fail closed rather than returning a vacuously "safe" empty result.
        raise DocumentProtectionError(
            failure_code=JobFailureCode.DOCUMENT_EXTRACTION_FAILED,
            stage="input_validation",
        )

    resolved_parent = parent or dlp_parent()

    # Each stage captures its failure and raises it only after the `except`
    # block has exited. Raising inside the handler would let Python attach the
    # provider exception to __context__, and `from None` does not prevent that
    # -- see _controlled_failure.
    failure: Optional[DocumentProtectionError] = None
    findings: List[PiiFinding] = []
    try:
        active_client = client if client is not None else _get_dlp_client()
        findings = detect_pii(text, client=active_client, parent=resolved_parent)
    except DocumentProtectionError as exc:
        # Already controlled; re-raise outside the handler too, and re-sever
        # its links in case it arrived carrying any.
        failure = _controlled_failure(
            failure_code=exc.failure_code,
            stage=exc.stage,
            cause_type=exc.cause_type,
        )
    except Exception as exc:
        failure = _controlled_failure(
            failure_code=JobFailureCode.PII_PROTECTION_FAILED,
            stage="detection",
            cause_type=type(exc).__name__,
        )
    if failure is not None:
        raise failure

    try:
        sanitized_text, mapping = tokenize_pii(text, findings)
    except Exception as exc:
        failure = _controlled_failure(
            failure_code=JobFailureCode.PII_PROTECTION_FAILED,
            stage="tokenization",
            cause_type=type(exc).__name__,
        )
    if failure is not None:
        raise failure

    return ProtectedDocument(
        sanitized_text=sanitized_text,
        token_map=PiiTokenMap(mapping),
        protection=ProtectionMetadata(
            engine=PROTECTION_ENGINE,
            finding_count=len(findings),
            token_count=len(mapping),
            entity_types=sorted({f.entity_type for f in findings}),
            sanitized_length=len(sanitized_text),
        ),
    )


def rehydrate_text(sanitized_text: str, token_map: PiiTokenMap) -> str:
    """Restore original values into sanitized text. Final-stage only.

    An exceptional operation, not part of ordinary analysis: it reverses the
    protection boundary and yields text containing real PII. It happens only
    when called deliberately and only when the caller supplies the map. The
    result is never logged and the map is never retained.
    """
    if not isinstance(token_map, PiiTokenMap):
        raise TypeError(
            "rehydration requires an explicit PiiTokenMap; refusing to "
            "rehydrate from a bare mapping"
        )
    return rehydrate_mapping_text(sanitized_text, token_map.as_dict())


def rehydrate_structure(obj: Any, token_map: PiiTokenMap) -> Any:
    """Recursively restore original values in a parsed structure.

    Carries the same warning as :func:`rehydrate_text`: final-stage, explicit,
    never automatic.
    """
    if not isinstance(token_map, PiiTokenMap):
        raise TypeError(
            "rehydration requires an explicit PiiTokenMap; refusing to "
            "rehydrate from a bare mapping"
        )
    return rehydrate_mapping_structure(obj, token_map.as_dict())
