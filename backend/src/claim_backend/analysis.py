"""Typed domain contract for VSO-facing claim analysis.

This module defines *what an analysis says*, nothing more. It performs no I/O,
makes no model calls, retrieves no regulations, and stores nothing. Like
:mod:`claim_backend.jobs` it imports no credential-bearing application code,
so it can be exercised in isolation.

Facts versus interpretation
---------------------------
The central design rule is that a model inference must never be presented as
an established fact, and never as a legal conclusion. Every assertion
therefore carries an :class:`AssertionBasis`:

* :attr:`AssertionBasis.DECISION_STATED` -- reported as explicitly stated in
  the VA decision.
* :attr:`AssertionBasis.EVIDENCE_STATED` -- reported as explicitly stated in a
  supporting evidence document.
* :attr:`AssertionBasis.MODEL_INFERRED` -- produced by model interpretation.
  Not a fact and not a legal determination.
* :attr:`AssertionBasis.REGULATORY_ANALYSIS` -- reserved for a future
  authoritative regulatory stage. Nothing in this step may emit it.

A :class:`PotentialIssue` is inherently interpretive, so its basis is
constrained to the two interpretive values. It is a *review item for a human
VSO*, never a finding of error. :class:`AnalysisResult` also carries
:attr:`AnalysisResult.regulatory_analysis_status`, which stays
``NOT_PERFORMED`` until such a stage exists -- so a consumer can always tell
that no regulatory determination backs the content.

Privacy contract
----------------
These models are not a storage mechanism for veteran information. There is no
field for a name, SSN, address, phone number, email, PII token map, raw
document, credential, prompt, or provider response, and every model sets
``extra="forbid"`` so none can be attached. The single free-text evidence
field is :attr:`EvidenceReference.redacted_excerpt`, which is length-capped
and actively rejects SSN-, email-, and phone-shaped content; only
already-tokenized text belongs there. Nothing in this module logs.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .jobs import utc_now

#: Version of this contract, published on every result so consumers can adapt.
ANALYSIS_SCHEMA_VERSION = "1.0"

#: Fixed text making the non-authoritative nature of an analysis explicit.
ANALYSIS_DISCLAIMER = (
    "Automated analysis for VSO review only. Items are model-identified and "
    "are not legal determinations, adjudications, or predictions of outcome. "
    "A qualified representative must verify every item against the source "
    "record."
)

#: Sentinel the extraction pipeline emits for a field it could not fill.
UNKNOWN_SENTINEL = "unknown"

_MAX_EXCERPT_LENGTH = 500

# Shapes that must never appear in the one free-text evidence field.
_SSN_PATTERN = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")


class AnalysisContractError(ValueError):
    """Raised when input to the adapter does not match the expected shape."""


class AssertionBasis(str, Enum):
    """Where an assertion comes from. See the module docstring."""

    DECISION_STATED = "DECISION_STATED"
    EVIDENCE_STATED = "EVIDENCE_STATED"
    MODEL_INFERRED = "MODEL_INFERRED"
    REGULATORY_ANALYSIS = "REGULATORY_ANALYSIS"


#: Bases that represent interpretation rather than a transcribed statement.
INTERPRETIVE_BASES = frozenset(
    {AssertionBasis.MODEL_INFERRED, AssertionBasis.REGULATORY_ANALYSIS}
)

#: Bases that represent something reported as stated in a document.
STATED_BASES = frozenset(
    {AssertionBasis.DECISION_STATED, AssertionBasis.EVIDENCE_STATED}
)


class ConfidenceLevel(str, Enum):
    """How reliable the *analysis* considers its own assertion to be.

    This is emphatically not a probability of legal correctness, entitlement,
    or appeal outcome. ``NOT_ASSESSED`` is the honest value when no upstream
    stage supplied confidence data, and is what the extraction adapter emits.
    """

    NOT_ASSESSED = "NOT_ASSESSED"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class EvidenceStrength(str, Enum):
    """How well the cited evidence supports an assertion.

    A judgement about evidentiary support only. It carries no implication
    about how a claim would or should be adjudicated. ``NOT_ASSESSED`` is the
    honest value when no stage has evaluated the evidence.
    """

    NOT_ASSESSED = "NOT_ASSESSED"
    INSUFFICIENT = "INSUFFICIENT"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class EvidenceSourceType(str, Enum):
    """The kind of record an evidence item came from."""

    RATING_DECISION = "RATING_DECISION"
    DECISION_NARRATIVE = "DECISION_NARRATIVE"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    SERVICE_RECORD = "SERVICE_RECORD"
    EXAMINATION_REPORT = "EXAMINATION_REPORT"
    LAY_STATEMENT = "LAY_STATEMENT"
    CLIENT_NARRATIVE = "CLIENT_NARRATIVE"
    UNSPECIFIED = "UNSPECIFIED"


class EvidenceRelationship(str, Enum):
    """How an evidence item relates to the assertion citing it."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTEXTUAL = "CONTEXTUAL"
    INSUFFICIENT = "INSUFFICIENT"


class DecisionOutcome(str, Enum):
    """What the decision did with a condition, as reported by extraction."""

    GRANTED = "GRANTED"
    DENIED = "DENIED"
    CONTINUED = "CONTINUED"
    INCREASED = "INCREASED"
    DECREASED = "DECREASED"
    DEFERRED = "DEFERRED"
    NOT_STATED = "NOT_STATED"


class FindingType(str, Enum):
    """The kind of observation a finding records."""

    RATING_ASSIGNED = "RATING_ASSIGNED"
    EFFECTIVE_DATE_ASSIGNED = "EFFECTIVE_DATE_ASSIGNED"
    DIAGNOSTIC_CODE_ASSIGNED = "DIAGNOSTIC_CODE_ASSIGNED"
    REGULATION_CITED = "REGULATION_CITED"
    CONDITION_ADDRESSED = "CONDITION_ADDRESSED"
    EVIDENCE_CONSIDERED = "EVIDENCE_CONSIDERED"
    OTHER = "OTHER"


class PotentialIssueType(str, Enum):
    """The category of a review item.

    Every member is phrased as an observation or an open question. None
    asserts that the VA erred; that determination is out of scope for this
    contract.
    """

    MISSING_DIAGNOSTIC_CODE = "MISSING_DIAGNOSTIC_CODE"
    MISSING_EFFECTIVE_DATE = "MISSING_EFFECTIVE_DATE"
    MISSING_RATING_PERCENTAGE = "MISSING_RATING_PERCENTAGE"
    MISSING_REGULATORY_CITATION = "MISSING_REGULATORY_CITATION"
    UNPARSABLE_EFFECTIVE_DATE = "UNPARSABLE_EFFECTIVE_DATE"
    UNPARSABLE_RATING_PERCENTAGE = "UNPARSABLE_RATING_PERCENTAGE"
    UNIDENTIFIED_CONDITION = "UNIDENTIFIED_CONDITION"
    POSSIBLE_UNADDRESSED_CONDITION = "POSSIBLE_UNADDRESSED_CONDITION"
    POSSIBLE_EARLIER_EFFECTIVE_DATE = "POSSIBLE_EARLIER_EFFECTIVE_DATE"
    POSSIBLE_RATING_INCONSISTENCY = "POSSIBLE_RATING_INCONSISTENCY"
    INCOMPLETE_EXTRACTION = "INCOMPLETE_EXTRACTION"
    OTHER = "OTHER"


class ReviewReason(str, Enum):
    """Why a human needs to look at a review item."""

    INCOMPLETE_INFORMATION = "INCOMPLETE_INFORMATION"
    INTERNAL_INCONSISTENCY = "INTERNAL_INCONSISTENCY"
    UNVERIFIED_EXTRACTION = "UNVERIFIED_EXTRACTION"
    REQUIRES_REGULATORY_ANALYSIS = "REQUIRES_REGULATORY_ANALYSIS"
    REQUIRES_ADDITIONAL_EVIDENCE = "REQUIRES_ADDITIONAL_EVIDENCE"
    REQUIRES_HUMAN_JUDGEMENT = "REQUIRES_HUMAN_JUDGEMENT"


class RegulatoryAnalysisStatus(str, Enum):
    """Whether an authoritative regulatory stage has run.

    Stays ``NOT_PERFORMED`` throughout this step, which is how a consumer
    knows that nothing in the result is backed by a regulatory determination.
    """

    NOT_PERFORMED = "NOT_PERFORMED"
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"


class _StrictModel(BaseModel):
    """Base for every analysis model: closed, validated on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConfidenceAssessment(_StrictModel):
    """A controlled expression of analytical confidence.

    ``score`` is optional and constrained to the inclusive range ``0.0``-``1.0``,
    where ``0.0`` means "no confidence in this assertion" and ``1.0`` means
    "no identified reason to doubt this assertion". It describes reliability of
    the analysis step that produced the assertion -- typically how cleanly a
    value was read out of a document.

    It explicitly does **not** express:

    * the probability that a claim is legally correct,
    * the probability that an appeal or supplemental claim would succeed,
    * entitlement to any benefit or evaluation.

    Leave ``score`` as ``None`` and ``level`` as ``NOT_ASSESSED`` when no stage
    has actually measured confidence. Inventing a number is worse than
    admitting absence.
    """

    level: ConfidenceLevel = ConfidenceLevel.NOT_ASSESSED
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional 0.0-1.0 reliability of the assertion; never a legal or outcome probability.",
    )
    rationale: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _require_level_when_scored(self) -> "ConfidenceAssessment":
        """A numeric score must be accompanied by a qualitative level."""
        if self.score is not None and self.level is ConfidenceLevel.NOT_ASSESSED:
            raise ValueError(
                "level must not be NOT_ASSESSED when a numeric score is supplied"
            )
        return self

    @classmethod
    def not_assessed(cls, rationale: Optional[str] = None) -> "ConfidenceAssessment":
        """The honest assessment when nothing measured confidence."""
        return cls(level=ConfidenceLevel.NOT_ASSESSED, score=None, rationale=rationale)


class EvidenceReference(_StrictModel):
    """A pointer to where evidence came from, not the evidence itself.

    Provenance is carried as a source *type* plus short, human-meaningful
    labels and locators, so an analysis can cite a document without any
    persistent document store existing. ``redacted_excerpt`` is the only place
    document text may appear, and only text that has already been through PII
    tokenization belongs there.
    """

    evidence_id: str = Field(min_length=1, max_length=64)
    source_type: EvidenceSourceType
    source_label: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Short human-facing descriptor, e.g. 'Rating decision, Reasons for Decision'.",
    )
    source_reference: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Locator within the source, e.g. 'page 3' or 'section 2'. Not a storage handle.",
    )
    redacted_excerpt: Optional[str] = Field(
        default=None,
        max_length=_MAX_EXCERPT_LENGTH,
        description="Optional PII-tokenized quotation. Never raw document text.",
    )
    relationship: EvidenceRelationship = EvidenceRelationship.CONTEXTUAL
    strength: EvidenceStrength = EvidenceStrength.NOT_ASSESSED
    basis: AssertionBasis = AssertionBasis.DECISION_STATED

    @field_validator("redacted_excerpt")
    @classmethod
    def _reject_raw_pii(cls, value: Optional[str]) -> Optional[str]:
        """Refuse excerpts that still look like they contain direct identifiers.

        A backstop, not a substitute for upstream tokenization: it catches the
        obvious shapes rather than every possible identifier.
        """
        if value is None:
            return None
        if _SSN_PATTERN.search(value):
            raise ValueError(
                "redacted_excerpt appears to contain an SSN; supply PII-tokenized text only"
            )
        if _EMAIL_PATTERN.search(value):
            raise ValueError(
                "redacted_excerpt appears to contain an email address; supply PII-tokenized text only"
            )
        if _PHONE_PATTERN.search(value):
            raise ValueError(
                "redacted_excerpt appears to contain a phone number; supply PII-tokenized text only"
            )
        return value


class ConditionAnalysis(_StrictModel):
    """A condition or issue the decision addressed, as reported by extraction.

    Fields the source did not state are ``None`` or empty rather than guessed.
    """

    condition_id: str = Field(min_length=1, max_length=64)
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Clinical/issue name, e.g. 'tinnitus'. Not a person's name.",
    )
    diagnostic_code: Optional[str] = Field(default=None, max_length=32)
    rating_percentage: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Assigned evaluation as a whole percent, when stated.",
    )
    effective_date: Optional[date] = None
    cfr_citations: List[str] = Field(default_factory=list)
    decision_outcome: DecisionOutcome = DecisionOutcome.NOT_STATED
    basis: AssertionBasis = AssertionBasis.DECISION_STATED
    evidence_ids: List[str] = Field(default_factory=list)


class Finding(_StrictModel):
    """A single observation the analysis records.

    ``basis`` keeps a transcribed statement distinguishable from an
    interpretation, and ``statement`` is a neutral description -- never an
    accusation or a conclusion of law.
    """

    finding_id: str = Field(min_length=1, max_length=64)
    finding_type: FindingType
    statement: str = Field(min_length=1, max_length=1000)
    basis: AssertionBasis
    condition_id: Optional[str] = Field(default=None, max_length=64)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: ConfidenceAssessment = Field(default_factory=ConfidenceAssessment)
    evidence_strength: EvidenceStrength = EvidenceStrength.NOT_ASSESSED


class PotentialIssue(_StrictModel):
    """A model-identified item for human VSO review.

    Deliberately framed as an open question. ``requires_review`` is pinned to
    ``True`` and ``basis`` is restricted to the interpretive values, so this
    model is structurally incapable of asserting an established error. Naming
    something a legal error would require an authoritative regulatory stage
    that does not exist yet.
    """

    issue_id: str = Field(min_length=1, max_length=64)
    issue_type: PotentialIssueType
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The open question for review, phrased as a question or observation.",
    )
    review_reason: ReviewReason
    basis: AssertionBasis = AssertionBasis.MODEL_INFERRED
    related_condition_id: Optional[str] = Field(default=None, max_length=64)
    related_finding_ids: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    confidence: ConfidenceAssessment = Field(default_factory=ConfidenceAssessment)
    evidence_strength: EvidenceStrength = EvidenceStrength.NOT_ASSESSED
    requires_review: bool = Field(
        default=True,
        description="Always True: a potential issue is a review item, never a conclusion.",
    )

    @field_validator("basis")
    @classmethod
    def _basis_must_be_interpretive(cls, value: AssertionBasis) -> AssertionBasis:
        if value not in INTERPRETIVE_BASES:
            raise ValueError(
                "A potential issue is interpretive; basis must be MODEL_INFERRED "
                "or REGULATORY_ANALYSIS, not a stated-fact basis"
            )
        return value

    @field_validator("requires_review")
    @classmethod
    def _must_require_review(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("requires_review cannot be disabled on a potential issue")
        return value


class AnalysisResult(_StrictModel):
    """The complete structured analysis for one job.

    Cross-references are validated: every ``evidence_id`` cited by a
    condition, finding, or issue must exist in :attr:`evidence`, and every
    referenced condition/finding must exist too. That keeps a result
    self-consistent without any database.
    """

    schema_version: str = ANALYSIS_SCHEMA_VERSION
    job_id: str = Field(
        min_length=1, max_length=64, description="The AnalysisJob.id this describes."
    )
    generated_at: datetime
    conditions: List[ConditionAnalysis] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    evidence: List[EvidenceReference] = Field(default_factory=list)
    potential_issues: List[PotentialIssue] = Field(default_factory=list)
    overall_confidence: ConfidenceAssessment = Field(
        default_factory=ConfidenceAssessment
    )
    overall_evidence_strength: EvidenceStrength = EvidenceStrength.NOT_ASSESSED
    regulatory_analysis_status: RegulatoryAnalysisStatus = (
        RegulatoryAnalysisStatus.NOT_PERFORMED
    )
    disclaimer: str = ANALYSIS_DISCLAIMER

    @field_validator("generated_at")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("generated_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "AnalysisResult":
        evidence_ids = {item.evidence_id for item in self.evidence}
        condition_ids = {item.condition_id for item in self.conditions}
        finding_ids = {item.finding_id for item in self.findings}

        def _check(referenced: Sequence[str], known: set, label: str) -> None:
            missing = sorted(set(referenced) - known)
            if missing:
                raise ValueError(f"unknown {label}: {', '.join(missing)}")

        for condition in self.conditions:
            _check(condition.evidence_ids, evidence_ids, "evidence_id")
        for finding in self.findings:
            _check(finding.evidence_ids, evidence_ids, "evidence_id")
            if finding.condition_id is not None:
                _check([finding.condition_id], condition_ids, "condition_id")
        for issue in self.potential_issues:
            _check(issue.supporting_evidence_ids, evidence_ids, "evidence_id")
            _check(issue.related_finding_ids, finding_ids, "finding_id")
            if issue.related_condition_id is not None:
                _check([issue.related_condition_id], condition_ids, "condition_id")

        for ids, label in (
            ([c.condition_id for c in self.conditions], "condition_id"),
            ([f.finding_id for f in self.findings], "finding_id"),
            ([e.evidence_id for e in self.evidence], "evidence_id"),
            ([i.issue_id for i in self.potential_issues], "issue_id"),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} values are not allowed")

        return self

    @property
    def stated_findings(self) -> List[Finding]:
        """Findings transcribed from a document rather than inferred."""
        return [f for f in self.findings if f.basis in STATED_BASES]

    @property
    def inferred_findings(self) -> List[Finding]:
        """Findings produced by interpretation."""
        return [f for f in self.findings if f.basis in INTERPRETIVE_BASES]


# ---------------------------------------------------------------------------
# Adapter from the existing extraction contract
# ---------------------------------------------------------------------------
#
# The extraction pipeline in `main` produces ExtractedData: a list of
# conditions whose five fields are all strings, using the literal "unknown"
# for anything the model could not fill. We accept that shape *structurally*
# (a Pydantic model, an object with attributes, or a plain mapping) rather
# than importing `main`, because importing `main` would drag Google Cloud
# credential validation into this module. The adapter adds no information: it
# re-types what is present, marks what is absent, and raises a review item
# wherever something is missing or unreadable.

_EXTRACTION_FIELDS = (
    "condition",
    "diagnostic_code",
    "effective_date",
    "cfr_citation",
    "percentage",
)

_PERCENTAGE_PATTERN = re.compile(r"^(\d{1,3})\s*(?:%|percent)?$", re.IGNORECASE)


def _is_unknown(raw: Any) -> bool:
    """True when the extraction did not supply a usable value."""
    if raw is None:
        return True
    if not isinstance(raw, str):
        return False
    return raw.strip().lower() in {"", UNKNOWN_SENTINEL, "n/a", "none", "null"}


def _clean(raw: Any) -> Optional[str]:
    """Normalise an extraction field to a trimmed string, or None if unknown."""
    if _is_unknown(raw):
        return None
    return str(raw).strip()


def _read_field(source: Any, name: str) -> Any:
    """Read a field from a Pydantic model, plain object, or mapping."""
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _has_field(source: Any, name: str) -> bool:
    """True when the field exists on a mapping or object.

    Absence is an error rather than an implicit "unknown": silently treating a
    malformed payload as a set of missing values would manufacture misleading
    review items.
    """
    if isinstance(source, Mapping):
        return name in source
    return hasattr(source, name)


def _iter_extracted_conditions(extraction: Any) -> List[Any]:
    """Pull the condition list out of an extraction result of any supported shape."""
    conditions = _read_field(extraction, "conditions")
    if conditions is None:
        raise AnalysisContractError(
            "extraction must expose a 'conditions' sequence"
        )
    if isinstance(conditions, (str, bytes, Mapping)):
        raise AnalysisContractError("'conditions' must be a sequence of conditions")
    return list(conditions)


def _parse_percentage(raw: Optional[str]) -> Tuple[Optional[int], bool]:
    """Return ``(percentage, unparsable)``.

    ``"10%"``, ``"10"``, and ``"10 percent"`` all yield ``10``. Anything else
    present-but-unreadable yields ``(None, True)`` so the caller can raise a
    review item instead of guessing.
    """
    if raw is None:
        return None, False
    match = _PERCENTAGE_PATTERN.match(raw.strip())
    if not match:
        return None, True
    value = int(match.group(1))
    if not 0 <= value <= 100:
        return None, True
    return value, False


def _parse_effective_date(raw: Optional[str]) -> Tuple[Optional[date], bool]:
    """Return ``(date, unparsable)`` for an ISO ``YYYY-MM-DD`` value.

    The extraction prompt requests ISO dates. Any other format is reported as
    unparsable rather than being guessed at, since misreading a VA effective
    date is consequential.
    """
    if raw is None:
        return None, False
    try:
        return date.fromisoformat(raw.strip()), False
    except ValueError:
        return None, True


class _IssueCollector:
    """Accumulates review items with stable, sequential identifiers."""

    def __init__(self) -> None:
        self.issues: List[PotentialIssue] = []

    def add(
        self,
        *,
        issue_type: PotentialIssueType,
        question: str,
        review_reason: ReviewReason,
        condition_id: Optional[str],
        finding_ids: Optional[List[str]] = None,
    ) -> None:
        self.issues.append(
            PotentialIssue(
                issue_id=f"issue-{len(self.issues) + 1}",
                issue_type=issue_type,
                question=question,
                review_reason=review_reason,
                basis=AssertionBasis.MODEL_INFERRED,
                related_condition_id=condition_id,
                related_finding_ids=finding_ids or [],
                supporting_evidence_ids=[],
                confidence=ConfidenceAssessment.not_assessed(
                    "Derived structurally from the extraction; not independently assessed."
                ),
                evidence_strength=EvidenceStrength.NOT_ASSESSED,
            )
        )


def build_analysis_from_extraction(
    extraction: Any,
    *,
    job_id: str,
    generated_at: Optional[datetime] = None,
) -> AnalysisResult:
    """Re-type an existing extraction result as a structured analysis.

    Deterministic and side-effect free: no model calls, no regulatory lookup,
    no randomness. Identifiers are sequential so the same input always yields
    the same output. ``generated_at`` is injectable for the same reason.

    What it does *not* do is invent data. A field the extraction reported as
    ``"unknown"`` becomes ``None``, and a field that is present but unreadable
    also becomes ``None`` -- in both cases accompanied by a
    :class:`PotentialIssue` so the gap is visible to a reviewer rather than
    silently swallowed. Confidence and evidence strength stay
    ``NOT_ASSESSED`` throughout, because the extraction contract supplies
    neither, and :attr:`AnalysisResult.regulatory_analysis_status` stays
    ``NOT_PERFORMED``.

    :param extraction: an object or mapping exposing ``conditions``, each with
        the five extraction fields. Accepted structurally so this module need
        not import the credential-bearing ``main`` module.
    :param job_id: the :class:`~claim_backend.jobs.AnalysisJob` id this
        analysis describes.
    :raises AnalysisContractError: when ``extraction`` has the wrong shape.
    """
    extracted = _iter_extracted_conditions(extraction)

    # One provenance record for the document the extraction ran against. The
    # excerpt is deliberately omitted: the pipeline keeps no document, and
    # retaining text here would reintroduce the exposure Step 1 removed.
    decision_evidence = EvidenceReference(
        evidence_id="evidence-1",
        source_type=EvidenceSourceType.RATING_DECISION,
        source_label="Submitted VA rating decision",
        source_reference=None,
        redacted_excerpt=None,
        relationship=EvidenceRelationship.CONTEXTUAL,
        strength=EvidenceStrength.NOT_ASSESSED,
        basis=AssertionBasis.DECISION_STATED,
    )

    conditions: List[ConditionAnalysis] = []
    findings: List[Finding] = []
    collector = _IssueCollector()

    for index, raw_condition in enumerate(extracted, start=1):
        for field in _EXTRACTION_FIELDS:
            if not _has_field(raw_condition, field):
                raise AnalysisContractError(
                    f"extracted condition {index} is missing field '{field}'"
                )

        name = _clean(_read_field(raw_condition, "condition"))
        condition_id = f"condition-{index}"

        if name is None:
            # Nothing to anchor a condition on; record the gap and move on
            # rather than fabricating a placeholder name.
            collector.add(
                issue_type=PotentialIssueType.UNIDENTIFIED_CONDITION,
                question=(
                    f"Extraction entry {index} has no identifiable condition name. "
                    "What condition does this entry refer to?"
                ),
                review_reason=ReviewReason.INCOMPLETE_INFORMATION,
                condition_id=None,
            )
            continue

        diagnostic_code = _clean(_read_field(raw_condition, "diagnostic_code"))
        citation = _clean(_read_field(raw_condition, "cfr_citation"))
        percentage, percentage_unparsable = _parse_percentage(
            _clean(_read_field(raw_condition, "percentage"))
        )
        effective_date, date_unparsable = _parse_effective_date(
            _clean(_read_field(raw_condition, "effective_date"))
        )

        conditions.append(
            ConditionAnalysis(
                condition_id=condition_id,
                name=name,
                diagnostic_code=diagnostic_code,
                rating_percentage=percentage,
                effective_date=effective_date,
                cfr_citations=[citation] if citation else [],
                # The extraction contract does not say whether a condition was
                # granted or denied, so this stays NOT_STATED.
                decision_outcome=DecisionOutcome.NOT_STATED,
                basis=AssertionBasis.DECISION_STATED,
                evidence_ids=[decision_evidence.evidence_id],
            )
        )

        # Findings mirror only what extraction actually reported, each marked
        # DECISION_STATED. No interpretation is added here.
        for finding_type, present, statement in (
            (
                FindingType.RATING_ASSIGNED,
                percentage is not None,
                f"Decision reports a {percentage}% evaluation for {name}.",
            ),
            (
                FindingType.DIAGNOSTIC_CODE_ASSIGNED,
                diagnostic_code is not None,
                f"Decision reports diagnostic code {diagnostic_code} for {name}.",
            ),
            (
                FindingType.EFFECTIVE_DATE_ASSIGNED,
                effective_date is not None,
                f"Decision reports an effective date of {effective_date} for {name}.",
            ),
            (
                FindingType.REGULATION_CITED,
                citation is not None,
                f"Decision cites {citation} for {name}.",
            ),
        ):
            if not present:
                continue
            findings.append(
                Finding(
                    finding_id=f"finding-{len(findings) + 1}",
                    finding_type=finding_type,
                    statement=statement,
                    basis=AssertionBasis.DECISION_STATED,
                    condition_id=condition_id,
                    evidence_ids=[decision_evidence.evidence_id],
                    confidence=ConfidenceAssessment.not_assessed(
                        "Transcribed by automated extraction; not independently verified."
                    ),
                    evidence_strength=EvidenceStrength.NOT_ASSESSED,
                )
            )

        for issue_type, question, reason, triggered in (
            (
                PotentialIssueType.MISSING_RATING_PERCENTAGE,
                f"No evaluation percentage was identified for {name}. "
                "Does the decision assign one?",
                ReviewReason.INCOMPLETE_INFORMATION,
                percentage is None and not percentage_unparsable,
            ),
            (
                PotentialIssueType.UNPARSABLE_RATING_PERCENTAGE,
                f"The evaluation percentage reported for {name} could not be "
                "read as a whole percent. What evaluation was assigned?",
                ReviewReason.UNVERIFIED_EXTRACTION,
                percentage_unparsable,
            ),
            (
                PotentialIssueType.MISSING_EFFECTIVE_DATE,
                f"No effective date was identified for {name}. "
                "Does the decision assign one?",
                ReviewReason.INCOMPLETE_INFORMATION,
                effective_date is None and not date_unparsable,
            ),
            (
                PotentialIssueType.UNPARSABLE_EFFECTIVE_DATE,
                f"The effective date reported for {name} could not be read as "
                "a calendar date. What effective date was assigned?",
                ReviewReason.UNVERIFIED_EXTRACTION,
                date_unparsable,
            ),
            (
                PotentialIssueType.MISSING_DIAGNOSTIC_CODE,
                f"No diagnostic code was identified for {name}. "
                "Which code did the decision apply?",
                ReviewReason.INCOMPLETE_INFORMATION,
                diagnostic_code is None,
            ),
            (
                PotentialIssueType.MISSING_REGULATORY_CITATION,
                f"No regulatory citation was identified for {name}. "
                "Which authority did the decision rely on?",
                ReviewReason.REQUIRES_REGULATORY_ANALYSIS,
                citation is None,
            ),
        ):
            if triggered:
                collector.add(
                    issue_type=issue_type,
                    question=question,
                    review_reason=reason,
                    condition_id=condition_id,
                )

    if not extracted:
        collector.add(
            issue_type=PotentialIssueType.INCOMPLETE_EXTRACTION,
            question=(
                "No conditions were identified in the submitted document. "
                "Does it contain rating decision content?"
            ),
            review_reason=ReviewReason.INCOMPLETE_INFORMATION,
            condition_id=None,
        )

    return AnalysisResult(
        job_id=job_id,
        generated_at=generated_at or utc_now(),
        conditions=conditions,
        findings=findings,
        evidence=[decision_evidence],
        potential_issues=collector.issues,
        # The extraction contract carries no confidence or evidence grading,
        # so neither is asserted here.
        overall_confidence=ConfidenceAssessment.not_assessed(
            "Automated extraction does not report confidence."
        ),
        overall_evidence_strength=EvidenceStrength.NOT_ASSESSED,
        regulatory_analysis_status=RegulatoryAnalysisStatus.NOT_PERFORMED,
    )
