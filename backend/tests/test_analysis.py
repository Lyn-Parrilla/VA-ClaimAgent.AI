"""Focused tests for the structured claim analysis contract.

Exercises claim_backend.analysis directly. The module imports no
credential-bearing code, so these run without Google Cloud configuration --
except the one test that deliberately feeds the *real* ExtractedData model
through the adapter to prove production compatibility.
"""

import re
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import BaseModel, ValidationError

from claim_backend.analysis import (
    ANALYSIS_DISCLAIMER,
    ANALYSIS_SCHEMA_VERSION,
    INTERPRETIVE_BASES,
    STATED_BASES,
    AnalysisContractError,
    AnalysisResult,
    AssertionBasis,
    ConditionAnalysis,
    ConfidenceAssessment,
    ConfidenceLevel,
    DecisionOutcome,
    EvidenceReference,
    EvidenceRelationship,
    EvidenceSourceType,
    EvidenceStrength,
    Finding,
    FindingType,
    PotentialIssue,
    PotentialIssueType,
    RegulatoryAnalysisStatus,
    ReviewReason,
    build_analysis_from_extraction,
)

FIXED_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Sensitive material that must have no representable home in the contract.
VETERAN_NAME = "John Doe"
SSN = "123-45-6789"
PII_MAP = {"<PERSON_1>": VETERAN_NAME, "<SSN_1>": SSN}

# Every model in the contract, for sweep-style security tests.
ALL_MODELS = [
    AnalysisResult,
    ConditionAnalysis,
    Finding,
    PotentialIssue,
    EvidenceReference,
    ConfidenceAssessment,
]


def _evidence(evidence_id: str = "evidence-1") -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.RATING_DECISION,
        source_label="Submitted VA rating decision",
    )


def _condition(condition_id: str = "condition-1") -> ConditionAnalysis:
    return ConditionAnalysis(
        condition_id=condition_id,
        name="tinnitus",
        diagnostic_code="6260",
        rating_percentage=10,
        effective_date=date(2023, 8, 14),
        cfr_citations=["38 CFR 4.87"],
        decision_outcome=DecisionOutcome.GRANTED,
        evidence_ids=["evidence-1"],
    )


def _finding(finding_id: str = "finding-1") -> Finding:
    return Finding(
        finding_id=finding_id,
        finding_type=FindingType.RATING_ASSIGNED,
        statement="Decision reports a 10% evaluation for tinnitus.",
        basis=AssertionBasis.DECISION_STATED,
        condition_id="condition-1",
        evidence_ids=["evidence-1"],
    )


def _issue(issue_id: str = "issue-1") -> PotentialIssue:
    return PotentialIssue(
        issue_id=issue_id,
        issue_type=PotentialIssueType.POSSIBLE_EARLIER_EFFECTIVE_DATE,
        question="Could an earlier effective date be supported? Requires review.",
        review_reason=ReviewReason.REQUIRES_HUMAN_JUDGEMENT,
        related_condition_id="condition-1",
        related_finding_ids=["finding-1"],
        supporting_evidence_ids=["evidence-1"],
    )


def _result(**overrides) -> AnalysisResult:
    payload = dict(
        job_id="job-1",
        generated_at=FIXED_TIME,
        conditions=[_condition()],
        findings=[_finding()],
        evidence=[_evidence()],
        potential_issues=[_issue()],
    )
    payload.update(overrides)
    return AnalysisResult(**payload)


# --- A. AnalysisResult construction ------------------------------------------


def test_analysis_result_constructs():
    result = _result()

    assert result.job_id == "job-1"
    assert result.schema_version == ANALYSIS_SCHEMA_VERSION
    assert result.generated_at == FIXED_TIME
    assert result.disclaimer == ANALYSIS_DISCLAIMER


def test_analysis_result_defaults_are_conservative():
    result = AnalysisResult(job_id="job-1", generated_at=FIXED_TIME)

    assert result.conditions == []
    assert result.findings == []
    assert result.overall_confidence.level is ConfidenceLevel.NOT_ASSESSED
    assert result.overall_confidence.score is None
    assert result.overall_evidence_strength is EvidenceStrength.NOT_ASSESSED
    assert result.regulatory_analysis_status is RegulatoryAnalysisStatus.NOT_PERFORMED


def test_generated_at_must_be_timezone_aware():
    with pytest.raises(ValidationError, match="timezone-aware"):
        AnalysisResult(job_id="job-1", generated_at=datetime(2026, 1, 1, 12, 0, 0))


def test_cross_references_must_resolve():
    with pytest.raises(ValidationError, match="unknown evidence_id"):
        _result(evidence=[])

    with pytest.raises(ValidationError, match="unknown condition_id"):
        AnalysisResult(
            job_id="job-1",
            generated_at=FIXED_TIME,
            evidence=[_evidence()],
            findings=[_finding()],
        )


def test_duplicate_identifiers_are_rejected():
    with pytest.raises(ValidationError, match="duplicate finding_id"):
        _result(findings=[_finding("finding-1"), _finding("finding-1")])


# --- B. nested findings ------------------------------------------------------


def test_findings_nest_inside_the_result():
    result = _result()

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert isinstance(finding, Finding)
    assert finding.finding_type is FindingType.RATING_ASSIGNED
    assert finding.condition_id == "condition-1"
    assert finding.evidence_ids == ["evidence-1"]


def test_finding_requires_an_explicit_basis():
    with pytest.raises(ValidationError):
        Finding(
            finding_id="finding-1",
            finding_type=FindingType.RATING_ASSIGNED,
            statement="x",
        )


# --- C. evidence items -------------------------------------------------------


def test_evidence_records_provenance_without_the_document():
    item = EvidenceReference(
        evidence_id="evidence-1",
        source_type=EvidenceSourceType.MEDICAL_RECORD,
        source_label="VA examination report",
        source_reference="page 3",
        relationship=EvidenceRelationship.SUPPORTS,
        strength=EvidenceStrength.MODERATE,
    )

    assert item.source_type is EvidenceSourceType.MEDICAL_RECORD
    assert item.source_reference == "page 3"
    assert item.redacted_excerpt is None


def test_tokenized_excerpt_is_allowed():
    item = EvidenceReference(
        evidence_id="evidence-1",
        source_type=EvidenceSourceType.RATING_DECISION,
        redacted_excerpt="<PERSON_1> is granted service connection for tinnitus.",
    )

    assert "<PERSON_1>" in item.redacted_excerpt


@pytest.mark.parametrize(
    "raw_excerpt",
    [
        f"Veteran {VETERAN_NAME} SSN {SSN}",
        "SSN 123456789 on file",
        "contact veteran@example.com for records",
        "reachable at 415-555-1234",
    ],
)
def test_excerpt_rejects_raw_direct_identifiers(raw_excerpt):
    """A backstop against untokenized text reaching the one free-text field."""
    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="evidence-1",
            source_type=EvidenceSourceType.RATING_DECISION,
            redacted_excerpt=raw_excerpt,
        )


def test_excerpt_is_length_capped():
    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="evidence-1",
            source_type=EvidenceSourceType.RATING_DECISION,
            redacted_excerpt="a" * 501,
        )


# --- D. potential issue representation ---------------------------------------


def test_potential_issue_links_question_evidence_and_finding():
    issue = _issue()

    assert issue.question.endswith("Requires review.")
    assert issue.review_reason is ReviewReason.REQUIRES_HUMAN_JUDGEMENT
    assert issue.related_finding_ids == ["finding-1"]
    assert issue.supporting_evidence_ids == ["evidence-1"]
    assert issue.related_condition_id == "condition-1"
    assert issue.confidence.level is ConfidenceLevel.NOT_ASSESSED
    assert issue.evidence_strength is EvidenceStrength.NOT_ASSESSED


def test_potential_issue_always_requires_review():
    assert _issue().requires_review is True

    with pytest.raises(ValidationError, match="cannot be disabled"):
        PotentialIssue(
            issue_id="issue-1",
            issue_type=PotentialIssueType.OTHER,
            question="q",
            review_reason=ReviewReason.REQUIRES_HUMAN_JUDGEMENT,
            requires_review=False,
        )


@pytest.mark.parametrize(
    "stated_basis", [AssertionBasis.DECISION_STATED, AssertionBasis.EVIDENCE_STATED]
)
def test_potential_issue_cannot_claim_to_be_a_stated_fact(stated_basis):
    """An interpretive review item must never masquerade as a transcribed fact."""
    with pytest.raises(ValidationError, match="interpretive"):
        PotentialIssue(
            issue_id="issue-1",
            issue_type=PotentialIssueType.POSSIBLE_RATING_INCONSISTENCY,
            question="q",
            review_reason=ReviewReason.INTERNAL_INCONSISTENCY,
            basis=stated_basis,
        )


def test_potential_issue_defaults_to_model_inferred():
    assert _issue().basis is AssertionBasis.MODEL_INFERRED


def test_issue_type_vocabulary_makes_no_legal_determination():
    """No member may assert that an error or violation occurred."""
    forbidden = ("ERROR", "VIOLATION", "ILLEGAL", "UNLAWFUL", "ENTITLED", "OWED")
    for member in PotentialIssueType:
        assert not any(word in member.value for word in forbidden), member.value


# --- E. controlled enum values -----------------------------------------------


def test_enum_vocabularies_are_closed():
    assert {b.value for b in AssertionBasis} == {
        "DECISION_STATED",
        "EVIDENCE_STATED",
        "MODEL_INFERRED",
        "REGULATORY_ANALYSIS",
    }
    assert {c.value for c in ConfidenceLevel} == {
        "NOT_ASSESSED",
        "LOW",
        "MODERATE",
        "HIGH",
    }
    assert {s.value for s in EvidenceStrength} == {
        "NOT_ASSESSED",
        "INSUFFICIENT",
        "WEAK",
        "MODERATE",
        "STRONG",
    }
    assert INTERPRETIVE_BASES.isdisjoint(STATED_BASES)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("source_type", "MADE_UP_SOURCE"),
        ("relationship", "MAYBE"),
        ("strength", "VERY_STRONG"),
        ("basis", "GUESSED"),
    ],
)
def test_evidence_rejects_off_vocabulary_values(field, bad_value):
    payload = {
        "evidence_id": "evidence-1",
        "source_type": EvidenceSourceType.RATING_DECISION,
        field: bad_value,
    }
    with pytest.raises(ValidationError):
        EvidenceReference(**payload)


def test_decision_outcome_defaults_to_not_stated():
    condition = ConditionAnalysis(condition_id="condition-1", name="tinnitus")
    assert condition.decision_outcome is DecisionOutcome.NOT_STATED


# --- F. confidence validation ------------------------------------------------


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_confidence_accepts_scores_in_range(score):
    assessment = ConfidenceAssessment(level=ConfidenceLevel.MODERATE, score=score)
    assert assessment.score == score


@pytest.mark.parametrize("score", [-0.01, 1.01, 2.0, -5])
def test_confidence_rejects_scores_out_of_range(score):
    with pytest.raises(ValidationError):
        ConfidenceAssessment(level=ConfidenceLevel.MODERATE, score=score)


def test_confidence_score_requires_a_qualitative_level():
    with pytest.raises(ValidationError, match="NOT_ASSESSED"):
        ConfidenceAssessment(level=ConfidenceLevel.NOT_ASSESSED, score=0.9)


def test_not_assessed_helper_reports_absence():
    assessment = ConfidenceAssessment.not_assessed("no upstream signal")

    assert assessment.level is ConfidenceLevel.NOT_ASSESSED
    assert assessment.score is None
    assert assessment.rationale == "no upstream signal"


def test_confidence_semantics_are_documented():
    """The range and the 'not a legal probability' caveat must be in the docs."""
    doc = ConfidenceAssessment.__doc__
    assert "0.0" in doc and "1.0" in doc
    assert "appeal" in doc.lower()
    assert "legally correct" in doc.lower()


def test_evidence_strength_semantics_are_documented():
    doc = EvidenceStrength.__doc__
    assert "adjudicated" in doc.lower()
    assert "NOT_ASSESSED" in doc


# --- G. evidence-strength validation -----------------------------------------


@pytest.mark.parametrize("strength", list(EvidenceStrength))
def test_all_evidence_strengths_are_usable(strength):
    finding = Finding(
        finding_id="finding-1",
        finding_type=FindingType.EVIDENCE_CONSIDERED,
        statement="s",
        basis=AssertionBasis.EVIDENCE_STATED,
        evidence_strength=strength,
    )
    assert finding.evidence_strength is strength


def test_evidence_strength_rejects_numeric_grades():
    with pytest.raises(ValidationError):
        Finding(
            finding_id="finding-1",
            finding_type=FindingType.EVIDENCE_CONSIDERED,
            statement="s",
            basis=AssertionBasis.EVIDENCE_STATED,
            evidence_strength=0.9,
        )


# --- H. invalid values rejected ----------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"condition_id": "", "name": "tinnitus"},
        {"condition_id": "condition-1", "name": ""},
        {"condition_id": "condition-1", "name": "tinnitus", "rating_percentage": 101},
        {"condition_id": "condition-1", "name": "tinnitus", "rating_percentage": -1},
        {"condition_id": "condition-1", "name": "tinnitus", "effective_date": "not-a-date"},
        {"condition_id": "condition-1", "name": "x" * 201},
    ],
)
def test_condition_rejects_invalid_values(payload):
    with pytest.raises(ValidationError):
        ConditionAnalysis(**payload)


def test_assignment_is_validated():
    condition = _condition()

    with pytest.raises(ValidationError):
        condition.rating_percentage = 150


# --- I, J, K, L, M. nothing sensitive is representable ------------------------


SENSITIVE_FIELD_NAMES = [
    # J. direct identifiers
    "veteran_name",
    "name_of_veteran",
    "ssn",
    "social_security_number",
    "address",
    "street_address",
    "phone",
    "phone_number",
    "email",
    "date_of_birth",
    "file_number",
    # K. PII mappings
    "pii_map",
    "entity_map",
    "pii_mapping",
    "token_map",
    # L. credentials
    "credentials",
    "api_key",
    "access_token",
    "service_account",
    "secret",
    # M. provider/model payloads
    "raw_response",
    "model_response",
    "provider_response",
    "completion",
    "prompt",
    "system_prompt",
    "messages",
    # arbitrary blobs
    "metadata",
    "extra",
    "context",
    "raw_document",
    "document_text",
]


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_models_forbid_extra_fields(model):
    assert model.model_config.get("extra") == "forbid"


@pytest.mark.parametrize("field_name", SENSITIVE_FIELD_NAMES)
def test_sensitive_fields_are_not_in_any_model_contract(field_name):
    """No model declares a home for identifiers, mappings, or credentials."""
    for model in ALL_MODELS:
        assert field_name not in model.model_fields, f"{model.__name__}.{field_name}"


@pytest.mark.parametrize("field_name", SENSITIVE_FIELD_NAMES)
def test_sensitive_fields_cannot_be_injected_into_a_result(field_name):
    with pytest.raises(ValidationError):
        AnalysisResult(
            job_id="job-1", generated_at=FIXED_TIME, **{field_name: "injected"}
        )


def test_pii_mapping_cannot_be_injected_anywhere():
    for model, base in (
        (AnalysisResult, {"job_id": "job-1", "generated_at": FIXED_TIME}),
        (ConditionAnalysis, {"condition_id": "condition-1", "name": "tinnitus"}),
        (
            EvidenceReference,
            {
                "evidence_id": "evidence-1",
                "source_type": EvidenceSourceType.RATING_DECISION,
            },
        ),
    ):
        with pytest.raises(ValidationError):
            model(**base, pii_map=PII_MAP)


def test_sensitive_fields_cannot_be_set_by_assignment():
    result = _result()

    for field_name in ("veteran_name", "pii_map", "api_key", "raw_response"):
        with pytest.raises(ValidationError):
            setattr(result, field_name, "injected")


def test_serialized_result_contains_no_sensitive_values():
    dumped = _result().model_dump_json()

    assert VETERAN_NAME not in dumped
    assert SSN not in dumped
    for value in PII_MAP.values():
        assert value not in dumped


def test_module_does_not_import_logging():
    """No logging means no path for an analysis payload to reach telemetry."""
    import claim_backend.analysis as module

    source = open(module.__file__).read()
    assert "import logging" not in source
    assert "logger" not in source
    assert "print(" not in source


def test_validation_errors_can_be_sanitized_before_api_exposure():
    """Step 3 adds no endpoint; if these models are ever exposed, the Step 2
    sanitizer strips the echoed value."""
    from claim_backend.jobs_api import _sanitize_validation_errors

    with pytest.raises(ValidationError) as exc_info:
        AnalysisResult(
            job_id="job-1",
            generated_at=FIXED_TIME,
            veteran_name=f"{VETERAN_NAME} {SSN}",
        )

    raw = exc_info.value.errors()
    assert any(SSN in str(error.get("input", "")) for error in raw)

    sanitized = _sanitize_validation_errors(raw)
    rendered = str(sanitized)
    assert SSN not in rendered
    assert VETERAN_NAME not in rendered
    for error in sanitized:
        assert set(error) <= {"type", "loc", "msg"}


# --- N. adapter from the existing extraction contract ------------------------


FULL_EXTRACTION = {
    "conditions": [
        {
            "condition": "Tinnitus",
            "diagnostic_code": "6260",
            "effective_date": "2023-08-14",
            "cfr_citation": "38 CFR 4.87",
            "percentage": "10%",
        }
    ]
}


def test_adapter_maps_a_complete_extraction():
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert result.job_id == "job-1"
    assert result.generated_at == FIXED_TIME
    condition = result.conditions[0]
    assert condition.name == "Tinnitus"
    assert condition.diagnostic_code == "6260"
    assert condition.rating_percentage == 10
    assert condition.effective_date == date(2023, 8, 14)
    assert condition.cfr_citations == ["38 CFR 4.87"]
    assert result.potential_issues == []


def test_adapter_accepts_the_real_extracted_data_model():
    """Proves compatibility with the production ExtractedData contract."""
    from claim_backend.main import ExtractedCondition, ExtractedData

    extraction = ExtractedData(
        conditions=[
            ExtractedCondition(
                condition="Tinnitus",
                diagnostic_code="6260",
                effective_date="2023-08-14",
                cfr_citation="38 CFR 4.87",
                percentage="10%",
            )
        ]
    )

    result = build_analysis_from_extraction(
        extraction, job_id="job-1", generated_at=FIXED_TIME
    )

    assert result.conditions[0].rating_percentage == 10
    assert result.conditions[0].effective_date == date(2023, 8, 14)


def test_adapter_creates_evidence_provenance_without_document_text():
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.source_type is EvidenceSourceType.RATING_DECISION
    # No document is retained anywhere.
    assert evidence.redacted_excerpt is None


def test_adapter_builds_findings_for_each_reported_value():
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert {f.finding_type for f in result.findings} == {
        FindingType.RATING_ASSIGNED,
        FindingType.DIAGNOSTIC_CODE_ASSIGNED,
        FindingType.EFFECTIVE_DATE_ASSIGNED,
        FindingType.REGULATION_CITED,
    }
    assert all(f.condition_id == "condition-1" for f in result.findings)


def test_adapter_is_deterministic():
    first = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )
    second = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert first.model_dump() == second.model_dump()


def test_adapter_handles_multiple_conditions():
    extraction = {
        "conditions": [
            FULL_EXTRACTION["conditions"][0],
            {
                "condition": "Lumbar strain",
                "diagnostic_code": "5237",
                "effective_date": "2022-01-05",
                "cfr_citation": "38 CFR 4.71a",
                "percentage": "20",
            },
        ]
    }

    result = build_analysis_from_extraction(
        extraction, job_id="job-1", generated_at=FIXED_TIME
    )

    assert [c.condition_id for c in result.conditions] == ["condition-1", "condition-2"]
    assert result.conditions[1].rating_percentage == 20


@pytest.mark.parametrize("bad", [{}, {"conditions": "text"}, {"conditions": {"a": 1}}])
def test_adapter_rejects_malformed_extraction(bad):
    with pytest.raises(AnalysisContractError):
        build_analysis_from_extraction(bad, job_id="job-1")


def test_adapter_rejects_condition_missing_contract_fields():
    with pytest.raises(AnalysisContractError, match="missing field"):
        build_analysis_from_extraction(
            {"conditions": [{"condition": "Tinnitus"}]}, job_id="job-1"
        )


# --- O. the adapter invents nothing ------------------------------------------


UNKNOWN_EXTRACTION = {
    "conditions": [
        {
            "condition": "Knee strain",
            "diagnostic_code": "unknown",
            "effective_date": "unknown",
            "cfr_citation": "unknown",
            "percentage": "unknown",
        }
    ]
}


def test_adapter_represents_unknown_fields_as_absent():
    result = build_analysis_from_extraction(
        UNKNOWN_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    condition = result.conditions[0]
    assert condition.name == "Knee strain"
    assert condition.diagnostic_code is None
    assert condition.rating_percentage is None
    assert condition.effective_date is None
    assert condition.cfr_citations == []


def test_adapter_flags_each_missing_field_for_review():
    result = build_analysis_from_extraction(
        UNKNOWN_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert {i.issue_type for i in result.potential_issues} == {
        PotentialIssueType.MISSING_RATING_PERCENTAGE,
        PotentialIssueType.MISSING_EFFECTIVE_DATE,
        PotentialIssueType.MISSING_DIAGNOSTIC_CODE,
        PotentialIssueType.MISSING_REGULATORY_CITATION,
    }


def test_adapter_builds_no_findings_for_absent_values():
    result = build_analysis_from_extraction(
        UNKNOWN_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )
    assert result.findings == []


def test_adapter_does_not_guess_a_non_iso_date():
    """A misread VA effective date is consequential, so it is surfaced, not guessed."""
    extraction = {
        "conditions": [
            {
                "condition": "Tinnitus",
                "diagnostic_code": "6260",
                "effective_date": "August 14, 2023",
                "cfr_citation": "38 CFR 4.87",
                "percentage": "10%",
            }
        ]
    }

    result = build_analysis_from_extraction(
        extraction, job_id="job-1", generated_at=FIXED_TIME
    )

    assert result.conditions[0].effective_date is None
    assert [i.issue_type for i in result.potential_issues] == [
        PotentialIssueType.UNPARSABLE_EFFECTIVE_DATE
    ]


def test_adapter_does_not_guess_an_unreadable_percentage():
    extraction = {
        "conditions": [
            {
                "condition": "Tinnitus",
                "diagnostic_code": "6260",
                "effective_date": "2023-08-14",
                "cfr_citation": "38 CFR 4.87",
                "percentage": "moderate",
            }
        ]
    }

    result = build_analysis_from_extraction(
        extraction, job_id="job-1", generated_at=FIXED_TIME
    )

    assert result.conditions[0].rating_percentage is None
    assert [i.issue_type for i in result.potential_issues] == [
        PotentialIssueType.UNPARSABLE_RATING_PERCENTAGE
    ]


def test_adapter_does_not_assert_a_decision_outcome():
    """Extraction never says granted/denied, so the adapter must not either."""
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )
    assert result.conditions[0].decision_outcome is DecisionOutcome.NOT_STATED


def test_adapter_asserts_no_confidence_or_evidence_strength():
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert result.overall_confidence.level is ConfidenceLevel.NOT_ASSESSED
    assert result.overall_confidence.score is None
    assert result.overall_evidence_strength is EvidenceStrength.NOT_ASSESSED
    for finding in result.findings:
        assert finding.confidence.score is None
        assert finding.evidence_strength is EvidenceStrength.NOT_ASSESSED


def test_adapter_reports_no_regulatory_analysis():
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )
    assert result.regulatory_analysis_status is RegulatoryAnalysisStatus.NOT_PERFORMED
    assert all(
        i.basis is not AssertionBasis.REGULATORY_ANALYSIS
        for i in result.potential_issues
    )


def test_adapter_flags_an_empty_extraction():
    result = build_analysis_from_extraction(
        {"conditions": []}, job_id="job-1", generated_at=FIXED_TIME
    )

    assert result.conditions == []
    assert [i.issue_type for i in result.potential_issues] == [
        PotentialIssueType.INCOMPLETE_EXTRACTION
    ]


def test_adapter_flags_an_unnamed_condition_without_inventing_one():
    result = build_analysis_from_extraction(
        {
            "conditions": [
                {
                    "condition": "unknown",
                    "diagnostic_code": "6260",
                    "effective_date": "unknown",
                    "cfr_citation": "unknown",
                    "percentage": "unknown",
                }
            ]
        },
        job_id="job-1",
        generated_at=FIXED_TIME,
    )

    assert result.conditions == []
    assert [i.issue_type for i in result.potential_issues] == [
        PotentialIssueType.UNIDENTIFIED_CONDITION
    ]


# --- P. facts stay distinguishable from inference ----------------------------


def test_adapter_marks_transcriptions_and_inferences_differently():
    result = build_analysis_from_extraction(
        UNKNOWN_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )

    assert all(f.basis is AssertionBasis.DECISION_STATED for f in result.findings)
    assert all(i.basis is AssertionBasis.MODEL_INFERRED for i in result.potential_issues)


def test_result_partitions_stated_from_inferred_findings():
    inferred = Finding(
        finding_id="finding-2",
        finding_type=FindingType.OTHER,
        statement="Model identifies a possible gap in the evidence discussion.",
        basis=AssertionBasis.MODEL_INFERRED,
    )
    result = _result(findings=[_finding(), inferred])

    assert [f.finding_id for f in result.stated_findings] == ["finding-1"]
    assert [f.finding_id for f in result.inferred_findings] == ["finding-2"]


def test_every_finding_carries_a_basis():
    result = build_analysis_from_extraction(
        FULL_EXTRACTION, job_id="job-1", generated_at=FIXED_TIME
    )
    for finding in result.findings:
        assert isinstance(finding.basis, AssertionBasis)


# --- Q. no outcome prediction ------------------------------------------------


FORBIDDEN_FIELD_PATTERN = re.compile(
    r"appeal|success|probability|odds|win|payout|award_amount|grant_chance|"
    r"likelihood_of|expected_rating|predicted",
    re.IGNORECASE,
)


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_no_model_exposes_an_outcome_prediction_field(model):
    for field_name in model.model_fields:
        assert not FORBIDDEN_FIELD_PATTERN.search(field_name), (
            f"{model.__name__}.{field_name} looks like an outcome prediction"
        )


def test_no_enum_member_implies_an_outcome_prediction():
    for enum_cls in (
        ConfidenceLevel,
        EvidenceStrength,
        PotentialIssueType,
        ReviewReason,
        FindingType,
        DecisionOutcome,
        AssertionBasis,
        RegulatoryAnalysisStatus,
    ):
        for member in enum_cls:
            assert not FORBIDDEN_FIELD_PATTERN.search(member.value), member.value


def test_disclaimer_disclaims_legal_conclusions_and_predictions():
    lowered = ANALYSIS_DISCLAIMER.lower()
    assert "not legal determinations" in lowered
    assert "predictions of outcome" in lowered
    assert "vso review" in lowered
