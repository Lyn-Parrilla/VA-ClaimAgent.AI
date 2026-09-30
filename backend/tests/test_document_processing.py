"""Focused tests for the secure document-processing boundary (Step 4).

Credential independence
-----------------------
Every test here injects a fake DLP client, so no Google Cloud client is ever
constructed, no credential is ever read, and no document text ever leaves the
process. `test_module_imports_without_gcp_credentials` and
`test_protection_works_with_all_gcp_env_absent` prove this explicitly by
deleting every GCP environment variable first.

This file is the isolated Step 4 suite. It is distinct from `test_main.py`,
which exercises the /api/extract pipeline and relies on the repo's stub
credentials from tests/conftest.py. Neither suite performs a live DLP call.
"""

import copy
import io
import pickle
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from claim_backend.document_processing import (
    DLP_BUILTIN_INFO_TYPES,
    DLP_INFO_TYPE_TO_ENTITY,
    PROTECTION_FAILURE_MESSAGE,
    DocumentProtectionError,
    PiiFinding,
    PiiTokenMap,
    ProtectedDocument,
    ProtectionMetadata,
    _controlled_failure,
    detect_pii,
    dlp_parent,
    protect_document_text,
    rehydrate_structure,
    rehydrate_text,
    tokenize_pii,
)
from claim_backend.jobs import JobFailureCode

# Synthetic values only. Nothing here is a real person's information, and
# nothing here leaves the test process.
VETERAN_NAME = "John Doe"
SSN = "123-45-6789"
DASHLESS_SSN = "987654321"
PHONE = "555-0199"
FULL_PHONE = "415-555-1234"
EMAIL = "veteran@example.com"

DOCUMENT = (
    f"{VETERAN_NAME}, SSN {SSN}, phone {PHONE}, is granted service connection "
    "for tinnitus, diagnostic code 6260, effective 2023-08-14, rated 10 percent."
)

GCP_ENV_VARS = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "CLOUD_ML_REGION",
    "ANTHROPIC_API_KEY",
    "LANGCHAIN_API_KEY",
)

TEST_PARENT = "projects/test-project/locations/global"


# --- fake DLP boundary -------------------------------------------------------


def _dlp_finding(info_type: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(
        info_type=SimpleNamespace(name=info_type),
        location=SimpleNamespace(
            codepoint_range=SimpleNamespace(start=start, end=end)
        ),
    )


class FakeDlpClient:
    """Stands in for dlp_v2.DlpServiceClient at the network boundary.

    Records the requests it receives so tests can assert on what *would* have
    been sent, while guaranteeing nothing actually is.
    """

    def __init__(self, findings=None, error: Exception = None):
        self._findings = findings or []
        self._error = error
        self.requests = []

    def inspect_content(self, request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            result=SimpleNamespace(findings=list(self._findings))
        )


def _client_for(text: str, spans) -> FakeDlpClient:
    """Build a fake client whose findings match real offsets within `text`."""
    findings = []
    for info_type, value in spans:
        start = text.index(value)
        findings.append(_dlp_finding(info_type, start, start + len(value)))
    return FakeDlpClient(findings=findings)


def _protect(text: str, spans, **kwargs) -> ProtectedDocument:
    return protect_document_text(
        text, client=_client_for(text, spans), parent=TEST_PARENT, **kwargs
    )


# --- W, X, Y, Z. credential independence and isolation -----------------------


def test_module_defines_its_contracts_without_gcp_credentials(monkeypatch):
    """Model definition and validation need no Google configuration.

    Deliberately does not use importlib.reload: rebinding the module dict
    would swap out the class objects this suite imported, breaking every
    isinstance and pytest.raises check. Cold-import is proven instead by
    test_module_imports_in_a_clean_interpreter_without_credentials, which
    uses a fresh process with a stripped environment.
    """
    for name in GCP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    import claim_backend.document_processing as module

    assert module.PROTECTION_ENGINE == "google-cloud-dlp"
    # Models still construct and validate with the environment stripped.
    assert ProtectedDocument(sanitized_text="safe").sanitized_text == "safe"
    assert module._dlp_client is None, "no DLP client should have been built"


def test_protection_works_with_all_gcp_env_absent(monkeypatch):
    """The whole boundary runs with zero Google configuration present."""
    for name in GCP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    assert SSN not in document.sanitized_text
    assert "<SSN_1>" in document.sanitized_text


def test_no_real_dlp_client_is_ever_constructed(monkeypatch):
    """A supplied client short-circuits the lazy real-client accessor."""
    import claim_backend.document_processing as module

    def _explode():
        raise AssertionError("a live DLP client must never be constructed in tests")

    monkeypatch.setattr(module, "_get_dlp_client", _explode)

    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])
    assert document.sanitized_text


def test_fake_client_receives_the_request_and_nothing_is_transmitted():
    client = _client_for(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["parent"] == TEST_PARENT
    # The text reaches only the in-process fake.
    assert request["item"]["value"] == DOCUMENT
    assert isinstance(client, FakeDlpClient)


def test_dlp_parent_reads_project_lazily(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-123")
    assert dlp_parent() == "projects/proj-123/locations/global"
    assert dlp_parent("other") == "projects/other/locations/global"


def test_dlp_parent_fails_closed_without_a_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    with pytest.raises(DocumentProtectionError) as exc_info:
        dlp_parent()

    assert exc_info.value.stage == "configuration"


# --- A. normal text protection -----------------------------------------------


def test_protects_ordinary_text():
    document = _protect(DOCUMENT, [("PERSON_NAME", VETERAN_NAME)])

    assert isinstance(document, ProtectedDocument)
    assert "<PERSON_1>" in document.sanitized_text
    assert VETERAN_NAME not in document.sanitized_text


def test_text_without_pii_passes_through_unchanged():
    text = "Service connection for tinnitus, diagnostic code 6260, is granted."
    document = protect_document_text(
        text, client=FakeDlpClient(findings=[]), parent=TEST_PARENT
    )

    assert document.sanitized_text == text
    assert len(document.token_map) == 0
    assert document.has_protected_content is False


def test_va_specifics_survive_protection():
    """Dates and diagnostic codes must not be tokenized away."""
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    assert "2023-08-14" in document.sanitized_text
    assert "6260" in document.sanitized_text
    assert "10 percent" in document.sanitized_text


def test_metadata_reports_counts_without_values():
    document = _protect(
        DOCUMENT,
        [("PERSON_NAME", VETERAN_NAME), ("US_SOCIAL_SECURITY_NUMBER", SSN)],
    )

    assert document.protection.engine == "google-cloud-dlp"
    assert document.protection.finding_count == 2
    assert document.protection.token_count == 2
    assert set(document.protection.entity_types) == {"PERSON", "SSN"}
    assert document.protection.sanitized_length == len(document.sanitized_text)

    dumped = document.protection.model_dump_json()
    assert SSN not in dumped
    assert VETERAN_NAME not in dumped


# --- B. SSN detection/tokenization -------------------------------------------


def test_dashed_ssn_is_tokenized():
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    assert SSN not in document.sanitized_text
    assert "<SSN_1>" in document.sanitized_text
    assert document.token_map.resolve("<SSN_1>") == SSN


def test_dashless_ssn_is_tokenized_via_the_custom_detector():
    text = f"Veteran SSN {DASHLESS_SSN} on file."
    document = _protect(text, [("CUSTOM_SSN", DASHLESS_SSN)])

    assert DASHLESS_SSN not in document.sanitized_text
    assert "<SSN_1>" in document.sanitized_text


# --- C. phone detection/tokenization -----------------------------------------


def test_local_phone_is_tokenized_via_the_custom_detector():
    document = _protect(DOCUMENT, [("CUSTOM_LOCAL_PHONE", PHONE)])

    assert PHONE not in document.sanitized_text
    assert "<PHONE_NUMBER_1>" in document.sanitized_text


def test_full_phone_is_tokenized():
    text = f"Call {FULL_PHONE} for records."
    document = _protect(text, [("PHONE_NUMBER", FULL_PHONE)])

    assert FULL_PHONE not in document.sanitized_text
    assert "<PHONE_NUMBER_1>" in document.sanitized_text


# --- D. multiple PII values --------------------------------------------------


def test_multiple_pii_values_are_each_tokenized():
    text = (
        f"{VETERAN_NAME}, SSN {SSN}, phone {PHONE}, email {EMAIL}, "
        "claims tinnitus."
    )
    document = _protect(
        text,
        [
            ("PERSON_NAME", VETERAN_NAME),
            ("US_SOCIAL_SECURITY_NUMBER", SSN),
            ("CUSTOM_LOCAL_PHONE", PHONE),
            ("EMAIL_ADDRESS", EMAIL),
        ],
    )

    for value in (VETERAN_NAME, SSN, PHONE, EMAIL):
        assert value not in document.sanitized_text
    assert len(document.token_map) == 4
    assert set(document.token_map.tokens()) == {
        "<PERSON_1>",
        "<SSN_1>",
        "<PHONE_NUMBER_1>",
        "<EMAIL_ADDRESS_1>",
    }


def test_repeated_entity_types_get_distinct_tokens():
    text = f"Primary {FULL_PHONE} and alternate {PHONE}."
    document = _protect(
        text, [("PHONE_NUMBER", FULL_PHONE), ("CUSTOM_LOCAL_PHONE", PHONE)]
    )

    assert set(document.token_map.tokens()) == {
        "<PHONE_NUMBER_1>",
        "<PHONE_NUMBER_2>",
    }
    assert FULL_PHONE not in document.sanitized_text
    assert PHONE not in document.sanitized_text


# --- 10. THE CORE INVARIANT --------------------------------------------------


@pytest.mark.parametrize(
    "info_type,value",
    [
        ("US_SOCIAL_SECURITY_NUMBER", SSN),
        ("CUSTOM_SSN", DASHLESS_SSN),
        ("CUSTOM_LOCAL_PHONE", PHONE),
        ("PHONE_NUMBER", FULL_PHONE),
        ("EMAIL_ADDRESS", EMAIL),
        ("PERSON_NAME", VETERAN_NAME),
    ],
)
def test_sanitized_output_never_contains_the_original_value(info_type, value):
    """raw input -> protection -> sanitized output, with no original surviving."""
    text = f"Record: {value} appears here."
    document = _protect(text, [(info_type, value)])

    assert value not in document.sanitized_text
    # The mapping legitimately holds it, for controlled rehydration only.
    assert value in document.token_map.as_dict().values()


def test_mapping_is_never_part_of_the_sanitized_representation():
    document = _protect(
        DOCUMENT,
        [("PERSON_NAME", VETERAN_NAME), ("US_SOCIAL_SECURITY_NUMBER", SSN)],
    )

    for rendered in (
        document.model_dump_json(),
        str(document.model_dump()),
        repr(document),
        str(document),
    ):
        assert SSN not in rendered
        assert VETERAN_NAME not in rendered


# --- E. mapping is temporary / in-memory -------------------------------------


def test_mapping_is_not_serialized_with_the_document():
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    assert "token_map" not in document.model_dump()
    assert "token_map" not in document.model_dump_json()
    assert set(document.model_dump()) == {"sanitized_text", "protection"}


def test_mapping_cannot_be_pickled_to_disk():
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    with pytest.raises(TypeError, match="never be persisted"):
        pickle.dumps(document.token_map)


def test_mapping_repr_redacts_its_values():
    token_map = PiiTokenMap({"<SSN_1>": SSN, "<PERSON_1>": VETERAN_NAME})

    for rendered in (repr(token_map), str(token_map), f"{token_map}"):
        assert SSN not in rendered
        assert VETERAN_NAME not in rendered
        assert "redacted" in rendered
        assert "tokens=2" in rendered


def test_mapping_iteration_yields_tokens_not_values():
    token_map = PiiTokenMap({"<SSN_1>": SSN})

    assert list(token_map) == ["<SSN_1>"]
    assert token_map.tokens() == ["<SSN_1>"]
    assert "<SSN_1>" in token_map


def test_mapping_accessors_are_explicit():
    token_map = PiiTokenMap({"<SSN_1>": SSN})

    assert token_map.resolve("<SSN_1>") == SSN
    assert token_map.resolve("<MISSING_1>") is None
    assert token_map.as_dict() == {"<SSN_1>": SSN}
    # as_dict returns a copy, so mutating it cannot corrupt the map.
    token_map.as_dict()["<SSN_1>"] = "tampered"
    assert token_map.resolve("<SSN_1>") == SSN


def test_mapping_copies_preserve_the_non_persistence_guarantee():
    token_map = PiiTokenMap({"<SSN_1>": SSN})

    for clone in (copy.copy(token_map), copy.deepcopy(token_map)):
        assert clone == token_map
        with pytest.raises(TypeError):
            pickle.dumps(clone)


def test_protection_holds_no_module_level_state():
    """Two runs share nothing; the map dies with its ProtectedDocument."""
    first = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])
    second = protect_document_text(
        "No identifiers here.", client=FakeDlpClient([]), parent=TEST_PARENT
    )

    assert len(second.token_map) == 0
    assert len(first.token_map) == 1


# --- F. sanitized text contains tokens ---------------------------------------


def test_sanitized_text_substitutes_tokens_in_place():
    text = f"SSN {SSN} end."
    document = _protect(text, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    assert document.sanitized_text == "SSN <SSN_1> end."


# --- G, H. rehydration is explicit and never automatic -----------------------


def test_rehydration_restores_the_original_when_explicitly_requested():
    document = _protect(
        DOCUMENT,
        [("PERSON_NAME", VETERAN_NAME), ("US_SOCIAL_SECURITY_NUMBER", SSN)],
    )

    restored = rehydrate_text(document.sanitized_text, document.token_map)

    assert restored == DOCUMENT
    assert VETERAN_NAME in restored
    assert SSN in restored


def test_rehydration_of_structures_is_recursive():
    document = _protect(DOCUMENT, [("PERSON_NAME", VETERAN_NAME)])
    payload = {"conditions": [{"note": "<PERSON_1> reports tinnitus"}]}

    restored = rehydrate_structure(payload, document.token_map)

    assert restored["conditions"][0]["note"] == f"{VETERAN_NAME} reports tinnitus"


def test_rehydration_does_not_happen_automatically():
    """Protection alone must never yield original values."""
    document = _protect(
        DOCUMENT,
        [("PERSON_NAME", VETERAN_NAME), ("US_SOCIAL_SECURITY_NUMBER", SSN)],
    )

    assert VETERAN_NAME not in document.sanitized_text
    assert SSN not in document.sanitized_text
    # No attribute or method on the result performs rehydration implicitly.
    assert not hasattr(document, "rehydrate")
    assert not hasattr(document, "original_text")
    assert not hasattr(document, "raw_text")


def test_rehydration_requires_the_explicit_token_map_type():
    """A bare dict is refused, so rehydration is always a deliberate act."""
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    with pytest.raises(TypeError, match="explicit PiiTokenMap"):
        rehydrate_text(document.sanitized_text, {"<SSN_1>": SSN})

    with pytest.raises(TypeError, match="explicit PiiTokenMap"):
        rehydrate_structure({"a": "<SSN_1>"}, {"<SSN_1>": SSN})


def test_rehydration_with_an_empty_map_is_a_no_op():
    assert rehydrate_text("<SSN_1> stays", PiiTokenMap()) == "<SSN_1> stays"


# --- I, AA. fail closed, without leaking the cause ---------------------------


def test_detection_failure_fails_closed():
    client = FakeDlpClient(error=RuntimeError("boom"))

    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    error = exc_info.value
    assert error.failure_code is JobFailureCode.PII_PROTECTION_FAILED
    assert error.stage == "detection"


def test_failure_returns_no_partially_protected_text():
    client = FakeDlpClient(error=RuntimeError("boom"))

    with pytest.raises(DocumentProtectionError):
        result = protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)
        assert result is None  # unreachable; documents intent


def test_failure_does_not_leak_the_underlying_exception_message():
    """A provider error can quote the payload, so its text must not travel."""
    leaky = RuntimeError(
        f"DLP request failed while inspecting: {VETERAN_NAME} SSN {SSN}"
    )
    client = FakeDlpClient(error=leaky)

    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    error = exc_info.value
    for rendered in (str(error), repr(error), str(error.args)):
        assert SSN not in rendered
        assert VETERAN_NAME not in rendered
        assert "DLP request failed" not in rendered
    assert str(error) == PROTECTION_FAILURE_MESSAGE
    # Only the safe class name is retained for diagnosis.
    assert error.cause_type == "RuntimeError"
    # Both chain links are severed so a traceback cannot resurface the message.
    assert error.__cause__ is None
    assert error.__context__ is None


def test_controlled_error_retains_no_exception_chain_at_all():
    """Regression: `raise ... from None` clears __cause__ but NOT __context__.

    Python attaches the exception being handled to __context__ whenever a new
    exception is raised inside an `except` block, and __suppress_context__ only
    hides it from traceback *display* -- it stays programmatically reachable.
    The boundary therefore builds its controlled error inside the handler and
    raises it after the handler exits. This test pins that behavior down by
    asserting on the actual attributes rather than trusting the raise syntax.
    """
    marker = "SENSITIVE_EXCEPTION_MARKER"
    client = FakeDlpClient(error=RuntimeError(marker))

    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    error = exc_info.value

    # 5 & 6: the original provider exception is retained through neither link.
    assert error.__cause__ is None
    assert error.__context__ is None

    # The marker is unreachable from the public surface.
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in str(error.args)

    # ...and unreachable by walking the chain, however deep.
    seen, node = [], error
    while node is not None:
        seen.append(node)
        node = node.__cause__ or node.__context__
        if node in seen:
            break
    assert seen == [error], "no other exception object should be reachable"

    # Only sanitized, controlled information survives.
    assert str(error) == PROTECTION_FAILURE_MESSAGE
    assert error.failure_code is JobFailureCode.PII_PROTECTION_FAILED
    assert error.stage == "detection"
    assert error.cause_type == "RuntimeError"


def test_tokenization_failure_also_retains_no_exception_chain(monkeypatch):
    """The second failure stage must be contained identically."""
    import claim_backend.document_processing as module

    marker = "SENSITIVE_EXCEPTION_MARKER"

    def _boom(text, findings):
        raise ValueError(marker)

    monkeypatch.setattr(module, "tokenize_pii", _boom)

    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(
            DOCUMENT,
            client=_client_for(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)]),
            parent=TEST_PARENT,
        )

    error = exc_info.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.stage == "tokenization"
    assert error.cause_type == "ValueError"
    assert marker not in str(error)
    assert marker not in repr(error)


def test_formatted_traceback_cannot_surface_the_marker():
    """Even a fully rendered traceback must not contain the provider message."""
    marker = "SENSITIVE_EXCEPTION_MARKER"
    client = FakeDlpClient(error=RuntimeError(marker))

    try:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)
    except DocumentProtectionError as error:
        rendered = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    assert marker not in rendered
    assert "During handling of the above exception" not in rendered
    assert PROTECTION_FAILURE_MESSAGE in rendered


def test_controlled_failure_helper_severs_links_on_its_own():
    """The helper is safe even if a future caller builds it inside a handler."""
    marker = "SENSITIVE_EXCEPTION_MARKER"

    try:
        raise RuntimeError(marker)
    except RuntimeError:
        built = _controlled_failure(
            failure_code=JobFailureCode.PII_PROTECTION_FAILED,
            stage="detection",
            cause_type="RuntimeError",
        )

    assert built.__cause__ is None
    assert built.__context__ is None
    assert built.__suppress_context__ is True
    assert marker not in str(built) and marker not in repr(built)


def test_failure_does_not_leak_document_text_into_the_traceback():
    client = FakeDlpClient(error=RuntimeError(DOCUMENT))

    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    rendered = str(exc_info.getrepr(style="long"))
    assert SSN not in rendered
    assert PHONE not in rendered


# --- J. controlled failure metadata ------------------------------------------


def test_failure_code_is_from_the_controlled_vocabulary():
    client = FakeDlpClient(error=ValueError("arbitrary text"))

    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    assert isinstance(exc_info.value.failure_code, JobFailureCode)


def test_failure_code_is_accepted_by_the_job_store_but_text_is_not():
    """The boundary hands the job store a code, never a message."""
    from claim_backend.jobs import JobStatus, JobStore

    store = JobStore(ttl_seconds=60)
    job = store.create()
    store.mark_processing(job.id)

    client = FakeDlpClient(error=RuntimeError(f"failed on {SSN}"))
    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(DOCUMENT, client=client, parent=TEST_PARENT)

    updated = store.mark_failed(job.id, exc_info.value.failure_code)
    assert updated.status is JobStatus.FAILED
    assert updated.error is JobFailureCode.PII_PROTECTION_FAILED

    dumped = updated.model_dump_json()
    assert SSN not in dumped
    assert "failed on" not in dumped

    # And the raw message is structurally unable to reach the job.
    with pytest.raises(TypeError):
        store.mark_failed(job.id, f"failed on {SSN}")


# --- K, L, M, N. nothing sensitive is representable --------------------------


def test_protected_document_has_no_original_text_field():
    assert set(ProtectedDocument.model_fields) == {
        "sanitized_text",
        "token_map",
        "protection",
    }
    for forbidden in ("original_text", "raw_text", "document_text", "source_text"):
        assert forbidden not in ProtectedDocument.model_fields


def test_protected_document_does_not_retain_the_original_text():
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    rendered = repr(document) + document.model_dump_json()
    assert DOCUMENT not in rendered
    assert SSN not in rendered


SENSITIVE_FIELDS = [
    "original_text",
    "raw_text",
    "document_text",
    "metadata",
    "extra",
    "context",
    "credentials",
    "api_key",
    "service_account",
    "access_token",
    "secret",
    "prompt",
    "system_prompt",
    "messages",
    "model_response",
    "raw_response",
    "completion",
    "storage_handle",
    "file_path",
    "bucket",
]


@pytest.mark.parametrize("field_name", SENSITIVE_FIELDS)
def test_protected_document_rejects_sensitive_or_arbitrary_fields(field_name):
    with pytest.raises(ValidationError):
        ProtectedDocument(sanitized_text="safe", **{field_name: "injected"})


@pytest.mark.parametrize("field_name", SENSITIVE_FIELDS)
def test_protection_metadata_rejects_sensitive_or_arbitrary_fields(field_name):
    with pytest.raises(ValidationError):
        ProtectionMetadata(**{field_name: "injected"})


def test_protected_document_is_frozen():
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    with pytest.raises(ValidationError):
        document.sanitized_text = DOCUMENT


def test_protected_document_rejects_assignment_of_new_attributes():
    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])

    with pytest.raises((ValidationError, AttributeError)):
        document.original_text = DOCUMENT


# --- O. no logging or printing -----------------------------------------------


def test_module_contains_no_logging_or_printing():
    import claim_backend.document_processing as module

    source = open(module.__file__).read()
    assert "import logging" not in source
    assert "logging.getLogger" not in source
    assert "print(" not in source
    assert "@traceable" not in source
    assert "langsmith" not in source


def test_protection_writes_nothing_to_stdout_or_stderr():
    out, err = io.StringIO(), io.StringIO()

    with redirect_stdout(out), redirect_stderr(err):
        document = _protect(
            DOCUMENT,
            [("PERSON_NAME", VETERAN_NAME), ("US_SOCIAL_SECURITY_NUMBER", SSN)],
        )
        rehydrate_text(document.sanitized_text, document.token_map)

    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_failure_path_writes_nothing_to_stdout_or_stderr():
    out, err = io.StringIO(), io.StringIO()

    with redirect_stdout(out), redirect_stderr(err):
        with pytest.raises(DocumentProtectionError):
            protect_document_text(
                DOCUMENT,
                client=FakeDlpClient(error=RuntimeError(SSN)),
                parent=TEST_PARENT,
            )

    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_logging_records_capture_nothing(caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])
        rehydrate_text(document.sanitized_text, document.token_map)

    assert SSN not in caplog.text
    assert VETERAN_NAME not in caplog.text


# --- P. empty / invalid input ------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 123, b"bytes", []])
def test_empty_or_invalid_input_fails_closed(bad):
    with pytest.raises(DocumentProtectionError) as exc_info:
        protect_document_text(bad, client=FakeDlpClient([]), parent=TEST_PARENT)

    assert exc_info.value.stage == "input_validation"
    assert exc_info.value.failure_code is JobFailureCode.DOCUMENT_EXTRACTION_FAILED


def test_invalid_input_never_reaches_the_dlp_client():
    client = FakeDlpClient([])

    with pytest.raises(DocumentProtectionError):
        protect_document_text("   ", client=client, parent=TEST_PARENT)

    assert client.requests == []


# --- Q. already-tokenized content --------------------------------------------


def test_already_tokenized_text_is_left_alone_when_nothing_is_detected():
    text = "<PERSON_1> reports tinnitus, effective 2023-08-14."
    document = protect_document_text(
        text, client=FakeDlpClient([]), parent=TEST_PARENT
    )

    assert document.sanitized_text == text
    assert len(document.token_map) == 0


def test_double_protection_is_stable():
    """Protecting an already-sanitized document adds no new tokens."""
    first = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])
    second = protect_document_text(
        first.sanitized_text, client=FakeDlpClient([]), parent=TEST_PARENT
    )

    assert second.sanitized_text == first.sanitized_text
    assert SSN not in second.sanitized_text


# --- R. existing DLP seams remain testable -----------------------------------


def test_detect_pii_maps_info_types_through_the_injected_client():
    client = FakeDlpClient(
        findings=[
            _dlp_finding("PERSON_NAME", 0, 8),
            _dlp_finding("CUSTOM_SSN", 13, 22),
            _dlp_finding("DATE", 23, 33),  # unmapped, must be dropped
            _dlp_finding("PERSON_NAME", 40, 40),  # empty span, must be dropped
        ]
    )

    findings = detect_pii("irrelevant", client=client, parent=TEST_PARENT)

    assert findings == [
        PiiFinding(entity_type="PERSON", start=0, end=8),
        PiiFinding(entity_type="SSN", start=13, end=22),
    ]


def test_dlp_configuration_is_preserved_from_the_original_implementation():
    assert DLP_BUILTIN_INFO_TYPES == [
        "PERSON_NAME",
        "US_SOCIAL_SECURITY_NUMBER",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "STREET_ADDRESS",
        "LOCATION",
    ]
    assert DLP_INFO_TYPE_TO_ENTITY["CUSTOM_SSN"] == "SSN"
    assert DLP_INFO_TYPE_TO_ENTITY["CUSTOM_LOCAL_PHONE"] == "PHONE_NUMBER"
    # DATE is deliberately absent so VA effective dates survive.
    assert "DATE" not in DLP_INFO_TYPE_TO_ENTITY


def test_tokenizer_behavior_matches_the_original_implementation():
    text = "John Doe SSN 123456789"
    findings = [
        PiiFinding(entity_type="PERSON", start=0, end=8),
        PiiFinding(entity_type="SSN", start=13, end=22),
    ]

    tokenized, mapping = tokenize_pii(text, findings)

    assert tokenized == "<PERSON_1> SSN <SSN_1>"
    assert mapping == {"<PERSON_1>": "John Doe", "<SSN_1>": "123456789"}


def test_overlapping_findings_prefer_the_longest_span():
    text = "Call 415-555-1234 today"
    findings = [
        PiiFinding(entity_type="PHONE_NUMBER", start=5, end=17),
        PiiFinding(entity_type="PHONE_NUMBER", start=5, end=13),
    ]

    tokenized, mapping = tokenize_pii(text, findings)

    assert tokenized == "Call <PHONE_NUMBER_1> today"
    assert mapping == {"<PHONE_NUMBER_1>": "415-555-1234"}


def test_main_reuses_this_single_dlp_implementation():
    """There must be exactly one detector/tokenizer in the project."""
    import claim_backend.document_processing as dp
    import claim_backend.main as main

    assert main._tokenize_pii is dp.tokenize_pii
    assert main._rehydrate_text is dp.rehydrate_mapping_text
    assert main._rehydrate_json is dp.rehydrate_mapping_structure
    assert main._get_dlp_client is dp._get_dlp_client
    assert main.DLP_BUILTIN_INFO_TYPES is dp.DLP_BUILTIN_INFO_TYPES
    assert main.DLP_CUSTOM_INFO_TYPES is dp.DLP_CUSTOM_INFO_TYPES
    assert main.DLP_INFO_TYPE_TO_ENTITY is dp.DLP_INFO_TYPE_TO_ENTITY
    assert main.PiiFinding is dp.PiiFinding


def test_no_persistent_files_are_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    document = _protect(DOCUMENT, [("US_SOCIAL_SECURITY_NUMBER", SSN)])
    rehydrate_text(document.sanitized_text, document.token_map)

    assert list(tmp_path.iterdir()) == []


def test_module_imports_in_a_clean_interpreter_without_credentials():
    """End-to-end proof: a fresh process with no GCP env can use the boundary."""
    code = (
        "import claim_backend.document_processing as dp\n"
        "from types import SimpleNamespace\n"
        "f = SimpleNamespace(info_type=SimpleNamespace(name='CUSTOM_SSN'),\n"
        "    location=SimpleNamespace(codepoint_range=SimpleNamespace(start=4, end=15)))\n"
        "class C:\n"
        "    def inspect_content(self, request):\n"
        "        return SimpleNamespace(result=SimpleNamespace(findings=[f]))\n"
        "d = dp.protect_document_text('SSN 123-45-6789', client=C(), parent='p')\n"
        "assert '123-45-6789' not in d.sanitized_text, 'LEAK'\n"
        "print('CLEAN_OK', d.sanitized_text)\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": "src",
        "HOME": "/tmp",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, result.stderr
    assert "CLEAN_OK" in result.stdout
    assert "123-45-6789" not in result.stdout
