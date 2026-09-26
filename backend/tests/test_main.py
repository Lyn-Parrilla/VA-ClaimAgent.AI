import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from claim_backend import main
from claim_backend.main import PiiFinding

client = TestClient(main.app)


def _make_thinking_block(text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", text=text)


def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _claude_response(payload: dict) -> SimpleNamespace:
    """Mimic Claude on Vertex returning a ThinkingBlock before the TextBlock."""
    return SimpleNamespace(
        content=[_make_thinking_block(), _make_text_block(json.dumps(payload))]
    )


def _fake_claude_happy(*, messages, system=None, max_tokens=4096):
    return _claude_response(
        {
            "conditions": [
                {
                    "condition": "Tinnitus",
                    "diagnostic_code": "6265",
                    "effective_date": "2023-08-14",
                    "cfr_citation": "38 CFR 4.87",
                    "percentage": "10%",
                }
            ]
        }
    )


def _fake_claude_adversarial(*, messages, system=None, max_tokens=4096):
    return _claude_response(
        {
            "conditions": [
                {
                    "condition": "unknown",
                    "diagnostic_code": "unknown",
                    "effective_date": "unknown",
                    "cfr_citation": "unknown",
                    "percentage": "unknown",
                }
            ]
        }
    )


def _fake_detect_pii(spans):
    """Build a detect_pii replacement returning fixed findings."""

    def _detect(text: str):
        return spans

    return _detect


def test_startup_check_requires_google_credentials(monkeypatch):
    """Missing GOOGLE_APPLICATION_CREDENTIALS must raise an informative error."""
    monkeypatch.setattr(main, "GOOGLE_APPLICATION_CREDENTIALS", None)

    with pytest.raises(main.StartupConfigurationError) as exc_info:
        main._validate_gcp_environment()

    assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc_info.value)


def test_startup_check_requires_existing_credentials_file(monkeypatch):
    """A credentials path that does not exist must be rejected at startup."""
    monkeypatch.setattr(
        main, "GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/service-account.json"
    )

    with pytest.raises(main.StartupConfigurationError) as exc_info:
        main._validate_gcp_environment()

    assert "not an existing file" in str(exc_info.value)


def test_tokenize_pii_does_not_merge_adjacent_acronyms():
    """A name followed by an acronym (e.g., SSN) must not become a single PERSON token."""
    text = "John Doe SSN 123456789"
    findings = [
        PiiFinding(entity_type="PERSON", start=0, end=8),
        PiiFinding(entity_type="SSN", start=13, end=22),
    ]

    tokenized, entity_map = main._tokenize_pii(text, findings)

    assert entity_map["<PERSON_1>"] == "John Doe"
    assert entity_map["<SSN_1>"] == "123456789"
    assert "John Doe SSN" not in entity_map.values()
    assert tokenized == "<PERSON_1> SSN <SSN_1>"


def test_tokenize_pii_prefers_longest_span_on_overlap():
    """Overlapping DLP findings at the same offset collapse to one token."""
    text = "Call 415-555-1234 today"
    findings = [
        PiiFinding(entity_type="PHONE_NUMBER", start=5, end=17),
        PiiFinding(entity_type="PHONE_NUMBER", start=5, end=13),
    ]

    tokenized, entity_map = main._tokenize_pii(text, findings)

    assert entity_map == {"<PHONE_NUMBER_1>": "415-555-1234"}
    assert tokenized == "Call <PHONE_NUMBER_1> today"


def test_detect_pii_maps_dlp_info_types(monkeypatch):
    """DLP info type names are mapped onto stable entity tokens, unknowns dropped."""

    def _fake_inspect_content(request):
        return SimpleNamespace(
            result=SimpleNamespace(
                findings=[
                    SimpleNamespace(
                        info_type=SimpleNamespace(name="PERSON_NAME"),
                        location=SimpleNamespace(
                            codepoint_range=SimpleNamespace(start=0, end=8)
                        ),
                    ),
                    SimpleNamespace(
                        info_type=SimpleNamespace(name="CUSTOM_SSN"),
                        location=SimpleNamespace(
                            codepoint_range=SimpleNamespace(start=13, end=22)
                        ),
                    ),
                    SimpleNamespace(
                        info_type=SimpleNamespace(name="DATE"),
                        location=SimpleNamespace(
                            codepoint_range=SimpleNamespace(start=23, end=33)
                        ),
                    ),
                ]
            )
        )

    monkeypatch.setattr(
        main,
        "_get_dlp_client",
        lambda: SimpleNamespace(inspect_content=_fake_inspect_content),
    )

    findings = main.detect_pii("John Doe SSN 123456789 2023-08-14")

    assert findings == [
        PiiFinding(entity_type="PERSON", start=0, end=8),
        PiiFinding(entity_type="SSN", start=13, end=22),
    ]


def test_extract_happy_path(monkeypatch):
    """Formal VA rating decision letter returns fully populated condition fields."""
    monkeypatch.setattr(main, "call_claude_llm", _fake_claude_happy)
    monkeypatch.setattr(
        main,
        "detect_pii",
        _fake_detect_pii([PiiFinding(entity_type="PERSON", start=4, end=12)]),
    )

    payload = {
        "text": (
            "The John Doe is service-connected for tinnitus, diagnostic code 6265, "
            "effective August 14, 2023, rated 10 percent under 38 CFR 4.87."
        )
    }
    response = client.post("/api/extract", json=payload)

    assert response.status_code == 200
    data = response.json()
    condition = data["extraction"]["conditions"][0]
    assert condition["condition"] == "Tinnitus"
    assert condition["diagnostic_code"] == "6265"
    assert condition["effective_date"] == "2023-08-14"
    assert condition["cfr_citation"] == "38 CFR 4.87"
    assert condition["percentage"] == "10%"
    # PII tokens are fully rehydrated and VA specifics survive untouched.
    assert "<PERSON_" not in data["redacted_text"]
    assert "John Doe" in data["redacted_text"]
    assert "August 14, 2023" in data["redacted_text"]
    assert "6265" in data["redacted_text"]


def test_extract_adversarial_triage(monkeypatch):
    """Unstructured triage narrative with PII edge cases returns unknown for formal fields."""
    monkeypatch.setattr(main, "call_claude_llm", _fake_claude_adversarial)

    text = "Hi, my knees have been hurting. SSN 123-45-6789. Call me at 555-0199."
    monkeypatch.setattr(
        main,
        "detect_pii",
        _fake_detect_pii(
            [
                PiiFinding(entity_type="SSN", start=36, end=47),
                PiiFinding(entity_type="PHONE_NUMBER", start=60, end=68),
            ]
        ),
    )

    response = client.post("/api/extract", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    condition = data["extraction"]["conditions"][0]
    assert condition["condition"] == "unknown"
    assert condition["diagnostic_code"] == "unknown"
    assert condition["effective_date"] == "unknown"
    assert condition["cfr_citation"] == "unknown"
    assert condition["percentage"] == "unknown"
    assert "<SSN_" not in data["redacted_text"]
    assert "<PHONE_NUMBER_" not in data["redacted_text"]
    assert "123-45-6789" in data["redacted_text"]
    assert "555-0199" in data["redacted_text"]


def test_extract_rejects_blank_text():
    """Whitespace-only input is rejected before any DLP or model call."""
    response = client.post("/api/extract", json={"text": "   "})
    assert response.status_code == 422
