"""Focused tests for the analysis job API.

These drive the real FastAPI app, so they also confirm the router is wired to
the single application-level JobStore rather than a per-request one.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from claim_backend.jobs import (
    JobFailureCode,
    JobStatus,
    get_job_store,
    reset_job_store,
)
from claim_backend.jobs_api import JobResponse
from claim_backend.main import app

# Material that must never appear in an API response.
VETERAN_TEXT = "John Doe, SSN 123-45-6789, is rated 10% for tinnitus."
SENSITIVE_FAILURE_TEXT = "Veteran John Doe SSN 123-45-6789"
PII_MAP = {"<PERSON_1>": "John Doe", "<SSN_1>": "123-45-6789"}

# The exact published shape of a job.
EXPECTED_KEYS = {
    "id",
    "status",
    "created_at",
    "updated_at",
    "expires_at",
    "completed_at",
    "failure_code",
}


@pytest.fixture
def client() -> TestClient:
    """A client over the real app, with the shared store reset per test."""
    reset_job_store()
    try:
        yield TestClient(app)
    finally:
        reset_job_store()


def _assert_utc(raw: str) -> datetime:
    """Parse an ISO timestamp from a response and assert it is UTC-aware."""
    parsed = datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    return parsed


# --- A, B, C. create ---------------------------------------------------------


def test_create_job_returns_201(client):
    assert client.post("/api/jobs").status_code == 201


def test_create_job_creates_a_queued_job(client):
    body = client.post("/api/jobs").json()

    assert body["status"] == "QUEUED"
    assert str(uuid.UUID(body["id"])) == body["id"]
    assert body["completed_at"] is None
    assert body["failure_code"] is None


def test_create_job_returns_exactly_the_safe_metadata(client):
    body = client.post("/api/jobs").json()
    assert set(body) == EXPECTED_KEYS


def test_create_job_timestamps_are_utc_and_ttl_applied(client):
    body = client.post("/api/jobs").json()

    created = _assert_utc(body["created_at"])
    updated = _assert_utc(body["updated_at"])
    expires = _assert_utc(body["expires_at"])

    assert updated == created
    assert expires - created == timedelta(seconds=get_job_store().ttl_seconds)


def test_create_job_accepts_an_explicitly_empty_body(client):
    assert client.post("/api/jobs", json={}).status_code == 201


def test_create_job_accepts_a_missing_body(client):
    assert client.post("/api/jobs").status_code == 201


@pytest.mark.parametrize(
    "payload",
    [
        {"text": VETERAN_TEXT},
        {"document": VETERAN_TEXT},
        {"pii_map": PII_MAP},
        {"prompt": "extract conditions"},
        {"model": "claude-sonnet-5"},
        {"max_tokens": 4096},
    ],
)
def test_create_job_rejects_any_payload(client, payload):
    """Document text, PII, prompts, and model parameters are all refused."""
    response = client.post("/api/jobs", json=payload)

    assert response.status_code == 422
    assert len(get_job_store()) == 0


def test_rejected_payload_is_not_echoed_back(client):
    """A 422 must not reflect the submitted value to the caller."""
    response = client.post("/api/jobs", json={"text": VETERAN_TEXT})

    assert response.status_code == 422
    assert "John Doe" not in response.text
    assert "123-45-6789" not in response.text
    assert "tinnitus" not in response.text
    # The useful, non-sensitive parts survive.
    assert "extra_forbidden" in response.text
    for error in response.json()["detail"]:
        assert set(error) <= {"type", "loc", "msg"}


# --- D, E, F. retrieve -------------------------------------------------------


def test_created_job_can_be_retrieved(client):
    created = client.post("/api/jobs").json()

    response = client.get(f"/api/jobs/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_returns_exactly_the_safe_metadata(client):
    created = client.post("/api/jobs").json()

    body = client.get(f"/api/jobs/{created['id']}").json()

    assert set(body) == EXPECTED_KEYS


def test_get_reflects_lifecycle_progress(client):
    created = client.post("/api/jobs").json()
    store = get_job_store()
    store.mark_processing(created["id"])

    processing = client.get(f"/api/jobs/{created['id']}").json()
    assert processing["status"] == "PROCESSING"
    assert processing["completed_at"] is None

    store.mark_completed(created["id"])
    completed = client.get(f"/api/jobs/{created['id']}").json()
    assert completed["status"] == "COMPLETED"
    _assert_utc(completed["completed_at"])


# --- G. missing jobs ---------------------------------------------------------


def test_unknown_job_returns_404(client):
    response = client.get(f"/api/jobs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_404_reveals_no_internals(client):
    response = client.get(f"/api/jobs/{uuid.uuid4()}")

    lowered = response.text.lower()
    for leak in ("traceback", "claim_backend", "jobstore", "keyerror", ".py"):
        assert leak not in lowered


def test_deleted_job_returns_404(client):
    created = client.post("/api/jobs").json()
    assert get_job_store().delete(created["id"]) is True

    assert client.get(f"/api/jobs/{created['id']}").status_code == 404


# --- H. malformed UUIDs ------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "12345",
        "00000000-0000-0000-0000-00000000000",  # one char short
        "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
        VETERAN_TEXT,
    ],
)
def test_malformed_uuid_returns_422_not_500(client, bad_id):
    response = client.get("/api/jobs/" + bad_id)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "uuid_parsing"
    assert "traceback" not in response.text.lower()


def test_path_traversal_attempt_is_rejected_without_error(client):
    """Normalized away by routing; must not 500 or reach the handler."""
    response = client.get("/api/jobs/../../etc/passwd")

    assert response.status_code in (404, 422)
    assert "traceback" not in response.text.lower()
    assert "root:" not in response.text


def test_malformed_uuid_does_not_echo_sensitive_input(client):
    response = client.get(f"/api/jobs/{SENSITIVE_FAILURE_TEXT}")

    assert response.status_code == 422
    assert "John Doe" not in response.text
    assert "123-45-6789" not in response.text


def test_uuid_is_accepted_case_insensitively(client):
    created = client.post("/api/jobs").json()

    response = client.get(f"/api/jobs/{created['id'].upper()}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


# --- I. single application-level store ---------------------------------------


def test_requests_share_one_application_level_store(client):
    first = client.post("/api/jobs").json()
    second = client.post("/api/jobs").json()

    # A per-request store would have lost `first` by now.
    assert client.get(f"/api/jobs/{first['id']}").status_code == 200
    assert client.get(f"/api/jobs/{second['id']}").status_code == 200
    assert len(get_job_store()) == 2


def test_api_writes_into_the_module_singleton(client):
    """Proves there is no second registry behind the API."""
    created = client.post("/api/jobs").json()

    stored = get_job_store().get(created["id"])

    assert stored is not None
    assert stored.id == created["id"]
    assert stored.status is JobStatus.QUEUED


def test_jobs_created_directly_in_the_store_are_visible_over_the_api(client):
    job = get_job_store().create()

    response = client.get(f"/api/jobs/{job.id}")

    assert response.status_code == 200
    assert response.json()["id"] == job.id


# --- J, K. controlled failure exposure ---------------------------------------


@pytest.mark.parametrize("code", list(JobFailureCode))
def test_failed_job_exposes_only_its_failure_code(client, code):
    created = client.post("/api/jobs").json()
    store = get_job_store()
    store.mark_processing(created["id"])
    store.mark_failed(created["id"], code)

    body = client.get(f"/api/jobs/{created['id']}").json()

    assert body["status"] == "FAILED"
    assert body["failure_code"] == code.value
    assert set(body) == EXPECTED_KEYS
    # FAILED is not a completion.
    assert body["completed_at"] is None


def test_api_cannot_surface_arbitrary_exception_text(client):
    """The store refuses free text, so no such value can reach a response."""
    created = client.post("/api/jobs").json()
    store = get_job_store()
    store.mark_processing(created["id"])

    with pytest.raises(TypeError):
        store.mark_failed(created["id"], SENSITIVE_FAILURE_TEXT)

    response = client.get(f"/api/jobs/{created['id']}")
    assert response.json()["failure_code"] is None
    assert SENSITIVE_FAILURE_TEXT not in response.text


def test_failure_code_field_is_constrained_to_the_enum():
    annotation = JobResponse.model_fields["failure_code"].annotation
    assert JobFailureCode in getattr(annotation, "__args__", (annotation,))


# --- L. no sensitive data in any response ------------------------------------


def test_response_model_publishes_only_metadata_fields():
    """Guards against a future storage field being exposed by accident."""
    assert set(JobResponse.model_fields) == EXPECTED_KEYS


def test_no_response_contains_sensitive_material(client):
    created = client.post("/api/jobs").json()
    store = get_job_store()
    store.mark_processing(created["id"])
    store.mark_failed(created["id"], JobFailureCode.PII_PROTECTION_FAILED)

    bodies = [
        client.post("/api/jobs").text,
        client.get(f"/api/jobs/{created['id']}").text,
        client.get(f"/api/jobs/{uuid.uuid4()}").text,
        client.post("/api/jobs", json={"text": VETERAN_TEXT}).text,
    ]

    for body in bodies:
        assert "John Doe" not in body
        assert "123-45-6789" not in body
        assert "tinnitus" not in body
        assert "<PERSON_1>" not in body
        assert "<SSN_1>" not in body
        for value in PII_MAP.values():
            assert value not in body


def test_openapi_job_schema_exposes_only_safe_fields():
    schema = app.openapi()["components"]["schemas"]["JobResponse"]
    assert set(schema["properties"]) == EXPECTED_KEYS


# --- N. existing contracts preserved -----------------------------------------


def test_extract_route_is_unchanged(client):
    spec = app.openapi()["paths"]["/api/extract"]["post"]

    assert set(spec["responses"]) == {"200", "422"}
    assert (
        spec["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ExtractRequest"
    )


def test_extract_validation_errors_still_use_the_default_shape(client):
    """The jobs-scoped sanitizer must not alter other routes."""
    response = client.post("/api/extract", json={})

    assert response.status_code == 422
    # FastAPI's default payload retains `input`; only /api/jobs is stripped.
    assert response.json()["detail"][0]["type"] == "missing"
    assert "input" in response.json()["detail"][0]


def test_health_check_still_works(client):
    assert client.get("/health-check").json() == {"status": "ok"}
