import json
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional

from anthropic import AnthropicVertex
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import dlp_v2
from langsmith import traceable
from pydantic import BaseModel, Field

from .jobs_api import register_jobs_api

# The backend root still holds .env and gcp-service-account.json, three levels up
# from this module under the src layout (backend/src/claim_backend/main.py).
BACKEND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

# Load .env from the backend root so credentials are never hardcoded.
load_dotenv(dotenv_path=os.path.join(BACKEND_ROOT, ".env"), override=True)

class StartupConfigurationError(RuntimeError):
    """Raised during startup when required Google Cloud configuration is missing."""


def _resolve_credentials_path(raw: Optional[str]) -> Optional[str]:
    """Resolve a relative credentials path against the backend root.

    This lets .env hold `GOOGLE_APPLICATION_CREDENTIALS=gcp-service-account.json`
    and still work regardless of the working directory uvicorn is started from.
    """
    if not raw:
        return raw
    if os.path.isabs(raw):
        return raw
    resolved = os.path.join(BACKEND_ROOT, raw)
    # Google's client libraries read this variable directly, so keep it in sync.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = resolved
    return resolved


GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = _resolve_credentials_path(
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
)
GOOGLE_CLOUD_PROJECT: Optional[str] = os.environ.get("GOOGLE_CLOUD_PROJECT")
CLOUD_ML_REGION: str = os.environ.get("CLOUD_ML_REGION", "us-east5")


def _validate_gcp_environment() -> None:
    """Fail fast at startup if Google Cloud credentials/config are not usable.

    Both Cloud DLP (PII detection) and Anthropic on Vertex AI authenticate via
    Application Default Credentials, so a missing service account file means the
    app can neither protect PII nor reach the model.
    """
    if not GOOGLE_APPLICATION_CREDENTIALS:
        raise StartupConfigurationError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. Point it at the JSON key "
            "file for a service account with the Cloud DLP and Vertex AI roles, "
            "e.g. GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json "
            "in backend/.env"
        )
    if not os.path.isfile(GOOGLE_APPLICATION_CREDENTIALS):
        raise StartupConfigurationError(
            "GOOGLE_APPLICATION_CREDENTIALS points to "
            f"'{GOOGLE_APPLICATION_CREDENTIALS}', which is not an existing file. "
            "Provide the absolute path to the service account JSON key file."
        )
    if not GOOGLE_CLOUD_PROJECT:
        raise StartupConfigurationError(
            "GOOGLE_CLOUD_PROJECT is not set. Set it to the GCP project ID that "
            "hosts Cloud DLP and Vertex AI, e.g. GOOGLE_CLOUD_PROJECT=my-project "
            "in backend/.env"
        )


_validate_gcp_environment()


app = FastAPI()

# CORS: explicitly whitelist the Vercel frontend only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://va-claim-agent-ai.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Analysis job API. Self-contained router over the in-memory job store; it is
# not yet connected to the extraction pipeline below.
register_jobs_api(app)

MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5@20260630")

# Clients are created lazily so importing this module never performs network or
# credential I/O, which keeps startup and tests fast.
_dlp_client: Optional[dlp_v2.DlpServiceClient] = None
_anthropic_client: Optional[AnthropicVertex] = None


def _get_dlp_client() -> dlp_v2.DlpServiceClient:
    global _dlp_client
    if _dlp_client is None:
        _dlp_client = dlp_v2.DlpServiceClient()
    return _dlp_client


def _get_anthropic_client() -> AnthropicVertex:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AnthropicVertex(
            project_id=GOOGLE_CLOUD_PROJECT, region=CLOUD_ML_REGION
        )
    return _anthropic_client


DLP_PARENT = f"projects/{GOOGLE_CLOUD_PROJECT}/locations/global"

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

# Map DLP info type names onto the stable token prefixes used in the entity map.
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


class PiiFinding(NamedTuple):
    """A single PII span detected by Cloud DLP, in codepoint offsets."""

    entity_type: str
    start: int
    end: int


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="VA rating decision letter text")


class ExtractedCondition(BaseModel):
    condition: str
    diagnostic_code: str
    effective_date: str
    cfr_citation: str
    percentage: str


class ExtractedData(BaseModel):
    conditions: List[ExtractedCondition]


class ExtractResponse(BaseModel):
    redacted_text: str
    extraction: ExtractedData


def detect_pii(text: str) -> List[PiiFinding]:
    """Inspect text with Cloud DLP and return the PII spans to tokenize.

    Intentionally not traced by LangSmith: the input here still contains raw PII.
    """
    response = _get_dlp_client().inspect_content(
        request={
            "parent": DLP_PARENT,
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


def _strip_markdown_json(raw: str) -> str:
    """Remove surrounding ```json ... ``` or ``` ... ``` code fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def _tokenize_pii(
    text: str, findings: List[PiiFinding]
) -> tuple[str, Dict[str, str]]:
    """Replace detected PII with safe tokens and return the tokenized text + map.

    The map is token -> original value, kept in request-local memory only.
    """
    pii_map: Dict[str, str] = {}
    type_counts: Dict[str, int] = {}
    # Walk backward through the text so earlier indices stay valid, preferring the
    # longest span whenever two findings start at the same offset.
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


def _rehydrate_text(text: str, entity_map: Dict[str, str]) -> str:
    """Replace safe tokens with original PII values, longest tokens first."""
    result = text
    for token in sorted(entity_map.keys(), key=len, reverse=True):
        result = result.replace(token, entity_map[token])
    return result


def _rehydrate_json(obj: Any, entity_map: Dict[str, str]) -> Any:
    """Recursively replace safe tokens in parsed JSON with the original PII values."""
    if isinstance(obj, dict):
        return {k: _rehydrate_json(v, entity_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rehydrate_json(item, entity_map) for item in obj]
    if isinstance(obj, str):
        result = obj
        for token in sorted(entity_map.keys(), key=len, reverse=True):
            result = result.replace(token, entity_map[token])
        return result
    return obj


@traceable(name="extract_va_claims")
def call_claude_llm(
    *,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    max_tokens: int = 4096,
):
    """Stateless helper to call Claude on Vertex AI.

    LangSmith environment variables are respected by the @traceable decorator.
    Only tokenized text ever reaches this call.
    """
    kwargs = {}
    if system:
        kwargs["system"] = system

    return _get_anthropic_client().messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        messages=messages,
        **kwargs,
    )


def _get_text_block(response) -> str:
    """Extract the text payload from the first TextBlock in the Anthropic response."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise HTTPException(status_code=500, detail="No text block in Anthropic response")


@app.get("/health-check")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/claude-test")
def claude_test() -> dict:
    try:
        response = call_claude_llm(
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=100,
        )
        return {"reply": _get_text_block(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text is required")

    # 1. Detect PII with Cloud DLP.
    findings = detect_pii(request.text)

    # 2. Tokenize PII in request-local memory.
    tokenized_text, entity_map = _tokenize_pii(request.text, findings)

    # 3. Prompt Claude for a strict JSON schema.
    system_prompt = (
        "You are a VA claims extraction assistant. "
        "The input is either a formal 'VA rating decision letter' or an unstructured "
        "'inbound client triage narrative'. "
        "For formal rating decision letters, extract each diagnosed condition, its "
        "diagnostic code, effective date, CFR citation, and disability percentage. "
        "For unstructured triage narratives, only extract a condition if it is explicitly "
        "stated as an official VA diagnosis or rating. "
        "Do NOT treat conversational symptom descriptions (for example: 'my knees have been hurting') "
        "as formal, diagnosed VA conditions. "
        "If the diagnosis, diagnostic code, effective date, CFR citation, or percentage is not "
        "explicitly stated as an official rating, you must return 'unknown' for that field. "
        "Return ONLY a valid JSON object with no Markdown, no explanation, and no text "
        "outside the JSON. Use this exact schema: "
        '{"conditions": [{"condition": "<condition name or unknown>", "diagnostic_code": "<code or unknown>", "effective_date": "<YYYY-MM-DD or unknown>", "cfr_citation": "<cfr or unknown>", "percentage": "<percentage or unknown>"}]}'
    )

    try:
        # 4. Call the LLM with LangSmith tracing.
        response = call_claude_llm(
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract from the following tokenized VA rating decision letter text:\n\n{tokenized_text}",
                }
            ],
        )

        # 5. Parse Claude's response.
        raw = _get_text_block(response)
        cleaned_json = _strip_markdown_json(raw)
        parsed = json.loads(cleaned_json)

        # 6. Rehydrate placeholders in both the raw redacted text and the JSON object.
        redacted_text = _rehydrate_text(tokenized_text, entity_map)
        rehydrated = _rehydrate_json(parsed, entity_map)
        extraction = ExtractedData.model_validate(rehydrated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    # entity_map and tokenized_text are discarded when this function returns.
    return ExtractResponse(redacted_text=redacted_text, extraction=extraction)
