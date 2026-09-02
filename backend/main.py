import json
import os
import re
from typing import List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Load .env from the same directory as this file so the API key is never hardcoded.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI()

# 1. CORS: explicitly whitelist the Vercel frontend only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://va-claim-agent-ai.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# 7. Load the Anthropic API key securely from the environment.
ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client: Optional[Anthropic] = (
    Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
)

# 2. Presidio PII redaction before any external API call.
nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}
analyzer = AnalyzerEngine(
    nlp_engine=NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
)
anonymizer = AnonymizerEngine()

REDACTION_OPERATORS = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
}

# Tight PII entity whitelist for VA letters.
PII_ENTITIES = ["PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS", "SSN"]

# Custom hyphenated SSN recognizer because the default US_SSN misses 123-45-6789.
SSN_PATTERN = Pattern(name="ssn_pattern", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.9)
ssn_recognizer = PatternRecognizer(
    supported_entity="SSN",
    patterns=[SSN_PATTERN],
    name="hyphenated_ssn_recognizer",
)
analyzer.registry.add_recognizer(ssn_recognizer)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="VA rating decision letter text")


class ExtractedCondition(BaseModel):
    condition: str
    diagnostic_code: str
    effective_date: str


class ExtractedData(BaseModel):
    conditions: List[ExtractedCondition]


class ExtractResponse(BaseModel):
    redacted_text: str
    extraction: ExtractedData


def _strip_markdown_json(raw: str) -> str:
    """Remove surrounding ```json ... ``` or ``` ... ``` code fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


@app.get("/health-check")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/claude-test")
def claude_test() -> dict:
    if anthropic_client is None:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )
        return {"reply": response.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    if anthropic_client is None:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text is required")

    # 2. Redact names, SSNs, phone numbers, and addresses before the LLM call.
    analyzer_results = analyzer.analyze(
        text=request.text, language="en", entities=PII_ENTITIES
    )
    redacted = anonymizer.anonymize(
        text=request.text,
        analyzer_results=analyzer_results,
        operators=REDACTION_OPERATORS,
    ).text

    # 5. Prompt Claude to return a strict JSON schema only.
    system_prompt = (
        "You are a VA claims extraction assistant. "
        "Parse the redacted VA rating decision letter text and extract each condition, "
        "its diagnostic code, and its effective date. "
        "Return ONLY a valid JSON object with no Markdown, no explanation, and no text "
        "outside the JSON. Use this exact schema: "
        '{"conditions": [{"condition": "<condition name>", "diagnostic_code": "<diagnostic code or unknown>", "effective_date": "<YYYY-MM-DD or unknown>"}]}'
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract from the following redacted VA rating decision letter text:\n\n{redacted}",
                }
            ],
        )
        raw = response.content[0].text
        # 6. Strip Markdown code fences before JSON validation.
        cleaned_json = _strip_markdown_json(raw)
        extraction = ExtractedData.model_validate_json(cleaned_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    return ExtractResponse(redacted_text=redacted, extraction=extraction)
