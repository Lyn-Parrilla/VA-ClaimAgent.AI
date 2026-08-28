import os

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/health-check")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/claude-test")
def claude_test() -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")

    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )
        return {"reply": response.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
