import os
import json
import logging
import requests
import openai

_log = logging.getLogger(__name__)

SUPPORTED_LLM_PROVIDERS = {"ollama", "openai"}

# ---- constants ----
OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "integer"},
        "confidence": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grade", "confidence"],
}


def _pick_llm_provider(specified: str | None) -> str:
    provider = (specified or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}. Must be one of {SUPPORTED_LLM_PROVIDERS}")

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set but provider=openai")

    return provider


def build_classification_prompt(text: str) -> str:
    return (
        "You are an assistant analyzing product reviews.\n\n"
        "Return a JSON object with fields:\n"
        "- grade: integer from 1 to 10, user satisfaction\n"
        "- confidence: float from 0 to 1, your confidence in grade\n"
        "- tags: list of short descriptive adjectives/keywords, empty if nothing\n"
        "---\n"
        f"Here is a review:\n\n{text}\n"
    )


def _openai_call(prompt: str, schema: dict | None) -> str | dict:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} if schema else None,
    )
    content = resp.choices[0].message.content or ""
    if schema:
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse OpenAI JSON output: {e}")
    return content


def _ollama_call(prompt: str, schema: dict | None) -> str | dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if schema:
        payload["format"] = schema

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")

    if schema:
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse Ollama JSON output: {e}")
    return content


def completion(prompt: str, schema: dict | None = None, provider: str | None = None) -> str | dict:
    provider = _pick_llm_provider(provider)
    try:
        if provider == "openai":
            return _openai_call(prompt, schema)
        else:
            return _ollama_call(prompt, schema)
    except Exception as e:
        _log.error(f"Completion error from {provider}: {e}")
        raise


def predict(text: str, provider: str | None = None):
    prompt = build_classification_prompt(text)
    response = completion(prompt, CLASSIFICATION_SCHEMA, provider)
    return (
        int(response.get("grade", 0)),
        float(response.get("confidence", 0.0)),
        response.get("tags", []),
    )
