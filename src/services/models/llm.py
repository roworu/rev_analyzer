import os
import json
import logging
import requests

_log = logging.getLogger(__name__)

# ollama settings
OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:4b")

# llm predict response format
CLASSIFICATION_FORMAT = {
    "type": "object",
    "properties": {
        "grade": {"type": "integer"},
        "confidence": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grade", "confidence"],
}


def build_classification_prompt(text: str) -> str:
    return (
        # 1) High-level role + task
        "You are an assistant that analyzes product reviews.\n"
        "Your task is to return a SINGLE valid JSON object.\n\n"

        # 2) Output format dexcription
        "Output format (JSON only, no extra text):\n"
        "{\n"
        '  "grade": integer (1–10),\n'
        '  "confidence": float (0.0–1.0),\n'
        '  "tags": array of short descriptive keywords\n'
        "}\n\n"

        # 3) Field semantics
        "Field definitions:\n"
        "- grade: overall user satisfaction (1 = very bad, 10 = excellent)\n"
        "- confidence: how confident you are in the grade based on the review clarity\n"
        "- tags: short adjectives or keywords; empty array if nothing stands out\n\n"

        # 4) Examples
        "Examples:\n\n"

        "Review:\n"
        "The product arrived late and the quality is terrible. Completely disappointed.\n"
        "Output:\n"
        "{\n"
        '  "grade": 2,\n'
        '  "confidence": 0.85,\n'
        '  "tags": ["late delivery", "poor quality", "disappointed"]\n'
        "}\n\n"

        "Review:\n"
        "Works fine.\n"
        "Output:\n"
        "{\n"
        '  "grade": 6,\n'
        '  "confidence": 0.4,\n'
        '  "tags": []\n'
        "}\n\n"

        # 5) Real input
        "Now analyze the following review:\n"
        "------------------------------\n"
        f"{text}\n"
    )


def ollama_call(prompt: str, format: dict | None) -> str | dict:
    
    try:
        # 1) check, that requessted model exists:
        avaliable_models = requests.post(f"{OLLAMA_URL}/api/tags")
        
        
        
        payload = {
            "model": DEFAULT_OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if format:
            payload["format"] = format
    
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
    
        if format:
            try:
                return json.loads(content)
            except Exception as e:
                raise ValueError(f"Failed to parse Ollama JSON output: {e}")

        return content
        
    except Exception as e:
        _log.exception(f"Unexpected error during Ollama call: {str(e)}")
        

def completion(prompt: str, format: dict | None = None) -> str | dict:
    # possibly implement here other providers as part of general `completion` function
    # currently - completions only handled by Ollama
    return ollama_call(prompt, format)


def predict(text: str):
    prompt = build_classification_prompt(text)
    response = completion(prompt, CLASSIFICATION_FORMAT)
    return (
        int(response.get("grade", 0)),
        float(response.get("confidence", 0.0)),
        response.get("tags", []),
    )
