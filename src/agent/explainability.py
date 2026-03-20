"""
src/agent/explainability.py
----------------------------
LLM-powered audit log generator for flagged phishing content.

generate_audit_log(text) calls an OpenAI-compatible API (Groq / Together AI /
OpenAI) and returns a dict with:
  - fraud_intent        : short label, e.g. "Authority Impersonation"
  - compliance_reasoning: 1-sentence policy explanation

The function is intentionally fast-path safe: if the LLM call fails for any
reason it returns graceful fallback strings instead of raising, so the rest of
the API response is never blocked.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import APIError, APIConnectionError, OpenAI, RateLimitError

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-compatible client (works with Groq, Together AI, OpenAI)
# ---------------------------------------------------------------------------
_client: OpenAI | None = None

def _get_client() -> OpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set – audit log will use fallback values.")
        return None
    _client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),  # None → OpenAI default
    )
    return _client


# ---------------------------------------------------------------------------
# Known fraud intent categories (helps the model stay consistent)
# ---------------------------------------------------------------------------
_INTENT_CATEGORIES = [
    "Authority Impersonation",
    "Forced Urgency",
    "Fear of Financial Loss",
    "Reward / FOMO",
    "Vendor / Insider Trust",
    "Credential Harvesting",
    "Malicious Link",
    "OTP Phishing",
    "Unknown",
]

_SYSTEM_PROMPT = (
    "You are a compliance analyst at a financial-crime intelligence unit. "
    "Your job is to analyse flagged messages and produce a strict audit report. "
    "You must respond ONLY with a valid JSON object — no markdown, no extra keys. "
    "The JSON must have exactly two keys:\n"
    '  "fraud_intent": one label from this list: '
    + str(_INTENT_CATEGORIES) + "\n"
    '  "compliance_reasoning": a single, concise sentence explaining why this '
    "message violates RBI / NPCI safety protocols or standard AML guidelines."
)


def generate_audit_log(text: str) -> dict[str, str]:
    """
    Call the LLM to produce a structured audit report for a flagged message.

    Parameters
    ----------
    text : str
        The phishing message that was flagged as KNOWN_SCAM.

    Returns
    -------
    dict with keys:
        fraud_intent        : str  – short category label
        compliance_reasoning: str  – 1-sentence policy explanation
    """
    _fallback = {
        "fraud_intent": "Unknown",
        "compliance_reasoning": "Automated audit unavailable – LLM service unreachable.",
    }

    client = _get_client()
    if client is None:
        return _fallback

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    user_prompt = (
        f"Flagged message:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Return the JSON audit report now."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,   # low temperature → consistent, factual output
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Strip optional markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)

        # Validate expected keys are present
        fraud_intent = str(parsed.get("fraud_intent", "Unknown")).strip()
        reasoning    = str(parsed.get("compliance_reasoning", "")).strip()

        if not reasoning:
            reasoning = "No reasoning provided by the model."

        logger.info("Audit log generated | intent=%s", fraud_intent)
        return {"fraud_intent": fraud_intent, "compliance_reasoning": reasoning}

    except (RateLimitError, APIConnectionError, APIError) as exc:
        logger.warning("LLM API error during audit log generation: %s", exc)
        return _fallback
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse audit log JSON: %s", exc)
        return _fallback
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error in generate_audit_log: %s", exc, exc_info=True)
        return _fallback
