# LLM-powered audit log generator that returns fraud intent and compliance reasoning for flagged phishing messages.

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import APIError, APIConnectionError, OpenAI, RateLimitError

load_dotenv()
logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - audit log will use fallback values.")
        return None
    _client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    return _client


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
    "You must respond ONLY with a valid JSON object - no markdown, no extra keys. "
    "The JSON must have exactly two keys:\n"
    '  "fraud_intent": one label from this list: '
    + str(_INTENT_CATEGORIES) + "\n"
    '  "compliance_reasoning": a single, concise sentence explaining why this '
    "message violates RBI / NPCI safety protocols or standard AML guidelines."
)


def generate_audit_log(text: str) -> dict[str, str]:
    _fallback = {
        "fraud_intent": "Unknown",
        "compliance_reasoning": "Automated audit unavailable - LLM service unreachable.",
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
            temperature=0.2,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)

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
