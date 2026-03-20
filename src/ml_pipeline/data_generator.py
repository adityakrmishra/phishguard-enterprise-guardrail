"""
src/ml_pipeline/data_generator.py
-----------------------------------
Synthetic Adversarial Dataset Generator
----------------------------------------
Generates polymorphic phishing messages targeting Indian fintech apps and
bank employees using an OpenAI-compatible API (OpenAI, Groq, Together AI).

Output: data/synthetic/red_team_phishing.csv
Columns: text | label | psychological_trigger

Usage:
    python -m src.ml_pipeline.data_generator

Environment variables (via .env):
    OPENAI_API_KEY    – your API key
    OPENAI_BASE_URL   – (optional) provider base URL, e.g. https://api.groq.com/openai/v1
    OPENAI_MODEL      – (optional) model name, default: gpt-4o-mini
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env from the project root

API_KEY  = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")          # None → defaults to OpenAI
MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OUTPUT_PATH = Path("data/synthetic/red_team_phishing.csv")
TOTAL_SAMPLES = 50
BATCH_SIZE    = 10   # messages to request per API call

# ---------------------------------------------------------------------------
# Psychological trigger categories
# ---------------------------------------------------------------------------
TRIGGERS: list[dict] = [
    {
        "name": "Forced Urgency",
        "description": (
            "Extreme time pressure – the victim must act within minutes or "
            "face irreversible consequences (account blocked, funds lost, legal action)."
        ),
    },
    {
        "name": "Authority Impersonation",
        "description": (
            "Impersonates a high-authority entity: RBI, NPCI, Income Tax Department, "
            "CBI, or a bank's CISO / CFO sending an internal memo to employees."
        ),
    },
    {
        "name": "Fear of Financial Loss",
        "description": (
            "Implies that the victim's savings, investments, or salary will be frozen "
            "or confiscated unless they take immediate action via a link or OTP."
        ),
    },
    {
        "name": "Reward / FOMO",
        "description": (
            "Promises an exclusive reward – cashback, UPI bonus, IPO allotment, "
            "or a limited-time interest rate – to exploit Fear Of Missing Out."
        ),
    },
    {
        "name": "Vendor / Insider Trust",
        "description": (
            "Targets bank employees by posing as an internal IT team, auditor, "
            "or trusted vendor requesting credentials or approval of a payment."
        ),
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an elite Red Team social engineer and security researcher.
Your task is to generate SYNTHETIC, LABELLED phishing messages for ML training purposes only.
These messages WILL NOT be sent to real people – they are adversarial examples for a
phishing detection model.

Target profile: Indian fintech apps (Paytm, PhonePe, GPay, CRED, Zepto) and
private-sector bank employees (HDFC, ICICI, Axis, Kotak).

Writing rules:
- Perfect grammar, no spelling mistakes (unlike real phishing).
- Sound like an official communication (formal tone, correct branding).
- Embed a plausible but fake action URL or phone number (e.g., secure-hdfc-login.in/verify).
- Use Indian cultural context: INR amounts, UPI IDs, Aadhaar, PAN, ITR, NACH mandates.
- Each message must be 2–4 sentences. Do NOT include a subject line.
- Vary sentence structure so no two messages sound the same (polymorphic).
"""

# ---------------------------------------------------------------------------
# Helper: build user prompt for one batch
# ---------------------------------------------------------------------------

def _build_user_prompt(trigger: dict, n: int) -> str:
    return f"""Generate exactly {n} distinct, polymorphic phishing messages.

Psychological trigger:  {trigger['name']}
Trigger description:    {trigger['description']}

Return ONLY a valid JSON array of {n} strings – no extra keys, no markdown fences.
Example format:
["Message one here.", "Message two here."]
"""


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_dataset(total: int = TOTAL_SAMPLES) -> pd.DataFrame:
    if not API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file in the project root with OPENAI_API_KEY=<your-key>. "
            "For Groq: also set OPENAI_BASE_URL=https://api.groq.com/openai/v1"
        )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Distribute samples across triggers as evenly as possible
    per_trigger = total // len(TRIGGERS)
    remainder   = total % len(TRIGGERS)

    records: list[dict] = []

    for i, trigger in enumerate(TRIGGERS):
        n_for_trigger = per_trigger + (1 if i < remainder else 0)
        logger.info(
            "Generating %d samples for trigger: '%s'", n_for_trigger, trigger["name"]
        )

        collected: list[str] = []
        attempts = 0

        while len(collected) < n_for_trigger and attempts < 5:
            need = n_for_trigger - len(collected)
            batch = min(need, BATCH_SIZE)

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": _build_user_prompt(trigger, batch)},
                    ],
                    temperature=1.0,   # high temperature → more diversity
                    max_tokens=2048,
                )

                raw = response.choices[0].message.content.strip()

                # Strip optional markdown fences
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                messages: list[str] = json.loads(raw)
                if not isinstance(messages, list):
                    raise ValueError("Model did not return a JSON array.")

                collected.extend(m.strip() for m in messages if isinstance(m, str) and m.strip())
                logger.info(
                    "  ✓ Collected %d/%d for '%s'",
                    len(collected), n_for_trigger, trigger["name"],
                )

            except (RateLimitError,) as exc:
                wait = 20
                logger.warning("Rate-limited. Waiting %ds… (%s)", wait, exc)
                time.sleep(wait)

            except (APIConnectionError, APIError) as exc:
                logger.error("API error: %s. Retrying in 5s…", exc)
                time.sleep(5)

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Parse error – retrying batch. (%s)", exc)

            attempts += 1

        # Trim to exact count if we over-collected
        for text in collected[:n_for_trigger]:
            records.append(
                {
                    "text":                   text,
                    "label":                  "scam",
                    "psychological_trigger":  trigger["name"],
                }
            )

    df = pd.DataFrame(records)
    logger.info("Dataset complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved %d rows → %s", len(df), path.resolve())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== PhishGuard Red-Team Dataset Generator ===")
    logger.info("Model: %s | Provider: %s", MODEL, BASE_URL or "OpenAI (default)")
    logger.info("Target samples: %d | Batch size: %d", TOTAL_SAMPLES, BATCH_SIZE)

    df = generate_dataset(TOTAL_SAMPLES)

    if df.empty:
        logger.error("No data was generated. Check your API key / network and retry.")
    else:
        save_dataset(df)
        print("\n--- Preview (first 5 rows) ---")
        print(df.head().to_string(index=False))
        print(f"\nTrigger distribution:\n{df['psychological_trigger'].value_counts().to_string()}")
