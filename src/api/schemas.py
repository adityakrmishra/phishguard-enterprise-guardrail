# Pydantic v2 request and response schema definitions for the PhishGuard analyze endpoint.

from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The URL, email body, or message content to analyse.",
        examples=["Click here to claim your prize: bit.ly/win-now"],
    )


class TransactionResponse(BaseModel):
    verdict: str = Field(
        ...,
        description="One of: SAFE | KNOWN_SCAM | ANOMALY_NEEDS_LLM",
    )
    distance: float = Field(
        ...,
        description="L2 distance to the nearest indexed template (lower = more similar).",
    )
    matched_template: str = Field(
        ...,
        description="The closest template found in the FAISS index.",
    )
    fraud_intent: Optional[str] = Field(
        default=None,
        description="Short fraud category label. Populated only for KNOWN_SCAM verdicts.",
    )
    compliance_reasoning: Optional[str] = Field(
        default=None,
        description="One-sentence compliance explanation. Populated only for KNOWN_SCAM verdicts.",
    )
