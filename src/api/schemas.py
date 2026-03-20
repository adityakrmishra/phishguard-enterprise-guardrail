"""
src/api/schemas.py
------------------
Pydantic v2 request/response models for the PhishGuard API.
"""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Payload sent by the client for phishing analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The URL, email body, or message content to analyse.",
        examples=["Click here to claim your prize: bit.ly/win-now"],
    )


class TransactionResponse(BaseModel):
    """Result returned by the /api/v1/analyze endpoint."""

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
