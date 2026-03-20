"""
src/api/routes.py
-----------------
FastAPI router for PhishGuard.

The SemanticRouter is instantiated once at module import time so the
embedding model and FAISS index are loaded into memory only once,
regardless of how many concurrent requests arrive.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import TransactionRequest, TransactionResponse
from src.dsa_router.vector_triage import SemanticRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single shared instance – loaded once when the module is first imported.
# ---------------------------------------------------------------------------
try:
    _semantic_router = SemanticRouter()
    logger.info("SemanticRouter loaded successfully.")
except Exception as exc:  # pragma: no cover
    logger.critical("Failed to initialise SemanticRouter: %s", exc, exc_info=True)
    _semantic_router = None  # type: ignore[assignment]

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=TransactionResponse,
    summary="Analyse text for phishing",
    status_code=status.HTTP_200_OK,
)
async def analyze(request: TransactionRequest) -> TransactionResponse:
    """
    Accepts a text payload and returns a phishing verdict via semantic triage.

    - **SAFE** – semantically close to known-good banking communications.
    - **KNOWN_SCAM** – semantically close to catalogued phishing templates.
    - **ANOMALY_NEEDS_LLM** – too dissimilar for fast-path routing; requires LLM review.
    """
    if _semantic_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SemanticRouter is unavailable. Check server logs.",
        )

    try:
        verdict, distance, matched_template = _semantic_router.triage(request.text)
    except Exception as exc:
        logger.error("Triage failed for input '%s…': %s", request.text[:40], exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during analysis. Please try again.",
        ) from exc

    return TransactionResponse(
        verdict=verdict,
        distance=round(distance, 6),
        matched_template=matched_template,
    )
