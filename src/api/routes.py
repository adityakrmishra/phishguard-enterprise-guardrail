# Two-stage phishing triage router: FAISS fast-path then DistilRoBERTa LoRA, with LLM audit logging for scam verdicts.

import logging

from fastapi import APIRouter, HTTPException, status

from src.agent.explainability import generate_audit_log
from src.api.schemas import TransactionRequest, TransactionResponse
from src.dsa_router.vector_triage import LABEL_ANOMALY, LABEL_KNOWN_SCAM, SemanticRouter
from src.ml_pipeline.inference import IntentClassifier

logger = logging.getLogger(__name__)

try:
    _semantic_router = SemanticRouter()
    logger.info("SemanticRouter loaded successfully.")
except Exception as exc:
    logger.critical("Failed to initialise SemanticRouter: %s", exc, exc_info=True)
    _semantic_router = None  # type: ignore[assignment]

try:
    _intent_classifier = IntentClassifier()
    logger.info("IntentClassifier loaded successfully.")
except FileNotFoundError as exc:
    logger.warning("IntentClassifier not available (%s).", exc)
    _intent_classifier = None
except Exception as exc:
    logger.error("IntentClassifier failed to load: %s", exc, exc_info=True)
    _intent_classifier = None

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=TransactionResponse,
    summary="Analyse text for phishing",
    status_code=status.HTTP_200_OK,
)
async def analyze(request: TransactionRequest) -> TransactionResponse:
    if _semantic_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SemanticRouter is unavailable. Check server logs.",
        )

    try:
        verdict, distance, matched_template = _semantic_router.triage(request.text)
    except Exception as exc:
        logger.error("Stage-1 triage failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during semantic triage.",
        ) from exc

    if verdict == LABEL_ANOMALY:
        if _intent_classifier is not None:
            try:
                llm_verdict, confidence = _intent_classifier.predict(request.text)
                verdict = llm_verdict
                matched_template = (
                    f"{matched_template} "
                    f"(Detected via LLM Analysis - confidence {confidence*100:.1f}%)"
                )
                logger.info("Stage-2 resolved ANOMALY -> %s (confidence=%.1f%%)", verdict, confidence * 100)
            except Exception as exc:
                logger.error("Stage-2 classification failed: %s", exc, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An error occurred during LLM analysis.",
                ) from exc
        else:
            logger.warning("IntentClassifier unavailable; returning un-resolved ANOMALY.")

    fraud_intent: str | None = None
    compliance_reasoning: str | None = None

    if verdict == LABEL_KNOWN_SCAM:
        try:
            audit = generate_audit_log(request.text)
            fraud_intent         = audit.get("fraud_intent")
            compliance_reasoning = audit.get("compliance_reasoning")
            logger.info("Audit log attached | intent=%s", fraud_intent)
        except Exception as exc:
            logger.error("Audit log generation error: %s", exc, exc_info=True)

    return TransactionResponse(
        verdict=verdict,
        distance=round(distance, 6),
        matched_template=matched_template,
        fraud_intent=fraud_intent,
        compliance_reasoning=compliance_reasoning,
    )
