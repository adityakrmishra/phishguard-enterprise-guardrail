"""
src/api/main.py
---------------
FastAPI application entry point for PhishGuard Enterprise Guardrail.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api import routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PhishGuard Enterprise Guardrail API",
    description=(
        "AI-powered phishing detection pipeline using semantic triage "
        "(FAISS nearest-neighbour) with optional LLM escalation."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Streamlit frontend (port 8501) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(routes.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Liveness probe – confirms the API process is running."""
    return HealthResponse(status="ok", message="PhishGuard API is up and running.")


# ---------------------------------------------------------------------------
# Dev entrypoint:  python -m src.api.main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
