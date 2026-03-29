# FAISS-backed semantic router that classifies text as SAFE, KNOWN_SCAM, or ANOMALY_NEEDS_LLM using sentence-transformer embeddings.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

LABEL_SAFE = "SAFE"
LABEL_KNOWN_SCAM = "KNOWN_SCAM"
LABEL_ANOMALY = "ANOMALY_NEEDS_LLM"

_SCAM_TEMPLATES: List[str] = [
    "URGENT: Your account has been suspended. Click here immediately to verify your details.",
    "Congratulations! You have been selected for a $5,000 wire transfer. Confirm your bank info now.",
    "Dear customer, your OTP is 748291. Share this with our agent to complete verification.",
    "Your Netflix subscription has expired. Update payment details to avoid service disruption: bit.ly/upd8",
    "IRS ALERT: Unpaid taxes detected. Call immediately or face arrest within 24 hours.",
    "Your PayPal account is limited. Verify now: www.paypa1-secure-login.com",
    "You have won a lottery prize. Send your SSN and bank details to claim your winnings.",
]


@dataclass
class _IndexEntry:
    text: str
    label: str


class SemanticRouter:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
    ) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()

        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self._dim)
        if use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

        self._entries: List[_IndexEntry] = []

        self.add_templates(_SCAM_TEMPLATES, LABEL_KNOWN_SCAM)
        logger.info(
            "SemanticRouter ready. Index size: %d vectors (%d-dim).",
            self._index.ntotal,
            self._dim,
        )

    def add_templates(self, texts: List[str], label: str) -> None:
        if not texts:
            return

        embeddings = self._embed(texts)
        self._index.add(embeddings)
        for text in texts:
            self._entries.append(_IndexEntry(text=text, label=label))

        logger.debug("Added %d '%s' templates. Total: %d.", len(texts), label, self._index.ntotal)

    def triage(
        self,
        text: str,
        scam_threshold: float = 0.75,
    ) -> Tuple[str, float, str]:
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty - returning ANOMALY.")
            return LABEL_ANOMALY, float("inf"), ""

        query = self._embed([text])
        distances, indices = self._index.search(query, k=1)

        distance: float = float(distances[0][0])
        nearest_idx: int = int(indices[0][0])
        nearest_entry = self._entries[nearest_idx]

        # Verdict is driven by the matched template's label and distance.
        # The index contains ONLY scam templates, so a close match means
        # KNOWN_SCAM; a distant match means the text is unlike any known
        # scam and should be escalated to the LLM for deeper review.
        if distance <= scam_threshold and nearest_entry.label == LABEL_KNOWN_SCAM:
            verdict = LABEL_KNOWN_SCAM
        else:
            verdict = LABEL_ANOMALY

        logger.debug(
            "triage | verdict=%s | dist=%.4f | nearest='%s'",
            verdict, distance, nearest_entry.text[:60],
        )
        return verdict, distance, nearest_entry.text

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    router = SemanticRouter()

    samples = [
        "Your account statement is ready for download.",
        "Click here NOW to unlock your suspended bank account!",
        "The quarterly GDP figures suggest a 3.2% contraction.",
    ]

    print("\n--- SemanticRouter Smoke Test ---")
    for sample in samples:
        verdict, dist, nearest = router.triage(sample)
        print(f"\nInput   : {sample}")
        print(f"Verdict : {verdict}  (L2={dist:.4f})")
        print(f"Nearest : {nearest[:80]}")
