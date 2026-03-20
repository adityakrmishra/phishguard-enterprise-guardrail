"""
src/ml_pipeline/inference.py
-----------------------------
IntentClassifier – loads the fine-tuned DistilRoBERTa LoRA adapter
and provides single-text inference for the FastAPI 'heavy brain' path.

Usage (standalone):
    python -m src.ml_pipeline.inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

ADAPTER_DIR  = Path("models/distilroberta-finetuned")
BASE_MODEL   = "distilroberta-base"
MAX_LEN      = 128
ID2LABEL     = {0: "SAFE", 1: "KNOWN_SCAM"}


class IntentClassifier:
    """
    Wraps the fine-tuned LoRA adapter for single-text phishing classification.

    Parameters
    ----------
    adapter_dir : Path
        Directory containing the saved PEFT adapter weights and tokeniser.
        Falls back to BASE_MODEL weights only if the adapter is not found.
    """

    def __init__(self, adapter_dir: Path = ADAPTER_DIR) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("IntentClassifier device: %s", self._device)

        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {adapter_dir.resolve()}\n"
                "Run `python -m src.ml_pipeline.train_lora` first."
            )

        logger.info("Loading tokeniser from %s", adapter_dir)
        self._tokeniser = AutoTokenizer.from_pretrained(str(adapter_dir))

        logger.info("Loading base model: %s", BASE_MODEL)
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=2,
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        )

        logger.info("Attaching LoRA adapter from %s", adapter_dir)
        self._model = PeftModel.from_pretrained(base, str(adapter_dir))
        self._model.to(self._device)
        self._model.eval()
        logger.info("IntentClassifier ready.")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Classify a single text string.

        Parameters
        ----------
        text : str
            The input message or URL to classify.

        Returns
        -------
        (verdict, confidence)
            verdict    : "SAFE" or "KNOWN_SCAM"
            confidence : softmax probability of the predicted class [0.0 – 1.0]
        """
        inputs = self._tokeniser(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits          # shape: (1, 2)

        probs      = torch.softmax(logits, dim=-1).squeeze()   # shape: (2,)
        pred_id    = int(probs.argmax().item())
        confidence = float(probs[pred_id].item())
        verdict    = ID2LABEL[pred_id]

        logger.debug(
            "IntentClassifier | verdict=%s | confidence=%.3f | text=%.60s",
            verdict, confidence, text,
        )
        return verdict, confidence


# ---------------------------------------------------------------------------
# Smoke test – python -m src.ml_pipeline.inference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    clf = IntentClassifier()
    samples = [
        "Your account statement for March 2024 is now available.",
        "URGENT: Your UPI account is blocked. Share OTP 928374 to restore access immediately.",
        "The quarterly GDP figures indicate a potential 3.2% economic contraction.",
    ]
    print("\n--- IntentClassifier Smoke Test ---")
    for s in samples:
        verdict, conf = clf.predict(s)
        print(f"[{verdict:10s}] ({conf*100:5.1f}%)  {s[:70]}")
