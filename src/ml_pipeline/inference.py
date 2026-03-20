# DistilRoBERTa LoRA adapter loader and single-text phishing inference class.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

ADAPTER_DIR  = Path("models/distilroberta-advanced")
BASE_MODEL   = "distilroberta-base"
MAX_LEN      = 128
ID2LABEL     = {0: "SAFE", 1: "KNOWN_SCAM"}


class IntentClassifier:
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
        inputs = self._tokeniser(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs      = torch.softmax(logits, dim=-1).squeeze()
        pred_id    = int(probs.argmax().item())
        confidence = float(probs[pred_id].item())
        verdict    = ID2LABEL[pred_id]

        logger.debug(
            "IntentClassifier | verdict=%s | confidence=%.3f | text=%.60s",
            verdict, confidence, text,
        )
        return verdict, confidence


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
