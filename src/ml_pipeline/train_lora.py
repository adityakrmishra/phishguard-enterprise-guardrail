# LoRA fine-tuning pipeline for DistilRoBERTa binary phishing classifier, saving adapter weights to models/distilroberta-finetuned/.

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_MODEL   = "distilroberta-base"
OUTPUT_DIR   = Path("models/distilroberta-finetuned")
DATA_PATH    = Path("data/synthetic/red_team_phishing.csv")
SEED         = 42
MAX_LEN      = 128
TEST_SIZE    = 0.15

LORA_R           = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.1
LORA_TARGET_MODS = ["query", "value"]

TRAIN_EPOCHS   = 3
TRAIN_BATCH    = 8
EVAL_BATCH     = 16
LR             = 2e-4
WEIGHT_DECAY   = 0.01

set_seed(SEED)

_SAFE_EXAMPLES: list[str] = [
    "Your account statement for March 2024 is now available online.",
    "Please log in to your banking portal to review your recent transactions.",
    "Your scheduled payment of Rs.5,000 to Rahul Sharma has been processed.",
    "We have updated our privacy policy. Please review the changes at your convenience.",
    "Your new debit card ending in 4321 has been dispatched and will arrive in 3-5 days.",
    "Monthly interest of Rs.312 has been credited to your savings account.",
    "Your Fixed Deposit of Rs.1,00,000 has been renewed for 12 months at 6.75% p.a.",
    "Your UPI payment of Rs.850 to Swiggy was successful. Ref: 402938471.",
    "Dear Customer, your net banking password will expire in 30 days. Please renew it.",
    "Your NACH mandate for Rs.2,500/month has been registered successfully.",
    "Your ITR for AY 2023-24 has been processed and a refund of Rs.3,200 is initiated.",
    "You have successfully logged in to PhonePe from a new device.",
    "Your home loan EMI of Rs.22,400 has been auto-debited for April.",
    "A cheque of Rs.15,000 issued by you has been cleared.",
    "Your KYC documents have been verified. Your account is now fully active.",
    "Congratulations! Your credit card limit has been increased to Rs.3,00,000.",
    "Your CRED coins balance is Rs.1,420. Redeem them before 31st March.",
    "Your Mutual Fund SIP of Rs.5,000 in Axis Bluechip Fund has been processed.",
    "Your nominee details have been updated successfully.",
    "Your two-factor authentication has been enabled on your account.",
]


def load_dataframe() -> pd.DataFrame:
    scam_df = pd.DataFrame()

    if DATA_PATH.exists():
        scam_df = pd.read_csv(DATA_PATH, usecols=["text"])
        scam_df["label"] = 1
        logger.info("Loaded %d scam examples from %s.", len(scam_df), DATA_PATH)
    else:
        logger.warning("%s not found - training with only placeholder scam examples.", DATA_PATH)
        placeholder_scam = [
            "URGENT: Your account has been suspended. Click here to verify now.",
            "Dear Customer, share your OTP 748291 with our agent immediately.",
            "Congratulations! You've won Rs.5,000 UPI cashback. Claim at: bit.ly/claim",
        ]
        scam_df = pd.DataFrame({"text": placeholder_scam, "label": 1})

    safe_df = pd.DataFrame({"text": _SAFE_EXAMPLES, "label": 0})

    df = pd.concat([scam_df[["text", "label"]], safe_df], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    logger.info(
        "Dataset: %d total | %d scam | %d safe",
        len(df), (df["label"] == 1).sum(), (df["label"] == 0).sum(),
    )
    return df


def build_hf_dataset(df: pd.DataFrame, tokeniser) -> DatasetDict:
    train_df, eval_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )

    def tokenise(batch):
        return tokeniser(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    eval_ds  = Dataset.from_pandas(eval_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(tokenise, batched=True, remove_columns=["text"])
    eval_ds  = eval_ds.map(tokenise,  batched=True, remove_columns=["text"])

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy  = float((preds == labels).mean())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train() -> None:
    logger.info("PhishGuard LoRA Fine-Tuning")
    logger.info("Base model : %s", BASE_MODEL)
    logger.info("Output dir : %s", OUTPUT_DIR.resolve())

    tokeniser = AutoTokenizer.from_pretrained(BASE_MODEL)

    df = load_dataframe()
    dataset = build_hf_dataset(df, tokeniser)
    logger.info("Train split: %d | Eval split: %d", len(dataset["train"]), len(dataset["eval"]))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "SAFE", 1: "SCAM"},
        label2id={"SAFE": 0, "SCAM": 1},
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODS,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokeniser,
        data_collator=DataCollatorWithPadding(tokeniser),
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training.")
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR))
    tokeniser.save_pretrained(str(OUTPUT_DIR))
    logger.info("LoRA adapter + tokeniser saved to %s", OUTPUT_DIR.resolve())

    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)


def smoke_test() -> None:
    logger.info("Smoke Test")

    if not OUTPUT_DIR.exists():
        logger.error("Model directory not found. Run train() first.")
        return

    tokeniser = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
    )
    model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))
    model.eval()

    samples = [
        ("Your account statement for March 2024 is now available.", 0),
        ("URGENT: Your UPI account is blocked. Share OTP 928374 to restore access.", 1),
    ]

    id2label = {0: "SAFE", 1: "SCAM"}

    for text, expected in samples:
        inputs = tokeniser(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = int(logits.argmax(-1).item())
        confidence = float(torch.softmax(logits, dim=-1).max().item())

        status = "PASS" if pred_id == expected else "FAIL"
        logger.info(
            "%s | Predicted: %-6s (%.1f%%) | Expected: %s | Text: %.60s",
            status,
            id2label[pred_id],
            confidence * 100,
            id2label[expected],
            text,
        )


if __name__ == "__main__":
    train()
    smoke_test()
