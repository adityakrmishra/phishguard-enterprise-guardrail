# Advanced multi-dataset LoRA fine-tuning pipeline combining UCI SMS Spam, mshenoda spam-messages, and local synthetic data.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Value, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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

BASE_MODEL    = "distilroberta-base"
OUTPUT_DIR    = Path("models/distilroberta-advanced")
SYNTHETIC_CSV = Path("data/synthetic/red_team_phishing.csv")
SEED          = 42
MAX_LEN       = 128
TEST_SIZE     = 0.15

LORA_R           = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.1
LORA_TARGET_MODS = ["query", "value"]

TRAIN_EPOCHS = 2
TRAIN_BATCH  = 16
EVAL_BATCH   = 32
LR           = 2e-4
WEIGHT_DECAY = 0.01

set_seed(SEED)


def load_uci_sms() -> Dataset:
    logger.info("Loading ucirvine/sms_spam ...")
    raw = load_dataset("ucirvine/sms_spam", split="train", trust_remote_code=True)
    logger.info("ucirvine/sms_spam: %d rows | columns: %s", len(raw), raw.column_names)

    def normalise(batch):
        label_col = "label" if "label" in raw.column_names else raw.column_names[1]
        text_col  = "sms"   if "sms"   in raw.column_names else raw.column_names[0]
        labels = []
        for v in batch[label_col]:
            if isinstance(v, int):
                labels.append(v)
            elif isinstance(v, str):
                labels.append(0 if v.strip().lower() == "ham" else 1)
            else:
                labels.append(int(v))
        return {"text": batch[text_col], "label": labels}

    ds = raw.map(normalise, batched=True, remove_columns=raw.column_names)
    logger.info("ucirvine/sms_spam normalised: %d rows", len(ds))
    return ds


def load_mshenoda_spam() -> Dataset:
    logger.info("Loading mshenoda/spam-messages ...")
    raw = load_dataset("mshenoda/spam-messages", split="train", trust_remote_code=True)
    logger.info("mshenoda/spam-messages: %d rows | columns: %s", len(raw), raw.column_names)

    text_col  = "text"  if "text"  in raw.column_names else raw.column_names[0]
    label_col = "label" if "label" in raw.column_names else raw.column_names[1]

    def normalise(batch):
        labels = []
        for v in batch[label_col]:
            if isinstance(v, int):
                labels.append(v)
            elif isinstance(v, str):
                labels.append(0 if v.strip().lower() in ("ham", "0", "safe") else 1)
            else:
                labels.append(int(v))
        return {"text": batch[text_col], "label": labels}

    ds = raw.map(normalise, batched=True, remove_columns=raw.column_names)
    logger.info("mshenoda/spam-messages normalised: %d rows", len(ds))
    return ds


def load_synthetic() -> Dataset:
    if not SYNTHETIC_CSV.exists():
        logger.warning("Synthetic CSV not found at %s - skipping.", SYNTHETIC_CSV)
        return Dataset.from_dict({"text": [], "label": []})

    df = pd.read_csv(SYNTHETIC_CSV)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    label_map = {"scam": 1, "spam": 1, "ham": 0, "safe": 0}
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().map(label_map).fillna(1).astype(int)
    else:
        df["label"] = 1

    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    logger.info("Synthetic dataset: %d rows", len(ds))
    return ds


def build_splits(combined: Dataset, tokeniser) -> tuple[Dataset, Dataset]:
    df = combined.to_pandas()
    train_df, eval_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )

    def tokenise(batch):
        return tokeniser(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    eval_ds  = Dataset.from_pandas(eval_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(tokenise, batched=True, remove_columns=["text"])
    eval_ds  = eval_ds.map(tokenise,  batched=True, remove_columns=["text"])
    return train_ds, eval_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy  = float((preds == labels).mean())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train() -> None:
    logger.info("PhishGuard Advanced LoRA Training")
    logger.info("Base model : %s", BASE_MODEL)
    logger.info("Output dir : %s", OUTPUT_DIR.resolve())

    ds_uci       = load_uci_sms()
    ds_mshenoda  = load_mshenoda_spam()
    ds_synthetic = load_synthetic()

    ds_uci       = ds_uci.cast_column("label", Value("int64"))
    ds_mshenoda  = ds_mshenoda.cast_column("label", Value("int64"))
    ds_synthetic = ds_synthetic.cast_column("label", Value("int64"))

    combined = concatenate_datasets([ds_uci, ds_mshenoda, ds_synthetic])
    combined = combined.shuffle(seed=SEED)
    logger.info(
        "Combined dataset: %d rows | label distribution: %s",
        len(combined),
        {k: int(v) for k, v in zip(*np.unique(combined["label"], return_counts=True))},
    )

    tokeniser = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds, eval_ds = build_splits(combined, tokeniser)
    logger.info("Train: %d | Eval: %d", len(train_ds), len(eval_ds))

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
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokeniser,
        data_collator=DataCollatorWithPadding(tokeniser),
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training.")
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR))
    tokeniser.save_pretrained(str(OUTPUT_DIR))
    logger.info("Advanced model saved to %s", OUTPUT_DIR.resolve())

    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)


if __name__ == "__main__":
    train()
