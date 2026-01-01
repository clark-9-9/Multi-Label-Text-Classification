#!/usr/bin/env python3
"""
main.py

Complete, single-file example for Multi-Label Text Classification using Hugging Face
Transformers Trainer API.

Key points implemented:
- Loads the CSV (raw text) from an in-memory string (io.StringIO) into pandas.
- Uses the 'ABSTRACT' column as the text input and the 6 topic columns as binary labels.
- Configures AutoModelForSequenceClassification with num_labels=6 and
  problem_type="multi_label_classification".
- Tokenization with AutoTokenizer (max_length=128, padding=True, truncation=True).
- A custom compute_metrics function (scikit-learn) that returns:
    - Macro-averaged F1-score
    - Macro-averaged ROC-AUC
- Simple 80/20 train/eval split (with safe duplication if dataset is too small).
- Trains for a small number of epochs (3) for demonstration.
- Prints final evaluation results clearly.
"""

import io
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import EvalPrediction
import math

# ----------------------------
# 1) RAW CSV (in-memory)
# ----------------------------
CSV_TEXT = """ID,TITLE,ABSTRACT,Computer Science,Physics,Mathematics,Statistics,Quantitative Biology,Quantitative Finance
1,Reconstructing Subject-Specific Effect Maps," Predictive models allow subject-specific inference when analyzing disease
related alterations in neuroimaging data. Given......",1,0,0,0,0,0
"""

# ----------------------------
# 2) Load data via pandas
# ----------------------------
# df = pd.read_csv(io.StringIO("train.csv"))
df = pd.read_csv("train.csv")

# Keep only ABSTRACT and the 6 label columns (in the requested order)
LABEL_COLS = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance",
]

if not set(LABEL_COLS).issubset(df.columns):
    raise ValueError("Expected label columns not found in the CSV data.")

# Clean abstract text column (strip whitespace)
df["text"] = df["ABSTRACT"].astype(str).str.strip()

# Convert labels to 0/1 ints (ensure numeric)
for c in LABEL_COLS:
    df[c] = df[c].astype(int).clip(0, 1)

# For demonstration: If dataset is extremely small (like 1 row),
# duplicate it a few times so training/evaluation can run meaningfully.
# (This preserves the original content while allowing a train/eval split.)
MIN_ROWS_REQUIRED = 4
if len(df) < MIN_ROWS_REQUIRED:
    times = math.ceil(MIN_ROWS_REQUIRED / len(df))
    df = pd.concat([df] * times, ignore_index=True)
    df = df.reset_index(drop=True)

# Build label vectors column
df["labels"] = df[LABEL_COLS].values.tolist()

# Keep only needed columns for datasets
df_for_ds = df[["text", "labels"]].copy()

# ----------------------------
# 3) Train / Eval split (80/20)
# ----------------------------
train_df, eval_df = train_test_split(df_for_ds, test_size=0.2, random_state=42, shuffle=True)

# Convert to Hugging Face datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))
dataset = DatasetDict({"train": train_ds, "validation": eval_ds})

# ----------------------------
# 4) Tokenizer & Model setup
# ----------------------------
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = len(LABEL_COLS)
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def preprocess_function(examples: Dict[str, List[Any]]):
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        padding=True,  # will be handled by data collator too
        truncation=True,
        max_length=MAX_LENGTH,
    )
    # Keep labels (list of lists) as they are
    tokenized["labels"] = examples["labels"]
    return tokenized

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Convert label lists to float tensors expected for multi-label
# (Trainer will collate them into tensors)
def convert_labels_to_float(example):
    example["labels"] = [float(x) for x in example["labels"]]
    return example

tokenized_datasets = tokenized_datasets.map(convert_labels_to_float, batched=False)

# Data collator (will pad to longest in batch)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model with problem_type set for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

# ----------------------------
# 5) Metrics
# ----------------------------
import numpy as np
import math
from scipy.special import expit  # sigmoid

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    p.predictions: logits (shape: batch_size x num_labels)
    p.label_ids: ground-truth labels (shape: batch_size x num_labels)
    We compute:
     - Macro-averaged F1 (multi-label): average the F1 per label
     - Macro-averaged ROC-AUC (multi-label): average roc_auc_score per label
    """
    logits = p.predictions
    labels = p.label_ids

    # For multi-label: apply sigmoid to logits to get probabilities
    probs = expit(logits)  # shape same as logits
    # Binarize predictions using 0.5 threshold for F1
    y_pred = (probs >= 0.5).astype(int)
    y_true = labels.astype(int)

    results: Dict[str, float] = {}

    # Macro F1
    try:
        mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception as e:
        # In rare cases (e.g., degenerate labels), fallback
        mac_f1 = float("nan")

    results["macro_f1"] = float(mac_f1)

    # Macro ROC-AUC: roc_auc_score supports multilabel with shape (#samples, #labels)
    # But it fails if a label has only one class present in y_true.
    # We'll compute roc_auc per label where possible and average those.
    per_label_roc = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        col_prob = probs[:, i]
        unique_vals = np.unique(col_true)
        if len(unique_vals) == 1:
            # ROC AUC is undefined when only one class present. Skip this label.
            continue
        try:
            score = roc_auc_score(col_true, col_prob)
            per_label_roc.append(score)
        except Exception:
            # Skip problematic labels
            continue

    if len(per_label_roc) == 0:
        results["macro_roc_auc"] = float("nan")
    else:
        results["macro_roc_auc"] = float(np.mean(per_label_roc))

    return results

# ----------------------------
# 6) TrainingArguments & Trainer
# ----------------------------
output_dir = "./multi_label_demo_output"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
    load_best_model_at_end=False,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ----------------------------
# 7) Train (small demo)
# ----------------------------
print("===== Starting training (demo) =====")
train_result = trainer.train()
print("===== Training complete =====")

# ----------------------------
# 8) Evaluate and print results
# ----------------------------
print("===== Running evaluation on validation set =====")
metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])

# We expect metrics to contain "eval_macro_f1" and "eval_macro_roc_auc"
# but Trainer prefixes metric names with "eval_" automatically
macro_f1 = metrics.get("eval_macro_f1", metrics.get("macro_f1", float("nan")))
macro_roc_auc = metrics.get("eval_macro_roc_auc", metrics.get("macro_roc_auc", float("nan")))

print("\n=== Final evaluation results ===")
print(f"Macro-Averaged F1-Score : {macro_f1:.6f}")
print(f"Macro-Averaged ROC-AUC  : {macro_roc_auc:.6f}")
print("\nFull metrics dict:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ----------------------------
# 9) Save model & tokenizer (optional)
# ----------------------------
print(f"\nSaving model & tokenizer to {output_dir} ...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nDemo complete. You can now inspect the output directory for the trained artifacts.")
