"""
Train baseline ML models on 10,000 support tickets dataset.
Targets:
1. ticket_category (7 classes)
2. priority (High, Medium, Low)
3. sentiment (5 classes)
4. escalation_risk (Yes vs No)
"""

import json
import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

DATA_PATH = "data/support_tickets_10k.csv"
MODEL_DIR = "models"


def clean_text(text: str) -> str:
    """Normalize and clean text input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"#ord\d+", " ORDER_REF ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records.")

    # Preprocessing
    df["cleaned_text"] = df["ticket_text"].apply(clean_text)

    # Train / Test split (8,000 train / 2,000 test)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["ticket_category"]
    )
    print(f"Train split: {len(train_df)} | Test split: {len(test_df)}")

    metrics_summary = {}

    targets = [
        {
            "name": "ticket_category",
            "column": "ticket_category",
            "file": "category_model.joblib",
        },
        {
            "name": "priority",
            "column": "priority",
            "file": "priority_model.joblib",
        },
        {
            "name": "sentiment",
            "column": "sentiment",
            "file": "sentiment_model.joblib",
        },
        {
            "name": "escalation_risk",
            "column": "escalated",
            "file": "escalation_model.joblib",
        },
    ]

    for target in targets:
        name = target["name"]
        col = target["column"]
        file_path = os.path.join(MODEL_DIR, target["file"])

        print(f"\n{'='*20} Training: {name} {'='*20}")
        X_train = train_df["cleaned_text"]
        y_train = train_df[col]
        X_test = test_df["cleaned_text"]
        y_test = test_df[col]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy:        {acc:.4f}")
        print(f"Macro F1-Score:  {f1_macro:.4f}")
        print(f"Weighted F1:     {f1_weighted:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Save model pipeline
        joblib.dump(pipeline, file_path)
        print(f"Saved {name} model to {file_path}")

        metrics_summary[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_f1": round(f1_weighted, 4),
            "classes": list(pipeline.classes_),
        }

    # Save metrics JSON
    metrics_path = os.path.join(MODEL_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved overall baseline metrics summary to {metrics_path}")


if __name__ == "__main__":
    train_and_evaluate()
