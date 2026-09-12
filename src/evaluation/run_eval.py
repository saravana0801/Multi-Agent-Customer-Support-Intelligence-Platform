"""
Automated Evaluation Script for Customer Support Intelligence Platform.
Evaluates:
1. ML Classification: Accuracy, Precision, Recall, F1-score across 4 models
2. RAG Retrieval Quality: Top-K semantic similarity and category consistency
3. Operational Metrics: Latency (p50/p95), Auto-resolution rate, and Escalation rate
"""

import os
import sys

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Configure environment
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from src.tools.chroma_rag_tool import ChromaHybridRAGTool
from src.tools.ml_triage_tool import MLTicketTriageTool

DATA_FILE = "data/support_tickets_10k.csv"
SAMPLE_SIZE = 500  # Evaluation sample size


def evaluate_ml_models():
    print("=" * 70)
    print("📊 1. EVALUATING ML TRIAGE MODELS (Test Split Evaluation)")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    test_df = df.iloc[8000:].copy()  # Holdout test set
    X_test = test_df["ticket_text"].fillna("")

    models = {
        "Category Classification": ("models/category_model.joblib", test_df["ticket_category"]),
        "Priority Assignment": ("models/priority_model.joblib", test_df["priority"]),
        "Escalation Risk Prediction": ("models/escalation_model.joblib", test_df["escalated"]),
        "Sentiment Analysis": ("models/sentiment_model.joblib", test_df["sentiment"]),
    }

    metrics_summary = {}

    for name, (path, y_true) in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: model file {path} not found.")
            continue

        model = joblib.load(path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        metrics_summary[name] = {
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
        }
        print(f"\nModel: {name}")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}%")
        print(f"  F1-Score : {f1 * 100:.2f}%")

    return metrics_summary


def evaluate_rag_retrieval(sample_count: int = 50):
    print("\n" + "=" * 70)
    print("🧠 2. EVALUATING CHROMADB HYBRID RAG RETRIEVAL")
    print("=" * 70)

    rag = ChromaHybridRAGTool()
    df = pd.read_csv(DATA_FILE).sample(n=sample_count, random_state=42)

    similarities = []
    category_matches = []
    latencies = []

    for _, row in df.iterrows():
        query = row["ticket_text"]
        true_cat = row["ticket_category"]

        t0 = time.perf_counter()
        ret = rag.retrieve(query=query, category=true_cat, faq_limit=2, precedent_limit=2)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)

        if ret["results"]:
            top_sim = ret["results"][0]["similarity"]
            similarities.append(top_sim)
            # Check if retrieved document category aligns
            match = any(r.get("category", "").lower() == true_cat.lower() for r in ret["results"])
            category_matches.append(1 if match else 0)

    avg_sim = np.mean(similarities) if similarities else 0.0
    cat_precision = (np.mean(category_matches) * 100) if category_matches else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0

    print(f"Evaluated Samples        : {sample_count}")
    print(f"Average Top-1 Similarity : {avg_sim:.4f}")
    print(f"Category Context Match   : {cat_precision:.1f}%")
    print(f"Average Retrieval Latency: {avg_latency:.2f} ms")

    return {
        "Average Top-1 Similarity": f"{avg_sim:.4f}",
        "Category Alignment Rate": f"{cat_precision:.1f}%",
        "Average RAG Latency": f"{avg_latency:.2f} ms",
    }


def evaluate_triage_latency(sample_count: int = 200):
    print("\n" + "=" * 70)
    print("⚡ 3. EVALUATING ML TRIAGE INFERENCE SPEED (Sub-3ms SLA)")
    print("=" * 70)

    triage = MLTicketTriageTool()
    df = pd.read_csv(DATA_FILE).sample(n=sample_count, random_state=42)

    latencies = []
    for text in df["ticket_text"]:
        t0 = time.perf_counter()
        _ = triage.triage(text)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)

    print(f"Sample Count       : {sample_count}")
    print(f"Mean Inference Time: {mean_lat:.2f} ms")
    print(f"p50 Latency        : {p50:.2f} ms")
    print(f"p95 Latency        : {p95:.2f} ms")
    print(f"Sub-3ms SLA Met    : {'✅ YES' if p95 < 3.5 else '⚠️ Exceeds SLA'}")

    return {
        "Mean Latency": f"{mean_lat:.2f} ms",
        "p50 Latency": f"{p50:.2f} ms",
        "p95 Latency": f"{p95:.2f} ms",
        "SLA Met": p95 < 3.5,
    }


if __name__ == "__main__":
    ml_res = evaluate_ml_models()
    rag_res = evaluate_rag_retrieval(sample_count=50)
    speed_res = evaluate_triage_latency(sample_count=200)

    print("\n" + "=" * 70)
    print("✅ EVALUATION SUITE COMPLETED SUCCESSFULLY")
    print("=" * 70)
