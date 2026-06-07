"""
train.py
--------
Master training script. Trains both Isolation Forest and Autoencoder,
logs everything to MLflow, and prints a comparison summary.

Run: python train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mlflow

from data.data_loader import load_data, preprocess
from models.isolation_forest import train_and_log as train_if
from models.autoencoder import train_and_log as train_ae


def main():
    print("=" * 60)
    print("  🚀 Real-Time Anomaly Detection — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load & preprocess data ──────────────────────────
    print("\n📂 Step 1: Loading data...")
    df = load_data()
    X_train, X_val, X_test, y_test, features = preprocess(df)

    # ── Step 2: Train Isolation Forest ──────────────────────────
    print("\n📂 Step 2: Training Isolation Forest...")
    mlflow.set_tracking_uri("http://localhost:5000")
    if_model, if_metrics = train_if(X_train, X_test, y_test)

    # ── Step 3: Train Autoencoder ────────────────────────────────
    print("\n📂 Step 3: Training Autoencoder...")
    ae_model, ae_metrics = train_ae(X_train, X_val, X_test, y_test, epochs=50)

    # ── Step 4: Comparison Summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Isolation Forest':>18} {'Autoencoder':>14}")
    print("-" * 50)
    for key in ["precision", "recall", "f1", "roc_auc"]:
        print(f"{key:<15} {if_metrics[key]:>18.4f} {ae_metrics[key]:>14.4f}")
    print("=" * 60)

    # Recommend best model
    best = "Autoencoder" if ae_metrics["roc_auc"] > if_metrics["roc_auc"] else "Isolation Forest"
    print(f"\n🏆 Best model by ROC-AUC: {best}")
    print("\n✅ Training complete! Models saved in models/saved/")
    print("   Run MLflow UI: mlflow ui --port 5000")
    print("   Run API:       uvicorn api.main:app --reload")
    print("   Run Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
