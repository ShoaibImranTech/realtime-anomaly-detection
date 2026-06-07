"""
api/main.py
-----------
FastAPI server with 3 endpoints:
  POST /predict        — score a single transaction
  POST /predict/batch  — score multiple transactions
  GET  /metrics        — model performance stats
  POST /retrain        — trigger retraining on new data
  GET  /health         — health check
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import json

from models.isolation_forest import IsolationForestModel
from models.autoencoder import AutoencoderModel

# ── App Setup ───────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Anomaly Detection API",
    description="Credit card fraud detection using Isolation Forest & Autoencoder",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ──────────────────────────────────────────────────
IF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "isolation_forest.pkl")
AE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "autoencoder.pt")
AE_THRESH_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "ae_threshold.json")

models = {}

def load_models():
    global models
    try:
        models["isolation_forest"] = IsolationForestModel.load(IF_MODEL_PATH)
        print("✅ Isolation Forest loaded")
    except Exception as e:
        print(f"⚠️  Could not load Isolation Forest: {e}")

    try:
        models["autoencoder"] = AutoencoderModel.load(AE_MODEL_PATH, AE_THRESH_PATH)
        print("✅ Autoencoder loaded")
    except Exception as e:
        print(f"⚠️  Could not load Autoencoder: {e}")

load_models()

# ── In-memory stats tracker ──────────────────────────────────────
stats = {
    "total_predictions": 0,
    "total_anomalies": 0,
    "latencies_ms": [],
}

# ── Schemas ──────────────────────────────────────────────────────
class Transaction(BaseModel):
    features: List[float] = Field(..., description="30 features: V1-V28 + Amount + Time")
    model: Optional[str] = Field("autoencoder", description="'autoencoder' or 'isolation_forest'")

class BatchTransactions(BaseModel):
    transactions: List[List[float]]
    model: Optional[str] = "autoencoder"

class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    label: str
    model_used: str
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    anomalies_detected: int


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "total_predictions": stats["total_predictions"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model_name = transaction.model
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded. Train first.")

    model = models[model_name]
    X = np.array(transaction.features).reshape(1, -1)

    start = time.perf_counter()
    anomaly_score = float(model.score(X)[0])
    prediction = int(model.predict(X)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    # Update stats
    stats["total_predictions"] += 1
    stats["latencies_ms"].append(latency_ms)
    if prediction == 1:
        stats["total_anomalies"] += 1

    return PredictionResponse(
        is_anomaly=bool(prediction),
        anomaly_score=round(anomaly_score, 6),
        label="FRAUD" if prediction == 1 else "NORMAL",
        model_used=model_name,
        inference_time_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchTransactions):
    model_name = batch.model
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded.")

    model = models[model_name]
    X = np.array(batch.transactions)

    start = time.perf_counter()
    scores = model.score(X)
    predictions = model.predict(X)
    latency_ms = (time.perf_counter() - start) * 1000

    results = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        results.append(PredictionResponse(
            is_anomaly=bool(pred),
            anomaly_score=round(float(score), 6),
            label="FRAUD" if pred == 1 else "NORMAL",
            model_used=model_name,
            inference_time_ms=round(latency_ms / len(predictions), 3),
        ))

    stats["total_predictions"] += len(predictions)
    stats["total_anomalies"] += int(predictions.sum())

    return BatchPredictionResponse(
        results=results,
        total=len(results),
        anomalies_detected=int(predictions.sum()),
    )


@app.get("/metrics")
def get_metrics():
    lats = stats["latencies_ms"]
    return {
        "total_predictions": stats["total_predictions"],
        "total_anomalies": stats["total_anomalies"],
        "anomaly_rate": round(
            stats["total_anomalies"] / max(stats["total_predictions"], 1), 4
        ),
        "avg_latency_ms": round(sum(lats) / max(len(lats), 1), 3),
        "p95_latency_ms": round(
            sorted(lats)[int(len(lats) * 0.95)] if lats else 0, 3
        ),
        "models_loaded": list(models.keys()),
    }


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """Triggers background retraining."""
    def _retrain():
        print("🔄 Background retraining started...")
        import subprocess
        subprocess.run(["python", "train.py"], check=True)
        load_models()
        print("✅ Retraining complete, models reloaded.")

    background_tasks.add_task(_retrain)
    return {"status": "Retraining started in background. Check logs."}
