<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=FF4444&center=true&vCenter=true&width=700&lines=Real-Time+Anomaly+Detection;Credit+Card+Fraud+Detection+System;Production+MLOps+Pipeline" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

<br/>

> **Production-grade fraud detection system** combining Isolation Forest and PyTorch Autoencoder,
> served at sub-20ms latency via FastAPI, with automated drift monitoring and full MLOps observability.

<br/>

| 🎯 97% ROC-AUC | ⚡ <20ms Latency | 🔬 2 Model Architectures | 📊 Live Drift Monitoring |
|:---:|:---:|:---:|:---:|

</div>

---

## 🧠 Overview

This system detects anomalous credit card transactions in real time using two complementary ML approaches trained exclusively on normal transaction patterns. When a transaction deviates from learned behavior, it is flagged as fraud — with inference completing in under 20 milliseconds.

The full stack covers **data ingestion → model training → REST API serving → drift monitoring → visual analytics** — a complete production ML pipeline from raw data to live dashboard.

---

## 📊 Model Performance

| Model | Precision | Recall | F1 Score | ROC-AUC | Avg Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| Isolation Forest | 0.85 | 0.78 | 0.81 | 0.94 | ~12ms |
| **Autoencoder (Best)** | **0.91** | **0.83** | **0.87** | **0.97** | ~18ms |

> Autoencoder threshold calibrated at the **95th percentile** of validation reconstruction errors, improving recall by ~5% over naive thresholding.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INCOMING TRANSACTION                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│   StandardScaler  ·  Feature Engineering  ·  Train/Test Split   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────┐           ┌──────────────────────┐
│   ISOLATION FOREST   │           │     AUTOENCODER       │
│   Scikit-learn       │           │     PyTorch           │
│   ROC-AUC: 0.94      │           │     ROC-AUC: 0.97     │
└──────────┬───────────┘           └──────────┬───────────┘
           └───────────────┬───────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI SERVER                            │
│        /predict  ·  /predict/batch  ·  /metrics  ·  /retrain   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────┐           ┌──────────────────────┐
│   EVIDENTLY AI       │           │       MLFLOW          │
│   Drift Monitoring   │           │   Experiment Tracker  │
│   KS Test · PSI      │           │   Params · Metrics    │
│   HTML Reports       │           │   Model Artifacts     │
└──────────────────────┘           └──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                          │
│     Live Scoring  ·  Anomaly Charts  ·  Drift Reports  ·  Batch │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
realtime-anomaly-detection/
│
├── 📂 data/
│   ├── data_loader.py          # Data ingestion, scaling, train/test split
│   └── processed/              # Preprocessed numpy arrays + drift reference CSV
│
├── 📂 models/
│   ├── isolation_forest.py     # Isolation Forest with MLflow experiment logging
│   ├── autoencoder.py          # PyTorch Autoencoder with threshold calibration
│   └── saved/                  # Serialized model artifacts (.pkl / .pt)
│
├── 📂 api/
│   └── main.py                 # FastAPI — /predict, /batch, /metrics, /retrain
│
├── 📂 monitoring/
│   ├── drift.py                # Evidently AI — KS test, PSI, HTML report generation
│   └── reports/                # Auto-generated drift reports (timestamped HTML)
│
├── 📂 dashboard/
│   └── app.py                  # Streamlit — live scoring, charts, drift UI, batch upload
│
├── 📂 notebooks/
│   └── exploration.ipynb       # EDA, ROC curves, reconstruction error analysis
│
├── train.py                    # Master pipeline — trains both models, prints comparison
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option A — Local

```bash
# Clone
git clone https://github.com/ShoaibImranTech/realtime-anomaly-detection.git
cd realtime-anomaly-detection

# Install
pip install -r requirements.txt

# (Optional) Add real dataset
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it in data/creditcard.csv
# Without it, realistic synthetic data is generated automatically

# Train both models
python train.py

# Terminal 1 — MLflow
mlflow ui --port 5000

# Terminal 2 — API
uvicorn api.main:app --reload --port 8000

# Terminal 3 — Dashboard
streamlit run dashboard/app.py
```

### Option B — Docker *(Recommended)*

```bash
git clone https://github.com/ShoaibImranTech/realtime-anomaly-detection.git
cd realtime-anomaly-detection
pip install -r requirements.txt && python train.py
docker-compose up --build
```

| Service | URL |
|:---|:---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI REST API | http://localhost:8000 |
| Interactive API Docs | http://localhost:8000/docs |
| MLflow Experiment UI | http://localhost:5000 |

---

## 🔌 API Reference

**`POST /predict`** — Score a single transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.3, 0.5, ...], "model": "autoencoder"}'
```

```json
{
  "is_anomaly": false,
  "anomaly_score": 0.0023,
  "label": "NORMAL",
  "model_used": "autoencoder",
  "inference_time_ms": 14.3
}
```

**`POST /predict/batch`** — Score multiple transactions simultaneously

**`GET /metrics`** — Live model performance and latency statistics

**`POST /retrain`** — Trigger background model retraining on new data

---

## 🔬 How It Works

**Isolation Forest** partitions the feature space randomly. Anomalous transactions — being rare and statistically different — get isolated in fewer partitions, producing a high anomaly score. Time complexity: O(n log n).

**Autoencoder** is a neural network trained exclusively on normal transactions. It learns a compressed latent representation of normal behavior. When a fraudulent transaction is passed through, the network fails to reconstruct it accurately — the resulting high reconstruction error triggers the anomaly flag. The detection threshold is set at the 95th percentile of validation reconstruction errors.

**Drift Monitoring** via Evidently AI continuously compares incoming production data against the training distribution using the Kolmogorov-Smirnov test and Population Stability Index (PSI). Significant distribution shifts generate timestamped HTML reports and alert the system to potential model degradation.

---

## 📈 MLflow Experiment Tracking

Both models are logged with full reproducibility:

- **Parameters** — contamination, n_estimators, epochs, threshold, learning rate
- **Metrics** — precision, recall, F1, ROC-AUC, training loss
- **Artifacts** — serialized model files, training curves

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| ML Models | Scikit-learn · PyTorch |
| API Server | FastAPI · Uvicorn |
| Drift Monitoring | Evidently AI |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit · Plotly |
| Containerization | Docker · Docker Compose |
| Language | Python 3.11 |

---

## 📄 License

MIT © [Muhammad Shoaib](https://github.com/ShoaibImranTech)
