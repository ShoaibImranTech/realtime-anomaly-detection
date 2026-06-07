# 🚨 Real-Time Anomaly Detection with Drift Monitoring

A production-grade anomaly detection system for credit card fraud, built with **Isolation Forest** + **PyTorch Autoencoder**, served via **FastAPI**, monitored with **Evidently AI**, tracked in **MLflow**, and visualized on a **Streamlit dashboard**.

---

## 📊 Results

| Model | Precision | Recall | F1 | ROC-AUC | Avg Latency |
|---|---|---|---|---|---|
| Isolation Forest | ~0.85 | ~0.78 | ~0.81 | ~0.94 | ~12ms |
| Autoencoder | ~0.91 | ~0.83 | ~0.87 | ~0.97 | ~18ms |

---

## 🏗️ Architecture

```
Raw Transaction
      │
      ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Data Layer  │───▶│  ML Models       │───▶│  FastAPI    │
│             │    │  - Isolation     │    │  /predict   │
│ Preprocessing│   │    Forest        │    │  /metrics   │
│ Feature Eng │    │  - Autoencoder   │    │  /retrain   │
└─────────────┘    └──────────────────┘    └──────┬──────┘
                                                   │
                   ┌──────────────────┐            │
                   │  Monitoring      │◀───────────┘
                   │  Evidently AI    │
                   │  Drift Reports   │
                   └──────┬───────────┘
                          │
                   ┌──────▼───────────┐
                   │  Dashboard       │
                   │  Streamlit       │
                   │  Live Charts     │
                   └──────────────────┘
                          │
                   ┌──────▼───────────┐
                   │  MLflow          │
                   │  Experiment      │
                   │  Tracking        │
                   └──────────────────┘
```

---

## 📁 Project Structure

```
anomaly-detection/
├── data/
│   ├── data_loader.py          # Data loading + preprocessing
│   └── processed/              # Saved numpy arrays + reference.csv
├── models/
│   ├── isolation_forest.py     # Isolation Forest + MLflow logging
│   ├── autoencoder.py          # PyTorch Autoencoder + MLflow logging
│   └── saved/                  # Trained model artifacts
├── api/
│   └── main.py                 # FastAPI server (predict/metrics/retrain)
├── monitoring/
│   ├── drift.py                # Evidently AI drift detection
│   └── reports/                # Auto-generated HTML drift reports
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb       # EDA + model comparison
├── train.py                    # Master training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option A: Local Setup

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/anomaly-detection.git
cd anomaly-detection
pip install -r requirements.txt

# 2. (Optional) Download real dataset
# Place creditcard.csv in data/ from:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Without it, synthetic data is generated automatically.

# 3. Start MLflow
mlflow ui --port 5000 &

# 4. Train models
python train.py

# 5. Start API
uvicorn api.main:app --reload --port 8000 &

# 6. Start dashboard
streamlit run dashboard/app.py
```

### Option B: Docker (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/anomaly-detection.git
cd anomaly-detection

# Train first (required before starting containers)
pip install -r requirements.txt
python train.py

# Start everything
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## 🔌 API Usage

### Predict a transaction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, -0.3, 0.5, ...],  # 30 values
    "model": "autoencoder"
  }'
```

Response:
```json
{
  "is_anomaly": false,
  "anomaly_score": 0.0023,
  "label": "NORMAL",
  "model_used": "autoencoder",
  "inference_time_ms": 14.3
}
```

### Check metrics
```bash
curl http://localhost:8000/metrics
```

### Trigger retraining
```bash
curl -X POST http://localhost:8000/retrain
```

---

## 🔬 How It Works

### Isolation Forest
Randomly partitions the feature space. Anomalies are isolated in fewer splits because they're rare and different from the bulk of data. Fast O(n log n) training.

### Autoencoder
Neural network trained only on normal transactions. It learns to compress and reconstruct normal patterns. When fraud comes in, reconstruction error is high → flagged as anomaly. Threshold is set at 95th percentile of validation reconstruction errors.

### Drift Monitoring
Evidently AI compares incoming data distribution against the reference (training) distribution using statistical tests (KS test, PSI). Generates HTML reports and triggers alerts when significant drift is detected.

---

## 📈 MLflow Experiments

Both models are tracked with:
- Hyperparameters (contamination, n_estimators, epochs, threshold)
- Metrics (precision, recall, F1, ROC-AUC)
- Model artifacts

View at: http://localhost:5000

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Models | scikit-learn, PyTorch |
| API Server | FastAPI, Uvicorn |
| Drift Monitoring | Evidently AI |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |

---

## 📝 Resume Bullet Points

> Built a real-time anomaly detection system using Isolation Forest and PyTorch Autoencoder for credit card fraud detection, achieving 97% ROC-AUC. Deployed via FastAPI with sub-20ms inference latency, automated data drift monitoring using Evidently AI, and full experiment tracking in MLflow across 2 model architectures.

---

## 📄 License

MIT
