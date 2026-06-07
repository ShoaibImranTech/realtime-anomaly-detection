"""
dashboard/app.py
-----------------
Streamlit dashboard showing:
  1. Live transaction scoring
  2. Real-time anomaly rate chart
  3. Model metrics
  4. Drift monitoring results
  5. Batch upload & scan

Run: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
import json
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200, r.json()
    except:
        return False, {}

def predict_transaction(features, model="autoencoder"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"features": features, "model": model}, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_api_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=2)
        return r.json()
    except:
        return {}

def generate_random_transaction(is_fraud=False):
    """Generate a random transaction feature vector."""
    if is_fraud:
        features = list(np.random.randn(28) * 2.5 + np.random.choice([-3, 3], size=28))
        features += [abs(np.random.exponential(300)), np.random.uniform(0, 172800)]
    else:
        features = list(np.random.randn(28) * 0.8)
        features += [abs(np.random.exponential(50)), np.random.uniform(0, 172800)]
    return features


# ── Session state init ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "anomaly_counts" not in st.session_state:
    st.session_state.anomaly_counts = []


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    api_ok, health_data = check_api()
    if api_ok:
        st.success("✅ API Connected")
        st.caption(f"Models: {', '.join(health_data.get('models_loaded', []))}")
    else:
        st.error("❌ API Offline — Run: uvicorn api.main:app --reload")

    selected_model = st.selectbox("Model", ["autoencoder", "isolation_forest"])

    st.divider()
    st.subheader("Simulate Transaction")
    simulate_mode = st.radio("Type", ["Random Normal", "Random Fraud", "Manual Input"])

    if simulate_mode == "Manual Input":
        st.caption("Enter 30 comma-separated feature values")
        raw_input = st.text_area("Features", value=",".join(["0.0"] * 30))

    if st.button("🔍 Run Prediction", type="primary"):
        if simulate_mode == "Random Normal":
            features = generate_random_transaction(is_fraud=False)
        elif simulate_mode == "Random Fraud":
            features = generate_random_transaction(is_fraud=True)
        else:
            try:
                features = [float(x.strip()) for x in raw_input.split(",")]
            except:
                st.error("Invalid input")
                features = None

        if features and api_ok:
            result = predict_transaction(features, model=selected_model)
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "label": result.get("label", "?"),
                "score": result.get("anomaly_score", 0),
                "latency": result.get("inference_time_ms", 0),
                "is_anomaly": result.get("is_anomaly", False),
            })
        elif not api_ok:
            st.warning("API not running. Start it first.")

    st.divider()
    if st.button("🔄 Trigger Retrain"):
        try:
            r = requests.post(f"{API_URL}/retrain")
            st.info(r.json().get("status", "Retrain triggered"))
        except:
            st.error("Could not reach API")


# ── Main Dashboard ────────────────────────────────────────────────
st.title("🚨 Real-Time Anomaly Detection Dashboard")
st.caption("Credit Card Fraud Detection — Isolation Forest + Autoencoder")

# ── Row 1: Key Metrics ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

metrics = get_api_metrics()
history = st.session_state.history
n_anomalies = sum(1 for h in history if h["is_anomaly"])

with col1:
    st.metric("Total Predictions", metrics.get("total_predictions", len(history)))
with col2:
    st.metric("Anomalies Detected", metrics.get("total_anomalies", n_anomalies))
with col3:
    rate = metrics.get("anomaly_rate", n_anomalies / max(len(history), 1))
    st.metric("Anomaly Rate", f"{rate:.1%}")
with col4:
    st.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms")

st.divider()

# ── Row 2: Live Transaction Log + Score Chart ─────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Live Transaction Log")
    if history:
        df_hist = pd.DataFrame(history[-20:][::-1])
        df_hist["status"] = df_hist["label"].apply(
            lambda x: "🚨 " + x if x == "FRAUD" else "✅ " + x
        )
        st.dataframe(
            df_hist[["time", "status", "score", "latency"]].rename(columns={
                "time": "Time", "status": "Result", "score": "Anomaly Score", "latency": "Latency (ms)"
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions yet. Use the sidebar to run predictions.")

with col_right:
    st.subheader("📈 Anomaly Score Over Time")
    if len(history) > 1:
        df_chart = pd.DataFrame(history[-50:])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart["time"], y=df_chart["score"],
            mode="lines+markers",
            marker=dict(
                color=["red" if a else "green" for a in df_chart["is_anomaly"]],
                size=8,
            ),
            line=dict(color="gray", width=1),
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run more predictions to see the chart.")

st.divider()

# ── Row 3: Drift Monitoring ───────────────────────────────────────
st.subheader("🔬 Drift Monitoring")

drift_col1, drift_col2 = st.columns([1, 1])

with drift_col1:
    st.markdown("Upload a CSV of recent transactions to check for data drift vs training distribution.")
    uploaded = st.file_uploader("Upload current data (CSV)", type="csv")

    if uploaded:
        current_df = pd.read_csv(uploaded)
        st.caption(f"Uploaded {len(current_df)} rows, {current_df.shape[1]} columns")

        if st.button("🔎 Run Drift Analysis"):
            with st.spinner("Running Evidently drift report..."):
                try:
                    from monitoring.drift import DriftMonitor
                    monitor = DriftMonitor()
                    summary = monitor.run_drift_report(current_df, save_html=True)

                    if summary["dataset_drift_detected"]:
                        st.error(f"🚨 DRIFT DETECTED — {summary['n_drifted_columns']}/{summary['total_columns']} columns drifted")
                    else:
                        st.success(f"✅ No drift detected ({summary['share_of_drifted_columns']:.1%} drift share)")

                    st.json(summary)
                    if "report_path" in summary:
                        st.caption(f"Full report: {summary['report_path']}")
                except Exception as e:
                    st.error(f"Drift analysis failed: {e}. Run train.py first to generate reference data.")

with drift_col2:
    st.markdown("**Simulate drift to test monitoring:**")
    drift_strength = st.slider("Drift Strength", 0.0, 5.0, 2.0, 0.5)
    n_sim = st.number_input("Simulated Samples", 100, 5000, 500, step=100)

    if st.button("⚡ Simulate & Detect Drift"):
        with st.spinner("Simulating..."):
            try:
                from monitoring.drift import DriftMonitor
                monitor = DriftMonitor()
                sim_data = monitor.simulate_drift(n_samples=int(n_sim), drift_strength=drift_strength)
                summary = monitor.run_drift_report(sim_data, save_html=True)

                if summary["dataset_drift_detected"]:
                    st.error(f"🚨 Drift detected as expected! {summary['n_drifted_columns']} drifted columns.")
                else:
                    st.warning("No drift detected with this strength level.")
                st.json(summary)
            except Exception as e:
                st.error(f"Error: {e}. Make sure train.py has been run first.")

st.divider()

# ── Row 4: Batch Upload & Score ───────────────────────────────────
st.subheader("📦 Batch Prediction")

batch_file = st.file_uploader("Upload transactions CSV (no Class column)", type="csv", key="batch")
if batch_file and api_ok:
    batch_df = pd.read_csv(batch_file)
    st.caption(f"{len(batch_df)} transactions loaded")

    if st.button("🚀 Score Batch"):
        with st.spinner("Scoring..."):
            payload = {"transactions": batch_df.values.tolist(), "model": selected_model}
            try:
                r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=30)
                result = r.json()
                results_df = pd.DataFrame(result["results"])
                st.success(f"✅ {result['anomalies_detected']} anomalies detected out of {result['total']}")
                st.dataframe(results_df, use_container_width=True)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button("⬇️ Download Results", csv, "anomaly_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("Built with FastAPI · PyTorch · Scikit-learn · Evidently AI · MLflow · Streamlit")
