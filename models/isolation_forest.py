"""
isolation_forest.py
--------------------
Isolation Forest model for anomaly detection.
Trains, evaluates, and saves the model with MLflow tracking.
"""

import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "isolation_forest.pkl")


class IsolationForestModel:
    def __init__(self, contamination: float = 0.02, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(self, X_train: np.ndarray):
        """Train on normal data only."""
        print("🌲 Training Isolation Forest...")
        self.model.fit(X_train)
        print("✅ Isolation Forest trained.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal."""
        raw = self.model.predict(X)          # sklearn: 1=normal, -1=anomaly
        return np.where(raw == -1, 1, 0)    # convert to 1=anomaly, 0=normal

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly scores (higher = more anomalous)."""
        return -self.model.score_samples(X)  # negate so higher = more anomalous

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate and return metrics dict."""
        preds = self.predict(X_test)
        scores = self.score(X_test)
        metrics = {
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
        }
        print("\n📊 Isolation Forest Evaluation:")
        print(classification_report(y_test, preds, target_names=["Normal", "Fraud"]))
        return metrics

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH):
        instance = cls()
        instance.model = joblib.load(path)
        print(f"✅ Isolation Forest loaded from {path}")
        return instance


def train_and_log(X_train, X_test, y_test, contamination=0.02, n_estimators=100):
    """Train with MLflow tracking."""
    mlflow.set_experiment("anomaly-detection")

    with mlflow.start_run(run_name="IsolationForest"):
        model = IsolationForestModel(contamination=contamination, n_estimators=n_estimators)
        model.train(X_train)

        metrics = model.evaluate(X_test, y_test)

        # Log params and metrics
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(model.model, "isolation_forest_model")

        model.save()
        print(f"📝 MLflow run logged | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return model, metrics
