"""
autoencoder.py
--------------
PyTorch Autoencoder for anomaly detection.
Trained on normal data only. High reconstruction error = anomaly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import os
import json
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "autoencoder.pt")
THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "saved", "ae_threshold.json")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        return error


class AutoencoderModel:
    def __init__(self, input_dim: int = 30, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim).to(self.device)
        self.threshold = None

    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        """Train autoencoder on normal data only."""
        print(f"🤖 Training Autoencoder on {self.device}...")

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        history = []

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

        # Set anomaly threshold from validation set (95th percentile of reconstruction error)
        self.model.eval()
        if X_val is not None:
            val_tensor = torch.FloatTensor(X_val).to(self.device)
            errors = self.model.reconstruction_error(val_tensor).cpu().numpy()
            self.threshold = float(np.percentile(errors, 95))
            print(f"✅ Anomaly threshold set at 95th percentile: {self.threshold:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1=anomaly, 0=normal based on threshold."""
        assert self.threshold is not None, "Train model first to set threshold."
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        errors = self.model.reconstruction_error(X_tensor).cpu().numpy()
        return (errors > self.threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns raw reconstruction errors (anomaly scores)."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self.model.reconstruction_error(X_tensor).cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        scores = self.score(X_test)
        metrics = {
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
        }
        print("\n📊 Autoencoder Evaluation:")
        print(classification_report(y_test, preds, target_names=["Normal", "Fraud"]))
        return metrics

    def save(self, model_path: str = MODEL_PATH, threshold_path: str = THRESHOLD_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        with open(threshold_path, "w") as f:
            json.dump({"threshold": self.threshold, "input_dim": self.input_dim}, f)
        print(f"💾 Autoencoder saved to {model_path}")

    @classmethod
    def load(cls, model_path: str = MODEL_PATH, threshold_path: str = THRESHOLD_PATH):
        with open(threshold_path, "r") as f:
            meta = json.load(f)
        instance = cls(input_dim=meta["input_dim"])
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        instance.threshold = meta["threshold"]
        print(f"✅ Autoencoder loaded from {model_path}")
        return instance


def train_and_log(X_train, X_val, X_test, y_test, epochs=50):
    """Train autoencoder with MLflow tracking."""
    mlflow.set_experiment("anomaly-detection")

    with mlflow.start_run(run_name="Autoencoder"):
        input_dim = X_train.shape[1]
        ae = AutoencoderModel(input_dim=input_dim, epochs=epochs)
        history = ae.train(X_train, X_val)

        metrics = ae.evaluate(X_test, y_test)

        mlflow.log_param("model_type", "Autoencoder")
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("threshold", ae.threshold)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("final_train_loss", history[-1])

        ae.save()
        print(f"📝 MLflow run logged | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return ae, metrics
