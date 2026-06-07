"""
monitoring/drift.py
--------------------
Data drift monitoring using Evidently AI.
Compares reference (training) data vs incoming production data.
Generates HTML reports and returns drift metrics as JSON.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummary,
    ColumnDriftMetric,
)

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "reference.csv")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


class DriftMonitor:
    def __init__(self, reference_path: str = REFERENCE_PATH):
        if not os.path.exists(reference_path):
            raise FileNotFoundError(
                f"Reference data not found at {reference_path}. Run train.py first."
            )
        self.reference_df = pd.read_csv(reference_path)
        self.feature_cols = list(self.reference_df.columns)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        print(f"✅ DriftMonitor initialized with {len(self.reference_df)} reference rows.")

    def run_drift_report(self, current_data: pd.DataFrame, save_html: bool = True) -> dict:
        """
        Runs full drift analysis comparing reference vs current data.
        Returns a summary dict and optionally saves an HTML report.
        """
        # Align columns
        current_data = current_data[self.feature_cols]

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftPreset(),
        ])
        report.run(reference_data=self.reference_df, current_data=current_data)

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
        drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
        n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift_detected": drift_detected,
            "share_of_drifted_columns": round(drift_share, 4),
            "n_drifted_columns": n_drifted,
            "total_columns": len(self.feature_cols),
            "current_rows": len(current_data),
            "reference_rows": len(self.reference_df),
        }

        if save_html:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = os.path.join(REPORTS_DIR, f"drift_report_{ts}.html")
            report.save_html(html_path)
            summary["report_path"] = html_path
            print(f"📄 Drift report saved: {html_path}")

        # Print summary
        status = "🚨 DRIFT DETECTED" if drift_detected else "✅ No drift"
        print(f"\n{status} | Drifted columns: {n_drifted}/{len(self.feature_cols)} ({drift_share:.1%})")

        return summary

    def run_column_drift(self, current_data: pd.DataFrame, columns: list = None) -> dict:
        """Check drift for specific columns."""
        if columns is None:
            columns = self.feature_cols[:5]  # default: first 5 features

        current_data = current_data[self.feature_cols]
        metrics = [ColumnDriftMetric(column_name=col) for col in columns if col in self.feature_cols]

        report = Report(metrics=metrics)
        report.run(reference_data=self.reference_df, current_data=current_data)

        result = report.as_dict()
        col_results = {}
        for i, col in enumerate(columns):
            if i < len(result["metrics"]):
                col_result = result["metrics"][i]["result"]
                col_results[col] = {
                    "drift_detected": col_result.get("drift_detected", False),
                    "stattest": col_result.get("stattest_name", "unknown"),
                    "p_value": round(col_result.get("p_value", 1.0), 4),
                }
        return col_results

    def run_data_quality(self, current_data: pd.DataFrame) -> dict:
        """Check data quality issues in incoming data."""
        current_data = current_data[self.feature_cols]

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=self.reference_df, current_data=current_data)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(REPORTS_DIR, f"quality_report_{ts}.html")
        report.save_html(html_path)
        print(f"📄 Quality report saved: {html_path}")

        return {"report_path": html_path, "timestamp": datetime.now().isoformat()}

    def simulate_drift(self, n_samples: int = 1000, drift_strength: float = 3.0) -> pd.DataFrame:
        """
        Simulate drifted data for testing.
        Shifts feature distributions by drift_strength standard deviations.
        """
        drifted = self.reference_df.sample(n=n_samples, replace=True).copy()
        # Shift half the columns to simulate concept drift
        cols_to_drift = self.feature_cols[:len(self.feature_cols)//2]
        for col in cols_to_drift:
            drifted[col] = drifted[col] + drift_strength * drifted[col].std()
        print(f"⚠️  Simulated drift on {len(cols_to_drift)} columns with strength={drift_strength}")
        return drifted


# ── CLI Usage ────────────────────────────────────────────────────
if __name__ == "__main__":
    monitor = DriftMonitor()

    print("\n--- Testing with NO drift (reference sample) ---")
    clean_sample = monitor.reference_df.sample(500)
    summary = monitor.run_drift_report(clean_sample)
    print(json.dumps(summary, indent=2))

    print("\n--- Testing with SIMULATED drift ---")
    drifted_sample = monitor.simulate_drift(n_samples=500, drift_strength=3.0)
    summary = monitor.run_drift_report(drifted_sample)
    print(json.dumps(summary, indent=2))
