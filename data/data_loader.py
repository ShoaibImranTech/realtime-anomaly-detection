"""
data_loader.py
--------------
Downloads and preprocesses the Credit Card Fraud dataset.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Since Kaggle requires auth, we simulate a realistic dataset with the same
statistical properties if the CSV is not present.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "processed")


def generate_synthetic_data(n_samples: int = 10000, fraud_ratio: float = 0.02) -> pd.DataFrame:
    """
    Generates synthetic credit card transaction data that mirrors
    the real Kaggle dataset structure (28 PCA features + Amount + Time + Class).
    Use this if you don't have the real CSV.
    """
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Normal transactions — tightly clustered
    normal = np.random.randn(n_normal, 28) * 0.8
    normal_amount = np.abs(np.random.exponential(50, n_normal))
    normal_time = np.sort(np.random.uniform(0, 172800, n_normal))

    # Fraudulent transactions — shifted distribution (anomalous)
    fraud = np.random.randn(n_fraud, 28) * 2.5 + np.random.choice([-3, 3], size=(n_fraud, 28))
    fraud_amount = np.abs(np.random.exponential(200, n_fraud))
    fraud_time = np.random.uniform(0, 172800, n_fraud)

    normal_df = pd.DataFrame(normal, columns=[f"V{i}" for i in range(1, 29)])
    normal_df["Amount"] = normal_amount
    normal_df["Time"] = normal_time
    normal_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud, columns=[f"V{i}" for i in range(1, 29)])
    fraud_df["Amount"] = fraud_amount
    fraud_df["Time"] = fraud_time
    fraud_df["Class"] = 1

    df = pd.concat([normal_df, fraud_df], ignore_index=True).sample(frac=1, random_state=42)
    return df


def load_data() -> pd.DataFrame:
    """Load real CSV if available, otherwise generate synthetic data."""
    if os.path.exists(DATA_PATH):
        print("✅ Loading real creditcard.csv...")
        df = pd.read_csv(DATA_PATH)
    else:
        print("⚠️  creditcard.csv not found. Generating synthetic data...")
        df = generate_synthetic_data()
    return df


def preprocess(df: pd.DataFrame):
    """
    Preprocesses the dataframe:
    - Scales 'Amount' and 'Time'
    - Splits into train (normal only) / test (mixed) sets
    - Returns reference data for drift monitoring
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    features = [col for col in df.columns if col != "Class"]
    X = df[features].values
    y = df["Class"].values

    # Train only on normal data (unsupervised anomaly detection)
    X_normal = X[y == 0]

    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    # Test set: mix of normal + fraud
    X_test, y_test = X, y

    # Save processed arrays
    np.save(os.path.join(PROCESSED_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_PATH, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_PATH, "y_test.npy"), y_test)

    # Save reference dataset for Evidently drift monitoring
    reference_df = pd.DataFrame(X_train, columns=features)
    reference_df.to_csv(os.path.join(PROCESSED_PATH, "reference.csv"), index=False)

    print(f"✅ Preprocessing done.")
    print(f"   Train (normal): {X_train.shape}")
    print(f"   Val   (normal): {X_val.shape}")
    print(f"   Test  (mixed):  {X_test.shape} | Fraud: {y_test.sum()}")

    return X_train, X_val, X_test, y_test, features


if __name__ == "__main__":
    df = load_data()
    preprocess(df)
