import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"

MODEL_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Remove duplicate transactions
df = df.drop_duplicates()

features = [
    "transaction_amount",
    "txn_hour",
    "txn_frequency_24h",
    "merchant_category_risk",
    "account_age_days",
    "avg_txn_amount_30d"
]

target = "is_fraud"

X = df[features].copy()
y = df[target]

# Log transform amount-related variables
X["transaction_amount"] = __import__("numpy").log1p(
    X["transaction_amount"]
)

X["avg_txn_amount_30d"] = __import__("numpy").log1p(
    X["avg_txn_amount_30d"]
)

# Stratified 75/25 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Final Logistic Regression pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# Save complete model + feature information
artifact = {
    "model": pipeline,
    "features": features
}

joblib.dump(artifact, MODEL_PATH)

print("=" * 50)
print("MODEL TRAINING COMPLETE")
print("=" * 50)
print(f"Rows used: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Features: {features}")
print(f"Model: Logistic Regression")
print(f"Saved to: {MODEL_PATH}")