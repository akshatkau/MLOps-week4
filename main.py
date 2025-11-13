import argparse
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# -------------------------------
# Parse arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to iris CSV file")
args = parser.parse_args()

data_path = args.data
print(f"Using dataset: {data_path}")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(data_path)
print(f"Loaded dataset with {df.shape[0]} rows")

# -------------------------------
# Prepare training data
# -------------------------------
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df["species"], random_state=123
)

X_train = train_df.drop(columns=["species"])
y_train = train_df["species"]
X_test = test_df.drop(columns=["species"])
y_test = test_df["species"]

# -------------------------------
# MLflow Tracking
# -------------------------------
mlflow.set_experiment("Iris-Data-Poisoning-Week8")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("dataset", data_path)
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("criterion", "entropy")

    # Train model
    clf = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=123)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save metrics to file
    metrics_df = pd.DataFrame([{"accuracy": accuracy}])
    metrics_df.to_csv("metrics.csv", index=False)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.joblib"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    # Log model to MLflow
    mlflow.log_artifact(model_path)
    mlflow.log_artifact("metrics.csv")
