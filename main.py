# --- Vertex AI Setup ---
from google.cloud import aiplatform

# Project info
PROJECT_ID = "gen-lang-client-0761066410"
REGION = "us-central1"
BUCKET_URI = "gs://21f3000376-week2"

# Init Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

# --- Libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

# --- Artifact paths ---
MODEL_ARTIFACT_DIR = "models/iris-data-week-2"
MODEL_DISPLAY_NAME = "iris-dt"

# --- Load dataset ---
iris_df = pd.read_csv("data/iris.csv")
print(f"Loaded dataset with {iris_df.shape[0]} rows")

# --- Train/test split ---
train_df, test_df = train_test_split(
    iris_df, test_size=0.3, stratify=iris_df["species"], random_state=123
)

X_train = train_df.drop(columns=["species"])
y_train = train_df["species"]
X_test = test_df.drop(columns=["species"])
y_test = test_df["species"]

# --- Train Decision Tree model ---
clf = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=123)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Decision Tree accuracy: {accuracy:.3f}")

# Save metrics for reproducibility
metrics_df = pd.DataFrame([{"accuracy": accuracy}])
metrics_df.to_csv("metrics.csv", index=False)

# --- Save trained model ---
os.makedirs("artifacts", exist_ok=True)
joblib.dump(clf, "artifacts/model.joblib")

print("Model saved to artifacts/model.joblib")
