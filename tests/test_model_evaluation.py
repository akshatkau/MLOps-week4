import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_model_accuracy_above_threshold():
    """Check if model accuracy is at least 85% on test data."""
    df = pd.read_csv("data/iris.csv")
    model = joblib.load("artifacts/model.joblib")

    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.85, f"Model accuracy too low: {accuracy:.2f}"
