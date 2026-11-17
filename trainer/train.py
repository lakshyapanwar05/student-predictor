import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.environ.get("DATA_PATH", "data/students.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
FEATURES = ["study_hours", "attendance", "previous_marks"]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]),
    "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracies[name] = round(accuracy_score(y_test, model.predict(X_test)), 4)
    print(f"{name}: accuracy = {accuracies[name]}")

best_name = max(accuracies, key=accuracies.get)
best_model = models[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_model.predict(X_test), target_names=["fail", "pass"]))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({
        "best_model": best_name,
        "accuracies": accuracies,
        "features": FEATURES,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }, f, indent=2)
print(f"Saved model and metrics to {MODEL_DIR}")
