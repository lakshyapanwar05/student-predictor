import json
import os

import joblib
import pandas as pd
import psycopg2
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
DB_HOST = os.environ.get("DB_HOST", "db")
DB_NAME = os.environ.get("DB_NAME", "predictions")
DB_USER = os.environ.get("DB_USER", "predictor")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "predictor")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Student Performance Predictor")
model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)
FEATURES = metrics["features"]

CREATE_SQL = """CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    study_hours REAL,
    attendance REAL,
    previous_marks REAL,
    passed BOOLEAN,
    probability REAL,
    created_at TIMESTAMP DEFAULT NOW())"""


class Student(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_marks: float = Field(ge=0, le=100)


def get_conn():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, connect_timeout=3,
    )


def log_prediction(s, passed, prob):
    try:
        conn = get_conn()
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_SQL)
            cur.execute(
                "INSERT INTO predictions (study_hours, attendance, previous_marks, passed, probability) "
                "VALUES (%s, %s, %s, %s, %s)",
                (s.study_hours, s.attendance, s.previous_marks, passed, prob),
            )
        conn.close()
        return True
    except Exception as e:
        print(f"DB logging failed: {e}", flush=True)
        return False


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
def predict(s: Student):
    X = pd.DataFrame([[s.study_hours, s.attendance, s.previous_marks]], columns=FEATURES)
    prob = float(model.predict_proba(X)[0][1])
    passed = prob >= 0.5
    logged = log_prediction(s, passed, prob)
    return {"passed": passed, "probability": round(prob, 4), "logged": logged}


@app.get("/history")
def history():
    try:
        conn = get_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT study_hours, attendance, previous_marks, passed, probability, created_at "
                "FROM predictions ORDER BY id DESC LIMIT 10"
            )
            rows = cur.fetchall()
        conn.close()
        return [
            {"study_hours": r[0], "attendance": r[1], "previous_marks": r[2],
             "passed": r[3], "probability": r[4], "created_at": str(r[5])}
            for r in rows
        ]
    except Exception:
        return []
